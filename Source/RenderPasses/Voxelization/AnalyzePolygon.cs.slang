#include "Math/LebedevQuadrature.slang"
#include "Math/Polygon.slang"
#include "Math/Triangle.slang"
#include "Voxel/VoxelGrid.slang"
#include "VoxelizationShared.slang"
import Utils.Sampling.UniformSampleGenerator;
import Scene.SceneTypes;
import Scene.Scene;
import Scene.Raster;
import Utils.Math.MathHelpers;
import Rendering.Materials.TexLODHelpers;
import Scene.Material.BasicMaterialData;

StructuredBuffer<Polygon> polygonBuffer;
StructuredBuffer<PolygonRange> polygonRangeBuffer;
StructuredBuffer<OctreeNode> octreeBuffer;
RWStructuredBuffer<VoxelData> gBuffer;
RWStructuredBuffer<Ellipsoid> pBuffer;
SamplerState sampler;

// 每面片状态用全局 scratch 缓冲（batch 内全局面片号 localHead+s 索引），面片数无上限
RWStructuredBuffer<uint> lobeOfBuffer;   // 每面片所属 lobe（Pass 1 写入，Pass 2 读取）

// 深度光栅化投影平面分辨率（复用 Projection.slang 的 PROJ_RES）
#define VIS_PROJ_RES PROJ_RES

cbuffer GridData
{
    float3 gridMin;
    float3 voxelSize;
    uint3 voxelCount;
};

cbuffer CB
{
    uint gBufferOffset;
    uint groupVoxelCount;
    uint sampleFrequency;
}

void Estimate(
    Ellipsoid e,
    PolygonRange range,
    inout SphericalFunc polygonFunc,
    inout SphericalFunc primitiveFunc,
    inout SphericalFunc totalFunc
)
{
    float3 cellMin = range.cellInt;
    float3 cellCenter = cellMin + 0.5f;
    for (uint i = 0; i < LEBEDEV_DIRECTION_COUNT; i += 2)
    {
        LebedevSample sample = LebedevSamples[i];
        float3 direction = sample.direction;
        Basis2 basis = orthonormal_basis(direction);
        uint hitCount = 0;
        for (uint j = 0; j < sampleFrequency; j++)
        {
            SamplingRay ray = rayToVoxel(Hammersley2D(j, sampleFrequency), direction, basis, cellCenter);
            float3 from = ray.origin;
            float3 to = from + 2 * ray.direction;
            float2 inOut = clip(cellMin, cellMin + 1.f, from, to);
            if (inOut.x >= 0)
            {
                float3 v = to - from;
                to = from + inOut.y * v - cellMin;
                from = from + inOut.x * v - cellMin;
                inOut = e.clip(from, to);
                if (inOut.x >= 0)
                    hitCount++;
            }
        }

        float weight = 8 * PI * sample.weight;
        float primitiveProjArea = PROJECT_CIRCLE_AREA * hitCount / (float)sampleFrequency;
        primitiveFunc.accumulate(weight * primitiveProjArea, direction);

        float totalProjArea = range.calcTotalProjArea(polygonBuffer, direction);
        float visibleProjArea = range.calcVisibleProjAreaRaster(polygonBuffer, direction);
        polygonFunc.accumulate(weight * visibleProjArea, direction);
        totalFunc.accumulate(weight * totalProjArea, direction);
    }
}

float4 sampleTextureArea(TextureHandle handle, float2 uv, float uvArea, float4 uniformValue)
{
    MaterialSystem ms = gScene.materials;
    TextureInfo info = ms.getTextureInfo(handle);
    float pixelArea = uvArea * info.width * info.height;
    float lodLevel = 0.5f * log2(max(pixelArea, 1));
    ITextureSampler lod = ExplicitLodTextureSampler(lodLevel);
    return ms.sampleTexture(handle, sampler, uv, uniformValue, lod);
}

// 构建三角形（TBN）与 UV 面片中心/面积（sampleNormal / sampleMaterial 共用）
void buildTriangleData(
    Polygon polygon, int3 cellInt, float nodeScale,
    out Triangle tri, out float2 center, out float area)
{
    uint3 vtxIndices = gScene.getIndices(polygon.triRef.meshID, polygon.triRef.triangleID);
    StaticVertexData[3] vertices = { gScene.getVertex(vtxIndices.x), gScene.getVertex(vtxIndices.y), gScene.getVertex(vtxIndices.z) };
    for (int i = 0; i < 3; i++)
    {
        tri.vertices[i] = (vertices[i].position - gridMin) / voxelSize; // 叶子体素坐标
        tri.uvs[i] = vertices[i].texCrd;
#if LERP_NORMAL
        tri.normals[i] = vertices[i].normal;
#else
        // buildTBN 的退化回退会读取 normals，out 参数下 `= {}` 初始化会被消除，故此处必须显式初始化
        tri.normals[i] = float3(0, 0, 1);
#endif
    }
    tri.buildTBN();

    float2 uvs[MAX_VERTEX_COUNT];
    for (uint i = 0; i < polygon.count; i++)
    {
        float3 leafVertex = polygon.vertices[i] * nodeScale;
        uvs[i] = tri.lerpUV(leafVertex);
    }
    area = 0;
    center = 0;
    for (uint i = 0; i < polygon.count; ++i)
    {
        float2 a = uvs[i];
        float2 b = uvs[(i + 1) % polygon.count];
        area += a.x * b.y - a.y * b.x;
        center += uvs[i];
    }
    center /= polygon.count;
    area = 0.5f * abs(area);
}

// Pass 1 用：仅算法线（不采样 baseColor/specular），用于定 lobe 与主视角法线
float3 sampleNormal(Polygon polygon, int3 cellInt, float nodeScale)
{
    Triangle tri = {};
    float2 center;
    float area;
    buildTriangleData(polygon, cellInt, nodeScale, tri, center, area);

#if LERP_NORMAL
    float dummy;
    float3 centroid = polygon.calcCentroid(dummy);
    float3 leafCentroid = centroid * nodeScale;
    return tri.lerpNormal(leafCentroid);
#else
    BasicMaterialData data = gScene.materials.getBasicMaterialData(polygon.triRef.materialID);
    float3 normal = sampleTextureArea(data.texNormalMap, center, area, float4(0, 0, 1, 0)).rgb;
    normal = normal * 2.f - 1.f;
    return calcShadingNormal(tri.TBN, normal);
#endif
}

ABSDFInput sampleMaterial(Polygon polygon, int3 cellInt, float nodeScale)
{
    Triangle tri = {};
    float2 center;
    float area;
    buildTriangleData(polygon, cellInt, nodeScale, tri, center, area);

    BasicMaterialData data = gScene.materials.getBasicMaterialData(polygon.triRef.materialID);

    float3 baseColor = sampleTextureArea(data.texBaseColor, center, area, data.baseColor).rgb;
    float4 spec = sampleTextureArea(data.texSpecular, center, area, data.specular);
    float3 normal;

#if LERP_NORMAL
    float dummy;
    float3 centroid = polygon.calcCentroid(dummy);
    float3 leafCentroid = centroid * nodeScale;
    normal = tri.lerpNormal(leafCentroid);
#else
    normal = sampleTextureArea(data.texNormalMap, center, area, float4(0, 0, 1, 0)).rgb;
    normal = normal * 2.f - 1.f;
    normal = calcShadingNormal(tri.TBN, normal);
#endif

    ABSDFInput input;
    input.baseColor = baseColor;
    input.specular = spec;
    input.normal = normal;
    input.area = polygon.calcArea();
    input.projArea = 0; // 由 calc() 在得到可见性后填入
    return input;
}

// 清空深度/归属数组
void clearDepth(
    inout float depthArr[VIS_PROJ_RES * VIS_PROJ_RES],
    inout uint idArr[VIS_PROJ_RES * VIS_PROJ_RES])
{
    for (uint i = 0; i < VIS_PROJ_RES * VIS_PROJ_RES; i++)
    {
        depthArr[i] = 1e30f;
        idArr[i] = 0xFFFFFFFFu;
    }
}

void calc(uint offset)
{
    if (offset >= groupVoxelCount)
        return;

    uint index = offset + gBufferOffset;
    VoxelData data = gBuffer[index];
    PolygonRange range = polygonRangeBuffer[index];

    // ---- Pass 1：每面片仅算法线，累计各 lobe 主视角法线 n0k = Σ area·n ----
    float3 n0Sum[LOBE_COUNT];
    float n0Area[LOBE_COUNT];
    for (uint k = 0; k < LOBE_COUNT; k++)
    {
        n0Sum[k] = float3(0);
        n0Area[k] = 0;
    }
    for (uint i = range.localHead; i < range.localHead + range.count; i++)
    {
        uint global = i; // batch 内全局面片号
        Polygon polygon = polygonBuffer[i];
        float3 normal = sampleNormal(polygon, range.cellInt, range.nodeScale);
        // 双面材质：面片同时向 ±n 两个 lobe 贡献（材质相同，法线相反）
        uint k0 = NormalIndex8(normal);
        uint k1 = NormalIndex8(-normal);
        lobeOfBuffer[global] = k0 | (k1 << 4); // 打包两个 lobe 索引
        float area = polygon.calcArea();
        n0Sum[k0] += area * normal;
        n0Area[k0] += area;
        n0Sum[k1] += area * (-normal);
        n0Area[k1] += area;
    }
    // 归一化得到每 lobe 主视角 n0k（无面片的 lobe 保持 0 向量，后续跳过）
    float3 n0[LOBE_COUNT];
    for (uint k = 0; k < LOBE_COUNT; k++)
    {
        n0[k] = (n0Area[k] > 0) ? safeNormalize(n0Sum[k]) : float3(0);
    }

    // ---- Pass 2：逐 lobe 深度光栅化 + 遍历像素按投影面积聚合材质 ----
    float depthArr[VIS_PROJ_RES * VIS_PROJ_RES];
    uint idArr[VIS_PROJ_RES * VIS_PROJ_RES];
    const uint pixelCount = VIS_PROJ_RES * VIS_PROJ_RES;
    const float pixelArea = PROJ_PIXEL_SIZE * PROJ_PIXEL_SIZE;

    // 总表面积（data.ABSDF.area 用于 solid 判定与 ellipsoid 拟合）
    for (uint i = range.localHead; i < range.localHead + range.count; i++)
    {
        data.ABSDF.area += polygonBuffer[i].calcArea();
    }

    for (uint k = 0; k < LOBE_COUNT; k++)
    {
        if (n0Area[k] <= 0)
            continue;

        // 沿 n0k 对全节点所有面片做深度光栅化，idArr[p] = 最前表面面片 global id
        clearDepth(depthArr, idArr);
        range.rasterizeDepth(polygonBuffer, n0[k], depthArr, idArr);

        // 直接遍历像素：光栅化已给出每像素可见面片，无需再回到面片维度
        // vpa_k = 沿 n0k 的总可见投影面积 VPA(n0k)（所有被覆盖像素，不管属于哪个 lobe）
        float vpa_k = 0;
        for (uint p = 0; p < pixelCount; p++)
        {
            uint global = idArr[p];
            if (global == 0xFFFFFFFFu)
                continue;
            vpa_k += pixelArea;

            // 解包 lobe 索引：面片同时归属正面 k0 与背面 k1（双面材质）
            uint packed = lobeOfBuffer[global];
            uint k0 = packed & 0xFu;
            uint k1 = (packed >> 4) & 0xFu;

            float sign;
            if (k == k0)
                sign = 1.0f;
            else if (k == k1)
                sign = -1.0f;
            else
                continue; // 该像素的面片不属于当前 lobe，不计入材质（但已计入 vpa_k）

            Polygon polygon = polygonBuffer[global];
            ABSDFInput input = sampleMaterial(polygon, range.cellInt, range.nodeScale);
            if (sign < 0.0f)
                input.normal = -input.normal; // 背面侧：法线取反，让 accumulate 归入当前 lobe
            input.projArea = pixelArea; // 像素投影面积

            data.ABSDF.accumulate(input); // NormalIndex8 归入 lobe（k==k0/k1 保证一致）
        }
        // 每 lobe 独立归一化：w_k = VPA_k / VPA(n0k)，分母是「沿 n0k 的总可见投影面积」
        data.ABSDF.lobes[k].normalizeSelf(vpa_k);
    }

    if (!data.IsSolid())
        return;

    data.ellipsoid.fit(polygonBuffer, range);

    SphericalFunc polyF = data.polygonsProjAreaFunc;
    SphericalFunc primF = data.primitiveProjAreaFunc;
    SphericalFunc totalF = data.totalProjAreaFunc;

    Estimate(data.ellipsoid, range, polyF, primF, totalF);

    data.polygonsProjAreaFunc = polyF;
    data.primitiveProjAreaFunc = primF;
    data.totalProjAreaFunc = totalF;

    gBuffer[index] = data;
    pBuffer[index] = data.ellipsoid;  // Per-node ellipsoid output
}

[numthreads(256, 1, 1)]
void main(uint3 offset: SV_DispatchThreadID)
{
    calc(offset.x);
}
