#include "Math/OctahedralProjection.slang"
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

void estimateOctahedralProjectionAreas(
    Ellipsoid e,
    PolygonRange range,
    inout ProjectionAreaFunc polygonFunc,
    inout ProjectionAreaFunc primitiveFunc,
    inout ProjectionAreaFunc totalFunc,
    uint sampleFrequency
)
{
    polygonFunc.init();
    primitiveFunc.init();
    totalFunc.init();

    for (uint y = 0; y < OCTA_RES; ++y)
    {
        for (uint x = 0; x < OCTA_RES; ++x)
        {
            uint index = y * OCTA_RES + x;
            float primitiveArea = 0.0f;
            float visibleArea = 0.0f;
            float totalArea = 0.0f;

            [unroll]
            for (uint sample = 0; sample < OCTA_SAMPLES_PER_TEXEL; ++sample)
            {
                float3 direction = octahedralTexelSampleDirection(uint2(x, y), sample);

                primitiveArea += estimatePrimitiveProjectionArea(
                    e, range, direction, sampleFrequency);
                visibleArea += range.calcVisibleProjAreaRaster(polygonBuffer, direction);
                totalArea += range.calcTotalProjArea(polygonBuffer, direction);
            }

            float invSampleCount = 1.0f / float(OCTA_SAMPLES_PER_TEXEL);
            primitiveFunc.coefficients[index] = primitiveArea * invSampleCount;
            polygonFunc.coefficients[index] = visibleArea * invSampleCount;
            totalFunc.coefficients[index] = totalArea * invSampleCount;
        }
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

void sampleTexture(Polygon polygon, int3 cellInt, float nodeScale, int offset, inout ABSDF ABSDF)
{
    TriangleRef triRef = polygon.triRef;
    MaterialSystem ms = gScene.materials;

    float2[MAX_VERTEX_COUNT] uvs;
    uint3 vtxIndices = gScene.getIndices(triRef.meshID, triRef.triangleID);
    Triangle tri = {};
    StaticVertexData[3] vertices = { gScene.getVertex(vtxIndices.x), gScene.getVertex(vtxIndices.y), gScene.getVertex(vtxIndices.z) };
    for (int i = 0; i < 3; i++)
    {
        tri.vertices[i] = (vertices[i].position - gridMin) / voxelSize; // 叶子体素坐标
        tri.uvs[i] = vertices[i].texCrd;
#if LERP_NORMAL
        tri.normals[i] = vertices[i].normal;
#endif
    }
    tri.buildTBN();
    for (uint i = 0; i < polygon.count; i++)
    {
        float3 leafVertex = polygon.vertices[i] * nodeScale;
        float2 uv = tri.lerpUV(leafVertex);
        uvs[i] = uv;
    }
    float area = 0;
    float2 center = 0;
    for (uint i = 0; i < polygon.count; ++i)
    {
        float2 a = uvs[i];
        float2 b = uvs[(i + 1) % polygon.count];
        area += a.x * b.y - a.y * b.x;
        center += uvs[i];
    }
    center /= polygon.count;
    area = 0.5f * abs(area);
    BasicMaterialData data = ms.getBasicMaterialData(triRef.materialID);

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

    ABSDFInput input = { baseColor, spec, normal, polygon.calcArea() };
    ABSDF.accumulate(input);
}

void calc(uint offset)
{
    if (offset >= groupVoxelCount)
        return;

    uint index = offset + gBufferOffset;
    VoxelData data = gBuffer[index];
    PolygonRange range = polygonRangeBuffer[index];

    // Each node independently: sample textures, fit ellipsoid, and fill the
    // filtered octahedral projection-area maps.
    // Same path for all levels (root, internal, leaf)
    for (uint i = range.localHead; i < range.localHead + range.count; i++)
    {
        Polygon polygon = polygonBuffer[i];
        sampleTexture(polygon, range.cellInt, range.nodeScale, index, data.ABSDF);
    }
    data.ABSDF.normalizeSelf();

    if (!data.IsSolid())
        return;

    data.ellipsoid.fit(polygonBuffer, range);

    ProjectionAreaFunc polyF = data.polygonsProjAreaFunc;
    ProjectionAreaFunc primF = data.primitiveProjAreaFunc;
    ProjectionAreaFunc totalF = data.totalProjAreaFunc;

    estimateOctahedralProjectionAreas(
        data.ellipsoid, range, polyF, primF, totalF, sampleFrequency);

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
