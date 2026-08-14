#include "Math/Polygon.slang"
#include "Math/Sampling.slang"
#include "Math/AABB.slang"
#include "Math/OctahedralMap.slang"
#include "Voxel/VoxelGrid.slang"
#include "VoxelizationShared.slang"
import Utils.Sampling.UniformSampleGenerator;
import Utils.Math.MathHelpers;

StructuredBuffer<Polygon> polygonBuffer;
StructuredBuffer<PolygonRange> polygonRangeBuffer;
RWStructuredBuffer<VoxelData> gBuffer;

cbuffer CB
{
    uint gBufferOffset;   // 本 batch 起始的全局 node 索引
    uint groupVoxelCount; // 本 batch 的节点数
    uint sampleFrequency; // primitive MC 采样频率（与 AnalyzePolygon 一致）
}

// 每个线程写一个 8x8 八面体采样点到 gBuffer 的 octa 字段
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint localThread = tid.x;
    uint localNode = localThread / OCTA_COUNT;
    uint sample = localThread % OCTA_COUNT;
    if (localNode >= groupVoxelCount)
        return;

    uint nodeIdx = gBufferOffset + localNode;

    // 非 solid 节点没有有效的 ellipsoid（AnalyzePolygon 未拟合），octa 值保持 init 的 0
    if (gBuffer[nodeIdx].ABSDF.area <= 0)
        return;

    int2 tc = int2(sample % OCTA_RES, sample / OCTA_RES);
    float2 uv = (float2(tc) + 0.5f) / OCTA_RES;
    float3 direction = octaDecode(uv);

    PolygonRange range = polygonRangeBuffer[nodeIdx];
    Ellipsoid e = gBuffer[nodeIdx].ellipsoid;

    // primitive: MC 采样（与 AnalyzePolygon::Estimate 完全一致）
    float3 cellMin = float3(range.cellInt);
    float3 cellCenter = cellMin + 0.5f;
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
    float primitive = PROJECT_CIRCLE_AREA * hitCount / (float)sampleFrequency;

    float total = range.calcTotalProjArea(polygonBuffer, direction);
    float visible = range.calcVisibleProjAreaRaster(polygonBuffer, direction);

    gBuffer[nodeIdx].primitiveProjAreaOcta.coefficients[sample] = primitive;
    gBuffer[nodeIdx].polygonsProjAreaOcta.coefficients[sample] = visible;
    gBuffer[nodeIdx].totalProjAreaOcta.coefficients[sample] = total;
}
