#include "Math/Polygon.slang"
#include "Math/Sampling.slang"
#include "Math/AABB.slang"
#include "Math/LebedevQuadrature.slang"
#include "Math/OctahedralMap.slang"
#include "Voxel/VoxelGrid.slang"
#include "VoxelizationShared.slang"
import Utils.Sampling.UniformSampleGenerator;
import Utils.Math.MathHelpers;

StructuredBuffer<Polygon> polygonBuffer;
StructuredBuffer<PolygonRange> polygonRangeBuffer;
StructuredBuffer<VoxelData> gBuffer;

RWTexture2D<float> outputMap; // 高分辨率 equirectangular 精确值
RWTexture2D<float> octaMap;   // 8x8 半八面体纹理：每个 texel 存该方向上的精确值

cbuffer CB
{
    uint targetGBufferIndex;
    uint2 mapResolution;
    uint functionType;       // 0=primitiveProjArea, 1=polygonsProjArea, 2=totalProjArea
    uint sampleFrequency;    // for primitiveProjArea ellipsoid test
}

// 精确值(逐方向求值)
//   0=primitiveProjArea(椭球 MC), 1=polygonsProjArea(光栅化可见), 2=totalProjArea(无遮挡求和)
float evalExact(uint ftype, VoxelData data, PolygonRange range, float3 direction)
{
    if (ftype == 0)
    {
        Ellipsoid e = data.ellipsoid;
        float3 cellMin = float3(range.cellInt);
        float3 cellCenter = cellMin + 0.5f;
        Basis2 basis = orthonormal_basis(direction);
        uint hitCount = 0;
        for (uint j = 0; j < sampleFrequency; j++)
        {
            SamplingRay ray = rayToVoxel(Hammersley2D(j, sampleFrequency), direction, basis, cellCenter);
            float3 from = ray.origin;
            float3 to = from + 2.0f * ray.direction;
            float2 inOut = clip(cellMin, cellMin + 1.0f, from, to);
            if (inOut.x >= 0)
            {
                float3 dv = to - from;
                to = from + inOut.y * dv - cellMin;
                from = from + inOut.x * dv - cellMin;
                inOut = e.clip(from, to);
                if (inOut.x >= 0)
                    hitCount++;
            }
        }
        return PROJECT_CIRCLE_AREA * hitCount / float(sampleFrequency);
    }
    else if (ftype == 1)
    {
        return range.calcVisibleProjAreaRaster(polygonBuffer, direction);
    }
    else
    {
        return range.calcTotalProjArea(polygonBuffer, direction);
    }
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    VoxelData data = gBuffer[targetGBufferIndex];
    PolygonRange range = polygonRangeBuffer[targetGBufferIndex];

    // 写 8x8 八面体纹理：每个 texel 存该方向上的精确值（实际采样值）
    // 前 64 个线程(tid.y==0)负责；mapResolution.x 最小为 64，恒有足够线程
    if (tid.y == 0 && tid.x < OCTA_COUNT)
    {
        int2 tc = int2(tid.x % OCTA_RES, tid.x / OCTA_RES);
        float2 uv = (float2(tc) + 0.5f) / OCTA_RES;
        octaMap[tc] = evalExact(functionType, data, range, octaDecode(uv));
    }

    if (tid.x >= mapResolution.x || tid.y >= mapResolution.y)
        return;

    // Only compute the top half of the texture (hemisphere: v ∈ [0, 0.5])
    // The bottom half is a mirror (even functions: f(d) = f(-d))
    float vNorm = (tid.y + 0.5f) / float(mapResolution.y);

    // Hemisphere: θ ∈ [0, π/2], φ ∈ [0, 2π]
    float theta = vNorm * PI * 0.5f;
    float phi = (tid.x + 0.5f) / float(mapResolution.x) * 2.0f * PI;

    // Direction with z as "up" axis (matches sphere display convention)
    float3 direction = float3(
        sin(theta) * cos(phi),
        sin(theta) * sin(phi),
        cos(theta)
    );

    outputMap[tid.xy] = evalExact(functionType, data, range, direction);
}
