#include "Math/Polygon.slang"
#include "Math/Sampling.slang"
#include "Math/AABB.slang"
#include "Math/LebedevQuadrature.slang"
#include "Voxel/VoxelGrid.slang"
#include "VoxelizationShared.slang"
import Utils.Sampling.UniformSampleGenerator;
import Utils.Math.MathHelpers;

StructuredBuffer<Polygon> polygonBuffer;
StructuredBuffer<PolygonRange> polygonRangeBuffer;
StructuredBuffer<VoxelData> gBuffer;

RWTexture2D<float> outputMap;

cbuffer CB
{
    uint targetGBufferIndex;
    uint2 mapResolution;
    uint functionType;       // 0=primitiveProjArea, 1=polygonsProjArea, 2=totalProjArea
    uint sampleFrequency;    // for primitiveProjArea ellipsoid test
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= mapResolution.x || tid.y >= mapResolution.y)
        return;

    // Only compute the top half of the texture (hemisphere: v ∈ [0, 0.5])
    // The bottom half is a mirror (even functions: f(d) = f(-d))
    float halfHeight = mapResolution.y / 2.0f;
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

    VoxelData data = gBuffer[targetGBufferIndex];
    PolygonRange range = polygonRangeBuffer[targetGBufferIndex];

    float exactValue = 0;

    if (functionType == 0)
    {
        // primitiveProjArea: ellipsoid hit test (same method as Estimate in AnalyzePolygon)
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
        exactValue = PROJECT_CIRCLE_AREA * hitCount / float(sampleFrequency);
    }
    else if (functionType == 1)
    {
        // polygonsProjArea (visible with occlusion): rasterization
        exactValue = range.calcVisibleProjAreaRaster(polygonBuffer, direction);
    }
    else
    {
        // totalProjArea: sum of all polygon projected areas (no occlusion)
        exactValue = range.calcTotalProjArea(polygonBuffer, direction);
    }

    outputMap[tid.xy] = exactValue;
}
