#include "Math/OctahedralProjection.slang"
#include "Math/Polygon.slang"
#include "Voxel/VoxelGrid.slang"
#include "VoxelizationShared.slang"

StructuredBuffer<Polygon> polygonBuffer;
StructuredBuffer<PolygonRange> polygonRangeBuffer;
StructuredBuffer<VoxelData> gBuffer;

RWTexture2D<float> outputMap;

cbuffer CB
{
    uint targetGBufferIndex;
    uint2 mapResolution;       // square octahedral UV map
    uint functionType;         // 0=primitiveProjArea, 1=polygonsProjArea, 2=totalProjArea
    uint sampleFrequency;      // for primitiveProjArea ellipsoid test
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= mapResolution.x || tid.y >= mapResolution.y)
        return;

    // The valid upper-hemisphere part of the folded octahedral map is a
    // diamond in the square UV domain. Keep the padded triangles black.
    float2 uv = (float2(tid.xy) + 0.5f) / float2(mapResolution);
    float2 signedUV = uv * 2.0f - 1.0f;
    if (abs(signedUV.x) + abs(signedUV.y) > 1.0f)
    {
        outputMap[tid.xy] = 0.0f;
        return;
    }

    float3 direction = octaDecode(uv);

    VoxelData data = gBuffer[targetGBufferIndex];
    PolygonRange range = polygonRangeBuffer[targetGBufferIndex];

    float exactValue = 0.0f;

    if (functionType == 0)
    {
        // Primitive projection area from the same ellipsoid ray estimator as
        // the GPU voxelization prefilter.
        exactValue = estimatePrimitiveProjectionArea(
            data.ellipsoid, range, direction, sampleFrequency);
    }
    else if (functionType == 1)
    {
        // Visible projected area, including rasterized occlusion.
        exactValue = range.calcVisibleProjAreaRaster(polygonBuffer, direction);
    }
    else
    {
        // Sum of projected polygon areas without occlusion.
        exactValue = range.calcTotalProjArea(polygonBuffer, direction);
    }

    outputMap[tid.xy] = exactValue;
}
