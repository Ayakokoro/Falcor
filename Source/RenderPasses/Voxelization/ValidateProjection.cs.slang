#include "Math/Polygon.slang"
#include "VoxelizationShared.slang"

StructuredBuffer<Polygon> polygonBuffer;
StructuredBuffer<PolygonRange> polygonRangeBuffer;
StructuredBuffer<VoxelData> gBuffer;
StructuredBuffer<float3> directions;

RWStructuredBuffer<DirectionValidation> validationOutput;

cbuffer CB
{
    uint targetGBufferIndex;
    uint directionCount;
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint dirIdx = tid.x;
    if (dirIdx >= directionCount)
        return;

    float3 dir = directions[dirIdx];

    VoxelData data = gBuffer[targetGBufferIndex];
    PolygonRange range = polygonRangeBuffer[targetGBufferIndex];

    DirectionValidation result;
    result.direction = dir;
    result.exactVisibleArea = range.calcVisibleProjAreaRaster(polygonBuffer, dir);
    result.shVisibleArea = data.polygonsProjAreaFunc.calc(dir);
    result.exactTotalArea = range.calcTotalProjArea(polygonBuffer, dir);
    result.shTotalArea = data.totalProjAreaFunc.calc(dir);

    validationOutput[dirIdx] = result;
}
