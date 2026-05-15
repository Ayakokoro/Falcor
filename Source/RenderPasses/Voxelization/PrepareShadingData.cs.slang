#include "Shading.slang"

StructuredBuffer<VoxelData> voxelDataBuffer;
RWStructuredBuffer<TEBSDF> gBuffer;
RWStructuredBuffer<Ellipsoid> pBuffer;

cbuffer CB
{
    uint voxelCount;
}

void prepare(uint offset)
{
    if (offset >= voxelCount)
        return;

    gBuffer[offset].init(voxelDataBuffer[offset]);
    pBuffer[offset] = voxelDataBuffer[offset].ellipsoid;
}

[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    prepare(tid.x);
}
