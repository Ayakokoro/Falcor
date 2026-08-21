#include "Shading.slang"

#ifndef RAY_MARCHING_BUFFER_COUNT
#define RAY_MARCHING_BUFFER_COUNT 1
#endif

StructuredBuffer<VoxelData> voxelDataBuffer[RAY_MARCHING_BUFFER_COUNT];
RWStructuredBuffer<TEBSDF> gBuffer[RAY_MARCHING_BUFFER_COUNT];
RWStructuredBuffer<Ellipsoid> pBuffer[RAY_MARCHING_BUFFER_COUNT];

cbuffer CB
{
    uint voxelCount;
    uint gbSplits[RAY_MARCHING_BUFFER_COUNT]; // exclusive global end index per buffer
    uint baseOffset;   // start index for this dispatch chunk
}

[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint offset = baseOffset + tid.x;
    if (offset >= voxelCount)
        return;

    uint bufferIndex = 0;
    [unroll]
    for (uint i = 0; i < RAY_MARCHING_BUFFER_COUNT; ++i)
    {
        if (offset < gbSplits[i])
        {
            bufferIndex = i;
            break;
        }
    }
    uint localIndex = offset - (bufferIndex == 0 ? 0 : gbSplits[bufferIndex - 1]);
    VoxelData data = voxelDataBuffer[bufferIndex][localIndex];

    TEBSDF gb;
    gb.init(data);
    gb.coverage = 12345.0f;  // sentinel
    gBuffer[bufferIndex][localIndex] = gb;
    pBuffer[bufferIndex][localIndex] = data.ellipsoid;
}
