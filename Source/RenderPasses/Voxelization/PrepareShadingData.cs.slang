#include "Shading.slang"

StructuredBuffer<VoxelData> voxelDataBuffer0;
StructuredBuffer<VoxelData> voxelDataBuffer1;
StructuredBuffer<VoxelData> voxelDataBuffer2;
StructuredBuffer<VoxelData> voxelDataBuffer3;
StructuredBuffer<VoxelData> voxelDataBuffer4;
RWStructuredBuffer<TEBSDF> gBuffer0;
RWStructuredBuffer<TEBSDF> gBuffer1;
RWStructuredBuffer<TEBSDF> gBuffer2;
RWStructuredBuffer<TEBSDF> gBuffer3;
RWStructuredBuffer<TEBSDF> gBuffer4;
RWStructuredBuffer<Ellipsoid> pBuffer0;
RWStructuredBuffer<Ellipsoid> pBuffer1;
RWStructuredBuffer<Ellipsoid> pBuffer2;
RWStructuredBuffer<Ellipsoid> pBuffer3;
RWStructuredBuffer<Ellipsoid> pBuffer4;

cbuffer CB
{
    uint voxelCount;
    uint gbSplit0;     // boundary between buffer [0] and [1]
    uint gbSplit1;     // boundary between buffer [1] and [2]
    uint gbSplit2;     // boundary between buffer [2] and [3]
    uint gbSplit3;     // boundary between buffer [3] and [4]
    uint baseOffset;   // start index for this dispatch chunk
}

[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint offset = baseOffset + tid.x;
    if (offset >= voxelCount)
        return;

    if (offset < gbSplit0)
    {
        TEBSDF gb;
        gb.init(voxelDataBuffer0[offset]);
        gb.coverage = 12345.0f;  // sentinel
        gBuffer0[offset] = gb;
        pBuffer0[offset] = voxelDataBuffer0[offset].ellipsoid;
    }
    else if (offset < gbSplit1)
    {
        uint idx = offset - gbSplit0;
        TEBSDF gb;
        gb.init(voxelDataBuffer1[idx]);
        gb.coverage = 12345.0f;
        gBuffer1[idx] = gb;
        pBuffer1[idx] = voxelDataBuffer1[idx].ellipsoid;
    }
    else if (offset < gbSplit2)
    {
        uint idx = offset - gbSplit1;
        TEBSDF gb;
        gb.init(voxelDataBuffer2[idx]);
        gb.coverage = 12345.0f;
        gBuffer2[idx] = gb;
        pBuffer2[idx] = voxelDataBuffer2[idx].ellipsoid;
    }
    else if (offset < gbSplit3)
    {
        uint idx = offset - gbSplit2;
        TEBSDF gb;
        gb.init(voxelDataBuffer3[idx]);
        gb.coverage = 12345.0f;
        gBuffer3[idx] = gb;
        pBuffer3[idx] = voxelDataBuffer3[idx].ellipsoid;
    }
    else
    {
        uint idx = offset - gbSplit3;
        TEBSDF gb;
        gb.init(voxelDataBuffer4[idx]);
        gb.coverage = 12345.0f;
        gBuffer4[idx] = gb;
        pBuffer4[idx] = voxelDataBuffer4[idx].ellipsoid;
    }
}
