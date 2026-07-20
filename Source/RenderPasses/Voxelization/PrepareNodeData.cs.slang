#include "Shading.slang"

StructuredBuffer<VoxelData> nodeDataBuffer;
StructuredBuffer<OctreeNode> octreeBuffer;
RWStructuredBuffer<TEBSDF> gBuffer;
RWStructuredBuffer<Ellipsoid> pBuffer;

cbuffer CB
{
    uint nodeCount;
}

void prepare(uint offset)
{
    if (offset >= nodeCount)
        return;

    // Each node (at any level) independently initializes from its VoxelData
    // nodeDataBuffer and gBuffer/pBuffer use the same BFS ordering
    gBuffer[offset].init(nodeDataBuffer[offset]);
    pBuffer[offset] = nodeDataBuffer[offset].ellipsoid;
}

[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    prepare(tid.x);
}
