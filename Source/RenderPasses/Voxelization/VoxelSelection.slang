#pragma once
#include "HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR
// Integer indices stay exact even for scenes with more than 2^24 nodes.
struct VoxelSelection
{
    uint gbIndex;
    uint nodeSize;
    uint valid;
    uint reserved;
    int3 nodeMinCell;
    uint reserved2;
    float3 viewDirection;
    float reserved3;
};
END_NAMESPACE_FALCOR
