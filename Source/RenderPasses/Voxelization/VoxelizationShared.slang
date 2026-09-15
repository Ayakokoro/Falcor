#pragma once
#include "HostDeviceShared.slang"
#include "Voxel/ABSDF.slang"
#include "Math/Ellipsoid.slang"
#include "Math/Triangle.slang"
#include "Math/OctahedralMap.slang"
BEGIN_NAMESPACE_FALCOR

typedef OctahedralFunc ProjectionAreaFunc;

struct VoxelData
{
    ABSDF ABSDF;
    Ellipsoid ellipsoid;
    ProjectionAreaFunc primitiveProjAreaFunc;
    ProjectionAreaFunc polygonsProjAreaFunc; // 所有多边形的可见投影面积
    ProjectionAreaFunc totalProjAreaFunc;    // 所有多边形的总投影面积

    MUTATING
    void init()
    {
        ABSDF.init();
        primitiveProjAreaFunc.init();
        polygonsProjAreaFunc.init();
        totalProjAreaFunc.init();
    }

    bool IsSolid() { return ABSDF.area > 0; }
};

struct DirectionValidation
{
    float3 direction;
    float exactVisibleArea; // ground truth visible projection area (rasterization)
    float approxVisibleArea; // bilinear interpolation of polygonsProjAreaFunc
    float exactTotalArea;   // ground truth total projection area (no occlusion)
    float approxTotalArea;  // bilinear interpolation of totalProjAreaFunc
};

struct OctreeNode
{
    uint childBase;   // Internal: first child index in next level (top-down BFS order)
    uint childMask;   // Bits 0-7: valid children. 0 = leaf node
    uint dataIndex;   // Index into gBuffer/pBuffer for this node's own data (independent per-node)
};

END_NAMESPACE_FALCOR
