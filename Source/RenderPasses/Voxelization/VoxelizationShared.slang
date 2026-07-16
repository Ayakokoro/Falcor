#pragma once
#include "HostDeviceShared.slang"
#include "Voxel/ABSDF.slang"
#include "Math/Ellipsoid.slang"
#include "Math/Triangle.slang"
#include "Math/SphericalHarmonics.slang"
BEGIN_NAMESPACE_FALCOR

typedef EvenSphericalHarmonics SphericalFunc;

struct VoxelData
{
    ABSDF ABSDF;
    Ellipsoid ellipsoid;
    SphericalFunc primitiveProjAreaFunc;
    SphericalFunc polygonsProjAreaFunc; // 所有多边形的可见投影面积
    SphericalFunc totalProjAreaFunc;    // 所有多边形的总投影面积

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

struct OctreeNode
{
    uint childBase;   // Internal: first child index in next level. Leaf: gBuffer index.
    uint childMask;   // Bits 0-7: valid children. 0 = leaf node (internal nodes have >=1 child).
};

END_NAMESPACE_FALCOR
