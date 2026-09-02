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

struct DirectionValidation
{
    float3 direction;
    float exactVisibleArea; // ground truth visible projection area (rasterization)
    float shVisibleArea;    // SH reconstruction from polygonsProjAreaFunc
    float exactTotalArea;   // ground truth total projection area (no occlusion)
    float shTotalArea;      // SH reconstruction from totalProjAreaFunc
};

struct OctreeNode
{
    uint childBase;   // Internal: first child index in next level (top-down BFS order)
    uint childMask;   // Bits 0-7: valid children. 0 = leaf node
    uint dataIndex;   // Index into gBuffer/pBuffer for this node's own data (independent per-node)
};

// Per-placement data for the single-bin multi-instance ray marcher. This is
// shared with C++ so the structured-buffer layout cannot drift between host
// upload code and Slang.
struct VoxelInstanceData
{
    float4x4 localToWorld;
    float4x4 worldToLocal;
    float4x4 normalTransform;
    float4 worldBoundsMin;
    float4 worldBoundsMax;
    uint instanceId;
    int screenLOD;
    uint enabled;
    uint padding;
};

END_NAMESPACE_FALCOR
