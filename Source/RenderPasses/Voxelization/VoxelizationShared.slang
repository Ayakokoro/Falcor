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

// Per-bin data after all voxel assets have been concatenated into the shared
// scene buffers. float4/uint4 fields keep the host/device layout explicit.
struct VoxelAssetData
{
    float4 gridMin;
    float4 voxelSize;
    uint4 voxelCountAndDepth; // xyz: LOD-0 dimensions, w: octree max depth
    uint4 resourceInfo;       // x: octree root, y: available LODs, z: data base, w: data count
};

// Per-placement data for the multi-asset ray marcher. assetIndex selects the
// bin metadata above; the octree and shading payload remain shared.
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
    uint assetIndex;
};

END_NAMESPACE_FALCOR
