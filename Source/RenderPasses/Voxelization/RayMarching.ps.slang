#include "Math/VoxelizationUtility.slang"
#include "VoxelizationShared.slang"
#include "MixLightSampler.slang"
import Rendering.Materials.StandardMaterial;
import Utils.Sampling.UniformSampleGenerator;

#ifndef MAX_BOUNCE
#define MAX_BOUNCE 3
#endif

#ifndef MAX_OCTREE_DEPTH
#define MAX_OCTREE_DEPTH 12
#endif

#ifndef RAY_MARCHING_BUFFER_COUNT
#define RAY_MARCHING_BUFFER_COUNT 1
#endif

// Compile-time switch for the instance-level BVH traversal. The host injects
// this via FullScreenPass::addDefine, driven by a UI checkbox (mUseInstanceBVH).
// 1 = the BVH traversal is compiled in; 0 = it is compiled out and traversal
// is always the linear scan. Changing it recompiles the pass shader.
#ifndef USE_INSTANCE_BVH
#define USE_INSTANCE_BVH 1
#endif

#define RAY_BIAS 1e-4f // cell-space t-offset to avoid self-intersection

static const float kSceneRayMax = 1e30f;
static const float kSceneHitTieEpsilon = 1e-5f;
static const uint kInvalidInstanceIndex = 0xffffffffu;
// Worst-case instance BVH stack depth for a balanced median tree: the tree
// over N instances has ⌈log2 N⌉ levels, plus a safety margin.
static const uint kInstanceBVHMaxStack = 64u;
// Safety margin for the near-to-far early-termination test.
static const float kBVHPruneEpsilon = 1e-6f;

StructuredBuffer<TEBSDF> gBuffer[RAY_MARCHING_BUFFER_COUNT];
StructuredBuffer<Ellipsoid> pBuffer[RAY_MARCHING_BUFFER_COUNT];
StructuredBuffer<OctreeNode> octreeBuffer;
StructuredBuffer<VoxelAssetData> voxelAssets;
StructuredBuffer<VoxelInstanceData> instances;
StructuredBuffer<InstanceBVHNode> instanceBVH;

RWStructuredBuffer<float4> selectedVoxel;

cbuffer CB
{
    uint2 pixelCount;
    float4x4 invVP;
    uint instanceCount;
    uint assetCount;
    uint instanceBVHNodeCount;
    float shadowBiasWorld;
    uint drawMode;
    uint frameIndex;
    float minPdf;
    float transmittanceThreshold;
    bool renderBackGround;
    float4 clearColor;
    uint2 selectedPixel;
    float tanHalfFovY;  // tan(FovY / 2), for primary ray cone
    int forcedLOD;      // -1=disabled, 0=finest leaf, N=stop N levels above leaf
    int maxLODLevel;    // cap screen-space LOD: -1=no cap, N=never select coarser than N
    uint gbSplits[RAY_MARCHING_BUFFER_COUNT]; // exclusive global end index per buffer
    float coverageBlend; // 0=raw coverage, 1=fill all empty cavity
}

// Ray cone: represents the angular footprint of a ray
// Half-angle tangent grows with BSDF roughness at each bounce
struct RayCone
{
    float tanHalfAngle; // tan(θ/2), where θ is the cone's full opening angle
    float bias;         // footprint radius at ray origin (cell-space), inherited from previous bounces
};

struct HitResult
{
    bool hit;
    int3 cellInt;    // in LOD-0 coordinate space
    float3 cell;     // in LOD-0 coordinate space
    uint gbIndex;    // global gBuffer/pBuffer index for the selected node
    uint nodeSize;   // size in LOD-0 cells of the node where hit occurred (1=leaf, >1=LOD stop)
    float lodLevel;  // selected discrete LOD level; 0=leaf
    float3 normalLocal; // geometric ellipsoid normal in asset-local space
};

// Scene hit expressed in world space while retaining the asset-local hit used
// to access the concatenated voxel hierarchies and material buffers.
struct WorldHitResult
{
    bool hit;
    uint instanceIndex;
    uint instanceId;
    float3 positionW;
    float tWorld;
    float3 directionLocal;
    float3 normalW;
    float rayConeBias;
    HitResult localHit;
};

// Intersections from several instances at nearly the same world-space ray
// depth form one event. The representative supplies material/shading data;
// unionCoverage prevents duplicated voxelized surfaces from being accumulated
// more than once by the Coverage view.
struct SceneHitEvent
{
    bool hit;
    WorldHitResult representative;
    float tMinWorld;
    float tMaxWorld;
    float mergeDistanceWorld;
    float unionCoverage;
    uint memberCount;
};

struct PsOut
{
    float4 color : SV_Target0;
};

bool outsideGrid(int3 cellInt, VoxelAssetData asset)
{
    return any(cellInt < int3(0)) || any(cellInt >= (int3)asset.voxelCountAndDepth.xyz);
}

uint getBufferIndex(uint globalIdx)
{
    [unroll]
    for (uint i = 0; i < RAY_MARCHING_BUFFER_COUNT; ++i)
    {
        if (globalIdx < gbSplits[i])
            return i;
    }
    return RAY_MARCHING_BUFFER_COUNT - 1;
}

uint getBufferOffset(uint globalIdx, uint bufferIndex)
{
    return globalIdx - (bufferIndex == 0 ? 0 : gbSplits[bufferIndex - 1]);
}

TEBSDF loadGBuffer(uint globalIdx)
{
    uint bufferIndex = getBufferIndex(globalIdx);
    return gBuffer[bufferIndex][getBufferOffset(globalIdx, bufferIndex)];
}

Ellipsoid loadPBuffer(uint globalIdx)
{
    uint bufferIndex = getBufferIndex(globalIdx);
    return pBuffer[bufferIndex][getBufferOffset(globalIdx, bufferIndex)];
}

// Extract max roughness across all lobes of a TEBSDF at gbIndex
float maxRoughness(uint gbIndex)
{
    TEBSDF bsdf = loadGBuffer(gbIndex);
    float r = 0;
    for (uint i = 0; i < bsdf.surface.lobeCount; i++)
        r = max(r, bsdf.surface.lobes[i].frostBite.rough);
    return r;
}

HitResult rayHitTE(
    Ellipsoid ellipsoid,
    float3 cell,
    float3 direction,
    int3 cellInt,
    uint gbIndex,
    uint nodeSize,
    VoxelAssetData asset
)
{
    // Compute node-local coordinates for ellipsoid intersection.
    // Ellipsoid is defined in node-local [0,1]³ space (fitted over the entire node),
    // but the incoming coordinates are in LOD-0 cell space.
    // Convert to node-local space, intersect, then convert back to LOD-0.
    int3 nodeMinCell = (cellInt / int3((int)nodeSize)) * int3((int)nodeSize);
    float3 localOrigin = (cell - float3(nodeMinCell)) / float(nodeSize);
    // Direction in node-local space: a step of 1 LOD-0 cell = step of 1/nodeSize in node-local
    float3 nodeDir = direction / float(nodeSize);

    int3 dirs = int3(sign(nodeDir));
    float3 border = (dirs + 1) / 2;
    float3 t = (border - localOrigin) / nodeDir;
    // t is in LOD-0 units; max distance to exit a node is ≤ √3 * nodeSize
    float tMin = 2.0f * float(nodeSize);
    if (nodeDir.x != 0 && t.x < tMin) tMin = t.x;
    if (nodeDir.y != 0 && t.y < tMin) tMin = t.y;
    if (nodeDir.z != 0 && t.z < tMin) tMin = t.z;
    float3 to = localOrigin + tMin * nodeDir;

    float2 inOut = ellipsoid.clip(localOrigin, to);

    // Convert intersection back to LOD-0 cell space
    float3 hitLocal = localOrigin + inOut.x * (to - localOrigin);
    float3 hitCell = float3(nodeMinCell) + hitLocal * float(nodeSize);
    int3 hitCellInt = int3(floor(hitCell));

    float3 normalLocal = float3(0.0f);
    if (inOut.x >= 0.0f)
    {
        // Ellipsoid::B is expressed in node coordinates. Converting its
        // gradient through the grid's voxel scale gives an asset-local
        // geometric normal. The scalar nodeSize cancels during normalization.
        float3 normalNode = mul(ellipsoid.B, hitLocal - ellipsoid.center);
        normalLocal = normalizeSafe(normalNode / asset.voxelSize.xyz);
        if (all(normalLocal == float3(0.0f)))
            normalLocal = -normalizeSafe(direction * asset.voxelSize.xyz);
    }

    HitResult result = {
        inOut.x >= 0,
        hitCellInt,
        hitCell,
        gbIndex,
        nodeSize,
        0.0f,
        normalLocal
    };
    return result;
}

HitResult rayHitVoxel(
    float3 cell,
    float3 direction,
    int3 cellInt,
    uint gbIndex,
    uint nodeSize,
    VoxelAssetData asset
)
{
#if !CHECK_ELLIPSOID
    float3 normalLocal = -normalizeSafe(direction * asset.voxelSize.xyz);
    return { true, cellInt, cellInt + 0.5f, gbIndex, nodeSize, 0.0f, normalLocal };
#else
    Ellipsoid ellipsoid = loadPBuffer(gbIndex);
    HitResult hit = rayHitTE(ellipsoid, cell, direction, cellInt, gbIndex, nodeSize, asset);
    if (hit.hit)
        return hit;
#endif
    return { false, cellInt, cellInt + 0.5f, 0, nodeSize, 0.0f, float3(0.0f) };
}

// --- Parametric octree traversal (negative direction DDA) ---
// Follows the Revelles 2000 parametric approach: axes are mirrored so that
// the ray always travels in the negative direction (d < 0 everywhere).
// Coordinate space: normalized to [1,2]³.
// Stack stores one entry per octree level, not per node.
#define S_MAX 23
#define EPSILON exp2(-float(S_MAX))

// Parametric octree traversal using Revelles 2000 negative-direction DDA.
// Merges the speed of parametric space stepping with grid-wide screen-space LOD,
// ellipsoid intersection tests, and optional near-origin hit rejection.
// from: starting point in LOD-0 cell space
HitResult rayMarching(
    VoxelAssetData asset,
    float3 from,
    float3 direction,
    float maxDistance,
    float rejectNearHitDistanceCells,
    int fixedLod
)
{
    direction = normalizeSafe(direction);
    if (all(direction == float3(0)))
        return { false, int3(from), from, 0, 0, 0.0f, float3(0.0f) };

    // A continuation ray no longer jumps directly to the boundary of a whole
    // LOD node. It traverses normally and rejects only actual primitive hits
    // that belong to the already-consumed origin event.
    float3 rayOrigin = from;

    // ===== Phase 1: normalize + mirror for negative-direction traversal =====
    uint octreeMaxDepth = asset.voxelCountAndDepth.w;
    uint availableLODLevels = asset.resourceInfo.y;
    uint rootSize = 1u << octreeMaxDepth;

    // LOD is selected once per frame from the projected grid size. The
    // optional arguments remain useful for explicit debug/test overrides, but
    // no ray-distance based LOD decision is made during traversal.
    int targetLOD = max(fixedLod, 0);
    if (forcedLOD >= 0)
        targetLOD = forcedLOD;
    if (maxLODLevel >= 0)
        targetLOD = min(targetLOD, maxLODLevel);
    targetLOD = clamp(targetLOD, 0, int(min(octreeMaxDepth, availableLODLevels)));

    // Normalize ray origin to [1,2]³ space
    float3 p = rayOrigin / float(rootSize) + 1.0f;
    p = clamp(p, 1.0f, 2.0f);

    float3 d = direction;
    if (abs(d.x) < EPSILON) d.x = d.x < 0.0f ? -EPSILON : EPSILON;
    if (abs(d.y) < EPSILON) d.y = d.y < 0.0f ? -EPSILON : EPSILON;
    if (abs(d.z) < EPSILON) d.z = d.z < 0.0f ? -EPSILON : EPSILON;

    precise float3 coef = -1.0f / abs(d);
    precise float3 bias = coef * p;

    // octant_mask: bit i = 1 means axis i WAS mirrored (original d[i] > 0).
    // Starts at 0; XOR toggles bits for axes whose original direction was positive.
    uint octant_mask = 0;
    if (d.x > 0.0f) { octant_mask ^= 1; bias.x = 3.0f * coef.x - bias.x; }
    if (d.y > 0.0f) { octant_mask ^= 2; bias.y = 3.0f * coef.y - bias.y; }
    if (d.z > 0.0f) { octant_mask ^= 4; bias.z = 3.0f * coef.z - bias.z; }

    // ===== Phase 2: root entry / exit in parametric space =====
    precise float t_min = max(max(2.0f * coef.x - bias.x, 2.0f * coef.y - bias.y), 2.0f * coef.z - bias.z);
    precise float t_max = min(min(1.0f * coef.x - bias.x, 1.0f * coef.y - bias.y), 1.0f * coef.z - bias.z);

    float h = t_max;
    t_min = max(t_min, 0.0f);

    // Clamp far end by maxDistance (convert cell-space distance → parametric t)
    if (maxDistance > 0)
        t_max = min(t_max, t_min + maxDistance / float(rootSize));

    if (t_min > t_max)
        return { false, int3(0), float3(0), 0, 0, 0.0f, float3(0.0f) };

    // ===== Phase 3: initialize state machine =====
    int2 stack[S_MAX + 1];

    uint parent = asset.resourceInfo.x;
    uint idx = 0;
    precise float3 pos = float3(1.0f, 1.0f, 1.0f);
    int scale = S_MAX - 1;
    precise float scale_exp2 = 0.5f;

    // Determine which sub-octant the ray enters first
    if (1.5f * coef.x - bias.x > t_min) { idx ^= 1; pos.x = 1.5f; }
    if (1.5f * coef.y - bias.y > t_min) { idx ^= 2; pos.y = 1.5f; }
    if (1.5f * coef.z - bias.z > t_min) { idx ^= 4; pos.z = 1.5f; }

    // ===== Phase 4: main traversal loop (INTERSECT → PUSH | ADVANCE → POP) =====
    uint maxIter = (1u << octreeMaxDepth) * 3u + 64u;
    uint iter = 0;

    while (scale < S_MAX)
    {
        if (++iter > maxIter) break;

        precise float3 t_corner = pos * coef - bias;
        precise float tc_max = min(min(t_corner.x, t_corner.y), t_corner.z);

        uint child_shift = idx ^ octant_mask;   // un-mirror child index → real memory layout
        OctreeNode node = octreeBuffer[parent];
        bool childExists = (node.childMask & (1u << child_shift)) != 0;

        // --- INTERSECT ---
        if (childExists && t_min <= t_max)
        {
            uint childgbIdx = node.childBase + countbits(node.childMask & ((1u << child_shift) - 1));
            OctreeNode childNode = octreeBuffer[childgbIdx];
            bool isLeaf = (childNode.childMask == 0);

            precise float tv_max = min(t_max, tc_max);
            precise float half_scale = scale_exp2 * 0.5f;

            // --- Node metadata for LOD decisions ---
            uint octreeDepth = S_MAX - scale;                    // 1..octreeMaxDepth
            uint nodeLOD     = octreeMaxDepth - octreeDepth;     // 0 at leaf
            uint nodeSize    = 1u << nodeLOD;                    // 1 at leaf

            // Structural ancestor nodes are present even when the CPU
            // voxelizer was asked to generate leaves only. Never stop at
            // such a node. The area check also protects the CHECK_ELLIPSOID=0
            // path, where rayHitVoxel would otherwise accept an empty node.
            TEBSDF nodeData = loadGBuffer(childNode.dataIndex);
            bool nodeHasData = (nodeLOD <= availableLODLevels) && (nodeData.area > 0.0f);

            bool selectedLOD = nodeHasData && ((int)nodeLOD == targetLOD);

            // Stop at the selected grid-wide LOD. Structural nodes without
            // data are never accepted as hits.
            bool shouldStop = nodeHasData && (isLeaf || selectedLOD);

            // --- Hit test when a stopping condition is met ---
            if (shouldStop && t_min <= tv_max)
            {
                float3 temp = pos;
                if ((octant_mask & 1) == 1) temp.x = 3.0f - scale_exp2 - pos.x;
                if ((octant_mask & 2) == 2) temp.y = 3.0f - scale_exp2 - pos.y;
                if ((octant_mask & 4) == 4) temp.z = 3.0f - scale_exp2 - pos.z;

                precise float3 hit_pos = min(max(p + t_min * d, temp + EPSILON), temp + scale_exp2 - EPSILON);

                precise float3 localHit = hit_pos - 1.0f;

                // in LOD-0 cell space, the hit point is:
                precise float3 hitCell  = localHit * rootSize;
                // in LOD-0 cell space, the hit cell is:
                int3 cellInt = int3(floor(hitCell));

                if (!outsideGrid(cellInt, asset))
                {
                    uint gbIndex = childNode.dataIndex;

                    HitResult hit = rayHitVoxel(hitCell, direction, cellInt, gbIndex, nodeSize, asset);
                    if (hit.hit)
                    {
                        hit.lodLevel = float(nodeLOD);
#if CHECK_ELLIPSOID
                        float hitDistanceCells = max(dot(hit.cell - from, direction), 0.0f);
#else
                        // The box fallback reports the voxel center as its hit
                        // position, so use the traversed node-entry distance for
                        // origin-event rejection instead.
                        float hitDistanceCells = max(t_min * float(rootSize), 0.0f);
#endif
                        bool rejectOriginEventHit = rejectNearHitDistanceCells >= 0.0f &&
                            hitDistanceCells <= rejectNearHitDistanceCells;
                        if (!rejectOriginEventHit)
                            return hit;
                    }
                }
            }

            // --- PUSH: descend into child ---
            // Unavailable internal nodes also come through this path because
            // selectedLOD is gated by nodeHasData. If the selected node misses
            // the ellipsoid, continue descending to finer data.
            bool wantPush = (!isLeaf && !selectedLOD) && (t_min <= tv_max);

            if (wantPush)
            {
                if (tc_max < h)
                {
                    stack[scale] = int2(parent, asint(t_max));
                }

                h = tc_max;

                parent = childgbIdx;
                idx = 0;
                scale--;
                scale_exp2 = half_scale;

                // Determine which sub-octant of the child the ray enters first.
                // t_center is the parametric t at the child-cell center.
                precise float3 t_center = half_scale * coef + t_corner;
                if (t_center.x > t_min) { idx ^= 1; pos.x += scale_exp2; }
                if (t_center.y > t_min) { idx ^= 2; pos.y += scale_exp2; }
                if (t_center.z > t_min) { idx ^= 4; pos.z += scale_exp2; }

                t_max = tv_max;
                continue;
            }
        }

        // --- ADVANCE: step to next sibling cell along the ray ---
        uint step_mask = 0;
        if (t_corner.x <= tc_max) { step_mask ^= 1; pos.x -= scale_exp2; }
        if (t_corner.y <= tc_max) { step_mask ^= 2; pos.y -= scale_exp2; }
        if (t_corner.z <= tc_max) { step_mask ^= 4; pos.z -= scale_exp2; }

        t_min = tc_max;
        idx ^= step_mask;

        // --- POP: crossed a parent boundary → return to ancestor level ---
        if ((idx & step_mask) != 0)
        {
            uint differing_bits = 0;
            if ((step_mask & 1) != 0) differing_bits |= asuint(pos.x) ^ asuint(pos.x + scale_exp2);
            if ((step_mask & 2) != 0) differing_bits |= asuint(pos.y) ^ asuint(pos.y + scale_exp2);
            if ((step_mask & 4) != 0) differing_bits |= asuint(pos.z) ^ asuint(pos.z + scale_exp2);

            scale = firstbithigh(differing_bits);
            scale_exp2 = asfloat((scale - S_MAX + 127) << 23);

            int2 stackEntry = stack[scale];
            parent = (uint)stackEntry.x;
            t_max = asfloat(stackEntry.y);

            uint shx = asuint(pos.x) >> scale;
            uint shy = asuint(pos.y) >> scale;
            uint shz = asuint(pos.z) >> scale;

            pos.x = asfloat(shx << scale);
            pos.y = asfloat(shy << scale);
            pos.z = asfloat(shz << scale);

            idx = (shx & 1) | ((shy & 1) << 1) | ((shz & 1) << 2);
            h = 0.0f;
        }
    }

    return { false, int3(0), float3(0), 0, 0, 0.0f, float3(0.0f) };
}

float3 localToWorldPoint(float3 pointLocal, VoxelInstanceData instance)
{
    return mul(instance.localToWorld, float4(pointLocal, 1.0f)).xyz;
}

float3 localToWorldVector(float3 vectorLocal, VoxelInstanceData instance)
{
    return mul(instance.localToWorld, float4(vectorLocal, 0.0f)).xyz;
}

float3 worldToLocalPoint(float3 pointWorld, VoxelInstanceData instance)
{
    return mul(instance.worldToLocal, float4(pointWorld, 1.0f)).xyz;
}

float3 worldToLocalVector(float3 vectorWorld, VoxelInstanceData instance)
{
    return mul(instance.worldToLocal, float4(vectorWorld, 0.0f)).xyz;
}

float3 localToWorldNormal(float3 normalLocal, VoxelInstanceData instance)
{
    return normalizeSafe(mul(instance.normalTransform, float4(normalLocal, 0.0f)).xyz);
}

WorldHitResult makeWorldHitResult(
    HitResult localHit,
    float3 rayOriginW,
    float3 rayDirectionW,
    uint instanceIndex,
    VoxelInstanceData instance,
    VoxelAssetData asset,
    float coneTan
)
{
    WorldHitResult result = {};
    result.instanceIndex = instanceIndex;
    result.instanceId = instance.instanceId;
    result.tWorld = 1e30f;
    result.directionLocal = normalizeSafe(worldToLocalVector(rayDirectionW, instance));
    result.localHit = localHit;

    if (!localHit.hit)
        return result;

    float3 positionLocal = asset.gridMin.xyz + localHit.cell * asset.voxelSize.xyz;
    result.positionW = localToWorldPoint(positionLocal, instance);
    result.tWorld = dot(result.positionW - rayOriginW, normalizeSafe(rayDirectionW));
    result.normalW = localToWorldNormal(localHit.normalLocal, instance);
    if (all(result.normalW == float3(0.0f)))
        result.normalW = -normalizeSafe(rayDirectionW);
    float3 directionCellPerWorld =
        worldToLocalVector(normalizeSafe(rayDirectionW), instance) / asset.voxelSize.xyz;
    result.rayConeBias = max(result.tWorld, 0.0f) * length(directionCellPerWorld) * coneTan;
    result.hit = true;
    return result;
}

float worldDistanceToCellDistance(
    float3 directionWorld,
    float distanceWorld,
    VoxelInstanceData instance,
    VoxelAssetData asset
)
{
    float3 directionCellPerWorld =
        worldToLocalVector(normalizeSafe(directionWorld), instance) / asset.voxelSize.xyz;
    return distanceWorld * length(directionCellPerWorld);
}

float worldVoxelSize(VoxelInstanceData instance, VoxelAssetData asset)
{
    // Rotation preserves the length of each scaled local voxel axis. Use the
    // largest world-space edge as a conservative emissive matching tolerance.
    float3 worldVoxelSizes = float3(
        length(mul((float3x3)instance.localToWorld, float3(asset.voxelSize.x, 0, 0))),
        length(mul((float3x3)instance.localToWorld, float3(0, asset.voxelSize.y, 0))),
        length(mul((float3x3)instance.localToWorld, float3(0, 0, asset.voxelSize.z)))
    );
    return max(max(worldVoxelSizes.x, worldVoxelSizes.y), max(worldVoxelSizes.z, 1e-6f));
}

float cellDistancePerWorldUnit(
    float3 directionWorld,
    VoxelInstanceData instance,
    VoxelAssetData asset
)
{
    float3 directionCellPerWorld =
        worldToLocalVector(normalizeSafe(directionWorld), instance) / asset.voxelSize.xyz;
    return length(directionCellPerWorld);
}

float worldRayAdvance(
    float3 directionWorld,
    VoxelInstanceData instance,
    VoxelAssetData asset
)
{
    // RAY_BIAS is defined in cell units. Convert it back to world units along
    // this ray, which is required for non-uniformly scaled instances.
    return RAY_BIAS / max(cellDistancePerWorldUnit(directionWorld, instance, asset), 1e-8f);
}

float worldDistanceForCellDistance(
    float3 directionWorld,
    float distanceCells,
    VoxelInstanceData instance,
    VoxelAssetData asset
)
{
    return distanceCells /
        max(cellDistancePerWorldUnit(directionWorld, instance, asset), 1e-8f);
}

float sceneHitMergeDistance(WorldHitResult hit, float3 rayDirectionW)
{
    if (!hit.hit)
        return kSceneHitTieEpsilon;

    VoxelInstanceData instance = instances[hit.instanceIndex];
    VoxelAssetData asset = voxelAssets[instance.assetIndex];
    return max(
        worldDistanceForCellDistance(rayDirectionW, 4.0f * RAY_BIAS, instance, asset),
        kSceneHitTieEpsilon
    );
}

float3 offsetRayOriginAlongNormal(
    WorldHitResult hit,
    float3 outgoingDirectionW,
    float normalBiasCells
)
{
    VoxelInstanceData instance = instances[hit.instanceIndex];
    VoxelAssetData asset = voxelAssets[instance.assetIndex];
    float3 directionW = normalizeSafe(outgoingDirectionW);
    float3 normalW = normalizeSafe(hit.normalW);
    if (all(normalW == float3(0.0f)))
        normalW = -directionW;

    float side = dot(directionW, normalW) >= 0.0f ? 1.0f : -1.0f;
    float biasWorld = worldDistanceForCellDistance(
        normalW,
        max(normalBiasCells, RAY_BIAS),
        instance,
        asset
    );
    return hit.positionW + side * biasWorld * normalW;
}

bool intersectRayBounds(
    float3 boundsMin,
    float3 boundsMax,
    float3 rayOrigin,
    float3 rayDirectionPerWorld,
    float rayTMin,
    float rayTMax,
    out float enterT,
    out float exitT
)
{
    const float kDirectionEpsilon = 1e-8f;
    enterT = rayTMin;
    exitT = rayTMax;

    if (abs(rayDirectionPerWorld.x) < kDirectionEpsilon &&
        (rayOrigin.x < boundsMin.x || rayOrigin.x > boundsMax.x))
    {
        return false;
    }
    if (abs(rayDirectionPerWorld.y) < kDirectionEpsilon &&
        (rayOrigin.y < boundsMin.y || rayOrigin.y > boundsMax.y))
    {
        return false;
    }
    if (abs(rayDirectionPerWorld.z) < kDirectionEpsilon &&
        (rayOrigin.z < boundsMin.z || rayOrigin.z > boundsMax.z))
    {
        return false;
    }

    float3 direction = rayDirectionPerWorld;
    if (abs(direction.x) < kDirectionEpsilon)
        direction.x = direction.x < 0.0f ? -kDirectionEpsilon : kDirectionEpsilon;
    if (abs(direction.y) < kDirectionEpsilon)
        direction.y = direction.y < 0.0f ? -kDirectionEpsilon : kDirectionEpsilon;
    if (abs(direction.z) < kDirectionEpsilon)
        direction.z = direction.z < 0.0f ? -kDirectionEpsilon : kDirectionEpsilon;

    float3 reciprocal = 1.0f / direction;
    float3 t0 = (boundsMin - rayOrigin) * reciprocal;
    float3 t1 = (boundsMax - rayOrigin) * reciprocal;
    float3 tNear = min(t0, t1);
    float3 tFar = max(t0, t1);

    // A parallel axis must not constrain the interval when the origin is
    // already inside that slab. The early-outs above handled the outside case.
    if (abs(rayDirectionPerWorld.x) < kDirectionEpsilon)
    {
        tNear.x = -kSceneRayMax;
        tFar.x = kSceneRayMax;
    }
    if (abs(rayDirectionPerWorld.y) < kDirectionEpsilon)
    {
        tNear.y = -kSceneRayMax;
        tFar.y = kSceneRayMax;
    }
    if (abs(rayDirectionPerWorld.z) < kDirectionEpsilon)
    {
        tNear.z = -kSceneRayMax;
        tFar.z = kSceneRayMax;
    }

    enterT = max(rayTMin, max(max(tNear.x, tNear.y), tNear.z));
    exitT = min(rayTMax, min(min(tFar.x, tFar.y), tFar.z));
    return exitT >= enterT;
}

bool intersectWorldBounds(
    float3 rayOriginW,
    float3 rayDirectionW,
    float rayTMin,
    float rayTMax,
    VoxelInstanceData instance,
    out float enterT,
    out float exitT
)
{
    return intersectRayBounds(
        instance.worldBoundsMin.xyz,
        instance.worldBoundsMax.xyz,
        rayOriginW,
        rayDirectionW,
        rayTMin,
        rayTMax,
        enterT,
        exitT
    );
}

// All t values are world-space distances along a normalized world ray.
WorldHitResult traceInstance(
    float3 rayOriginW,
    float3 rayDirectionW,
    float rayTMin,
    float rayTMax,
    float coneTan,
    uint instanceIndex,
    float rejectOriginEventDistanceWorld
)
{
    VoxelInstanceData instance = instances[instanceIndex];
    WorldHitResult result = {};
    result.instanceIndex = instanceIndex;
    result.instanceId = instance.instanceId;
    result.tWorld = kSceneRayMax;

    float3 directionW = normalizeSafe(rayDirectionW);
    if (instance.enabled == 0 || instance.assetIndex >= assetCount ||
        all(directionW == float3(0)) || rayTMax < rayTMin)
        return result;
    VoxelAssetData asset = voxelAssets[instance.assetIndex];

    float broadEnterT = 0.0f;
    float broadExitT = 0.0f;
    if (!intersectWorldBounds(
        rayOriginW,
        directionW,
        rayTMin,
        rayTMax,
        instance,
        broadEnterT,
        broadExitT
    ))
    {
        return result;
    }

    // Convert the world ray into LOD-0 cell coordinates without normalizing
    // first. Its length is the exact cells-per-world-unit conversion for this
    // direction under a non-uniform instance transform.
    float3 originLocal = worldToLocalPoint(rayOriginW, instance);
    float3 originCell = (originLocal - asset.gridMin.xyz) / asset.voxelSize.xyz;
    float3 directionCellPerWorld =
        worldToLocalVector(directionW, instance) / asset.voxelSize.xyz;
    float cellsPerWorld = length(directionCellPerWorld);
    if (!(cellsPerWorld > 1e-8f))
        return result;

    float exactEnterT = 0.0f;
    float exactExitT = 0.0f;
    if (!intersectRayBounds(
        float3(0.0f),
        float3(asset.voxelCountAndDepth.xyz),
        originCell,
        directionCellPerWorld,
        broadEnterT,
        broadExitT,
        exactEnterT,
        exactExitT
    ))
    {
        return result;
    }

    float traceDistanceCells = max((exactExitT - exactEnterT) * cellsPerWorld, 0.0f);
    if (!(traceDistanceCells > RAY_BIAS))
        return result;

    float3 directionCell = directionCellPerWorld / cellsPerWorld;
    float startBiasCells = min(RAY_BIAS, traceDistanceCells * 0.25f);
    float3 fromCell = originCell + exactEnterT * directionCellPerWorld +
        startBiasCells * directionCell;

    bool startsInsideGrid = all(originCell >= float3(0.0f)) &&
        all(originCell < float3(asset.voxelCountAndDepth.xyz));
    float rejectNearHitDistanceCells = -1.0f;
    if (startsInsideGrid && rejectOriginEventDistanceWorld >= 0.0f)
    {
        rejectNearHitDistanceCells = max(
            rejectOriginEventDistanceWorld * cellsPerWorld,
            4.0f * RAY_BIAS
        );
    }

    HitResult localHit = rayMarching(
        asset,
        fromCell,
        directionCell,
        max(traceDistanceCells - startBiasCells, 0.0f),
        rejectNearHitDistanceCells,
        instance.screenLOD
    );
    result = makeWorldHitResult(
        localHit,
        rayOriginW,
        directionW,
        instanceIndex,
        instance,
        asset,
        coneTan
    );

    // Validate the returned world hit against the original finite query
    // interval as a final guard.
    if (result.hit &&
        (result.tWorld < exactEnterT - kSceneHitTieEpsilon ||
         result.tWorld > rayTMax + kSceneHitTieEpsilon ||
         result.tWorld > exactExitT + kSceneHitTieEpsilon))
    {
        result.hit = false;
        result.tWorld = kSceneRayMax;
    }
    return result;
}

// Returns the nearest world-space hit event. Hits from different instances are
// grouped when their ray depths differ only by a scale-aware merge tolerance.
// A non-negative rejectOriginEventDistanceWorld consumes near-origin members
// of the previous event without pre-skipping an entire LOD node.
#if USE_INSTANCE_BVH
SceneHitEvent traceSceneBVH(
    float3 rayOriginW,
    float3 rayDirectionW,
    float rayTMin,
    float rayTMax,
    float coneTan,
    float rejectOriginEventDistanceWorld
);
#endif

void mergeCandidate(
    inout SceneHitEvent event,
    WorldHitResult candidate,
    float3 directionW
);

SceneHitEvent traceScene(
    float3 rayOriginW,
    float3 rayDirectionW,
    float rayTMin,
    float rayTMax,
    float coneTan,
    float rejectOriginEventDistanceWorld
)
{
    SceneHitEvent event = {};
    event.representative.instanceIndex = kInvalidInstanceIndex;
    event.representative.instanceId = 0xffffffffu;
    event.representative.tWorld = kSceneRayMax;
    event.tMinWorld = kSceneRayMax;
    event.tMaxWorld = -kSceneRayMax;

    float3 directionW = normalizeSafe(rayDirectionW);
    if (all(directionW == float3(0)) || rayTMax < rayTMin)
        return event;

#if USE_INSTANCE_BVH
    if (instanceBVHNodeCount > 0)
        return traceSceneBVH(rayOriginW, directionW, rayTMin, rayTMax, coneTan, rejectOriginEventDistanceWorld);
#endif

    for (uint i = 0; i < instanceCount; ++i)
    {
        WorldHitResult candidate = traceInstance(
            rayOriginW,
            directionW,
            rayTMin,
            rayTMax,
            coneTan,
            i,
            rejectOriginEventDistanceWorld
        );
        if (!candidate.hit)
            continue;

        mergeCandidate(event, candidate, directionW);
    }
    return event;
}

// Merge a single traced instance candidate into the running scene event.
// Shared by both the linear scan and the BVH traversal so the two paths keep
// identical event semantics.
void mergeCandidate(
    inout SceneHitEvent event,
    WorldHitResult candidate,
    float3 directionW
)
{
    if (!candidate.hit)
        return;

    float candidateMergeDistance = sceneHitMergeDistance(candidate, directionW);
    float eventTolerance = max(event.mergeDistanceWorld, candidateMergeDistance);
    bool startsCloserEvent = !event.hit ||
        candidate.tWorld < event.tMinWorld - eventTolerance;
    bool joinsCurrentEvent = event.hit && !startsCloserEvent &&
        candidate.tWorld <= event.tMinWorld + eventTolerance;

    if (!startsCloserEvent && !joinsCurrentEvent)
        return;

    float candidateCoverage = 0.0f;
    if (drawMode == uint(ABSDFDrawMode::Coverage))
    {
        TEBSDF candidateBSDF = loadGBuffer(candidate.localHit.gbIndex);
        candidateCoverage = saturate(
            candidateBSDF.calcCoverage(candidate.directionLocal, coverageBlend)
        );
    }

    if (startsCloserEvent)
    {
        event.hit = true;
        event.representative = candidate;
        event.tMinWorld = candidate.tWorld;
        event.tMaxWorld = candidate.tWorld;
        event.mergeDistanceWorld = candidateMergeDistance;
        event.unionCoverage = candidateCoverage;
        event.memberCount = 1u;
        return;
    }

    event.tMinWorld = min(event.tMinWorld, candidate.tWorld);
    event.tMaxWorld = max(event.tMaxWorld, candidate.tWorld);
    event.mergeDistanceWorld = max(event.mergeDistanceWorld, candidateMergeDistance);
    event.unionCoverage = max(event.unionCoverage, candidateCoverage);
    event.memberCount++;

    bool stableRepresentative = candidate.instanceId < event.representative.instanceId ||
        (candidate.instanceId == event.representative.instanceId &&
         candidate.instanceIndex < event.representative.instanceIndex);
    if (stableRepresentative)
        event.representative = candidate;
}

#if USE_INSTANCE_BVH
// Traverse the instance-level BVH near-to-far. Nodes are pruned once their
// AABB entry lies beyond the running event's merge window, so spatially distant
// clusters of instances are skipped without ever tracing them.
SceneHitEvent traceSceneBVH(
    float3 rayOriginW,
    float3 rayDirectionW,
    float rayTMin,
    float rayTMax,
    float coneTan,
    float rejectOriginEventDistanceWorld
)
{
    SceneHitEvent event = {};
    event.representative.instanceIndex = kInvalidInstanceIndex;
    event.representative.instanceId = 0xffffffffu;
    event.representative.tWorld = kSceneRayMax;
    event.tMinWorld = kSceneRayMax;
    event.tMaxWorld = -kSceneRayMax;

    float3 directionW = normalizeSafe(rayDirectionW);
    if (all(directionW == float3(0)) || rayTMax < rayTMin)
        return event;

    // Fixed-size stack. instanceBVHNodeCount is a CB-sized constant so this
    // unrolls to a fully compile-time stack depth.
    uint stack[kInstanceBVHMaxStack];
    uint stackTop = 0;
    stack[stackTop++] = 0u; // root

    while (stackTop > 0)
    {
        const uint nodeIndex = stack[--stackTop];
        InstanceBVHNode node = instanceBVH[nodeIndex];

        float nodeEnterT = 0.0f;
        float nodeExitT = 0.0f;
        if (!intersectRayBounds(
            node.boundsMin.xyz,
            node.boundsMax.xyz,
            rayOriginW,
            directionW,
            rayTMin,
            rayTMax,
            nodeEnterT,
            nodeExitT
        ))
            continue;

        // Prune this node: its box begins beyond the running event depth plus
        // the accumulated merge tolerance, so any surface inside it lies beyond
        // the event's accept window and cannot join. Only this node is skipped
        // -- a plain stack does not guarantee global near-to-far order, so
        // remaining entries may still be nearer and must still be visited.
        // Assumes no *later* candidate needs a larger merge window than the
        // accumulated max, i.e. instance voxel scales are comparable.
        if (event.hit &&
            nodeEnterT > event.tMinWorld + event.mergeDistanceWorld + kBVHPruneEpsilon)
            continue;

        if (node.info.x > 0u) // leaf
        {
            WorldHitResult candidate = traceInstance(
                rayOriginW,
                directionW,
                rayTMin,
                rayTMax,
                coneTan,
                node.info.z, // instance position (== index into instances[])
                rejectOriginEventDistanceWorld
            );
            mergeCandidate(event, candidate, directionW);
            continue;
        }

        // Interior: push both intersecting children so the nearer one is
        // popped first.
        const uint leftChild = nodeIndex + 1u;
        const uint rightChild = nodeIndex + node.info.y;

        float leftEnterT = 0.0f;
        float rightEnterT = 0.0f;
        float childExitT = 0.0f;
        InstanceBVHNode leftNode = instanceBVH[leftChild];
        InstanceBVHNode rightNode = instanceBVH[rightChild];

        bool leftHit = intersectRayBounds(
            leftNode.boundsMin.xyz, leftNode.boundsMax.xyz,
            rayOriginW, directionW, rayTMin, rayTMax,
            leftEnterT, childExitT);
        bool rightHit = intersectRayBounds(
            rightNode.boundsMin.xyz, rightNode.boundsMax.xyz,
            rayOriginW, directionW, rayTMin, rayTMax,
            rightEnterT, childExitT);

        if (leftHit && rightHit)
        {
            if (leftEnterT <= rightEnterT)
            {
                stack[stackTop++] = rightChild;
                stack[stackTop++] = leftChild;
            }
            else
            {
                stack[stackTop++] = leftChild;
                stack[stackTop++] = rightChild;
            }
        }
        else if (leftHit)
        {
            stack[stackTop++] = leftChild;
        }
        else if (rightHit)
        {
            stack[stackTop++] = rightChild;
        }
    }

    return event;
}
#endif // USE_INSTANCE_BVH

float calcTrasmissionVisibility(WorldHitResult hit, float3 directionWorld, RayCone cone)
{
    if (!hit.hit)
        return 1.0f;

    float3 directionW = normalizeSafe(directionWorld);
    float3 originW = offsetRayOriginAlongNormal(hit, directionW, RAY_BIAS);
    SceneHitEvent blocker = traceScene(
        originW,
        directionW,
        0.0f,
        kSceneRayMax,
        cone.tanHalfAngle,
        sceneHitMergeDistance(hit, directionW)
    );
    return blocker.hit ? 0.0f : 1.0f;
}

float calcLightVisibility(
    WorldHitResult hit,
    float3 directionWorld,
    float distanceWorld,
    RayCone cone
)
{
    if (!hit.hit)
        return 1.0f;

    VoxelInstanceData instance = instances[hit.instanceIndex];
    VoxelAssetData asset = voxelAssets[instance.assetIndex];
    float3 directionW = normalizeSafe(directionWorld);
    if (all(directionW == float3(0)))
        return 1.0f;

    // Spawn on the outgoing side of the geometric surface. Converting the
    // normal offset from cells to world units keeps it stable under a
    // non-uniform instance transform.
    float3 originW = offsetRayOriginAlongNormal(
        hit,
        directionW,
        max(
            worldDistanceToCellDistance(hit.normalW, shadowBiasWorld, instance, asset),
            RAY_BIAS
        )
    );

    const float ignoreDistanceCells = 2.0f;
    float ignoreDistanceWorld = ignoreDistanceCells /
        max(cellDistancePerWorldUnit(directionW, instance, asset), 1e-8f);
    if (distanceWorld >= 0.0f && distanceWorld <= ignoreDistanceWorld)
        return 1.0f;

    float maxDistanceWorld = distanceWorld >= 0.0f
        ? max(distanceWorld - ignoreDistanceWorld, 0.0f)
        : kSceneRayMax;
    SceneHitEvent blocker = traceScene(
        originW,
        directionW,
        0.0f,
        maxDistanceWorld,
        cone.tanHalfAngle,
        sceneHitMergeDistance(hit, directionW)
    );
    return blocker.hit ? 0.0f : 1.0f;
}

float3 coverageTest(SceneHitEvent firstEvent, float3 directionWorld, RayCone cone)
{
    if (!firstEvent.hit)
        return valueToColor(0.0f);

    float3 directionW = normalizeSafe(directionWorld);
    SceneHitEvent currentEvent = firstEvent;
    float coverage = 0.0f;

    // Include the first hit and then continue through the complete instance
    // scene. MAX_BOUNCE therefore controls the number of additional layers.
    for (uint layer = 0; layer < MAX_BOUNCE + 1; ++layer)
    {
        WorldHitResult currentHit = currentEvent.representative;
        VoxelInstanceData instance = instances[currentHit.instanceIndex];
        VoxelAssetData asset = voxelAssets[instance.assetIndex];
        coverage += (1.0f - coverage) * currentEvent.unionCoverage;
        if (coverage >= 1.0f - transmittanceThreshold)
        {
            coverage = 1.0f;
            break;
        }

        float3 nextOriginW = currentHit.positionW +
            worldRayAdvance(directionW, instance, asset) * directionW;
        SceneHitEvent nextEvent = traceScene(
            nextOriginW,
            directionW,
            0.0f,
            kSceneRayMax,
            cone.tanHalfAngle,
            max(
                currentEvent.mergeDistanceWorld,
                sceneHitMergeDistance(currentHit, directionW)
            )
        );
        if (!nextEvent.hit)
            break;
        currentEvent = nextEvent;
    }
    return valueToColor(coverage);
}

float3 debugShading(HitResult hit, float3 v, VoxelInstanceData instance)
{
    uint offset = hit.gbIndex;
    TEBSDF bsdf = loadGBuffer(offset);
    Ellipsoid e = loadPBuffer(offset);
    float result = 0;
    switch (drawMode)
    {
    case ABSDFDrawMode::Normal:
        return localToWorldNormal(bsdf.surface.getMainNormal(v), instance) * 0.5 + 0.5;
    case ABSDFDrawMode::Ellipsoid:
        float3 localPos = hit.cell - hit.cellInt;
        float3 d = localPos - e.center;
        float3 p = d - v * dot(d, v);
        float3 Bp = mul(e.B, p);
        float3 Bn = mul(e.B, v);
        float denom = max(dot(v, Bn), 1e-8f);
        float val = dot(p, Bp) - (dot(p, Bn) * dot(p, Bn)) / denom;
        float t = 1.0f - safeSqrt(val);
        return pdfToColor(t, 1.0f);
    case ABSDFDrawMode::PolygonsProjArea:
        result = bsdf.polygonsProjAreaFunc.calc(v);
        break;
    case ABSDFDrawMode::TotalProjArea:
        result = bsdf.totalProjAreaFunc.calc(v);
        break;
    case ABSDFDrawMode::PrimitiveProjArea:
        result = bsdf.primitiveProjAreaFunc.calc(v);
        break;
    case ABSDFDrawMode::InternalVisibility:
        result = bsdf.calcInternalVisibility(v);
        break;
    case ABSDFDrawMode::TotalArea:
        result = bsdf.area;
        break;
    default:
        break;
    }
    return valueToColor(result);
}

float3 shading(uint offset, float3 l, float3 v, float3 h, float3 lightColor)
{
    TEBSDF BSDF = loadGBuffer(offset);
    BSDF.updateCoverage(v, transmittanceThreshold, coverageBlend);
    return lightColor * BSDF.eval(l, v, h);
}

float3 shadingDirect(
    WorldHitResult worldHit,
    float3 v,
    inout UniformSampleGenerator sg,
    RayCone cone
)
{
    HitResult hit = worldHit.localHit;
    VoxelInstanceData instance = instances[worldHit.instanceIndex];
    uint offset = hit.gbIndex;
    TEBSDF BSDF = loadGBuffer(offset);
    LightSample ls = {};
    float3 viewLocal = normalize(v);
    float3 viewWorld = normalize(localToWorldVector(viewLocal, instance));
    bool valid = sampleLight(
        worldHit.positionW,
        viewLocal,
        viewWorld,
        BSDF,
        sg,
        ls,
        instance.localToWorld,
        instance.worldToLocal
    );

    if (!valid || ls.pdf <= 0 || all(ls.h == 0))
        return 0;

    float3 lightLocal = normalize(worldToLocalVector(ls.l, instance));
    // Under non-uniform scale, transforming a world half-vector is not
    // equivalent to the half-vector of the transformed directions.
    float3 halfLocal = normalize(lightLocal + viewLocal);
    ls.pdf = max(ls.pdf, minPdf);
    float visibility = calcLightVisibility(
        worldHit,
        ls.l,
        ls.distance,
        cone
    );

    if (visibility > 0)
        return visibility * shading(offset, lightLocal, viewLocal, halfLocal, ls.lightColor) / ls.pdf;
    return 0;
}

float3 shadingGlobal(
    SceneHitEvent firstEvent,
    inout UniformSampleGenerator sg,
    RayCone cone
)
{
    WorldHitResult[MAX_BOUNCE + 1] hits;
    float[MAX_BOUNCE + 1] eventMergeDistances;
    float3[MAX_BOUNCE + 1] sampledDirectionsLocal;
    float3[MAX_BOUNCE + 1] sampledDirectionsWorld;
    float3[MAX_BOUNCE + 1] sampledHalfVectorsLocal;
    float[MAX_BOUNCE + 1] pdfs;
    hits[0] = firstEvent.representative;
    eventMergeDistances[0] = firstEvent.mergeDistanceWorld;
    uint shadingPointCount = 1;
    bool escapedScene = false;

    for (uint i = 0; i < MAX_BOUNCE; ++i)
    {
        WorldHitResult currentHit = hits[i];
        VoxelInstanceData instance = instances[currentHit.instanceIndex];
        uint offset = currentHit.localHit.gbIndex;
        TEBSDF BSDF = loadGBuffer(offset);
        float3 currentV = -currentHit.directionLocal;
        BSDF.updateCoverage(currentV, transmittanceThreshold, coverageBlend);
        BSDF.sample(
            currentV,
            sampledDirectionsLocal[i],
            sampledHalfVectorsLocal[i],
            sg
        );

        pdfs[i] = BSDF.evalPdf(
            sampledDirectionsLocal[i],
            currentV,
            sampledHalfVectorsLocal[i]
        );
        if (pdfs[i] <= 0)
            break;
        pdfs[i] = max(pdfs[i], minPdf);

        sampledDirectionsWorld[i] = normalizeSafe(
            localToWorldVector(sampledDirectionsLocal[i], instance)
        );
        if (all(sampledDirectionsWorld[i] == float3(0.0f)))
            break;

        float3 nextOriginW = offsetRayOriginAlongNormal(
            currentHit,
            sampledDirectionsWorld[i],
            RAY_BIAS
        );
        SceneHitEvent nextEvent = traceScene(
            nextOriginW,
            sampledDirectionsWorld[i],
            0.0f,
            kSceneRayMax,
            cone.tanHalfAngle,
            max(
                eventMergeDistances[i],
                sceneHitMergeDistance(currentHit, sampledDirectionsWorld[i])
            )
        );
        if (!nextEvent.hit)
        {
            escapedScene = true;
            break;
        }

        hits[i + 1] = nextEvent.representative;
        eventMergeDistances[i + 1] = nextEvent.mergeDistanceWorld;
        shadingPointCount = i + 2;
    }

    float3 reflectLights[MAX_BOUNCE + 1];
    int lastShadingPoint = int(shadingPointCount) - 1;
    for (int i = lastShadingPoint; i >= 0; --i)
    {
        reflectLights[i] = 0;
        WorldHitResult currentHit = hits[i];
        uint currentOffset = currentHit.localHit.gbIndex;
        float3 currentV = -currentHit.directionLocal;

        reflectLights[i] += shadingDirect(currentHit, currentV, sg, cone);
        if (i == lastShadingPoint)
        {
#if USE_ENV_MAP
            if (escapedScene && all(sampledHalfVectorsLocal[i] == float3(0.0f)))
            {
                float3 incidentLight = gEnvMapSampler.eval(sampledDirectionsWorld[i]);
                reflectLights[i] += shading(
                    currentOffset,
                    sampledDirectionsLocal[i],
                    currentV,
                    sampledHalfVectorsLocal[i],
                    incidentLight
                ) / pdfs[i];
            }
#endif
        }
        else
        {
            if (pdfs[i] <= 0)
                continue;
            float3 incidentLight = reflectLights[i + 1];
            // Clamp fireflies from low-pdf BSDF samples: a lit diffuse
            // surface is ~0.25, so 10.0 catches spikes while leaving normal
            // indirect paths untouched.
            incidentLight = min(incidentLight, 10.0);
            reflectLights[i] += shading(
                currentOffset,
                sampledDirectionsLocal[i],
                currentV,
                sampledHalfVectorsLocal[i],
                incidentLight
            ) / pdfs[i];
        }
    }
    return reflectLights[0];
}

float3 screenCoordToWorld(float2 screenCoord, float NDCDepth)
{
    float4 NDC = float4(2 * screenCoord.x - 1, 1 - 2 * screenCoord.y, NDCDepth, 1);
    float4 temp = mul(invVP, NDC);
    return (temp / temp.w).xyz;
}

float2 randomJitter()
{
    UniformSampleGenerator sg = UniformSampleGenerator(0, frameIndex);
    float2 jitter = Next2(sg) - 0.5f;
    jitter /= pixelCount;
    return jitter;
}

float4 renderBackground(float3 direction)
{
#if USE_ENV_MAP
    if (renderBackGround)
    {
        float3 color = gScene.envMap.eval(direction);
        return float4(color, 1);
    }
#endif
    return clearColor;
}

// Simple Lambert + shadow-ray shading for LOD heatmap visualization.
// Samples the scene light and traces a visibility ray; baseColor is the
// LOD heatmap hue modulated by N·L and shadow.
float3 lambertShading(
    WorldHitResult worldHit,
    float3 v,
    inout UniformSampleGenerator sg,
    RayCone cone,
    float3 baseColor
)
{
    HitResult hit = worldHit.localHit;
    VoxelInstanceData instance = instances[worldHit.instanceIndex];
    TEBSDF BSDF = loadGBuffer(hit.gbIndex);
    float3 viewLocal = normalize(v);
    float3 viewWorld = normalize(localToWorldVector(viewLocal, instance));

    LightSample ls = {};
    bool valid = sampleLight(
        worldHit.positionW,
        viewLocal,
        viewWorld,
        BSDF,
        sg,
        ls,
        instance.localToWorld,
        instance.worldToLocal
    );

    // Ambient fallback (no valid light sample)
    float3 ambient = baseColor * 0.04;

    if (!valid || ls.pdf <= 0)
        return ambient;

    float3 lightLocal = normalize(worldToLocalVector(ls.l, instance));
    float3 N = BSDF.surface.getMainNormal(lightLocal);
    float NdotL = saturate(dot(N, lightLocal));

    float visibility = calcLightVisibility(
        worldHit,
        ls.l,
        ls.distance,
        cone
    );

    // Lambert: diffuse = baseColor * (N·L) * lightColor * visibility / (π * pdf)
    float3 diffuse = baseColor * NdotL * ls.lightColor * visibility * (1.0 / Pi) / max(ls.pdf, minPdf);

    return ambient + diffuse;
}

PsOut main(float2 texCoord: TEXCOORD, float4 pos: SV_POSITION)
{
    PsOut psOut = {};
    uint2 pixel = uint2(texCoord * pixelCount);
    float2 jitter = randomJitter();
    texCoord = saturate(texCoord + jitter);
    float3 rayOriginWorld = screenCoordToWorld(texCoord, 0);
    float3 rayEndWorld = screenCoordToWorld(texCoord, 1);
    float3 directionWorld = normalize(rayEndWorld - rayOriginWorld);
    psOut.color = renderBackground(directionWorld);

    if (assetCount == 0 || instanceCount == 0)
        return psOut;

    // Primary ray cone: pixel footprint
    // tan(coneHalfAngle) = tan(HalfFovY) / pixelHeight at distance=1
    float coneTan = tanHalfFovY / pixelCount.y;

    UniformSampleGenerator sg = UniformSampleGenerator(pixel, frameIndex);
    float primaryMaxDistance = length(rayEndWorld - rayOriginWorld);
    SceneHitEvent primaryEvent = traceScene(
        rayOriginWorld,
        directionWorld,
        0.0f,
        primaryMaxDistance,
        coneTan,
        -1.0f
    );
    WorldHitResult worldHit = primaryEvent.representative;
    VoxelInstanceData selectedInstance = {};
    VoxelAssetData selectedAsset = {};
    if (worldHit.hit)
    {
        selectedInstance = instances[worldHit.instanceIndex];
        selectedAsset = voxelAssets[selectedInstance.assetIndex];
    }

    if (all(pixel == selectedPixel))
    {
        if (worldHit.hit)
        {
            // DisplayNDF evaluates the selected lobe in asset-local space.
            selectedVoxel[0] = float4(-worldHit.directionLocal, (float)worldHit.localHit.gbIndex);
            selectedVoxel[1] = float4(float3(worldHit.localHit.cellInt), (float)worldHit.instanceId);
        }
        else
        {
            selectedVoxel[0] = float4(-1);
            selectedVoxel[1] = float4(-1);
        }
    }

    if (worldHit.hit)
    {
        // Keep traversal and shading in asset-local space; primary-hit
        // consumers can use the world-space result above.
        HitResult hit = worldHit.localHit;
        float3 directionLocal = worldHit.directionLocal;
        RayCone primaryCone = { coneTan, worldHit.rayConeBias };
        float3 color = 0;
        switch (drawMode)
        {
        case ABSDFDrawMode::Default:
#if USE_EMISSIVE_LIGHTS
        {
            if (checkHitEmissive(
                worldHit.positionW,
                worldVoxelSize(selectedInstance, selectedAsset),
                color
            ))
            {
                psOut.color = float4(color, 1);
                return psOut;
            }
        }
#endif
            color = shadingGlobal(primaryEvent, sg, primaryCone);
            break;
        case ABSDFDrawMode::Coverage:
            color = coverageTest(primaryEvent, directionWorld, primaryCone);
            break;
        case ABSDFDrawMode::LODHeatmap:
        {
            Ellipsoid e = loadPBuffer(hit.gbIndex);
            TEBSDF bsdf = loadGBuffer(hit.gbIndex);

            SurfaceBRDF surface = bsdf.surface;


        }
            break;
        case ABSDFDrawMode::LODLevel:
        {
            float t = saturate(
                hit.lodLevel / max(float(selectedAsset.voxelCountAndDepth.w), 1.0f)
            );
            uint lodLevel = firstbithigh(hit.nodeSize);

            const float3 colors[] = {
                float3(1.0, 1.0, 0.0), // c0: 底部 (黄色)
                float3(0.0, 1.0, 0.0), // c1: 绿色段
                float3(0.0, 1.0, 1.0), // c2: 青色段
                float3(0.0, 0.0, 1.0), // c3: 蓝色段
                float3(1.0, 0.0, 1.0), // c4: 品红色段
                float3(1.0, 0.0, 0.0), // c5: 顶部 (红色)
            };

            float3 baseColor;
            float sectionT; // 当前段内的局部 t 值

            if (t < 0.2f) { // 第1段: 黄色到绿色
                sectionT = t / 0.2f;
                baseColor = lerp(colors[0], colors[1], sectionT);
            } else if (t < 0.4f) { // 第2段: 绿色到青色
                sectionT = (t - 0.2f) / 0.2f;
                baseColor = lerp(colors[1], colors[2], sectionT);
            } else if (t < 0.6f) { // 第3段: 青色到蓝色
                sectionT = (t - 0.4f) / 0.2f;
                baseColor = lerp(colors[2], colors[3], sectionT);
            } else if (t < 0.8f) { // 第4段: 蓝色到品红色
                sectionT = (t - 0.6f) / 0.2f;
                baseColor = lerp(colors[3], colors[4], sectionT);
            } else { // 第5段: 品红色到红色
                sectionT = (t - 0.8f) / 0.2f;
                baseColor = lerp(colors[4], colors[5], sectionT);
            }

            baseColor = colors[min(lodLevel, 5u)]; // Clamp to max color index
            color = lambertShading(worldHit, -directionLocal, sg, primaryCone, baseColor);
        }
            break;
        default:
            color = debugShading(hit, -directionLocal, selectedInstance);
            break;
        }
        psOut.color = float4(color, 1);
    }
    return psOut;
}
