#include "Voxel/VoxelGrid.slang"
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

#define RAY_BIAS 1e-4f // cell-space t-offset to avoid self-intersection

StructuredBuffer<TEBSDF> gBuffer[RAY_MARCHING_BUFFER_COUNT];
StructuredBuffer<Ellipsoid> pBuffer[RAY_MARCHING_BUFFER_COUNT];
StructuredBuffer<OctreeNode> octreeBuffer;

RWStructuredBuffer<float4> selectedVoxel;

cbuffer GridData
{
    float3 gridMin;
    float3 voxelSize;
    uint3 voxelCount;
    uint octreeMaxDepth;
}

cbuffer CB
{
    uint2 pixelCount;
    float4x4 invVP;
    float4x4 instanceTransform;        // local grid -> world
    float4x4 inverseInstanceTransform; // world -> local grid
    float4x4 normalTransform;           // local normal -> world normal
    float shadowBias;
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
    int screenLOD;      // per-frame grid-wide LOD selected from projected grid area
    uint availableLODLevels; // generated parent levels; 0 means leaf data only
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
    uint gbIndex;    // gBuffer offset at leaf node
    uint nodeSize;   // size in LOD-0 cells of the node where hit occurred (1=leaf, >1=LOD stop)
    float lodLevel;  // selected discrete LOD level; 0=leaf
};

struct PsOut
{
    float4 color : SV_Target0;
};

bool outsideGrid(int3 cellInt)
{
    return any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount);
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

HitResult rayHitTE(Ellipsoid ellipsoid, float3 cell, float3 direction, int3 cellInt, uint gbIndex, uint nodeSize)
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

    HitResult result = { inOut.x >= 0, hitCellInt, hitCell, gbIndex, nodeSize, 0.0f };
    return result;
}

HitResult rayHitVoxel(float3 cell, float3 direction, int3 cellInt, uint gbIndex, uint nodeSize)
{
#if !CHECK_ELLIPSOID
    return { true, cellInt, cellInt + 0.5f, gbIndex, nodeSize, 0.0f };
#else
    Ellipsoid ellipsoid = loadPBuffer(gbIndex);
    HitResult hit = rayHitTE(ellipsoid, cell, direction, cellInt, gbIndex, nodeSize);
    if (hit.hit)
        return hit;
#endif
    return { false, cellInt, cellInt + 0.5f, 0, nodeSize, 0.0f };
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
// ellipsoid intersection tests, and ignoreNodeSize pre-skip from the original
// parameterized stack-based traversal.
// from: starting point in LOD-0 cell space
HitResult rayMarching(float3 from, float3 direction, float maxDistance, uint ignoreNodeSize, int fixedLod)
{
    direction = normalizeSafe(direction);
    if (all(direction == float3(0)))
        return { false, int3(from), from, 0, 0, 0.0f };

    // ===== Phase 1: ignoreNodeSize pre-skip (unchanged logic) =====
    float3 rayOrigin = from;

    if (ignoreNodeSize > 0)
    {
        float3 cell = from;
        int3 cellInt = int3(floor(from));
        int3 dirs = int3(sign(direction));
        float3 reciprocal = 1.0f / direction;

        if (ignoreNodeSize == 1)
        {
            leave(cell, cellInt, dirs, direction, reciprocal);
        }
        else
        {
            float3 cellFloat = cell;
            int3 blockMin;
            blockMin.x = int(floor(cellFloat.x / float(ignoreNodeSize))) * int(ignoreNodeSize);
            blockMin.y = int(floor(cellFloat.y / float(ignoreNodeSize))) * int(ignoreNodeSize);
            blockMin.z = int(floor(cellFloat.z / float(ignoreNodeSize))) * int(ignoreNodeSize);

            int3 exitFace;
            exitFace.x = dirs.x > 0 ? blockMin.x + int(ignoreNodeSize) : blockMin.x;
            exitFace.y = dirs.y > 0 ? blockMin.y + int(ignoreNodeSize) : blockMin.y;
            exitFace.z = dirs.z > 0 ? blockMin.z + int(ignoreNodeSize) : blockMin.z;

            float3 tToExit = (float3(exitFace) - cell) * reciprocal;
            float tMin = 2.0f * float(ignoreNodeSize);
            if (direction.x != 0 && tToExit.x < tMin) tMin = tToExit.x;
            if (direction.y != 0 && tToExit.y < tMin) tMin = tToExit.y;
            if (direction.z != 0 && tToExit.z < tMin) tMin = tToExit.z;
            if (tMin < 0) tMin = 0;

            cell += tMin * direction;
            cellInt = int3(floor(cell + direction * RAY_BIAS));
        }

        rayOrigin = cell + direction * RAY_BIAS;
    }

    // ===== Phase 2: normalize + mirror for negative-direction traversal =====
    uint rootSize = 1u << octreeMaxDepth;

    // LOD is selected once per frame from the projected grid size. The
    // optional arguments remain useful for explicit debug/test overrides, but
    // no ray-distance based LOD decision is made during traversal.
    int targetLOD = screenLOD;
    if (forcedLOD >= 0)
        targetLOD = forcedLOD;
    if (fixedLod >= 0)
        targetLOD = fixedLod;
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

    // ===== Phase 3: root entry / exit in parametric space =====
    precise float t_min = max(max(2.0f * coef.x - bias.x, 2.0f * coef.y - bias.y), 2.0f * coef.z - bias.z);
    precise float t_max = min(min(1.0f * coef.x - bias.x, 1.0f * coef.y - bias.y), 1.0f * coef.z - bias.z);

    float h = t_max;
    t_min = max(t_min, 0.0f);

    // Clamp far end by maxDistance (convert cell-space distance → parametric t)
    if (maxDistance > 0)
        t_max = min(t_max, t_min + maxDistance / float(rootSize));

    if (t_min > t_max)
        return { false, int3(0), float3(0), 0, 0, 0.0f };

    // ===== Phase 4: initialize state machine =====
    int2 stack[S_MAX + 1];

    uint parent = 0;
    uint idx = 0;
    precise float3 pos = float3(1.0f, 1.0f, 1.0f);
    int scale = S_MAX - 1;
    precise float scale_exp2 = 0.5f;

    // Determine which sub-octant the ray enters first
    if (1.5f * coef.x - bias.x > t_min) { idx ^= 1; pos.x = 1.5f; }
    if (1.5f * coef.y - bias.y > t_min) { idx ^= 2; pos.y = 1.5f; }
    if (1.5f * coef.z - bias.z > t_min) { idx ^= 4; pos.z = 1.5f; }

    // ===== Phase 5: main traversal loop (INTERSECT → PUSH | ADVANCE → POP) =====
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

                if (!outsideGrid(cellInt))
                {
                    uint gbIndex = childNode.dataIndex;

                    HitResult hit = rayHitVoxel(hitCell, direction, cellInt, gbIndex, nodeSize);
                    if (hit.hit)
                    {
                        hit.lodLevel = float(nodeLOD);
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

    return { false, int3(0), float3(0), 0, 0, 0.0f };
}

float calcTrasmissionVisibility(HitResult hit, float3 direction, RayCone cone)
{
    hit = rayMarching(hit.cell, direction, -1, hit.nodeSize, -1);
    return hit.hit ? 0 : 1;
}

float calcLightVisibility(HitResult hit, float3 direction, float3 mainNormal, float distance, RayCone cone)
{
    float ignoreDistance = 2;
    if (distance >= 0 && distance <= ignoreDistance)
        return 1;
    float3 from = hit.cell + shadowBias * mainNormal;
    hit = rayMarching(from, direction, distance - ignoreDistance, hit.nodeSize, -1);
    return hit.hit ? 0 : 1;
}

float3 coverageTest(HitResult hit, float3 direction, RayCone cone)
{
    float coverage = 0;
    uint offset = hit.gbIndex;
    float times = MAX_BOUNCE;
    while (times > 0)
    {
        TEBSDF bsdf = loadGBuffer(offset);
        coverage += (1 - coverage) * bsdf.calcCoverage(direction, coverageBlend);
        if (coverage >= 1 - transmittanceThreshold)
        {
            coverage = 1;
            break;
        }
        hit = rayMarching(hit.cell, direction, -1, hit.nodeSize, -1);
        if (!hit.hit)
            break;
        offset = hit.gbIndex;
        times--;
    }
    return valueToColor(coverage);
}

float3 debugShading(HitResult hit, float3 v)
{
    uint offset = hit.gbIndex;
    TEBSDF bsdf = loadGBuffer(offset);
    Ellipsoid e = loadPBuffer(offset);
    float result = 0;
    switch (drawMode)
    {
    case ABSDFDrawMode::Normal:
        return bsdf.surface.getMainNormal(v) * 0.5 + 0.5;
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

float3 localToWorldPoint(float3 pointLocal)
{
    return mul(instanceTransform, float4(pointLocal, 1.0f)).xyz;
}

float3 localToWorldVector(float3 vectorLocal)
{
    return mul(instanceTransform, float4(vectorLocal, 0.0f)).xyz;
}

float3 worldToLocalPoint(float3 pointWorld)
{
    return mul(inverseInstanceTransform, float4(pointWorld, 1.0f)).xyz;
}

float3 worldToLocalVector(float3 vectorWorld)
{
    return mul(inverseInstanceTransform, float4(vectorWorld, 0.0f)).xyz;
}

float3 localToWorldNormal(float3 normalLocal)
{
    return normalize(mul(normalTransform, float4(normalLocal, 0.0f)).xyz);
}

float3 shadingDirect(HitResult hit, float3 v, inout UniformSampleGenerator sg, RayCone cone)
{
    uint offset = hit.gbIndex;
    TEBSDF BSDF = loadGBuffer(offset);
    float3 posLocal = gridMin + hit.cell * voxelSize;
    float3 posW = localToWorldPoint(posLocal);
    LightSample ls = {};
    bool valid = sampleLight(posW, v, BSDF, sg, ls);

    if (!valid || ls.pdf <= 0 || all(ls.h == 0))
        return 0;

    ls.pdf = max(ls.pdf, minPdf);
    float3 mainNormal = BSDF.surface.getMainNormal(ls.l);
    float visibility = calcLightVisibility(hit, ls.l, mainNormal, ls.distance / voxelSize.x, cone);

    if (visibility > 0)
        return visibility * shading(offset, ls.l, v, ls.h, ls.lightColor) / ls.pdf;
    return 0;
}

float3 shadingGlobal(HitResult hit, float3 v, inout UniformSampleGenerator sg, RayCone cone)
{
    HitResult[MAX_BOUNCE + 1] hits;
    float3[MAX_BOUNCE + 1] ls;
    float3[MAX_BOUNCE + 1] hs;
    float[MAX_BOUNCE + 1] pdfs;
    uint bounce = MAX_BOUNCE;
    uint shadingPointCount = bounce + 1;
    hits[0] = hit;
    bool leaveGrid = false;

    for (int i = 0; i < bounce; i++)
    {
        uint offset = hits[i].gbIndex;
        TEBSDF BSDF = loadGBuffer(offset);
        float3 currentV = i == 0 ? v : -ls[i - 1];
        BSDF.updateCoverage(currentV, transmittanceThreshold, coverageBlend);
        BSDF.sample(currentV, ls[i], hs[i], sg);

        pdfs[i] = BSDF.evalPdf(ls[i], currentV, hs[i]);
        if (pdfs[i] <= 0)
        {
            shadingPointCount = i + 1;
            break;
        }
        pdfs[i] = max(pdfs[i], minPdf);

        hits[i + 1] = rayMarching(hits[i].cell, ls[i], -1, hits[i].nodeSize, -1);
        if (!hits[i + 1].hit)
        {
            leaveGrid = true;
            shadingPointCount = i + 1;
            break;
        }
    }

    float3 reflectLights[MAX_BOUNCE + 1];
    for (int i = shadingPointCount - 1; i >= 0; i--)
    {
        reflectLights[i] = 0;
        HitResult currentHit = hits[i];
        uint currentOffset = currentHit.gbIndex;
        float3 currentV = i == 0 ? v : -ls[i - 1];

        reflectLights[i] += shadingDirect(currentHit, currentV, sg, cone);
        if (i == shadingPointCount - 1)
        {
#if USE_ENV_MAP
            if (leaveGrid && all(hs[i] == 0))
            {
                float3 incidentLight = gEnvMapSampler.eval(ls[i]);
                reflectLights[i] += shading(currentOffset, ls[i], currentV, hs[i], incidentLight) / pdfs[i];
            }
#endif
        }
        else
        {
            if (pdfs[i] <= 0)
                continue;
            HitResult nextHit = hits[i + 1];
            if (nextHit.hit)
            {
                float3 incidentLight = reflectLights[i + 1];
                // Clamp fireflies from low-pdf BSDF samples:
                // a lit diffuse surface is ~0.25, so 10.0 catches spikes
                // while leaving normal indirect paths untouched.
                incidentLight = min(incidentLight, 10.0);
                reflectLights[i] += shading(currentOffset, ls[i], currentV, hs[i], incidentLight) / pdfs[i];
            }
        }
    }
    return reflectLights[0];
}

float3 screenCoordToCell(float2 screenCoord, float NDCDepth)
{
    float4 NDC = float4(2 * screenCoord.x - 1, 1 - 2 * screenCoord.y, NDCDepth, 1);
    float4 temp = mul(invVP, NDC);
    float3 posWorld = (temp / temp.w).xyz;
    float3 posLocal = worldToLocalPoint(posWorld);
    float3 cell = (posLocal - gridMin) / voxelSize;
    return cell;
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
float3 lambertShading(HitResult hit, float3 v, inout UniformSampleGenerator sg, RayCone cone, float3 baseColor)
{
    TEBSDF BSDF = loadGBuffer(hit.gbIndex);
    float3 posLocal = gridMin + hit.cell * voxelSize;
    float3 posW = localToWorldPoint(posLocal);

    LightSample ls = {};
    bool valid = sampleLight(posW, v, BSDF, sg, ls);

    // Ambient fallback (no valid light sample)
    float3 ambient = baseColor * 0.04;

    if (!valid || ls.pdf <= 0)
        return ambient;

    float3 N = BSDF.surface.getMainNormal(ls.l);
    float NdotL = saturate(dot(N, ls.l));

    float visibility = calcLightVisibility(hit, ls.l, N, ls.distance / voxelSize.x, cone);

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
    float3 from = screenCoordToCell(texCoord, 0);
    float3 to = screenCoordToCell(texCoord, 1);
    float3 delta = to - from;
    float3 direction = normalize(delta);
    psOut.color = renderBackground(normalize(localToWorldVector(direction)));

    if (octreeMaxDepth == 0)
        return psOut;

    float2 inOut = clip(float3(0), float3(voxelCount), from, to);
    if (inOut.x < 0)
        return psOut;

    // Primary ray cone: pixel footprint
    // tan(coneHalfAngle) = tan(HalfFovY) / pixelHeight at distance=1
    float coneTan = tanHalfFovY / pixelCount.y;

    UniformSampleGenerator sg = UniformSampleGenerator(pixel, frameIndex);
    from = from + (inOut.x + 1e-6) * delta;

    // Error: need world space distance
    float initialDistance = (inOut.x + 1e-6) * length(delta);
    RayCone primaryCone = { coneTan, initialDistance * coneTan };
    HitResult hit = rayMarching(from, direction, -1, 0, -1);

    if (all(pixel == selectedPixel))
    {
        if (hit.hit)
        {
            selectedVoxel[0] = float4(normalize(localToWorldVector(-direction)), (float)hit.gbIndex);
            selectedVoxel[1] = float4(float3(hit.cellInt), 0);
        }
        else
        {
            selectedVoxel[0] = float4(-1);
            selectedVoxel[1] = float4(-1);
        }
    }

    if (hit.hit)
    {
        float3 color = 0;
        switch (drawMode)
        {
        case ABSDFDrawMode::Default:
#if USE_EMISSIVE_LIGHTS
        {
            float3 posLocal = gridMin + hit.cell * voxelSize;
            float3 posW = localToWorldPoint(posLocal);
            if (checkHitEmissive(posW, voxelSize.x, color))
            {
                psOut.color = float4(color, 1);
                return psOut;
            }
        }
#endif
            color = shadingGlobal(hit, -direction, sg, primaryCone);
            break;
        case ABSDFDrawMode::Coverage:
            color = coverageTest(hit, direction, primaryCone);
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
            float t = saturate(hit.lodLevel / max(float(octreeMaxDepth), 1.0f));
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
            color = lambertShading(hit, -direction, sg, primaryCone, baseColor);
        }
            break;
        default:
            color = debugShading(hit, -direction);
            break;
        }
        psOut.color = float4(color, 1);
    }
    return psOut;
}
