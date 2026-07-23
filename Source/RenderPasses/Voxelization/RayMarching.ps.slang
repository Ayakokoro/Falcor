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

#define USE_BRUTE_FORCE 0  // 1 = DDA cell-by-cell, 0 = parameterized octree traversal

StructuredBuffer<TEBSDF> gBuffer;
StructuredBuffer<Ellipsoid> pBuffer;
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
}

// Ray cone: represents the angular footprint of a ray
// Half-angle tangent grows with BSDF roughness at each bounce
struct RayCone
{
    float tanHalfAngle; // tan(θ/2), where θ is the cone's full opening angle
};

struct HitResult
{
    bool hit;
    int3 cellInt;    // in LOD-0 coordinate space
    float3 cell;     // in LOD-0 coordinate space
    uint gbIndex;    // gBuffer offset at leaf node
};

struct PsOut
{
    float4 color : SV_Target0;
};

bool outsideGrid(int3 cellInt)
{
    return any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount);
}

// Point query: descend octree from root to check if cellInt is solid.
// Returns gbIndex if solid, 0xFFFFFFFF if empty.
// When forcedLOD > 0, stops at the target LOD level instead of descending to leaf.
uint lookupCell(int3 cellInt)
{
    uint targetLevel = (forcedLOD > 0) ? (octreeMaxDepth - (uint)forcedLOD) : octreeMaxDepth;
    uint nodeIndex = 0;
    for (uint level = 0; level < octreeMaxDepth; level++)
    {
        OctreeNode node = octreeBuffer[nodeIndex];

        if (node.childMask == 0 || level == targetLevel)
            return node.dataIndex;

        uint childShift = octreeMaxDepth - level - 1;
        int3 localCell = (cellInt >> childShift) & 1;
        uint childSlot = localCell.x + localCell.y * 2 + localCell.z * 4;

        if ((node.childMask & (1u << childSlot)) != 0 )
            nodeIndex = node.childBase + countbits(node.childMask & ((1u << childSlot) - 1));
        else
            return 0xFFFFFFFF;
    }
    return octreeBuffer[nodeIndex].dataIndex;
}

// Brute-force DDA traversal: step one LOD-0 cell at a time, query octree per cell.
// Used for A/B testing against the parameterized traversal.
HitResult rayMarchingBruteForce(float3 from, float3 direction, float maxDistance, bool ignoreFirst, RayCone cone, float initialDistance)
{
    direction = normalizeSafe(direction);
    if (all(direction == float3(0)))
        return { false, int3(from), from, 0 };

    float3 cell = from;
    int3 cellInt = int3(floor(from));
    float distance = initialDistance;

    float3 reciprocal_direction = 1.0f / direction;
    int3 directions = int3(sign(direction));
    if (ignoreFirst)
        distance += leave(cell, cellInt, directions, direction, reciprocal_direction);

    while (true)
    {
        if (outsideGrid(cellInt))
            break;
        if (maxDistance > 0 && distance > maxDistance)
            break;

        uint gbIndex = lookupCell(cellInt);
        if (gbIndex != 0xFFFFFFFF)
        {
            HitResult hit;
            if (forcedLOD > 0)
            {
                // Convert to node-normalized space for coarser-LOD ellipsoid test
                uint nodeSize = 1u << (uint)forcedLOD;
                int3 nodeMin = (cellInt >> (uint)forcedLOD) << (uint)forcedLOD;
                float3 localHit = (cell - float3(nodeMin)) / float(nodeSize);
#if CHECK_ELLIPSOID
                Ellipsoid e = pBuffer[gbIndex];
                float3 fr = localHit;
                int3 sgn = int3(sign(direction));
                float3 bdr = float3(sgn + 1) * 0.5f;
                float3 tBdr = (bdr - fr) / direction;
                float tMinLoc = 2.0f;
                if (direction.x != 0 && tBdr.x < tMinLoc) tMinLoc = tBdr.x;
                if (direction.y != 0 && tBdr.y < tMinLoc) tMinLoc = tBdr.y;
                if (direction.z != 0 && tBdr.z < tMinLoc) tMinLoc = tBdr.z;
                if (tMinLoc < 0) tMinLoc = 0;
                float3 to = fr + tMinLoc * direction;
                float2 inOut = e.clip(fr, to);
                if (inOut.x >= 0)
                {
                    float3 hitLocal = fr + inOut.x * (to - fr);
                    float3 hitCell = float3(nodeMin) + hitLocal * float(nodeSize);
                    hit = { true, int3(floor(hitCell)), hitCell, gbIndex };
                }
                else
                    hit = { false, int3(cell), cell, 0 };
#else
                hit = { true, nodeMin, float3(nodeMin) + localHit * float(nodeSize), gbIndex };
#endif
            }
            else
            {
                hit = rayHitVoxel(cell, direction, cellInt, gbIndex);
            }
            if (hit.hit)
                return hit;
        }

        distance += leave(cell, cellInt, directions, direction, reciprocal_direction);
    }
    return { false, cellInt, cellInt + 0.5f, 0 };
}

// Extract max roughness across all lobes of a TEBSDF at gbIndex
float maxRoughness(uint gbIndex)
{
    TEBSDF bsdf = gBuffer[gbIndex];
    float r = 0;
    for (uint i = 0; i < bsdf.surface.lobeCount; i++)
        r = max(r, bsdf.surface.lobes[i].frostBite.rough);
    return r;
}

HitResult rayHitTE(Ellipsoid ellipsoid, float3 cell, float3 direction, int3 cellInt, uint gbIndex)
{
    float3 from = cell - cellInt;
    int3 directions = int3(sign(direction));
    float3 border = (directions + 1) / 2;
    float3 t = (border - from) / direction;
    float tMin = 2.0f;
    if (direction.x != 0 && t.x < tMin) tMin = t.x;
    if (direction.y != 0 && t.y < tMin) tMin = t.y;
    if (direction.z != 0 && t.z < tMin) tMin = t.z;
    float3 to = from + tMin * direction;
    float2 inOut = ellipsoid.clip(from, to);
    HitResult result = { inOut.x >= 0, cellInt, cell + inOut.x * (to - from), gbIndex };
    return result;
}

HitResult rayHitVoxel(float3 cell, float3 direction, int3 cellInt, uint gbIndex)
{
#if !CHECK_ELLIPSOID
    return { true, cellInt, cellInt + 0.5f, gbIndex };
#else
    Ellipsoid ellipsoid = pBuffer[gbIndex];
    HitResult hit = rayHitTE(ellipsoid, cell, direction, cellInt, gbIndex);
    if (hit.hit)
        return hit;
#endif
    return { false, cellInt, cellInt + 0.5f, 0 };
}

// Ray-slab AABB intersection in cell space
// tMin is clamped to 0 (only considers what's ahead of the ray origin)
bool intersectAABB(float3 rayOrigin, float3 reciprocal, float3 aabbMin, float3 aabbMax,
                   out float tMin, out float tMax)
{
    float3 t0 = (aabbMin - rayOrigin) * reciprocal;
    float3 t1 = (aabbMax - rayOrigin) * reciprocal;
    float3 tNear = min(t0, t1);
    float3 tFar = max(t0, t1);
    tMin = max(max(tNear.x, tNear.y), max(tNear.z, 0.0f));
    tMax = min(min(tFar.x, tFar.y), tFar.z);
    return tMin <= tMax;
}

#define MAX_OCTREE_STACK 128

struct StackEntry
{
    uint nodeIndex;
    uint level;
    int3 nodeMin; // AABB min corner in LOD-0 cell coords
    float tMin;   // ray entry t for this node (cell-space)
};

// Parameterized octree traversal using a stack ordered by ray-t
// Each internal node: intersect children AABBs → sort by tMin → push (far to near)
// Leaf node: test voxel; on miss the next stack entry is already the next cell along the ray
HitResult rayMarchingParam(float3 from, float3 direction, float maxDistance, bool ignoreFirst, RayCone cone, float initialDistance)
{
    direction = normalizeSafe(direction);
    if (all(direction == float3(0)))
        return { false, int3(from), from, 0 };

    float3 reciprocal = 1.0f / direction;
    float3 rayOrigin = from;

    if (ignoreFirst)
    {
        float3 cell = from;
        int3 cellInt = int3(floor(from));
        int3 dirs = int3(sign(direction));
        leave(cell, cellInt, dirs, direction, reciprocal);
        rayOrigin = cell + direction * 1e-4f;
    }

    // Ray must intersect the octree root AABB
    uint rootSize = 1u << octreeMaxDepth;
    float rootTMin, rootTMax;
    if (!intersectAABB(rayOrigin, reciprocal, float3(0, 0, 0), float3(rootSize, rootSize, rootSize),
                       rootTMin, rootTMax))
        return { false, int3(rayOrigin), rayOrigin, 0 };

    StackEntry stack[MAX_OCTREE_STACK];
    uint stackTop = 0;
    stack[stackTop++] = { 0, 0, int3(0, 0, 0), rootTMin };

    while (stackTop > 0)
    {
        StackEntry entry = stack[--stackTop];

        if (maxDistance > 0 && entry.tMin > maxDistance)
            continue;

        OctreeNode node = octreeBuffer[entry.nodeIndex];

        // --- Leaf node ---
        if (entry.level == octreeMaxDepth || node.childMask == 0)
        {
            uint gbIndex = node.dataIndex;
            int3 cellInt = entry.nodeMin;
            if (!outsideGrid(cellInt))
            {
                float3 hitPoint = rayOrigin + (entry.tMin + 1e-5f) * direction;
                HitResult hit = rayHitVoxel(hitPoint, direction, cellInt, gbIndex);
                if (hit.hit)
                    return hit;
            }
            continue;
        }

        // --- Forced LOD stop: treat coarser node as a "voxel" ---
        uint nodeLOD = octreeMaxDepth - entry.level;
        if (forcedLOD > 0 && (int)nodeLOD == forcedLOD)
        {
            uint gbIndex = node.dataIndex;
            uint nodeSize = 1u << (octreeMaxDepth - entry.level);
            float3 hitPoint = rayOrigin + (entry.tMin + 1e-5f) * direction;

            // Convert to node-normalized space [0,1]^3
            // nodeMin is in leaf-voxel space; node spans [nodeMin, nodeMin+nodeSize]
            float3 localHit = (hitPoint - float3(entry.nodeMin)) / float(nodeSize);

#if CHECK_ELLIPSOID
            Ellipsoid e = pBuffer[gbIndex];
            float3 from = localHit;
            int3 signs = int3(sign(direction));
            float3 border = float3(signs + 1) * 0.5f;
            float3 tToBorder = (border - from) / direction;
            float tMinLocal = 2.0f;
            if (direction.x != 0 && tToBorder.x < tMinLocal) tMinLocal = tToBorder.x;
            if (direction.y != 0 && tToBorder.y < tMinLocal) tMinLocal = tToBorder.y;
            if (direction.z != 0 && tToBorder.z < tMinLocal) tMinLocal = tToBorder.z;
            if (tMinLocal < 0) tMinLocal = 0;
            float3 to = from + tMinLocal * direction;
            float2 inOut = e.clip(from, to);
            if (inOut.x < 0)
                continue; // ellipsoid miss at this LOD
            float3 hitLocal = from + inOut.x * (to - from);
#else
            float3 hitLocal = localHit;
#endif
            float3 hitCell = float3(entry.nodeMin) + hitLocal * float(nodeSize);
            return { true, int3(floor(hitCell)), hitCell, gbIndex };
        }

        // --- Internal node: collect intersecting children ---
        uint childLevel = entry.level + 1;
        uint childSize = 1u << (octreeMaxDepth - childLevel);
        int3 nodeMin = entry.nodeMin;

        uint childSlots[8];
        float childTMins[8];
        uint childCount = 0;
        uint mask = node.childMask;

        for (uint s = 0; s < 8; s++)
        {
            if (!(mask & (1u << s)))
                continue;

            int3 childMin = nodeMin + int3(
                (s & 1) ? int(childSize) : 0,
                ((s >> 1) & 1) ? int(childSize) : 0,
                ((s >> 2) & 1) ? int(childSize) : 0
            );

            float cTMin, cTMax;
            if (intersectAABB(rayOrigin, reciprocal,
                              float3(childMin), float3(childMin) + float3(float(childSize)),
                              cTMin, cTMax))
            {
                childSlots[childCount] = s;
                childTMins[childCount] = cTMin;
                childCount++;
            }
        }

        // Sort children by tMin ascending (insertion sort, ≤8 elements)
        for (uint i = 1; i < childCount; i++)
        {
            uint slot = childSlots[i];
            float t = childTMins[i];
            int j = int(i) - 1;
            while (j >= 0 && childTMins[j] > t)
            {
                childSlots[j + 1] = childSlots[j];
                childTMins[j + 1] = childTMins[j];
                j--;
            }
            childSlots[j + 1] = slot;
            childTMins[j + 1] = t;
        }

        // Push children in reverse order (largest tMin first → nearest pops first)
        for (int i = int(childCount) - 1; i >= 0; i--)
        {
            uint s = childSlots[i];
            uint childIdx = node.childBase + countbits(mask & ((1u << s) - 1));
            int3 childMin = nodeMin + int3(
                (s & 1) ? int(childSize) : 0,
                ((s >> 1) & 1) ? int(childSize) : 0,
                ((s >> 2) & 1) ? int(childSize) : 0
            );
            stack[stackTop++] = { childIdx, childLevel, childMin, childTMins[i] };
        }
    }

    return { false, int3(rayOrigin), rayOrigin, 0 };
}

// Dispatcher: toggle USE_BRUTE_FORCE at top of file to switch traversal modes
HitResult rayMarching(float3 from, float3 direction, float maxDistance, bool ignoreFirst, RayCone cone, float initialDistance)
{
#if USE_BRUTE_FORCE
    return rayMarchingBruteForce(from, direction, maxDistance, ignoreFirst, cone, initialDistance);
#else
    return rayMarchingParam(from, direction, maxDistance, ignoreFirst, cone, initialDistance);
#endif
}

float calcTrasmissionVisibility(HitResult hit, float3 direction, RayCone cone)
{
    hit = rayMarching(hit.cell, direction, -1, true, cone, 0);
    return hit.hit ? 0 : 1;
}

float calcLightVisibility(HitResult hit, float3 direction, float3 mainNormal, float distance, RayCone cone)
{
    float ignoreDistance = 2;
    if (distance >= 0 && distance <= ignoreDistance)
        return 1;
    float3 from = hit.cell + shadowBias * mainNormal;
    hit = rayMarching(from, direction, distance - ignoreDistance, true, cone, 0);
    return hit.hit ? 0 : 1;
}

float3 coverageTest(HitResult hit, float3 direction, RayCone cone)
{
    float coverage = 0;
    uint offset = hit.gbIndex;
    float times = MAX_BOUNCE;
    while (times > 0)
    {
        TEBSDF bsdf = gBuffer[offset];
        coverage += (1 - coverage) * bsdf.calcCoverage(direction);
        if (coverage >= 1 - transmittanceThreshold)
        {
            coverage = 1;
            break;
        }
        hit = rayMarching(hit.cell, direction, -1, true, cone, 0);
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
    TEBSDF bsdf = gBuffer[offset];
    Ellipsoid e = pBuffer[offset];
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
    TEBSDF BSDF = gBuffer[offset];
    BSDF.updateCoverage(v, transmittanceThreshold);
    return lightColor * BSDF.eval(l, v, h);
}

float3 shadingDirect(HitResult hit, float3 v, inout UniformSampleGenerator sg, RayCone cone)
{
    uint offset = hit.gbIndex;
    TEBSDF BSDF = gBuffer[offset];
    float3 posW = gridMin + hit.cell * voxelSize;
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
        TEBSDF BSDF = gBuffer[offset];
        float3 currentV = i == 0 ? v : -ls[i - 1];
        BSDF.updateCoverage(currentV, transmittanceThreshold);
        BSDF.sample(currentV, ls[i], hs[i], sg);

        pdfs[i] = BSDF.evalPdf(ls[i], currentV, hs[i]);
        if (pdfs[i] <= 0)
        {
            shadingPointCount = i + 1;
            break;
        }
        pdfs[i] = max(pdfs[i], minPdf);

        // Widen cone based on BSDF roughness before next bounce
        float rough = maxRoughness(offset);
        float spreadTan = tan(min(rough, 0.995f) * 1.57f); // rough→[0, ~86°], tan≈14 max
        RayCone nextCone = { max(cone.tanHalfAngle, spreadTan) };

        hits[i + 1] = rayMarching(hits[i].cell, ls[i], -1, true, nextCone, 0);
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
    float4 posWorld = temp / temp.w;
    float3 cell = (posWorld.xyz - gridMin) / voxelSize;
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
    psOut.color = renderBackground(direction);

    if (octreeMaxDepth == 0)
        return psOut;

    float2 inOut = clip(float3(0), float3(voxelCount), from, to);
    if (inOut.x < 0)
        return psOut;

    // Primary ray cone: pixel footprint
    // tan(coneHalfAngle) = tan(HalfFovY) / pixelHeight at distance=1
    RayCone primaryCone = { tanHalfFovY / pixelCount.y };

    UniformSampleGenerator sg = UniformSampleGenerator(pixel, frameIndex);
    from = from + (inOut.x + 1e-6) * delta;
    float initialDistance = (inOut.x + 1e-6) * length(delta);
    HitResult hit = rayMarching(from, direction, -1, false, primaryCone, initialDistance);

    if (all(pixel == selectedPixel))
    {
        if (hit.hit)
        {
            selectedVoxel[0] = float4(-direction, (float)hit.gbIndex);
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
            float3 posW = gridMin + hit.cell * voxelSize;
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
            // // With octree, show depth of the leaf node as a heatmap
            // // Compute octree depth by checking cellInt relative to grid
            // uint depth = octreeMaxDepth;
            // // Simple: use voxelCount to infer approximate depth
            // float3 normalizedCell = float3(hit.cellInt) / float3(voxelCount);
            // float approxDepth = normalizedCell.x + normalizedCell.y + normalizedCell.z;
            // const float3 kDepthColors[] = {
            //     float3(0.0, 1.0, 0.0), // green  — fine
            //     float3(0.0, 0.8, 1.0), // cyan
            //     float3(0.0, 0.0, 1.0), // blue
            //     float3(0.8, 0.8, 0.0), // yellow
            //     float3(1.0, 0.5, 0.0), // orange
            //     float3(1.0, 0.0, 0.0), // red
            //     float3(1.0, 0.0, 1.0), // magenta
            //     float3(1.0, 1.0, 1.0), // white  — coarse
            // };
            // uint idx = uint(approxDepth * 8) % 8;
            // color = kDepthColors[idx];
            color = maxRoughness(hit.gbIndex);
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
