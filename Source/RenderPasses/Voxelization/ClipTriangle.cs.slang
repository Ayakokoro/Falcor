#include "VoxelizationShared.slang"
#include "Voxel/VoxelGrid.slang"
#include "Math/VoxelizationUtility.slang"
#include "Math/AABB.slang"

import Scene.SceneTypes;
import Scene.Scene;
import Scene.Raster;
import Utils.Math.MathHelpers;

// Flat buffers written by LoadMesh.cs.slang
RWStructuredBuffer<float3> positions;
RWStructuredBuffer<float3> normals;
RWStructuredBuffer<float2> texCoords;
RWStructuredBuffer<uint3> triangles;

// Output buffers
RWStructuredBuffer<Polygon> polygonBuffer;
RWStructuredBuffer<PolygonRange> polygonRangeBuffer;
RWStructuredBuffer<uint> polygonCountBuffer;
RWStructuredBuffer<int> vBuffer;

// Visited buffer (Pass 1): one uint per voxel, 0=unvisited, nonzero=visited.
// Uses InterlockedOr for lock-free warp-safe marking — no spin-wait, no CAS deadlock.
// Kept for debug/fallback; primary compaction path uses rawMarkBuffer instead (no full readback).
RWStructuredBuffer<uint> visitedBuffer;

// Streaming mark buffer (Pass 1): compact list of solid voxel linear indices.
// Each time a triangle clips to >=3 vertices in a cell, the cell index is appended here
// via InterlockedAdd. May contain duplicates (same cell hit by multiple triangles).
// CPU reads back ONLY this buffer (and its counter) — typically KB-scale vs MB/GB visitedBuffer.
RWStructuredBuffer<uint> rawMarkBuffer;
RWStructuredBuffer<uint> rawMarkCounter;   // atomic counter: total entries written to rawMarkBuffer

// Compact list of solid voxel linear indices (voxels with visited=1 after Pass 1).
// Maps [0, validSolidVoxelCount) → linear index into the voxel grid.
// Used by Pass 2 for per-voxel dispatch (one thread per solid voxel).
StructuredBuffer<uint> solidVoxelOffsetBuffer;

// cbuffer for Pass 2/3 voxel dispatch parameters (ignored in Pass 1)
cbuffer VoxelDispatchConfig
{
    uint validSolidVoxelCount; // number of entries in solidVoxelOffsetBuffer
}

cbuffer GridData
{
    float3 gridMin;
    float3 voxelSize;
    uint3 voxelCount;
};

// MeshData: per-mesh geometry info
// meshID is set by C++ loop (each dispatch processes one mesh)
cbuffer MeshData
{
    uint vertexCount;
    uint triangleCount;
    uint vbOffset;       // vertex buffer offset (into positions/normals/texCoords)
    uint ibOffset;       // index buffer offset (into scene's index buffer)
    uint triOffset;      // triangle array offset (into triangles[] flat buffer)
    bool use16BitIndices;
    uint materialID;
    uint meshID;         // global mesh ID, set by C++ for-loop
};

// Pass 0: count polygons only (atomic increment)
// Pass 1: write polygons to polygonBuffer (needs polygonRangeBuffer.localHead)
cbuffer PassConfig
{
    uint writePass;
}

// Load triangle from flat buffers (written by LoadMesh.cs.slang)
// triangles[] stores global indices (vbOffset + localIndex) per mesh
// positions[] is indexed directly by this global index
Triangle loadTriangle(uint triangleId)
{
    Triangle tri = {};
    uint3 triIndices = triangles[triangleId + triOffset];
    for (int i = 0; i < 3; i++)
    {
        uint vi = triIndices[i]; // global index into flat positions/normals/texCoords
        tri.vertices[i] = (positions[vi] - gridMin) / voxelSize;
        tri.uvs[i] = texCoords[vi];
        tri.normals[i] = normals[vi];
    }
    tri.buildTBN();
    return tri;
}

// AABB-based voxel traversal (matches CPU PolygonGenerator::clip exactly).
// Computes triangle's integer bounding box, then iterates EVERY cell inside it.
// This guarantees 100% coverage of intersected voxels — no DDA blind spots.
// Each thread handles one triangle independently; InterlockedOr is warp-safe.
void voxelizeTriangleAABB(Triangle tri, uint triangleId, int3 voxelCount)
{
    // Compute integer AABB of triangle vertices (identical to CPU::calcAABBInt)
    // NOTE: No clamp to grid bounds — must match CPU behavior exactly.
    // CPU's calcAABBInt() does NOT clamp; out-of-bounds cells are handled
    // naturally by boxClipTriangle returning count=0.
    // We add an explicit bounds guard before InterlockedOr instead (below).
    int3 vmin = int3(INT_MAX, INT_MAX, INT_MAX);
    int3 vmax = int3(INT_MIN, INT_MIN, INT_MIN);
    for (int i = 0; i < 3; i++)
    {
        int3 vi = int3(floor(tri.vertices[i]));
        vmin = min(vmin, vi);
        vmax = max(vmax, vi);
    }

    // Brute-force iterate every cell in AABB (same as CPU PolygonGenerator::clip)
    for (int z = vmin.z; z <= vmax.z; z++)
    {
        for (int y = vmin.y; y <= vmax.y; y++)
        {
            for (int x = vmin.x; x <= vmax.x; x++)
            {
                int3 cell = int3(x, y, z);
                Polygon clipped = boxClipTriangle(float3(cell), float3(cell) + 1.0f, tri);

                if (clipped.count >= 3)
                {
                    // Bounds guard: only mark cells within valid grid range.
                    // This matches CPU behavior where out-of-bounds cells visited
                    // by calcAABBInt() would produce empty clips anyway,
                    // but we must avoid writing outside visitedBuffer/rawMarkBuffer.
                    if (all(cell >= int3(0,0,0)) && all(cell < int3(voxelCount)))
                    {
                        int index = CellToIndex(cell, voxelCount);
                        // Lock-free visited marking (warp-safe: no spin-wait, no CAS deadlock)
                        InterlockedOr(visitedBuffer[index], 1u);
                        // Streaming output: append cell index to compact mark buffer.
                        // InterlockedAdd is warp-safe (no spin-wait, just atomic increment).
                        uint markIdx = InterlockedAdd(rawMarkCounter[0], 1u);
                        rawMarkBuffer[markIdx] = (uint)index;
                    }
                }
            }
        }
    }
}

// Clip triangle against one specific voxel cell.
// Returns true if the clipped polygon has >= 3 vertices.
// Caller (main() Pass 2) handles buffer writing with thread-local counter — no atomics needed
// because each solid voxel has exactly one dedicated thread (per-voxel dispatch).
bool clipTrianglePerVoxel(Triangle tri, uint triangleId, int3 cellInt, out Polygon polygon)
{
    polygon = boxClipTriangle(float3(cellInt), float3(cellInt) + 1.f, tri);

    if (polygon.count < 3)
        return false;

    polygon.normal = float3(tri.TBN[0][2], tri.TBN[1][2], tri.TBN[2][2]);
    polygon.triRef.meshID = meshID;
    polygon.triRef.triangleID = triangleId;
    polygon.triRef.materialID = materialID;

    return true;
}

void voxelizeTriangle(uint triangleId)
{
    if (triangleId >= triangleCount)
        return;

    Triangle tri = loadTriangle(triangleId);
    // AABB brute-force iteration — matches CPU PolygonGenerator::clip exactly.
    // Visits every cell in the triangle's integer bounding box, no DDA blind spots.
    voxelizeTriangleAABB(tri, triangleId, int3(voxelCount));
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (writePass == 1)
    {
        // ========== Pass 2: per-voxel COUNT (lock-free, no polygon writes) ==========
        // Each thread owns exactly one solid voxel. Iterates ALL triangles,
        // clips each against its assigned cell, and counts valid polygons.
        // Result: polygonCountBuffer[voxelOffset] = exact count for this voxel.
        uint voxelOffset = solidVoxelOffsetBuffer[tid.x];
        PolygonRange range = polygonRangeBuffer[voxelOffset];

        uint localPolyCount = 0;
        for (uint triangleId = 0; triangleId < triangleCount; triangleId++)
        {
            Triangle tri = loadTriangle(triangleId);
            Polygon poly;
            if (clipTrianglePerVoxel(tri, triangleId, range.cellInt, poly))
                localPolyCount++;
        }

        // Write final count for this voxel (single non-atomic write)
        polygonCountBuffer[voxelOffset] = localPolyCount;
        return;
    }

    if (writePass == 2)
    {
        // ========== Pass 3: per-voxel WRITE EXACT (zero waste) ==========
        // Each thread owns exactly one solid voxel. Iterates ALL triangles,
        // clips each against its assigned cell, and writes polygons to
        // the EXACT pre-allocated range [localHead, localHead + count).
        //
        // No atomics needed: each thread writes to a private range computed
        // by CPU exclusive prefix sum from Pass 2 counts. No max limit.

        uint voxelOffset = solidVoxelOffsetBuffer[tid.x];
        PolygonRange range = polygonRangeBuffer[voxelOffset];

        uint localHead = range.localHead;   // exact offset from prefix sum
        uint localIdx = 0;

        for (uint triangleId = 0; triangleId < triangleCount; triangleId++)
        {
            Triangle tri = loadTriangle(triangleId);
            Polygon poly;
            if (clipTrianglePerVoxel(tri, triangleId, range.cellInt, poly))
            {
                polygonBuffer[localHead + localIdx] = poly;
                localIdx++;
            }
        }
        return;
    }

    // Pass 0 (writePass == 0): per-triangle dispatch — AABB traversal + InterlockedOr visited marking
    voxelizeTriangle(tid.x);
}
