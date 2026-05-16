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
RWStructuredBuffer<uint> solidVoxelCount; // single-element buffer

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

int tryGetOffset(int3 cellInt)
{
    int index = CellToIndex(cellInt, voxelCount);

    int v = vBuffer[index];
    if (v >= 0)
        return v;

    // -2 means "allocation in progress" (lock)
    int original;
    InterlockedCompareExchange(vBuffer[index], -1, -2, original);
    if (original == -1)
    {
        // Successfully claimed this voxel
        uint offset;
        InterlockedAdd(solidVoxelCount[0], 1u, offset);
        vBuffer[index] = (int)offset;
        return (int)offset;
    }

    // Wait for other thread to finish allocation
    while ((v = vBuffer[index]) == -2)
    {
    }
    return v;
}

// Load triangle from flat buffers (written by LoadMesh.cs.slang)
// triangles[] stores LOCAL indices (0..vertexCount-1) per mesh
// positions[] is at vbOffset+localIndex, so add vbOffset when reading
Triangle loadTriangle(uint triangleId)
{
    Triangle tri = {};
    uint3 triIndices = triangles[triangleId + triOffset];
    for (int i = 0; i < 3; i++)
    {
        uint vi = triIndices[i]; // local index within this mesh's vertex buffer
        uint flatVi = vbOffset + vi; // absolute index into flat positions/normals/texCoords
        tri.vertices[i] = (positions[flatVi] - gridMin) / voxelSize;
        tri.uvs[i] = texCoords[flatVi];
        tri.normals[i] = normals[flatVi];
    }
    tri.buildTBN();
    return tri;
}

void clipTriangle(Triangle tri, uint triangleId)
{
    AABBInt aabb = tri.calcAABBInt();
    for (int i = 0; i < aabb.count(); i++)
    {
        int3 cellInt = aabb.indexToCell(i);
        float3 minPoint = float3(cellInt);
        Polygon polygon = boxClipTriangle(minPoint, minPoint + 1.f, tri);

        polygon.normal = float3(tri.TBN[0][2], tri.TBN[1][2], tri.TBN[2][2]);

        // ========== Critical fix: set triRef ==========
        polygon.triRef.meshID = meshID;           // set by C++ for-loop (one dispatch per mesh)
        polygon.triRef.triangleID = triangleId;   // local triangle ID within this mesh
        polygon.triRef.materialID = materialID;   // from MeshData cbuffer
        // ==============================================

        if (polygon.count >= 3)
        {
            int offset = tryGetOffset(cellInt);

            // Atomic: get write position and increment counter
            uint writeIndex;
            InterlockedAdd(polygonCountBuffer[offset], 1u, writeIndex);

            // Pass 1 only: actually write the polygon
            if (writePass == 1)
            {
                writeIndex += polygonRangeBuffer[offset].localHead;
                polygonBuffer[writeIndex] = polygon;
            }
        }
    }
}

void voxelizeTriangle(uint triangleId)
{
    if (triangleId >= triangleCount)
        return;

    Triangle tri = loadTriangle(triangleId);
    clipTriangle(tri, triangleId);
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    voxelizeTriangle(tid.x);
}
