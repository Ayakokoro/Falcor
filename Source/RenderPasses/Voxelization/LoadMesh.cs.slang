#include "Voxel/VoxelGrid.slang"
#include "Scene/VertexAttrib.slangh"

import Scene.SceneTypes;
import Scene.Scene;
import Scene.Raster;
import Utils.Math.MathHelpers;

RWStructuredBuffer<float3> positions;
RWStructuredBuffer<float3> normals;
RWStructuredBuffer<float2> texCoords;
RWStructuredBuffer<uint3> triangles;

cbuffer MeshData
{
    uint vertexCount;
    uint triangleCount;
    uint vbOffset;
    uint ibOffset;
    uint triOffset;
    bool use16BitIndices;
};

void loadTriangle(uint triangleId)
{
    if (triangleId >= triangleCount)
        return;

    uint3 triangle = gScene.getLocalIndices(ibOffset, triangleId, use16BitIndices);
    triangle += vbOffset; // adjust to flat buffer index space
    triangles[triangleId + triOffset] = triangle;
    for (uint i = 0; i < 3; i++)
    {
        StaticVertexData vertex = gScene.getVertex(triangle[i]);
        positions[triangle[i]] = vertex.position;
        texCoords[triangle[i]] = vertex.texCrd;
        normals[triangle[i]] = vertex.normal;
    }
}

[numthreads(256, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    loadTriangle(tid.x);
}
