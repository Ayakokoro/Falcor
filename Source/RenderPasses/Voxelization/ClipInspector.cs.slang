#include "Math/VoxelizationUtility.slang"
#include "VoxelInspectorShared.slang"
import Scene.Scene;

RWStructuredBuffer<InspectorTriangle> clippedTriangles;
RWStructuredBuffer<uint> clipStats; // generated triangles, overflow, source polygons, reserved
cbuffer CB
{
    float3 nodeMinW;
    float3 nodeExtentW;
    uint instanceIndex;
    uint triangleCount;
    uint triangleOffset;
    uint capacity;
};

[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint triangleID = tid.x + triangleOffset;
    if (triangleID >= triangleCount) return;
    GeometryInstanceID instanceID = { instanceIndex };
    float3 positions[3];
    gScene.getVertexPositionsW(instanceID, triangleID, positions);
    Triangle tri = {};
    for (uint i = 0; i < 3; ++i)
        tri.vertices[i] = (positions[i] - nodeMinW) / nodeExtentW;
    float3 lo = min(tri.vertices[0], min(tri.vertices[1], tri.vertices[2]));
    float3 hi = max(tri.vertices[0], max(tri.vertices[1], tri.vertices[2]));
    if (any(lo > 1.f) || any(hi < 0.f)) return;

    float3 vertices[MAX_VERTEX_COUNT_PROCESSING];
    uint count = boxClipTriangleVertices(float3(0), float3(1), tri, vertices);
    if (count < 3) return;
    uint unused;
    InterlockedAdd(clipStats[2], 1, unused);
    for (uint i = 1; i + 1 < count; ++i)
    {
        if (length(cross(vertices[i] - vertices[0], vertices[i + 1] - vertices[0])) < 1e-10f) continue;
        uint index;
        InterlockedAdd(clipStats[0], 1, index);
        if (index >= capacity)
        {
            InterlockedOr(clipStats[1], 1, unused);
            continue;
        }
        InspectorTriangle result;
        result.p0 = vertices[0]; result.p1 = vertices[i]; result.p2 = vertices[i + 1];
        result.b0 = calcBarycentricCoordinates(tri.vertices[0], tri.vertices[1], tri.vertices[2], result.p0);
        result.b1 = calcBarycentricCoordinates(tri.vertices[0], tri.vertices[1], tri.vertices[2], result.p1);
        result.b2 = calcBarycentricCoordinates(tri.vertices[0], tri.vertices[1], tri.vertices[2], result.p2);
        result.instanceID = instanceIndex;
        result.triangleID = triangleID;
        clippedTriangles[index] = result;
    }
}
