// The same median-split BVH used to sample the mesh SDF, with original per-corner UVs.
struct SurfaceUVNode
{
    float3 lower;
    uint first; // Right child for an interior node; first triangle for a leaf.
    float3 upper;
    uint count;
};

struct SurfaceUVTriangle
{
    float4 a;       // xyz: first vertex; w: its U coordinate.
    float4 ab;      // xyz: edge to second vertex; w: first vertex's V coordinate.
    float4 ac;      // xyz: edge to third vertex; w: scale for normalized edge calculations.
    float4 uvEdges; // UV differences along AB and AC.
};

StructuredBuffer<SurfaceUVNode> gSurfaceUVNodes;
StructuredBuffer<SurfaceUVTriangle> gSurfaceUVTriangles;

struct MeshSurfaceUV
{
    float2 uv;
    float3 gradU;
    float3 gradV;
};

float uvBoxDistanceSquared(float3 p, SurfaceUVNode node)
{
    float3 delta = p - clamp(p, node.lower, node.upper);
    return dot(delta, delta);
}

float uvEdgeParameter(float3 q, float3 edge)
{
    float length2 = dot(edge, edge);
    return length2 > 0.f ? saturate(dot(q, edge) / length2) : 0.f;
}

// Return the two barycentric coordinates (B,C) of the point on this triangle nearest to p.
float2 closestSurfaceBarycentrics(float3 p, SurfaceUVTriangle tri)
{
    float3 ab = tri.ab.xyz / tri.ac.w;
    float3 ac = tri.ac.xyz / tri.ac.w;
    float3 q = (p - tri.a.xyz) / tri.ac.w;
    float3 n = cross(ab, ac);
    float area2 = dot(n, n);
    if (area2 > 0.f)
    {
        float2 bary = float2(dot(cross(q, ac), n), dot(cross(ab, q), n)) / area2;
        if (all(bary >= 0.f) && bary.x + bary.y <= 1.f) return bary;
    }

    float b = uvEdgeParameter(q, ab);
    float c = uvEdgeParameter(q, ac);
    float bc = uvEdgeParameter(q - ab, ac - ab);
    float3 toAB = q - b * ab;
    float3 toAC = q - c * ac;
    float3 toBC = (q - ab) - bc * (ac - ab);
    float dAB = dot(toAB, toAB), dAC = dot(toAC, toAC), dBC = dot(toBC, toBC);
    if (dAB <= min(dAC, dBC)) return float2(b, 0.f);
    return dAC <= dBC ? float2(0.f, c) : float2(1.f - bc, bc);
}

MeshSurfaceUV meshSurfaceUV(float3 worldPosition)
{
    // The builder halves the triangle range at each split and uses at most 32-bit indices.
    uint pending[32];
    uint pendingCount = 0;
    uint nodeIndex = 0;
    uint closest = 0xffffffffu;
    float best = 3.402823466e38f;
    float2 closestBary = float2(0.f);
    [loop]
    for (;;)
    {
        SurfaceUVNode node = gSurfaceUVNodes[nodeIndex];
        if (uvBoxDistanceSquared(worldPosition, node) <= best)
        {
            if (node.count == 0)
            {
                uint left = nodeIndex + 1, right = node.first;
                float dLeft = uvBoxDistanceSquared(worldPosition, gSurfaceUVNodes[left]);
                float dRight = uvBoxDistanceSquared(worldPosition, gSurfaceUVNodes[right]);
                if (max(dLeft, dRight) <= best)
                    pending[pendingCount++] = dLeft <= dRight ? right : left;
                if (min(dLeft, dRight) <= best)
                {
                    nodeIndex = dLeft <= dRight ? left : right;
                    continue;
                }
            }
            else
            {
                [loop]
                for (uint lane = 0; lane < node.count; ++lane)
                {
                    uint id = node.first + lane;
                    SurfaceUVTriangle tri = gSurfaceUVTriangles[id];
                    float2 bary = closestSurfaceBarycentrics(worldPosition, tri);
                    float3 delta = (worldPosition - tri.a.xyz) - (bary.x * tri.ab.xyz + bary.y * tri.ac.xyz);
                    float distance2 = dot(delta, delta);
                    if (distance2 < best || (distance2 == best && id < closest))
                    {
                        best = distance2;
                        closest = id;
                        closestBary = bary;
                    }
                }
            }
        }
        if (pendingCount == 0) break;
        nodeIndex = pending[--pendingCount];
    }

    MeshSurfaceUV result = {};
    if (closest == 0xffffffffu) return result;
    SurfaceUVTriangle tri = gSurfaceUVTriangles[closest];
    result.uv = float2(tri.a.w, tri.ab.w) + closestBary.x * tri.uvEdges.xy + closestBary.y * tri.uvEdges.zw;
    float3 ab = tri.ab.xyz / tri.ac.w, ac = tri.ac.xyz / tri.ac.w;
    float3 n = cross(ab, ac);
    float area2 = dot(n, n);
    if (area2 > 0.f)
    {
        float3 gradB = (cross(ac, n) / area2) / tri.ac.w;
        float3 gradC = (cross(n, ab) / area2) / tri.ac.w;
        result.gradU = tri.uvEdges.x * gradB + tri.uvEdges.z * gradC;
        result.gradV = tri.uvEdges.y * gradB + tri.uvEdges.w * gradC;
    }
    return result;
}
