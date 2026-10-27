#pragma once
#include "Polygon.slang"
#include "Triangle.slang"

#define MAX_VERTEX_COUNT_PROCESSING 12

inline bool approximatelyEqual(float a, float b, float tolerance = 1e-6) { return abs(a - b) <= tolerance; }
inline bool approximatelyEqual(float3 a, float3 b, float tolerance = 1e-6)
{
    return approximatelyEqual(a.x, b.x, tolerance) && approximatelyEqual(a.y, b.y, tolerance) &&
           approximatelyEqual(a.z, b.z, tolerance);
}

inline float3 intersectAxisPlane(float3 s, float3 e, int axis, float bound)
{
    float denom = e[axis] - s[axis];
    if (denom == 0.0)
        return s;
    float t = (bound - s[axis]) / denom;
    return lerp(s, e, t);
}

inline uint removeAdjacentRepeatPoints(inout float3 v[MAX_VERTEX_COUNT_PROCESSING], uint count)
{
    if (count == 0)
        return 0;

    uint writeIndex = 0;
    for (uint i = 0; i < count; ++i)
    {
        uint next = (i + 1) % count;
        if (approximatelyEqual(v[i], v[next]))
            continue;
        v[writeIndex++] = v[i];
    }
    return writeIndex;
}

inline uint planeClip(
    float3 inPoly[MAX_VERTEX_COUNT_PROCESSING],
    uint inCount,
    out float3 outPoly[MAX_VERTEX_COUNT_PROCESSING],
    int axis,
    float bound,
    bool greater
)
{
    uint outCount = 0;

    if (inCount == 0)
        return 0;

    float3 S = inPoly[inCount - 1];
    bool Sin = greater ? (S[axis] >= bound) : (S[axis] <= bound);

    for (uint i = 0; i < inCount; ++i)
    {
        float3 E = inPoly[i];
        bool Ein = greater ? (E[axis] >= bound) : (E[axis] <= bound);

        if (Ein)
        {
            if (!Sin && outCount < MAX_VERTEX_COUNT_PROCESSING)
            {
                outPoly[outCount++] = intersectAxisPlane(S, E, axis, bound);
            }
            if (outCount < MAX_VERTEX_COUNT_PROCESSING)
            {
                outPoly[outCount++] = E;
            }
        }
        else
        {
            if (Sin && outCount < MAX_VERTEX_COUNT_PROCESSING)
            {
                outPoly[outCount++] = intersectAxisPlane(S, E, axis, bound);
            }
        }

        S = E;
        Sin = Ein;
    }

    return outCount;
}

// 用轴对齐包围盒裁剪三角形
inline uint boxClipTriangleVertices(
    float3 minPoint, float3 maxPoint, Triangle tri, out float3 vertices[MAX_VERTEX_COUNT_PROCESSING])
{
    float3 temp[MAX_VERTEX_COUNT_PROCESSING];

    uint vCount = 3;
    vertices[0] = tri.vertices[0];
    vertices[1] = tri.vertices[1];
    vertices[2] = tri.vertices[2];

    // 六个平面
    float bounds[6] = { minPoint.x, maxPoint.x, minPoint.y, maxPoint.y, minPoint.z, maxPoint.z };

    bool greater = true;

    for (uint i = 0; i < 6; ++i)
    {
        uint newCount = planeClip(vertices, vCount, temp, (int)(i >> 1), bounds[i], greater);

        if (newCount == 0)
        {
            return 0;
        }

        // 把 temp 拷回 vertices
        vCount = newCount;
        for (uint k = 0; k < vCount; ++k)
        {
            vertices[k] = temp[k];
        }

        greater = !greater;
    }

    if (vCount > 3)
        vCount = removeAdjacentRepeatPoints(vertices, vCount);

    return vCount;
}

inline Polygon boxClipTriangle(float3 minPoint, float3 maxPoint, Triangle tri)
{
    Polygon poly = {};
    float3 vertices[MAX_VERTEX_COUNT_PROCESSING];
    uint vCount = boxClipTriangleVertices(minPoint, maxPoint, tri, vertices);

    for (uint i = 0; i < vCount; ++i)
    {
        poly.add(vertices[i]);
    }
    return poly;
}
