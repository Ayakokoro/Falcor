#pragma once
#include "Types.h"
#include "Triangle.h"

namespace VoxelizationUtility {

// Suzuki–Abe polygon clipping against one half-space
inline void planeClip(const float3* in, size_t inCount, float3* out, size_t& outCount,
                      int axis, float bound, bool greater) {
    const float epsilon = 0.0f;
    outCount = 0;
    if (inCount == 0) return;

    float3 S = in[inCount - 1];
    bool Sin = greater ? S[axis] >= bound - epsilon : S[axis] <= bound + epsilon;

    auto intersectPlane = [](const float3& s, const float3& e, int ax, float bv) -> float3 {
        float denom = e[ax] - s[ax];
        if (denom == 0.0f) return s;
        float t = (bv - s[ax]) / denom;
        return lerp(s, e, t);
    };

    for (size_t i = 0; i < inCount; ++i) {
        const float3& E = in[i];
        bool Ein = greater ? E[axis] >= bound - epsilon : E[axis] <= bound + epsilon;
        if (Ein) {
            if (!Sin) out[outCount++] = intersectPlane(S, E, axis, bound);
            out[outCount++] = E;
        } else {
            if (Sin) out[outCount++] = intersectPlane(S, E, axis, bound);
        }
        S = E;
        Sin = Ein;
    }
}

// Clip triangle against AABB, producing a convex polygon
inline Polygon BoxClipTriangle(const float3& minPoint, const float3& maxPoint, Triangle& tri) {
    float3 vertices[12], temp[12];
    size_t vertexCount = 3, tempCount = 0;

    Polygon polygon = {}; polygon.init();
    vertices[0] = tri.vertices[0];
    vertices[1] = tri.vertices[1];
    vertices[2] = tri.vertices[2];

    float bounds[6] = { minPoint.x, maxPoint.x, minPoint.y, maxPoint.y, minPoint.z, maxPoint.z };
    bool greater = true;
    for (uint i = 0; i < 6; i++) {
        planeClip(vertices, vertexCount, temp, tempCount, i >> 1, bounds[i], greater);
        for (size_t j = 0; j < tempCount; j++) vertices[j] = temp[j];
        vertexCount = tempCount;
        if (vertexCount == 0) return polygon;
        greater = !greater;
    }

    // Deduplicate nearly-equal vertices
    if (vertexCount > 3) {
        float3 dedup[12]; size_t dedupCount = 0;
        auto approxEq = [](const float3& a, const float3& b, float tol = 1e-6f) {
            return std::abs(a.x - b.x) <= tol && std::abs(a.y - b.y) <= tol && std::abs(a.z - b.z) <= tol;
        };
        if (!approxEq(vertices[0], vertices[vertexCount - 1]))
            dedup[dedupCount++] = vertices[0];
        for (size_t i = 1; i < vertexCount; i++)
            if (!approxEq(vertices[i], vertices[i - 1]))
                dedup[dedupCount++] = vertices[i];
        if (dedupCount != vertexCount) {
            for (size_t i = 0; i < dedupCount; i++) vertices[i] = dedup[i];
            vertexCount = dedupCount;
        }
    }

    uint n = std::min((uint)MAX_VERTEX_COUNT, (uint)vertexCount);
    polygon.count = n;
    for (uint i = 0; i < n; i++) polygon.vertices[i] = vertices[i];
    polygon.triRef.init();
    return polygon;
}

} // namespace VoxelizationUtility
