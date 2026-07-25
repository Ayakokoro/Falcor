#pragma once
#include "Types.h"
#include "AABB.h"

#define MAX_VERTEX_COUNT 6

struct TriangleRef {
    uint meshID = 0;
    uint triangleID = 0;
    uint materialID = 0;

    void init() { meshID = triangleID = materialID = 0; }
};

struct Triangle {
    float3 vertices[3];
    float3 normals[3];
    float2 uvs[3];
    float3x3 TBN;

    void init() {
        for (int i = 0; i < 3; i++) {
            vertices[i] = float3(0);
            uvs[i] = float2(0);
            normals[i] = float3(0);
        }
    }

    bool contains2D(const float3& p) const {
        float3 n = cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
        float s0 = dot(n, cross(vertices[0] - p, vertices[1] - p));
        float s1 = dot(n, cross(vertices[1] - p, vertices[2] - p));
        float s2 = dot(n, cross(vertices[2] - p, vertices[0] - p));
        return (s0 >= 0) && (s1 >= 0) && (s2 >= 0);
    }

    float3 barycentricCoordinates(const float3& p) const {
        float3 ab = vertices[1] - vertices[0];
        float3 ac = vertices[2] - vertices[0];
        float3 ap = p - vertices[0];
        float d00 = dot(ab, ab), d01 = dot(ab, ac), d11 = dot(ac, ac);
        float d20 = dot(ap, ab), d21 = dot(ap, ac);
        float denom = d00 * d11 - d01 * d01;
        if (std::abs(denom) < 1e-8f)
            return float3(1.0f / 3.0f);
        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        return float3(1.0f - v - w, v, w);
    }

    float2 lerpUV(const float3& p) const {
        float3 coord = barycentricCoordinates(p);
        return uvs[0] * coord.x + uvs[1] * coord.y + uvs[2] * coord.z;
    }

    float3 lerpNormal(const float3& p) const {
        float3 coord = barycentricCoordinates(p);
        return safeNormalize(normals[0] * coord.x + normals[1] * coord.y + normals[2] * coord.z);
    }

    AABBInt calcAABBInt() const {
        AABBInt aabb; aabb.init();
        for (int i = 0; i < 3; i++)
            aabb.accumulate(floorToInt3(vertices[i]));
        aabb.complete();
        return aabb;
    }

    void buildTBN() {
        float3 e1 = vertices[1] - vertices[0];
        float3 e2 = vertices[2] - vertices[0];
        float2 duv1 = uvs[1] - uvs[0];
        float2 duv2 = uvs[2] - uvs[0];
        float D = duv1.x * duv2.y - duv1.y * duv2.x;
        float3 N = cross(e1, e2);
        float len = length(N);
        if (len == 0) {
            N = normals[0] + normals[1] + normals[2];
            len = length(N);
            if (len == 0) N = float3(0, 0, 1), len = 1;
        }
        N /= len;
        float3 T, B;
        if (std::abs(D) < 1e-8f) {
            T = safeNormalize(e1 - N * dot(N, e1));
            B = safeNormalize(cross(N, T));
        } else {
            float r = 1.0f / D;
            float3 Tp = (e1 * duv2.y - e2 * duv1.y) * r;
            float3 Bp = (e2 * duv1.x - e1 * duv2.x) * r;
            T = safeNormalize(Tp - N * dot(N, Tp));
            B = safeNormalize(Bp - N * dot(N, Bp));
        }
        TBN[0] = float3(T.x, B.x, N.x);
        TBN[1] = float3(T.y, B.y, N.y);
        TBN[2] = float3(T.z, B.z, N.z);
    }
};

struct Polygon {
    TriangleRef triRef;
    float3 vertices[MAX_VERTEX_COUNT];
    uint count = 0;
    float3 normal;

    void init() { count = 0; }

    float calcArea() const {
        float3 s(0);
        for (uint i = 0; i < count; ++i) {
            const float3& a = vertices[i];
            const float3& b = vertices[(i + 1) % count];
            s += cross(a, b);
        }
        return safeSqrt(dot(s, s)) / 2.0f;
    }

    float calcProjArea(const float3& direction) const {
        return calcArea() * std::abs(dot(direction, normal));
    }

    float3 calcCentroid(float& totalArea) const {
        const float3& v0 = vertices[0];
        float3 sum(0);
        totalArea = 0;
        for (uint i = 1; i + 1 < count; ++i) {
            const float3& a = v0;
            const float3& b = vertices[i];
            const float3& c = vertices[i + 1];
            float3 s = cross(b - a, c - a);
            float area = safeSqrt(dot(s, s)) * 0.5f;
            if (area <= 0) continue;
            sum += (a + b + c) / 3.0f * area;
            totalArea += area;
        }
        return (totalArea > 0) ? sum / totalArea : v0;
    }

    void add(const float3& point) {
        if (count < MAX_VERTEX_COUNT)
            vertices[count++] = point;
    }
};
