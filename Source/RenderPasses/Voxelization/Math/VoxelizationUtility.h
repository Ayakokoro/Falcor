#pragma once
#include "Falcor.h"
#include "Profiler.h"
#include "Random.h"
#include "Polygon.slang"
#include "SphericalHarmonics.slang"
#include "Sampling.slang"
#include "Triangle.slang"
#include <unordered_set>
#include <tuple>
using namespace Falcor;

struct Float3Hash
{
    size_t operator()(const float3& v) const noexcept
    {
        size_t h1 = std::hash<float>{}(v.x);
        size_t h2 = std::hash<float>{}(v.y);
        size_t h3 = std::hash<float>{}(v.z);
        size_t h = h1;
        h ^= h2 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= h3 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};
struct Float3Equal
{
    bool operator()(const float3& a, const float3& b) const noexcept { return a.x == b.x && a.y == b.y && a.z == b.z; }
};

class VoxelizationUtility
{
private:
    static float2 intersect(float p1, float p2, float p0, float q)
    {
        float u1 = (p1 - p0) / q;
        float u2 = (p2 - p0) / q;
        if (u1 > u2)
        {
            float temp = u1;
            u1 = u2;
            u2 = temp;
        }
        return float2(u1, u2);
    }

    static bool approximatelyEqual(float a, float b, float tolerance = 1e-6) { return abs(a - b) <= tolerance; }
    static bool approximatelyEqual(float3 a, float3 b, float tolerance = 1e-6)
    {
        return approximatelyEqual(a.x, b.x, tolerance) && approximatelyEqual(a.y, b.y, tolerance) &&
               approximatelyEqual(a.z, b.z, tolerance);
    }

    static float3 intersectAxisPlane(const float3& s, const float3& e, int axis, float planeVal)
    {
        float denom = e[axis] - s[axis];
        if (denom == 0.0)
            return s;
        float t = (planeVal - s[axis]) / denom;
        return math::lerp(s, e, t);
    }

    static void planeClip(const float3* inData, size_t inCount, float3* outData, size_t& outCount, int axis, float bound,
                          bool greater)
    {
        float epsilon = 0; //非0值反而导致更多体素多边形投影面积失真
        outCount = 0;
        if (inCount == 0)
            return;

        float3 S = inData[inCount - 1];
        bool Sin = greater ? S[axis] >= bound - epsilon : S[axis] <= bound + epsilon;

        for (size_t i = 0; i < inCount; ++i)
        {
            const float3 E = inData[i];
            const bool Ein = greater ? E[axis] >= bound - epsilon : E[axis] <= bound + epsilon;

            if (Ein)
            {
                if (!Sin)
                    outData[outCount++] = intersectAxisPlane(S, E, axis, bound);
                outData[outCount++] = E;
            }
            else
            {
                if (Sin)
                    outData[outCount++] = intersectAxisPlane(S, E, axis, bound);
            }

            S = E;
            Sin = Ein;
        }
    }

public:
    static void RemoveRepeatPoints(std::vector<float3>& points)
    {
        std::unordered_set<float3, Float3Hash, Float3Equal> temp(points.begin(), points.end());
        points.assign(temp.begin(), temp.end());
    }

    static void RemoveAdjacentRepeatPoints(std::vector<float3>& points)
    {
        std::vector<float3> temp;
        temp.reserve(points.size());
        //默认points中至少有1个元素
        if (!approximatelyEqual(points[0], points.back()))
            temp.push_back(points[0]);
        for (size_t i = 1; i < points.size(); i++)
        {
            if (!approximatelyEqual(points[i], points[i - 1]))
                temp.push_back(points[i]);
        }
        if(points.size() != temp.size())
            points.swap(temp);
    }

    /// <summary>
    /// 用轴对齐长方体裁剪有向线段
    /// </summary>
    static float2 BoxClip(float3 minPoint, float3 maxPoint, float3 from, float3 to)
    {
        float uIn = 0, uOut = 1;
        float3 v = to - from;

        float2 u12 = intersect(minPoint.x, maxPoint.x, from.x, v.x);
        uIn = math::max(uIn, u12.x);
        uOut = math::min(uOut, u12.y);
        if (uIn > uOut)
            return float2(-1, -1);

        u12 = intersect(minPoint.y, maxPoint.y, from.y, v.y);
        uIn = math::max(uIn, u12.x);
        uOut = math::min(uOut, u12.y);
        if (uIn > uOut)
            return float2(-1, -1);

        u12 = intersect(minPoint.z, maxPoint.z, from.z, v.z);
        uIn = math::max(uIn, u12.x);
        uOut = math::min(uOut, u12.y);
        if (uIn > uOut)
            return float2(-1, -1);

        return float2(uIn, uOut);
    }

    /// <summary>
    /// 用轴对齐长方体裁剪三角形
    /// </summary>
    static Polygon BoxClipTriangle(float3 minPoint, float3 maxPoint, Triangle& tri)
    {
        float3 vertices[12];
        float3 temp[12];
        size_t vertexCount = 3;
        size_t tempCount = 0;

        Polygon polygon = {};
        polygon.init();

        vertices[0] = tri.vertices[0];
        vertices[1] = tri.vertices[1];
        vertices[2] = tri.vertices[2];

        float bounds[6] = {minPoint.x, maxPoint.x, minPoint.y, maxPoint.y, minPoint.z, maxPoint.z};
        bool greater = true;
        for (uint i = 0; i < 6; i++)
        {
            planeClip(vertices, vertexCount, temp, tempCount, i >> 1, bounds[i], greater);
            // swap buffers
            for (size_t j = 0; j < tempCount; j++)
                vertices[j] = temp[j];
            vertexCount = tempCount;
            if (vertexCount == 0)
                return polygon;
            greater = !greater;
        }

        if (vertexCount > 3)
        {
            float3 dedup[12];
            size_t dedupCount = 0;
            if (!approximatelyEqual(vertices[0], vertices[vertexCount - 1]))
                dedup[dedupCount++] = vertices[0];
            for (size_t i = 1; i < vertexCount; i++)
            {
                if (!approximatelyEqual(vertices[i], vertices[i - 1]))
                    dedup[dedupCount++] = vertices[i];
            }
            if (dedupCount != vertexCount)
            {
                for (size_t i = 0; i < dedupCount; i++)
                    vertices[i] = dedup[i];
                vertexCount = dedupCount;
            }
        }

        uint n = math::min(MAX_VERTEX_COUNT, (int)vertexCount);
        polygon.count = n;
        for (uint i = 0; i < n; i++)
            polygon.vertices[i] = vertices[i];
        polygon.triRef.init();
        return polygon;
    }
};
