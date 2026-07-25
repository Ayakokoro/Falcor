#pragma once
#include "Types.h"
#include "Triangle.h"

#define PROJ_RES  64
#define PROJ_PLANE_SIZE  2.0f
#define PROJ_HALF_SIZE   (PROJ_PLANE_SIZE * 0.5f)
#define PROJ_PIXEL_SIZE  (PROJ_PLANE_SIZE / float(PROJ_RES))
#define PROJ_WORDS       ((PROJ_RES * PROJ_RES + 31u) / 32u)

struct Basis2 {
    float3 u, w;
};

inline Basis2 orthonormal_basis(const float3& v) {
    float3 axis = (std::abs(v.z) < 0.999f) ? float3(0, 0, 1) : float3(1, 0, 0);
    Basis2 b;
    b.u = safeNormalize(cross(axis, v));
    b.w = safeNormalize(cross(v, b.u));
    return b;
}

// Rasterization helpers
inline float edgeFunc(const float2& a, const float2& b, const float2& p) {
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

inline bool pointInTriangle(const float2& p, const float2& a, const float2& b, const float2& c) {
    float e0 = edgeFunc(a, b, p);
    float e1 = edgeFunc(b, c, p);
    float e2 = edgeFunc(c, a, p);
    const float eps = 1e-6f;
    bool hasNeg = (e0 < -eps) || (e1 < -eps) || (e2 < -eps);
    bool hasPos = (e0 >  eps) || (e1 >  eps) || (e2 >  eps);
    return !(hasNeg && hasPos);
}

inline int clampInt(int x, int lo, int hi) { return std::max(lo, std::min(hi, x)); }

inline void maskSet(uint32_t mask[PROJ_WORDS], uint32_t linearIndex) {
    uint32_t word = linearIndex >> 5;
    uint32_t bit  = linearIndex & 31u;
    mask[word] |= (1u << bit);
}

inline uint32_t maskCountBits(const uint32_t mask[PROJ_WORDS]) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < PROJ_WORDS; ++i)
        sum += countbits(mask[i]);
    return sum;
}

inline void rasterizeTriToMask(const float2& a, const float2& b, const float2& c, uint32_t mask[PROJ_WORDS]) {
    float2 mn = glm::min(a, glm::min(b, c));
    float2 mx = glm::max(a, glm::max(b, c));
    int ix0 = (int)std::floor((mn.x + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE);
    int iy0 = (int)std::floor((mn.y + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE);
    int ix1 = (int)std::floor((mx.x + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE);
    int iy1 = (int)std::floor((mx.y + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE);
    int Rm1 = PROJ_RES - 1;
    ix0 = clampInt(ix0, 0, Rm1); ix1 = clampInt(ix1, 0, Rm1);
    iy0 = clampInt(iy0, 0, Rm1); iy1 = clampInt(iy1, 0, Rm1);
    for (int iy = iy0; iy <= iy1; ++iy) {
        float y = -PROJ_HALF_SIZE + (float(iy) + 0.5f) * PROJ_PIXEL_SIZE;
        for (int ix = ix0; ix <= ix1; ++ix) {
            float x = -PROJ_HALF_SIZE + (float(ix) + 0.5f) * PROJ_PIXEL_SIZE;
            if (pointInTriangle(float2(x, y), a, b, c))
                maskSet(mask, uint32_t(iy) * PROJ_RES + uint32_t(ix));
        }
    }
}

struct PolygonRange {
    uint localHead = 0;
    uint count = 0;
    int3 cellInt = int3(0);
    float nodeScale = 1.0f;

    void init(const int3& c) {
        localHead = 0; count = 0;
        cellInt = c; nodeScale = 1.0f;
    }

    float3 calcCentroid(const std::vector<Polygon>& polygons, float& totalArea) const {
        float3 sum(0); totalArea = 0;
        for (uint i = 0; i < count; ++i) {
            float area;
            float3 c = polygons[i + localHead].calcCentroid(area);
            sum += c * area;
            totalArea += area;
        }
        return (totalArea > 0) ? sum / totalArea : float3(cellInt) + float3(0.5f);
    }

    float calcTotalProjArea(const std::vector<Polygon>& polygons, const float3& direction) const {
        float sum = 0;
        for (uint i = 0; i < count; ++i)
            sum += polygons[i + localHead].calcProjArea(direction);
        return sum;
    }

    uint32_t calcVisibleProjCellsRaster(const std::vector<Polygon>& polygons, const float3& direction) const {
        Basis2 b = orthonormal_basis(safeNormalize(direction));
        float3 cubeCenter = float3(cellInt) + float3(0.5f);
        uint32_t mask[PROJ_WORDS] = {};
        for (uint pi = 0; pi < count; ++pi) {
            const Polygon& poly = polygons[localHead + pi];
            uint n = poly.count;
            if (n < 3) continue;
            float2 pv[MAX_VERTEX_COUNT];
            for (uint k = 0; k < n; ++k) {
                float3 p = poly.vertices[k] - cubeCenter;
                pv[k] = float2(dot(p, b.u), dot(p, b.w));
            }
            for (uint k = 1; k + 1 < n; ++k)
                rasterizeTriToMask(pv[0], pv[k], pv[k + 1], mask);
        }
        return maskCountBits(mask);
    }

    float calcVisibleProjAreaRaster(const std::vector<Polygon>& polygons, const float3& direction) const {
        return float(calcVisibleProjCellsRaster(polygons, direction)) * (PROJ_PIXEL_SIZE * PROJ_PIXEL_SIZE);
    }
};
