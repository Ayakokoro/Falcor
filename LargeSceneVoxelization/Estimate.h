#pragma once
#include "Types.h"
#include "Sampling.h"
#include "Ellipsoid.h"
#include "SphericalHarmonics.h"
#include "LebedevQuadrature.h"
#include "VoxelData.h"
#include "Polygon.h"

// AABB clip: returns (tEnter, tExit) for ray from->to against unit voxel [cellMin, cellMin+1]
inline float2 clipAABB(const float3& cellMin, const float3& cellMax,
                       const float3& from, const float3& to) {
    float uIn = 0.0f, uOut = 1.0f;
    float3 v = to - from;
    for (int a = 0; a < 3; a++) {
        if (std::abs(v[a]) < 1e-12f) {
            if (from[a] < cellMin[a] || from[a] > cellMax[a])
                return float2(-1, -1);
        } else {
            float t0 = (cellMin[a] - from[a]) / v[a];
            float t1 = (cellMax[a] - from[a]) / v[a];
            if (t0 > t1) std::swap(t0, t1);
            uIn = std::max(uIn, t0);
            uOut = std::min(uOut, t1);
            if (uIn > uOut) return float2(-1, -1);
        }
    }
    return float2(uIn, uOut);
}

// CPU port of AnalyzePolygon.cs.slang:Estimate()
// Performs Lebedev quadrature to estimate projected area SH functions
inline void Estimate(Ellipsoid& e, PolygonRange& range,
                     SphericalFunc& polygonFunc,
                     SphericalFunc& primitiveFunc,
                     SphericalFunc& totalFunc,
                     const std::vector<Polygon>& polygonBuffer,
                     uint sampleFrequency) {
    float3 cellMin = float3(range.cellInt);
    float3 cellCenter = cellMin + float3(0.5f);

    for (uint i = 0; i < LEBEDEV_DIRECTION_COUNT; i += 2) {
        const LebedevSample& sample = LebedevSamples[i];
        float3 direction = sample.direction;
        Basis2 basis = orthonormal_basis(direction);

        uint hitCount = 0;
        for (uint j = 0; j < sampleFrequency; j++) {
            SamplingRay ray = rayToVoxel(Hammersley2D(j, sampleFrequency), direction, basis, cellCenter);
            float3 to = ray.origin + 2.0f * ray.direction;
            float2 inOut = clipAABB(cellMin, cellMin + float3(1.0f), ray.origin, to);
            if (inOut.x >= 0) {
                float3 v = to - ray.origin;
                float3 from = ray.origin + inOut.x * v - cellMin;
                float3 tto  = ray.origin + inOut.y * v - cellMin;
                inOut = e.clip(from, tto);
                if (inOut.x >= 0) hitCount++;
            }
        }

        float weight = 8.0f * M_PI_F * sample.weight;
        float primitiveProjArea = PROJECT_CIRCLE_AREA * (float)hitCount / (float)sampleFrequency;
        primitiveFunc.accumulate(weight * primitiveProjArea, direction);

        float totalProjArea = range.calcTotalProjArea(polygonBuffer, direction);
        float visibleProjArea = range.calcVisibleProjAreaRaster(polygonBuffer, direction);
        polygonFunc.accumulate(weight * visibleProjArea, direction);
        totalFunc.accumulate(weight * totalProjArea, direction);
    }
}
