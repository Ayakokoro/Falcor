#pragma once

#include "Ellipsoid.h"
#include "OctahedralMap.h"
#include "Polygon.h"
#include "Sampling.h"
#include "Types.h"
#include "VoxelData.h"
#include "BakeTiming.h"
#include <array>

// AABB clip: returns (tEnter, tExit) for ray from->to against the box.
inline float2 clipAABB(const float3& cellMin, const float3& cellMax,
                       const float3& from, const float3& to)
{
    float uIn = 0.0f;
    float uOut = 1.0f;
    const float3 v = to - from;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (std::abs(v[axis]) < 1e-12f)
        {
            if (from[axis] < cellMin[axis] || from[axis] > cellMax[axis])
                return float2(-1.0f);
        }
        else
        {
            float t0 = (cellMin[axis] - from[axis]) / v[axis];
            float t1 = (cellMax[axis] - from[axis]) / v[axis];
            if (t0 > t1)
                std::swap(t0, t1);
            uIn = std::max(uIn, t0);
            uOut = std::min(uOut, t1);
            if (uIn > uOut)
                return float2(-1.0f);
        }
    }
    return float2(uIn, uOut);
}

// Estimate the ellipsoid's projected area for one direction. This is the
// same Monte Carlo estimator that was previously used while accumulating SH,
// but it now returns one actual function value for the octahedral table.
inline float primitiveProjectionArea(const Ellipsoid& ellipsoid,
                                     const PolygonRange& range,
                                     const float3& direction,
                                     const Basis2& basis,
                                     const std::vector<VoxelRaySample>& samples)
{
    const float3 cellMin = float3(range.cellInt);
    const float3 cellCenter = cellMin + float3(0.5f);
    uint32_t hitCount = 0;

    for (const auto& sample : samples)
    {
        const SamplingRay ray = rayToVoxel(sample, direction, basis, cellCenter);
        const float3 to = ray.origin + 2.0f * ray.direction;
        float2 inOut = clipAABB(cellMin, cellMin + float3(1.0f), ray.origin, to);
        if (inOut.x < 0.0f)
            continue;

        const float3 delta = to - ray.origin;
        const float3 fromLocal = ray.origin + inOut.x * delta - cellMin;
        const float3 toLocal = ray.origin + inOut.y * delta - cellMin;
        if (ellipsoid.intersectsSegment(fromLocal, toLocal))
            ++hitCount;
    }

    return PROJECT_CIRCLE_AREA * static_cast<float>(hitCount) /
        static_cast<float>(samples.size());
}

// Arbitrary-direction diagnostics share the same disk samples as the baker.
inline float primitiveProjectionArea(const Ellipsoid& ellipsoid,
                                     const PolygonRange& range,
                                     const float3& direction,
                                     uint32_t sampleFrequency)
{
    const auto& samples = voxelRaySamples(sampleFrequency);
    const Basis2 basis = orthonormal_basis(safeNormalize(direction));
    return primitiveProjectionArea(ellipsoid, range, direction, basis, samples);
}

struct OctahedralProjectionDirection {
    float3 direction;
    Basis2 basis;
};

inline const auto& octahedralProjectionDirections()
{
    static_assert(OCTA_SAMPLES_PER_TEXEL > 0);
    static const auto directions = [] {
        std::array<OctahedralProjectionDirection, OCTA_COUNT * OCTA_SAMPLES_PER_TEXEL> result;
        for (uint32_t y = 0; y < OCTA_RES; ++y)
            for (uint32_t x = 0; x < OCTA_RES; ++x)
                for (uint32_t sample = 0; sample < OCTA_SAMPLES_PER_TEXEL; ++sample) {
                    const float2 localUV = Hammersley2D(sample, OCTA_SAMPLES_PER_TEXEL);
                    const float2 uv((static_cast<float>(x) + localUV.x) / static_cast<float>(OCTA_RES),
                                    (static_cast<float>(y) + localUV.y) / static_cast<float>(OCTA_RES));
                    auto& entry = result[(y * OCTA_RES + x) * OCTA_SAMPLES_PER_TEXEL + sample];
                    entry.direction = octahedralDecode(uv);
                    entry.basis = orthonormal_basis(safeNormalize(entry.direction));
                }
        return result;
    }();
    return directions;
}

// Fill three direction-dependent projection-area functions. Each texel is
// supersampled inside its octahedral cell and stores the average area; no
// Lebedev weight or SH basis evaluation is involved.
inline void computeOctahedralProjectionAreas(
    VoxelData& voxelData,
    const PolygonRange& range,
    const std::vector<Polygon>& polygons,
    uint32_t sampleFrequency,
    BakeTiming::Worker* timings = nullptr)
{
    voxelData.primitiveProjAreaFunc.init();
    voxelData.polygonsProjAreaFunc.init();
    voxelData.totalProjAreaFunc.init();

    const uint32_t frequency = std::max(1u, sampleFrequency);
    constexpr uint32_t samplesPerTexel = OCTA_SAMPLES_PER_TEXEL;
    static_assert(samplesPerTexel > 0);
    const auto& samples = voxelRaySamples(frequency);
    const auto& directions = octahedralProjectionDirections();
    if (timings)
        timings->primitiveSamples += uint64_t(OCTA_RES) * OCTA_RES * samplesPerTexel * frequency;

    for (uint32_t y = 0; y < OCTA_RES; ++y)
    {
        for (uint32_t x = 0; x < OCTA_RES; ++x)
        {
            const uint32_t index = y * OCTA_RES + x;
            float primitiveArea = 0.0f;
            float visibleArea = 0.0f;
            float totalArea = 0.0f;

            for (uint32_t sample = 0; sample < samplesPerTexel; ++sample)
            {
                const auto& entry = directions[index * samplesPerTexel + sample];
                const float3 direction = entry.direction;

                BakeTiming::Scope timer(timings, BakeTiming::Task::PrimitiveProjection);
                primitiveArea += primitiveProjectionArea(
                    voxelData.ellipsoid, range, direction, entry.basis, samples);
                timer.next(timings, BakeTiming::Task::VisibleProjection);
                visibleArea += range.calcVisibleProjAreaRaster(
                    polygons, direction);
                timer.next(timings, BakeTiming::Task::TotalProjection);
                totalArea += range.calcTotalProjArea(polygons, direction);
            }

            const float inverseSampleCount = 1.0f / float(samplesPerTexel);
            voxelData.primitiveProjAreaFunc.coefficients[index] =
                primitiveArea * inverseSampleCount;
            voxelData.polygonsProjAreaFunc.coefficients[index] =
                visibleArea * inverseSampleCount;
            voxelData.totalProjAreaFunc.coefficients[index] =
                totalArea * inverseSampleCount;
        }
    }
}
