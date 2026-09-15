#pragma once

#include "Types.h"

#include <algorithm>
#include <cstdint>
#include <cmath>

// Projection area is an even directional function for the CPU voxel data:
// A(d) == A(-d).  Store the upper hemisphere in a folded octahedral map.
// Keep this value synchronized with the Slang implementation in
// Source/RenderPasses/Voxelization/Math/OctahedralMap.slang.
constexpr uint32_t OCTA_RES = 4;
constexpr uint32_t OCTA_COUNT = OCTA_RES * OCTA_RES;
// Deterministic directions sampled inside each texel during CPU prefiltering.
// Increase this value for smoother maps at the cost of proportionally more
// projection-area evaluations.
constexpr uint32_t OCTA_SAMPLES_PER_TEXEL = 4;

inline float2 octahedralEncode(const float3& direction)
{
    float3 d = safeNormalize(direction);
    if (d.z < 0.0f)
        d = -d;

    const float denominator = std::abs(d.x) + std::abs(d.y) + d.z;
    if (denominator <= 1e-8f)
        return float2(0.5f);

    const float2 p = float2(d.x, d.y) / denominator;
    const float2 s(p.x - p.y, p.x + p.y);
    return 0.5f * s + float2(0.5f);
}

inline float3 octahedralDecode(const float2& uv)
{
    const float2 s = uv * 2.0f - float2(1.0f);
    const float2 p(
        0.5f * (s.x + s.y),
        0.5f * (s.y - s.x));
    const float z = 1.0f - std::abs(p.x) - std::abs(p.y);
    return safeNormalize(float3(p.x, p.y, std::max(z, 0.0f)));
}

struct OctahedralFunc
{
    float coefficients[OCTA_COUNT] = {};

    void init()
    {
        for (uint32_t i = 0; i < OCTA_COUNT; ++i)
            coefficients[i] = 0.0f;
    }

    float eval(const float3& direction) const
    {
        if (dot(direction, direction) <= 1e-12f)
            return 0.0f;

        const float2 uv = octahedralEncode(direction);
        const float2 grid = uv * float(OCTA_RES) - float2(0.5f);
        const float2 base = glm::floor(grid);
        const int rawX0 = static_cast<int>(base.x);
        const int rawY0 = static_cast<int>(base.y);
        const int x0 = std::clamp(rawX0, 0, int(OCTA_RES) - 1);
        const int y0 = std::clamp(rawY0, 0, int(OCTA_RES) - 1);
        const int x1 = std::clamp(rawX0 + 1, 0, int(OCTA_RES) - 1);
        const int y1 = std::clamp(rawY0 + 1, 0, int(OCTA_RES) - 1);
        const float fx = grid.x - std::floor(grid.x);
        const float fy = grid.y - std::floor(grid.y);

        auto sample = [this](int x, int y) -> float
        {
            return coefficients[uint32_t(y) * OCTA_RES + uint32_t(x)];
        };

        const float a = lerp(sample(x0, y0), sample(x1, y0), fx);
        const float b = lerp(sample(x0, y1), sample(x1, y1), fx);
        return lerp(a, b, fy);
    }
};

static_assert(sizeof(OctahedralFunc) == OCTA_COUNT * sizeof(float));
