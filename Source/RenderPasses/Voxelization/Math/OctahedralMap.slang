#pragma once

#include "../HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

#define OCTA_RES 4
#define OCTA_COUNT (OCTA_RES * OCTA_RES)
// Number of deterministic sub-texel directions used when filtering each
// stored projection-area texel. Keep this synchronized with
// LargeSceneVoxelization/OctahedralMap.h.
#define OCTA_SAMPLES_PER_TEXEL 4

// The projection-area functions are even: directions on the lower
// hemisphere are folded onto the upper hemisphere before lookup.
#if !defined(HOST_CODE)
inline float2 octaEncode(float3 direction)
{
    direction = normalize(direction);
    if (direction.z < 0.0f)
        direction = -direction;

    const float denominator = abs(direction.x) + abs(direction.y) + direction.z;
    float2 p = direction.xy / max(denominator, 1e-8f);
    float2 s = float2(p.x - p.y, p.x + p.y);
    return s * 0.5f + 0.5f;
}

inline float3 octaDecode(float2 uv)
{
    float2 s = uv * 2.0f - 1.0f;
    float2 p = float2(s.x + s.y, s.y - s.x) * 0.5f;
    float z = 1.0f - abs(p.x) - abs(p.y);
    return normalize(float3(p.x, p.y, max(z, 0.0f)));
}

// This is intentionally the same sequence as the CPU Hammersley2D helper in
// LargeSceneVoxelization/Sampling.h (the first coordinate is i/N, not the
// center of the interval). It is used both for the directional prefilter and
// for the primitive-area Monte Carlo estimator so the CPU and GPU maps match.
inline uint octahedralRadicalInverseVdC(uint bits)
{
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555u) << 1) | ((bits & 0xAAAAAAAAu) >> 1);
    bits = ((bits & 0x33333333u) << 2) | ((bits & 0xCCCCCCCCu) >> 2);
    bits = ((bits & 0x0F0F0F0Fu) << 4) | ((bits & 0xF0F0F0F0u) >> 4);
    bits = ((bits & 0x00FF00FFu) << 8) | ((bits & 0xFF00FF00u) >> 8);
    return bits;
}

inline float2 octahedralHammersley2D(uint index, uint sampleCount)
{
    sampleCount = max(sampleCount, 1u);
    return float2(
        float(index) / float(sampleCount),
        float(octahedralRadicalInverseVdC(index)) * 2.3283064365386963e-10f);
}

inline float3 octahedralTexelSampleDirection(uint2 texel, uint sampleIndex)
{
    float2 localUV = octahedralHammersley2D(sampleIndex, OCTA_SAMPLES_PER_TEXEL);
    float2 uv = (float2(texel) + localUV) / float(OCTA_RES);
    return octaDecode(uv);
}
#endif

struct OctahedralFunc
{
    float coefficients[OCTA_COUNT];

    MUTATING
    void init()
    {
        for (int i = 0; i < OCTA_COUNT; ++i)
            coefficients[i] = 0.0f;
    }

#if !defined(HOST_CODE)
    float calc(float3 direction)
    {
        float2 uv = octaEncode(direction);
        float2 grid = uv * float(OCTA_RES) - 0.5f;
        int2 i0 = int2(floor(grid));
        float2 fraction = grid - float2(i0);
        int2 i1 = min(i0 + 1, int2(OCTA_RES - 1));
        i0 = max(i0, int2(0));

        float v00 = coefficients[i0.y * OCTA_RES + i0.x];
        float v10 = coefficients[i0.y * OCTA_RES + i1.x];
        float v01 = coefficients[i1.y * OCTA_RES + i0.x];
        float v11 = coefficients[i1.y * OCTA_RES + i1.x];
        return lerp(
            lerp(v00, v10, fraction.x),
            lerp(v01, v11, fraction.x),
            fraction.y);
    }
#endif
};

END_NAMESPACE_FALCOR
