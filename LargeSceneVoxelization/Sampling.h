#pragma once
#include "Types.h"
#include "Polygon.h"

struct SamplingRay {
    float3 origin;
    float3 direction;
};

inline float2 sampleDisk(const float2& u) {
    float r = std::sqrt(u.x);
    float phi = 2.0f * M_PI_F * u.y;
    return float2(r * std::cos(phi), r * std::sin(phi));
}

// Hammersley low-discrepancy sequence
inline float2 Hammersley2D(uint i, uint numSamples) {
    float u = (float)i / (float)numSamples;
    uint bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555u) << 1) | ((bits & 0xAAAAAAAAu) >> 1);
    bits = ((bits & 0x33333333u) << 2) | ((bits & 0xCCCCCCCCu) >> 2);
    bits = ((bits & 0x0F0F0F0Fu) << 4) | ((bits & 0xF0F0F0F0u) >> 4);
    bits = ((bits & 0x00FF00FFu) << 8) | ((bits & 0xFF00FF00u) >> 8);
    float v = (float)bits * 2.3283064365386963e-10f;
    return float2(u, v);
}

// Ray from disk on unit hemisphere toward voxel center
inline SamplingRay rayToVoxel(const float2& uDisk, const float3& direction,
                               const Basis2& basis, const float3& cellCenter) {
    float2 p = sampleDisk(uDisk);
    const float r = 0.8660254f;
    float uu = p.x * r, ww = p.y * r;
    float h = std::sqrt(std::max(0.0f, r * r - uu * uu - ww * ww));
    float3 point = h * direction + uu * basis.u + ww * basis.w;
    return { point + cellCenter, -direction };
}
