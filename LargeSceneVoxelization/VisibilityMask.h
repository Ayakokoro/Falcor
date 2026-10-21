#pragma once
#include "Types.h"
#include "VisibilityMaskFormat.h"

// One bit has the same surface-sample identity in all direction texels.
// This CPU payload is deliberately separate from the GPU-shared VoxelData.
namespace VisibilityMask {

inline float2 octWrap(const float2& p) {
    return float2((1.0f - std::abs(p.y)) * (p.x >= 0.0f ? 1.0f : -1.0f),
                  (1.0f - std::abs(p.x)) * (p.y >= 0.0f ? 1.0f : -1.0f));
}

// Matches MathHelpers.slang::oct_to_ndir_unorm(), including the lower fold.
inline float3 direction(uint32_t index) {
    const float2 p = 2.0f * float2((float(index % Resolution) + 0.5f) / Resolution,
                                  (float(index / Resolution) + 0.5f) / Resolution) - 1.0f;
    float3 n(p, 1.0f - std::abs(p.x) - std::abs(p.y));
    if (n.z < 0.0f) {
        const float2 wrapped = octWrap(p);
        n.x = wrapped.x;
        n.y = wrapped.y;
    }
    return safeNormalize(n);
}

// Nearest texel lookup, with the same convention as ndir_to_oct_unorm().
// DirectionCount is the invalid-direction sentinel.
inline uint32_t directionIndex(const float3& n) {
    const float norm = std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
    if (!(norm > 0.0f) || !std::isfinite(norm)) return DirectionCount;
    float2 p(n.x / norm, n.y / norm);
    if (n.z < 0.0f) p = octWrap(p);
    const float2 uv = glm::clamp(p * 0.5f + 0.5f, float2(0), float2(1));
    const uint32_t x = std::min(uint32_t(uv.x * Resolution), Resolution - 1);
    const uint32_t y = std::min(uint32_t(uv.y * Resolution), Resolution - 1);
    return y * Resolution + x;
}

struct Query {
    float value = 1.0f; // Bright fallback, NOT a measurement of full visibility.
    uint32_t outgoingSamples = 0;
    bool supported = false;
};

// Caller handles the lobe's incident/outgoing hemisphere and BRDF cosines.
inline Query query(const Data& data, uint32_t compactLobe,
                   uint32_t incidentDirection, uint32_t outgoingDirection) {
    Query result;
    const uint32_t bits = lobeBits(compactLobe, data.lobeCount());
    if (!bits || incidentDirection >= DirectionCount || outgoingDirection >= DirectionCount ||
        !(data.validLobes() & (1u << compactLobe)))
        return result;
    const uint32_t visible = data.masks[outgoingDirection] & bits;
    result.outgoingSamples = countbits(visible);
    if (result.outgoingSamples == 0) return result;
    result.value = float(countbits(data.masks[incidentDirection] & visible)) / result.outgoingSamples;
    result.supported = true;
    return result;
}

} // namespace VisibilityMask
