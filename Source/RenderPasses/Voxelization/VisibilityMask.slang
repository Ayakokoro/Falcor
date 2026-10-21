#pragma once
#include "HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

// Mirrored by VisibilityMaskFormat.h; host static_asserts check these values.
static const uint kVisibilityMaskResolution = 4;
static const uint kVisibilityMaskDirectionCount = 16;
static const uint kVisibilityMaskSampleCount = 32;
static const uint kVisibilityMaskMaxLobes = 8;
static const uint kVisibilityMaskWordsPerRecord = 17;

enum class InternalVisibilityMode : uint32_t
{
    Legacy = 0,
    Bitmask = 1,
    Off = 2,
};
FALCOR_ENUM_INFO(
    InternalVisibilityMode,
    {
        { InternalVisibilityMode::Legacy, "Legacy one-way (VPA/TPA)" },
        { InternalVisibilityMode::Bitmask, "Bitmask" },
        { InternalVisibilityMode::Off, "Off" },
    }
);
FALCOR_ENUM_REGISTER(InternalVisibilityMode);

END_NAMESPACE_FALCOR

#if !defined(HOST_CODE)
import Utils.Math.MathHelpers;

// Only the two requested direction words and the metadata are fetched.
struct VisibilityMaskPair
{
    uint incident;
    uint outgoing;
    uint metadata;
};

struct VisibilityMaskQuery
{
    float value;
    uint outgoingSamples;
    bool supported;
};

uint visibilityMaskDirectionIndex(float3 direction)
{
    float norm = dot(abs(direction), float3(1));
    if (!(norm > 0.0f) || !isfinite(norm))
        return kVisibilityMaskDirectionCount;
    float2 uv = saturate(ndir_to_oct_unorm(direction));
    uint2 xy = min(uint2(uv * kVisibilityMaskResolution), uint2(kVisibilityMaskResolution - 1));
    return xy.y * kVisibilityMaskResolution + xy.x;
}

VisibilityMaskQuery queryVisibilityMask(VisibilityMaskPair pair, uint compactLobe, uint materialLobeCount)
{
    // Unsupported is a bright fallback, not a measured fully visible result.
    VisibilityMaskQuery result = { 1.0f, 0, false };
    uint count = countbits(pair.metadata & 0xffu);
    if (count == 0 || count > kVisibilityMaskMaxLobes || count != materialLobeCount || compactLobe >= count)
        return result;
    if ((pair.metadata & (1u << (8 + compactLobe))) == 0)
        return result;
    uint begin = kVisibilityMaskSampleCount * compactLobe / count;
    uint end = kVisibilityMaskSampleCount * (compactLobe + 1) / count;
    uint bits = (0xffffffffu >> (kVisibilityMaskSampleCount - (end - begin))) << begin;
    uint visible = pair.outgoing & bits;
    result.outgoingSamples = countbits(visible);
    if (result.outgoingSamples == 0)
        return result;
    result.value = float(countbits(pair.incident & visible)) / float(result.outgoingSamples);
    result.supported = true;
    return result;
}
#endif
