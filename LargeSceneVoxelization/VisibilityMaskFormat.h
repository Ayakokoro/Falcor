#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Integer-only VBM1 payload shared by the standalone baker and the renderer.
// Keep GLM/Types.h out of this header: Falcor uses its own math types and C++17.
namespace VisibilityMask {

constexpr uint32_t Resolution = 4;
constexpr uint32_t DirectionCount = Resolution * Resolution;
constexpr uint32_t SampleCount = 32;
constexpr uint32_t MaxLobes = 8;
constexpr uint32_t WordsPerRecord = DirectionCount + 1;

struct Data {
    // One bit has the same surface-sample identity in all direction texels.
    std::array<uint32_t, DirectionCount> masks{};
    // Bits 0..7: original ABSDF valid-lobe mask (ascending compaction order).
    // Bits 8..15: fully sampled lobes, indexed AFTER compaction. Rest: zero.
    uint32_t metadata = 0;

    uint32_t sourceLobes() const { return metadata & 0xffu; }
    uint32_t lobeCount() const {
        uint32_t bits = sourceLobes(), count = 0;
        for (; bits; bits &= bits - 1) ++count;
        return count;
    }
    uint32_t validLobes() const { return (metadata >> 8) & 0xffu; }
};
static_assert(sizeof(Data) == WordsPerRecord * sizeof(uint32_t) && sizeof(Data) == 68);
static_assert(offsetof(Data, metadata) == 64 && std::is_trivially_copyable_v<Data>);

inline uint32_t lobeBits(uint32_t compactLobe, uint32_t lobeCount) {
    if (lobeCount == 0 || lobeCount > MaxLobes || compactLobe >= lobeCount)
        return 0;
    const uint32_t begin = SampleCount * compactLobe / lobeCount;
    const uint32_t end = SampleCount * (compactLobe + 1) / lobeCount;
    // Also handles a single lobe without ever shifting by 32.
    return (0xffffffffu >> (SampleCount - (end - begin))) << begin;
}

} // namespace VisibilityMask
