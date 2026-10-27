#pragma once
#include "Types.h"
#include "Polygon.h"
#include <memory>
#include <mutex>
#include <unordered_map>

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

// Direction-independent part of a disk/hemisphere ray. Keep the original
// formulas and sample order. Fast-math may round an uncached/fused expression
// differently from a stored float, so bitwise identity is not guaranteed.
struct VoxelRaySample {
    float uu, ww, h;
};

inline VoxelRaySample makeVoxelRaySample(const float2& uDisk) {
    float2 p = sampleDisk(uDisk);
    const float r = 0.8660254f;
    float uu = p.x * r, ww = p.y * r;
    float h = std::sqrt(std::max(0.0f, r * r - uu * uu - ww * ww));
    return {uu, ww, h};
}

// Immutable tables are shared by workers. The thread-local fast path avoids a
// lock/refcount operation per node; weak entries release unused frequencies.
// The returned reference stays valid until this thread requests another count.
inline const std::vector<VoxelRaySample>& voxelRaySamples(uint32_t frequency) {
    using Table = std::vector<VoxelRaySample>;
    static std::mutex mutex;
    static std::unordered_map<uint32_t, std::weak_ptr<const Table>> tables;
    thread_local std::shared_ptr<const Table> current;
    frequency = std::max(1u, frequency);
    if (!current || current->size() != frequency) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = tables.begin(); it != tables.end();) {
            if (it->second.expired()) it = tables.erase(it);
            else ++it;
        }
        auto& entry = tables[frequency];
        auto table = entry.lock();
        if (!table) {
            auto samples = std::make_shared<Table>();
            samples->reserve(frequency);
            for (uint32_t i = 0; i < frequency; ++i)
                samples->push_back(makeVoxelRaySample(Hammersley2D(i, frequency)));
            table = samples;
            entry = table;
        }
        current = std::move(table);
    }
    return *current;
}

// Ray from disk on unit hemisphere toward voxel center. Translation and the
// subsequent AABB clipping remain per node, including their float rounding.
inline SamplingRay rayToVoxel(const VoxelRaySample& sample, const float3& direction,
                               const Basis2& basis, const float3& cellCenter) {
    float3 point = sample.h * direction + sample.uu * basis.u + sample.ww * basis.w;
    return { point + cellCenter, -direction };
}

inline SamplingRay rayToVoxel(const float2& uDisk, const float3& direction,
                               const Basis2& basis, const float3& cellCenter) {
    return rayToVoxel(makeVoxelRaySample(uDisk), direction, basis, cellCenter);
}
