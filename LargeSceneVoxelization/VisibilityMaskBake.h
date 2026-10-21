#pragma once
#include "VisibilityMask.h"
#include "VisibleABSDF.h"
#include "Polygon.h"
#include <limits>
#include <stdexcept>
#include <tuple>

namespace VisibilityMaskBake {

// Also used as the base of the legacy material/correlation scratch, so the
// sixteen mask projections can overwrite material maps without another arena.
struct ProjectionMaps {
    static constexpr uint32_t PixelCount = PROJ_RES * PROJ_RES;
    std::vector<float> depths;
    std::vector<uint32_t> ids;
    std::vector<float3> directions;
    std::vector<Basis2> bases;

    explicit ProjectionMaps(uint32_t count = VisibilityMask::DirectionCount)
        : depths(size_t(count) * PixelCount), ids(size_t(count) * PixelCount),
          directions(count), bases(count) {}

    float* depthData(uint32_t k) { return depths.data() + size_t(k) * PixelCount; }
    uint32_t* idData(uint32_t k) { return ids.data() + size_t(k) * PixelCount; }
    const float* depthData(uint32_t k) const { return depths.data() + size_t(k) * PixelCount; }
    const uint32_t* idData(uint32_t k) const { return ids.data() + size_t(k) * PixelCount; }

    void rasterize(uint32_t k, const PolygonRange& range,
                   const std::vector<Polygon>& polygons, const float3& dir) {
        directions[k] = dir;
        bases[k] = orthonormal_basis(dir);
        std::fill_n(depthData(k), PixelCount, std::numeric_limits<float>::max());
        std::fill_n(idData(k), PixelCount, UINT32_MAX);
        range.rasterizeDepth(polygons, dir, depthData(k), idData(k));
    }
};

// The only occluder available is the polygon owning this depth texel. Correct
// its depth to the query location using its plane; do not use a large slope
// bias that would merge nearby layers. Empty texels / uncovered texel corners
// mean no *observed* occluder, a deliberately bright raster approximation.
inline bool visible(const ProjectionMaps& maps, uint32_t d,
                    const std::vector<Polygon>& polygons, const float3& center,
                    const float3& point, uint32_t sourcePolygon) {
    const Basis2& basis = maps.bases[d];
    const float3 relative = point - center;
    const float2 p(dot(relative, basis.u), dot(relative, basis.w));
    const int x = int(std::floor((p.x + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE));
    const int y = int(std::floor((p.y + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE));
    if (x < 0 || y < 0 || x >= PROJ_RES || y >= PROJ_RES) return false;
    const uint32_t pixel = uint32_t(y * PROJ_RES + x);
    const uint32_t owner = maps.idData(d)[pixel];
    if (owner == UINT32_MAX || owner == sourcePolygon) return true;
    const Polygon& occluder = polygons[owner];
    const float nd = dot(occluder.normal, maps.directions[d]);
    if (std::abs(nd) < 1e-6f) return true;
    const float2 texel(-PROJ_HALF_SIZE + (float(x) + 0.5f) * PROJ_PIXEL_SIZE,
                       -PROJ_HALF_SIZE + (float(y) + 0.5f) * PROJ_PIXEL_SIZE);
    const float3 offset = (p.x - texel.x) * basis.u + (p.y - texel.y) * basis.w;
    const float ownerDepth = maps.depthData(d)[pixel] + dot(occluder.normal, offset) / nd;
    constexpr float DepthEpsilon = 1e-5f; // Units of the normalized node edge.
    if (-dot(relative, maps.directions[d]) <= ownerDepth + DepthEpsilon) return true;

    float2 projected[MAX_VERTEX_COUNT];
    for (uint32_t v = 0; v < occluder.count; ++v) {
        const float3 q = occluder.vertices[v] - center;
        projected[v] = float2(dot(q, basis.u), dot(q, basis.w));
    }
    for (uint32_t v = 1; v + 1 < occluder.count; ++v)
        if (pointInTriangle(p, projected[0], projected[v], projected[v + 1])) return false;
    return true;
}

struct Stats {
    uint64_t candidates = 0;
    uint64_t acceptedSamples = 0;
    uint64_t activeLobes = 0;
    uint64_t invalidLobes = 0;
    uint64_t frontDirections = 0;
    uint64_t unsupportedFrontDirections = 0;

    Stats& operator+=(const Stats& other) {
        candidates += other.candidates;
        acceptedSamples += other.acceptedSamples;
        activeLobes += other.activeLobes;
        invalidLobes += other.invalidLobes;
        frontDirections += other.frontDirections;
        unsupportedFrontDirections += other.unsupportedFrontDirections;
        return *this;
    }
};

// Per-node deterministic PRNG; no shared state or dependence on worker order.
struct Random {
    uint32_t state;
    explicit Random(uint32_t seed) : state(seed) {}
    double next() {
        state += 0x9e3779b9u;
        uint32_t x = state;
        x = (x ^ (x >> 16)) * 0x21f0aaadu;
        x = (x ^ (x >> 15)) * 0x735a2d97u;
        x ^= x >> 15;
        return (double(x) + 0.5) * (1.0 / 4294967296.0);
    }
};

inline double triangleArea(const Polygon& p, uint32_t fan) {
    return 0.5 * double(length(cross(p.vertices[fan] - p.vertices[0],
                                     p.vertices[fan + 1] - p.vertices[0])));
}

// Bounded rejection sampling. All accepted points in a lobe have equal area
// weight; a lobe that cannot fill its allocation is invalidated, not padded.
constexpr uint32_t AttemptsPerSample = 256;
struct AreaEntry { uint32_t polygon; double cumulativeArea; };

inline VisibilityMask::Data bake(const VisibleABSDF& absdf, const PolygonRange& range,
                                 const std::vector<Polygon>& polygons,
                                 const std::vector<uint32_t>& lobePairs,
                                 ProjectionMaps& maps, uint32_t seed, Stats& stats) {
    if (range.localHead != 0 || range.count != polygons.size() || lobePairs.size() != polygons.size() ||
        maps.directions.size() < VisibilityMask::DirectionCount)
        throw std::invalid_argument("Visibility masks require one complete node polygon block and 16 projections.");
    VisibilityMask::Data result;
    std::array<uint32_t, VISIBLE_LOBE_COUNT> sourceLobes{};
    uint32_t lobeCount = 0;
    for (uint32_t k = 0; k < VISIBLE_LOBE_COUNT; ++k) {
        if (!absdf.lobes[k].isValid()) continue;
        result.metadata |= 1u << k;
        sourceLobes[lobeCount++] = k;
    }
    stats.activeLobes += lobeCount;
    if (lobeCount == 0) return result;

    std::vector<uint8_t> eligible(polygons.size(), 0);
    for (uint32_t d = 0; d < VisibilityMask::DirectionCount; ++d) {
        maps.rasterize(d, range, polygons, VisibilityMask::direction(d));
        const uint32_t* owners = maps.idData(d);
        for (uint32_t p = 0; p < ProjectionMaps::PixelCount; ++p)
            if (owners[p] != UINT32_MAX) eligible[owners[p]] = 1;
    }

    // Shard merge order depends on clip-worker partitioning. Give the area
    // CDF a stable source-triangle order before drawing deterministic samples.
    std::vector<uint32_t> candidates;
    for (uint32_t p = 0; p < polygons.size(); ++p)
        if (eligible[p]) candidates.push_back(p);
    std::stable_sort(candidates.begin(), candidates.end(), [&](uint32_t a, uint32_t b) {
        const auto& x = polygons[a].triRef;
        const auto& y = polygons[b].triRef;
        return std::tie(x.instanceIdx, x.meshID, x.triangleID, x.materialID) <
               std::tie(y.instanceIdx, y.meshID, y.triangleID, y.materialID);
    });

    const float3 center = float3(range.cellInt) + float3(0.5f);
    std::vector<AreaEntry> distribution;
    for (uint32_t k = 0; k < lobeCount; ++k) {
        const uint32_t original = sourceLobes[k];
        const uint32_t bits = VisibilityMask::lobeBits(k, lobeCount);
        const uint32_t begin = VisibilityMask::SampleCount * k / lobeCount;
        const uint32_t desired = countbits(bits);
        uint32_t front = 0;
        for (uint32_t d = 0; d < VisibilityMask::DirectionCount; ++d)
            if (dot(absdf.lobes[original].normal, maps.directions[d]) > 1e-5f)
                front |= 1u << d;
        stats.frontDirections += countbits(front);

        distribution.clear();
        double totalArea = 0;
        for (uint32_t p : candidates) {
            if ((lobePairs[p] & 15u) != original && ((lobePairs[p] >> 4) & 15u) != original)
                continue;
            double area = 0;
            for (uint32_t v = 1; v + 1 < polygons[p].count; ++v) area += triangleArea(polygons[p], v);
            if (!(area > 0) || !std::isfinite(area)) continue;
            totalArea += area;
            distribution.push_back({p, totalArea});
        }

        Random random(seed ^ (0x85ebca6bu * (original + 1)));
        uint32_t accepted = 0;
        for (uint32_t attempt = 0; totalArea > 0 && front && accepted < desired &&
             attempt < desired * AttemptsPerSample; ++attempt) {
            ++stats.candidates;
            const double target = random.next() * totalArea;
            auto entry = std::lower_bound(distribution.begin(), distribution.end(), target,
                [](const AreaEntry& e, double value) { return e.cumulativeArea < value; });
            const Polygon& polygon = polygons[entry->polygon];
            double localArea = target - (entry == distribution.begin() ? 0.0 : (entry - 1)->cumulativeArea);
            uint32_t fan = 1;
            while (fan + 2 < polygon.count) {
                const double area = triangleArea(polygon, fan);
                if (localArea < area) break;
                localArea -= area;
                ++fan;
            }
            const float root = float(std::sqrt(random.next()));
            const float v = float(random.next());
            const float3 point = (1.0f - root) * polygon.vertices[0] +
                root * (1.0f - v) * polygon.vertices[fan] + root * v * polygon.vertices[fan + 1];
            uint32_t signature = 0;
            for (uint32_t d = 0; d < VisibilityMask::DirectionCount; ++d)
                if (visible(maps, d, polygons, center, point, entry->polygon)) signature |= 1u << d;
            if (!(signature & front)) continue;
            const uint32_t sampleBit = 1u << (begin + accepted++);
            for (uint32_t d = 0; d < VisibilityMask::DirectionCount; ++d)
                if (signature & (1u << d)) result.masks[d] |= sampleBit;
        }
        if (accepted == desired) {
            result.metadata |= 1u << (8 + k);
            stats.acceptedSamples += accepted;
        } else {
            ++stats.invalidLobes;
            for (auto& mask : result.masks) mask &= ~bits;
        }
        for (uint32_t d = 0; d < VisibilityMask::DirectionCount; ++d)
            if ((front & (1u << d)) && !(result.masks[d] & bits)) ++stats.unsupportedFrontDirections;
    }
    return result;
}

} // namespace VisibilityMaskBake
