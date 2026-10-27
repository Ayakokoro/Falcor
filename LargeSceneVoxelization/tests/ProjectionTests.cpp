#include "Estimate.h"
#include "GridData.h"
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <limits>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Keep the pre-optimization sampling arithmetic as an independent reference.
SamplingRay originalRay(const float2& u, const float3& direction,
                        const Basis2& basis, const float3& center) {
    float radius = std::sqrt(u.x);
    float phi = 2.0f * M_PI_F * u.y;
    float2 p(radius * std::cos(phi), radius * std::sin(phi));
    const float r = 0.8660254f;
    float uu = p.x * r, ww = p.y * r;
    float h = std::sqrt(std::max(0.0f, r * r - uu * uu - ww * ww));
    float3 point = h * direction + uu * basis.u + ww * basis.w;
    return {point + center, -direction};
}

template<bool Cached>
float originalProjection(const Ellipsoid& ellipsoid, const PolygonRange& range,
                         const float3& direction, uint32_t frequency) {
    frequency = std::max(1u, frequency);
    const float3 cellMin = float3(range.cellInt);
    const float3 center = cellMin + float3(0.5f);
    const auto basis = orthonormal_basis(safeNormalize(direction));
    const auto& samples = voxelRaySamples(frequency);
    uint32_t hits = 0;
    for (uint32_t i = 0; i < frequency; ++i) {
        SamplingRay ray;
        if constexpr (Cached) ray = rayToVoxel(samples[i], direction, basis, center);
        else ray = originalRay(Hammersley2D(i, frequency), direction, basis, center);
        const float3 to = ray.origin + 2.0f * ray.direction;
        const auto interval = clipAABB(cellMin, cellMin + float3(1), ray.origin, to);
        if (interval.x < 0) continue;
        const float3 delta = to - ray.origin;
        const auto fromLocal = ray.origin + interval.x * delta - cellMin;
        const auto toLocal = ray.origin + interval.y * delta - cellMin;
        if (ellipsoid.clip(fromLocal, toLocal).x >= 0) ++hits;
    }
    return PROJECT_CIRCLE_AREA * float(hits) / float(frequency);
}

struct Random {
    std::mt19937 engine{0x92a4b381u};
    float value(float low, float high) { return std::uniform_real_distribution<float>(low, high)(engine); }
    float3 point(float low, float high) { return float3(value(low, high), value(low, high), value(low, high)); }
    Ellipsoid ellipsoid() {
        Ellipsoid e;
        e.center = point(0.1f, 0.9f);
        const auto normal = safeNormalize(point(-1, 1));
        const auto basis = orthonormal_basis(normal);
        const float3x3 rotation(basis.u, basis.w, normal);
        const float3 axes(std::exp2(value(-9, 0)), std::exp2(value(-9, 0)), std::exp2(value(-9, 0)));
        e.B = rotation * diag3(1.0f / (axes * axes)) * transpose(rotation);
        e.B = 0.5f * (e.B + transpose(e.B));
        return e;
    }
};

void testSamples() {
    const auto& directions = octahedralProjectionDirections();
    for (uint32_t index = 0; index < OCTA_COUNT; ++index)
        for (uint32_t sub = 0; sub < OCTA_SAMPLES_PER_TEXEL; ++sub) {
            const float2 u = Hammersley2D(sub, OCTA_SAMPLES_PER_TEXEL);
            const float3 reference = octahedralDecode(float2((float(index % OCTA_RES) + u.x) / float(OCTA_RES),
                                                             (float(index / OCTA_RES) + u.y) / float(OCTA_RES)));
            const auto& actual = directions[index * OCTA_SAMPLES_PER_TEXEL + sub];
            require(actual.direction == reference, "Cached octahedral direction changed.");
            const auto basis = orthonormal_basis(safeNormalize(reference));
            for (uint32_t axis = 0; axis < 3; ++axis) {
                require(std::abs(actual.basis.u[axis] - basis.u[axis]) <= 4 * std::numeric_limits<float>::epsilon(),
                        "Cached basis changed beyond float rounding.");
                require(std::abs(actual.basis.w[axis] - basis.w[axis]) <= 4 * std::numeric_limits<float>::epsilon(),
                        "Cached basis changed beyond float rounding.");
            }
        }
    for (uint32_t count : {0u, 1u, 8u, 31u, 256u, 1024u}) {
        const auto& samples = voxelRaySamples(count);
        require(samples.size() == std::max(1u, count), "Cache count/clamping failed.");
        for (const auto& dir : directions)
            for (int cell : {0, 31, 511, 65535})
                for (uint32_t i = 0; i < samples.size(); ++i) {
                    const float3 center = float3(float(cell)) + float3(0.5f);
                    const auto expected = originalRay(Hammersley2D(i, uint32_t(samples.size())), dir.direction, dir.basis, center);
                    const auto actual = rayToVoxel(samples[i], dir.direction, dir.basis, center);
                    // /fp:fast can fuse/reassociate across the uncached sampler.
                    // Storing a table introduces float rounding, but must not
                    // move a ray beyond a few ULPs of the old position.
                    for (uint32_t axis = 0; axis < 3; ++axis)
                        require(std::abs(actual.origin[axis] - expected.origin[axis]) <=
                            4 * std::numeric_limits<float>::epsilon() * std::max(1.0f, std::abs(expected.origin[axis])),
                            "Cached ray moved beyond float rounding tolerance.");
                    require(actual.direction == expected.direction, "Cached ray direction changed.");
                }
    }
    const auto* shared = voxelRaySamples(1024).data();
    std::array<const VoxelRaySample*, 8> addresses{};
    std::vector<std::thread> threads;
    for (size_t i = 0; i < addresses.size(); ++i)
        threads.emplace_back([&, i] { addresses[i] = voxelRaySamples(1024).data(); });
    for (auto& thread : threads) thread.join();
    for (auto* address : addresses) require(address == shared, "Workers did not share the sample table.");
    require(voxelRaySamples(17).size() == 17 && voxelRaySamples(1024).size() == 1024,
            "Switching sample counts reused a stale table.");
}

void testIntersection() {
    Ellipsoid sphere;
    sphere.center = float3(0);
    sphere.B = identity3x3();
    require(sphere.intersectsSegment(float3(-2, 0, 0), float3(2, 0, 0)), "Through segment missed.");
    require(sphere.intersectsSegment(float3(-2, 1, 0), float3(2, 1, 0)), "Tangent missed.");
    require(!sphere.intersectsSegment(float3(-2, 1.1f, 0), float3(2, 1.1f, 0)), "Outside segment hit.");
    require(sphere.intersectsSegment(float3(0), float3(0)), "Inside point missed.");
    require(sphere.intersectsSegment(float3(1, 0, 0), float3(1, 0, 0)), "Boundary point missed.");
    require(!sphere.intersectsSegment(float3(2, 0, 0), float3(2, 0, 0)), "Outside point hit.");
    // Probe both sides of tangencies and segment endpoints at float-ULP spacing.
    for (float boundary : {1.0f, -1.0f}) {
        float coordinate = boundary;
        for (uint32_t i = 0; i < 40; ++i) {
            for (float outside : {-3.0f, 3.0f}) {
                const float3 a(outside, 0, 0), b(coordinate, 0, 0);
                require(sphere.intersectsSegment(a, b) == (sphere.clip(a, b).x >= 0), "Endpoint classification changed.");
                require(sphere.intersectsSegment(b, a) == (sphere.clip(b, a).x >= 0), "Reversed endpoint changed.");
            }
            const float3 a(-2, coordinate, 0), b(2, coordinate, 0);
            require(sphere.intersectsSegment(a, b) == (sphere.clip(a, b).x >= 0), "Near-tangent classification changed.");
            coordinate = std::nextafter(coordinate, i < 20 ? 0.0f : 2.0f * boundary);
        }
    }
    Ellipsoid degenerate = sphere;
    degenerate.B = diag3(float3(1, 1, 0));
    require(degenerate.intersectsSegment(float3(0, 0, -2), float3(0, 0, 2)), "Degenerate inside segment missed.");
    require(!degenerate.intersectsSegment(float3(2, 0, -2), float3(2, 0, 2)), "Degenerate outside segment hit.");
    auto invalid = sphere;
    invalid.center.x = std::bit_cast<float>(0x7fc00000u);
    require(!invalid.intersectsSegment(float3(0), float3(1)), "Nonfinite ellipsoid hit.");

    Random random;
    for (uint32_t i = 0; i < 2000; ++i) {
        const auto e = random.ellipsoid();
        for (uint32_t j = 0; j < 256; ++j) {
            const auto from = random.point(-0.5f, 1.5f), to = random.point(-0.5f, 1.5f);
            // An independent double-precision closest-point oracle checks the
            // geometric query. Thin rotated ellipsoids amplify float cancellation;
            // classify only outside a conservative float roundoff interval.
            const glm::dmat3 matrix(e.B);
            const glm::dvec3 f = glm::dvec3(from) - glm::dvec3(e.center);
            const glm::dvec3 v = glm::dvec3(to) - glm::dvec3(from);
            const double a = glm::dot(v, matrix * v);
            const double h = glm::dot(v, matrix * f);
            const double t = std::clamp(-h / a, 0.0, 1.0);
            const glm::dvec3 p = f + t * v;
            const double minimum = glm::dot(p, matrix * p) - 1.0;
            double scale = 1;
            for (uint32_t column = 0; column < 3; ++column)
                for (uint32_t row = 0; row < 3; ++row)
                    scale += std::abs(f[row] * matrix[column][row] * f[column]) +
                        2 * t * std::abs(v[row] * matrix[column][row] * f[column]) +
                        t * t * std::abs(v[row] * matrix[column][row] * v[column]);
            const double bound = 32 * std::numeric_limits<float>::epsilon() * scale;
            require(e.intersectsSegment(from, to) == (minimum <= 0) || std::abs(minimum) <= bound,
                    "Random ellipsoid query disagrees with double oracle outside roundoff interval.");
        }
    }
}

struct Case { Ellipsoid ellipsoid; PolygonRange range; };

std::vector<Case> syntheticCases() {
    std::vector<Case> cases;
    Random random;
    for (uint32_t i = 0; i < 64; ++i) {
        Case item;
        item.ellipsoid = random.ellipsoid();
        item.range.init(int3(int(i * 8), int(i * 3), int(i * 5)));
        cases.push_back(item);
    }
    return cases;
}

// Optional real-scene kernel benchmark reads a previous bake; never modifies it.
std::vector<Case> readCases(const char* path) {
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "Cannot open benchmark voxel file.");
    GridData grid;
    uint32_t depth;
    input.read(reinterpret_cast<char*>(&grid), sizeof(grid));
    input.read(reinterpret_cast<char*>(&depth), sizeof(depth));
    require(input && depth <= 16 && grid.solidVoxelCount < 100000000, "Invalid benchmark voxel header.");
    std::vector<uint32_t> counts(depth + 1);
    input.read(reinterpret_cast<char*>(counts.data()), counts.size() * sizeof(uint32_t));
    std::vector<OctreeNode> nodes(grid.solidVoxelCount);
    input.read(reinterpret_cast<char*>(nodes.data()), nodes.size() * sizeof(OctreeNode));
    require(bool(input), "Truncated benchmark octree.");
    const auto voxelStart = input.tellg();
    std::vector<int3> cells(nodes.size(), int3(0));
    for (size_t i = 0; i < nodes.size(); ++i) {
        uint32_t child = nodes[i].childBase;
        for (uint32_t c = 0; c < 8; ++c) if (nodes[i].childMask & (1u << c)) {
            require(child < nodes.size(), "Invalid child index.");
            cells[child++] = cells[i] * 2 + int3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
        }
    }
    const size_t leafStart = nodes.size() - counts.back();
    std::vector<Case> result;
    for (size_t i = leafStart; i < nodes.size(); i += std::max(size_t(1), size_t(counts.back()) / 1024)) {
        VoxelData data;
        input.seekg(voxelStart + std::streamoff(uint64_t(nodes[i].dataIndex) * sizeof(VoxelData)));
        input.read(reinterpret_cast<char*>(&data), sizeof(data));
        require(bool(input), "Truncated benchmark voxel data.");
        if (!data.isSolid()) continue;
        Case item;
        item.ellipsoid = data.ellipsoid;
        item.range.init(cells[i]);
        result.push_back(item);
    }
    require(!result.empty(), "No solid benchmark leaves.");
    return result;
}

void compareProjections(const std::vector<Case>& cases, uint32_t frequency, bool benchmark) {
    const auto& directions = octahedralProjectionDirections();
    std::array<std::vector<float>, 3> areas;
    std::array<double, 3> times{};
    // Initialize shared caches before measuring steady-state kernels.
    voxelRaySamples(frequency);
    for (uint32_t mode = 0; mode < 3; ++mode) {
        areas[mode].reserve(cases.size() * directions.size());
        const auto start = BakeTiming::Clock::now();
        for (const auto& item : cases)
            for (const auto& dir : directions) {
                float area;
                if (mode == 0) area = originalProjection<false>(item.ellipsoid, item.range, dir.direction, frequency);
                else if (mode == 1) area = originalProjection<true>(item.ellipsoid, item.range, dir.direction, frequency);
                else area = primitiveProjectionArea(item.ellipsoid, item.range, dir.direction, dir.basis, voxelRaySamples(frequency));
                areas[mode].push_back(area);
            }
        times[mode] = BakeTiming::secondsSince(start);
    }
    size_t changedCache = 0, changedPredicate = 0;
    float maxCacheError = 0, maxPredicateError = 0;
    for (size_t i = 0; i < areas[0].size(); ++i) {
        changedCache += areas[0][i] != areas[1][i];
        changedPredicate += areas[1][i] != areas[2][i];
        maxCacheError = std::max(maxCacheError, std::abs(areas[0][i] - areas[1][i]));
        maxPredicateError = std::max(maxPredicateError, std::abs(areas[1][i] - areas[2][i]));
    }
    std::cout << "Projection comparison: nodes=" << cases.size() << ", samples=" << frequency
              << ", cache changes=" << changedCache << ", predicate changes=" << changedPredicate
              << ", max cache error=" << maxCacheError << ", max predicate error=" << maxPredicateError << '\n';
    if (benchmark) std::cout << "Kernel seconds: original=" << times[0] << ", cache=" << times[1]
                            << ", cache+predicate=" << times[2] << ", speedup=" << times[0] / times[2] << '\n';
    // Same sample count/order and estimator. Fast-math reassociation may move
    // grazing samples across the surface; bound the area delta in sample units.
    const float tolerance = PROJECT_CIRCLE_AREA * float(std::max(1u, frequency / 256)) / float(frequency) + 1e-6f;
    require(maxCacheError <= tolerance && maxPredicateError <= tolerance,
            "Projected area difference exceeds the sample-quantization regression bound.");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--benchmark") {
            compareProjections(readCases(argv[2]), 1024, true);
        } else {
            testIntersection();
            testSamples();
            const auto cases = syntheticCases();
            for (uint32_t count : {1u, 31u, 1024u}) compareProjections(cases, count, false);
            std::cout << "Projection tests passed: shared samples, boundaries, degenerate segments and reference counts.\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
