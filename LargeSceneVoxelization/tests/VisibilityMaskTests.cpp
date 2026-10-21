#include "VisibilityMaskBake.h"
#include "VisibilityMaskFile.h"
#include <iostream>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<typename F> void requireThrows(F&& f, const char* message) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}

Polygon quad(float x0, float y0, float x1, float y1, float z, float slope = 0) {
    Polygon p{};
    p.add(float3(x0, y0, z + slope * (x0 - 0.5f)));
    p.add(float3(x1, y0, z + slope * (x1 - 0.5f)));
    p.add(float3(x1, y1, z + slope * (x1 - 0.5f)));
    p.add(float3(x0, y1, z + slope * (x0 - 0.5f)));
    p.normal = safeNormalize(float3(-slope, 0, 1));
    return p;
}

ABSDF oneLobe(const float3& normal = float3(0, 0, 1)) {
    ABSDF result;
    result.init();
    auto& lobe = result.lobes[NormalIndex8(normal)];
    lobe.normal = normal;
    lobe.weight = 1;
    return result;
}

VisibilityMask::Data bake(const std::vector<Polygon>& polygons, const ABSDF& material,
                          VisibilityMaskBake::Stats& stats, uint32_t seed = 1234) {
    PolygonRange range;
    range.init(int3(0));
    range.count = uint32_t(polygons.size());
    std::vector<uint32_t> pairs;
    for (const auto& polygon : polygons)
        pairs.push_back(NormalIndex8(polygon.normal) | (NormalIndex8(-polygon.normal) << 4));
    VisibilityMaskBake::ProjectionMaps scratch;
    return VisibilityMaskBake::bake(material, range, polygons, pairs, scratch, seed, stats);
}

void testQueries() {
    for (uint32_t count = 1; count <= 8; ++count) {
        uint32_t all = 0;
        for (uint32_t k = 0; k < count; ++k) {
            const uint32_t bits = VisibilityMask::lobeBits(k, count);
            require(bits != 0 && !(all & bits), "Lobe allocations overlap or are empty.");
            all |= bits;
        }
        require(all == UINT32_MAX, "Lobe allocations do not cover 32 samples.");
    }
    VisibilityMask::Data data;
    data.metadata = (1u << 7) | (1u << 8); // Source lobe 7 -> compact lobe 0.
    data.masks[0] = 0x0000ffffu;
    data.masks[1] = 0xffff0000u;
    data.masks[2] = 0x00ffff00u;
    require(VisibilityMask::query(data, 0, 0, 0).value == 1, "Diagonal visibility must be one.");
    require(VisibilityMask::query(data, 0, 1, 0).value == 0, "Disjoint visible sets must give zero.");
    require(VisibilityMask::query(data, 0, 2, 0).value == 0.5f, "Half overlap must give one half.");
    require(!VisibilityMask::query(data, 0, 0, 3).supported, "Empty view must be unsupported.");
    data.metadata &= 0xffu;
    require(!VisibilityMask::query(data, 0, 0, 0).supported, "Invalid lobe must not be queried.");
    require(!VisibilityMask::query(data, 32, 0, 0).supported, "Out-of-range lobe must be rejected.");
    for (uint32_t d = 0; d < VisibilityMask::DirectionCount; ++d)
        require(VisibilityMask::directionIndex(VisibilityMask::direction(d)) == d, "Octahedral texel round trip failed.");
    require(VisibilityMask::directionIndex(float3(0)) == VisibilityMask::DirectionCount, "Zero direction accepted.");
    require(VisibilityMask::directionIndex(float3(0, 0, -1)) == 15, "Lower octahedral fold mismatch.");
}

void testLayers() {
    // The small rear sheets remain fully hidden by the large front sheet in
    // all four positive-Z bake directions. Extra area must not attenuate it.
    for (uint32_t layers : {1u, 8u, 32u, 256u}) {
        std::vector<Polygon> polygons{quad(0.05f, 0.05f, 0.95f, 0.95f, 0.75f)};
        for (uint32_t i = 1; i < layers; ++i)
            polygons.push_back(quad(0.4f, 0.4f, 0.6f, 0.6f, 0.70f - float(i) * 0.0001f));
        VisibilityMaskBake::Stats stats;
        const auto data = bake(polygons, oneLobe(), stats);
        require(data.validLobes() == 1 && stats.acceptedSamples == 32, "Layer bake did not fill its samples.");
        for (uint32_t i = 0; i < VisibilityMask::DirectionCount; ++i) {
            if (VisibilityMask::direction(i).z <= 0) continue;
            for (uint32_t o = 0; o < VisibilityMask::DirectionCount; ++o) {
                if (VisibilityMask::direction(o).z <= 0) continue;
                auto q = VisibilityMask::query(data, 0, i, o);
                require(q.supported && q.value == 1.0f, "Hidden rear layers darken the visible front sheet.");
            }
        }
    }
}

void testThinLayersAndEdges() {
    for (float slope : {0.0f, 0.2f}) {
        std::vector<Polygon> polygons{quad(0.1f, 0.1f, 0.9f, 0.9f, 0.5f, slope),
                                      quad(0.1f, 0.1f, 0.9f, 0.9f, 0.4998f, slope)};
        PolygonRange range;
        range.init(int3(0));
        range.count = 2;
        VisibilityMaskBake::ProjectionMaps maps;
        for (uint32_t d : {5u, 6u, 9u, 10u}) {
            maps.rasterize(d, range, polygons, VisibilityMask::direction(d));
            require(!VisibilityMaskBake::visible(maps, d, polygons, float3(0.5f), float3(0.5f, 0.5f, 0.4998f), 1),
                    "Depth reprojection merged two thin layers.");
        }
        polygons.resize(1);
        range.count = 1;
        for (uint32_t d : {5u, 6u, 9u, 10u}) {
            maps.rasterize(d, range, polygons, VisibilityMask::direction(d));
            for (uint32_t v = 0; v < polygons[0].count; ++v) {
                require(VisibilityMaskBake::visible(maps, d, polygons, float3(0.5f), polygons[0].vertices[v], 0),
                        "An unoccluded edge became invisible due to an empty depth texel.");
            }
        }
    }
}

void testOppositeShadows() {
    std::vector<Polygon> polygons{quad(0.1f, 0.45f, 0.9f, 0.55f, 0.1f)};
    Polygon wall{};
    wall.normal = float3(1, 0, 0);
    wall.add(float3(0.5f, 0, 0.1f));
    wall.add(float3(0.5f, 1, 0.1f));
    wall.add(float3(0.5f, 1, 1));
    wall.add(float3(0.5f, 0, 1));
    polygons.push_back(wall);
    VisibilityMaskBake::Stats stats;
    const auto data = bake(polygons, oneLobe(), stats);
    const auto q = VisibilityMask::query(data, 0, 5, 6);
    require(q.supported && q.value < 0.15f, "Opposite wall shadows did not produce near-disjoint visible sets.");
    require(VisibilityMask::query(data, 0, 5, 5).value == 1, "Occluded geometry broke diagonal visibility.");
    VisibilityMaskBake::Stats repeatedStats;
    const auto repeated = bake(polygons, oneLobe(), repeatedStats);
    require(data.masks == repeated.masks && data.metadata == repeated.metadata, "Repeated bake is not deterministic.");
}

void testUnsupportedLobe() {
    VisibilityMaskBake::Stats stats;
    const auto data = bake({quad(0.54321f, 0.51234f, 0.543211f, 0.512341f, 0.5f)}, oneLobe(), stats);
    require(data.lobeCount() == 1 && data.validLobes() == 0 && stats.invalidLobes == 1,
            "Unresolved subpixel geometry must invalidate the lobe.");
    for (uint32_t mask : data.masks) require(mask == 0, "Invalid lobe has sample bits.");
}

void testAreaMeasure() {
    // Two coplanar patches with areas 0.02 and 0.03. Both are visible from
    // direction 10. A finite wall shadows only the smaller patch from 6.
    // Area sampling predicts Q=0.6; uniform polygon sampling predicts 0.5.
    std::vector<Polygon> polygons{quad(0.1f, 0.45f, 0.3f, 0.55f, 0.1f),
                                  quad(0.6f, 0.45f, 0.9f, 0.55f, 0.1f)};
    Polygon wall{};
    wall.normal = float3(1, 0, 0);
    wall.add(float3(0.5f, 0, 0.1f));
    wall.add(float3(0.5f, 0.4f, 0.1f));
    wall.add(float3(0.5f, 0.4f, 1));
    wall.add(float3(0.5f, 0, 1));
    polygons.push_back(wall);
    double mean = 0;
    for (uint32_t seed = 0; seed < 128; ++seed) {
        VisibilityMaskBake::Stats stats;
        const auto data = bake(polygons, oneLobe(), stats, seed);
        const auto q = VisibilityMask::query(data, 0, 6, 10);
        require(q.supported && q.outgoingSamples == 32, "Area-measure fixture lost an outgoing surface sample.");
        mean += q.value / 128.0;
    }
    require(std::abs(mean - 0.6) < 0.035, "Visibility samples do not follow surface-area measure.");
}

void testFiles(const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);
    const auto path = (dir / "roundtrip.vbm").string();
    std::vector<VisibilityMask::Data> records(2);
    records[0].metadata = 0x0104;
    records[0].masks.fill(0xffffffffu);
    VisibilityMaskFile::Fingerprint fingerprint;
    const std::string source = "paired voxel bytes";
    fingerprint.append(source.data(), source.size());
    VisibilityMaskFile::write(path, records, fingerprint);
    const auto loaded = VisibilityMaskFile::read(path, 2, fingerprint);
    require(loaded[0].masks == records[0].masks && loaded[0].metadata == records[0].metadata,
            "Mask serialization round trip failed.");
    auto stale = fingerprint;
    ++stale.hash;
    requireThrows([&] { VisibilityMaskFile::read(path, 2, stale); }, "Stale companion accepted.");
    requireThrows([&] { VisibilityMaskFile::read(path, 3, fingerprint); }, "Wrong node count accepted.");
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    requireThrows([&] { VisibilityMaskFile::read(path, 2, fingerprint); }, "Truncated companion accepted.");
    records[0].metadata = 0x0004; // Invalidate lobe but leave its bits set.
    VisibilityMaskFile::write(path, records, fingerprint);
    requireThrows([&] { VisibilityMaskFile::read(path, 2, fingerprint); }, "Invalid payload metadata accepted.");
    std::filesystem::remove(path);
}
} // namespace

int main(int argc, char** argv) {
    try {
        testQueries();
        testLayers();
        testThinLayersAndEdges();
        testOppositeShadows();
        testUnsupportedLobe();
        testAreaMeasure();
        testFiles(argc > 1 ? argv[1] : "visibility-mask-test-output");
        std::cout << "Visibility mask tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Visibility mask test failed: " << error.what() << '\n';
        return 1;
    }
}
