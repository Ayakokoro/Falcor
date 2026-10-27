#pragma once
#include "VisibleABSDF.h"
#include "VisibilityMaskBake.h"
#include "VisibleMaterialFile.h"
#include "BakeTiming.h"

// Geometry is normalized to the selected node, just like the existing area baker.
inline VisibleMaterialFile::Data bakeVisibleMaterial(
    const std::vector<VisibleABSDFInput>& materialSamples,
    const std::vector<Polygon>& nodePolys, const PolygonRange& range,
    uint32_t level, VisibilityMaskBake::ProjectionMaps& scratch,
    VisibilityMask::Data* visibilityMask = nullptr, BakeTiming::Worker* timings = nullptr) {
    BakeTiming::Scope timer(timings, BakeTiming::Task::VisibleSetup);
    VisibleABSDF material;
    const auto& cellInt = range.cellInt;
    std::vector<uint32_t> lobePairs;
    lobePairs.reserve(nodePolys.size());

    float3 mainNormalSum[VISIBLE_LOBE_COUNT] = {};
    float mainNormalArea[VISIBLE_LOBE_COUNT] = {};

    for (const auto& input : materialSamples) {
        uint32_t k0 = VisibleNormalIndex8(input.normal);
        uint32_t k1 = VisibleNormalIndex8(-input.normal);
        lobePairs.push_back(k0 | (k1 << 4));

        float area = input.area;
        material.area += area;
        mainNormalSum[k0] += area * input.normal;
        mainNormalArea[k0] += area;
        mainNormalSum[k1] += area * (-input.normal);
        mainNormalArea[k1] += area;
    }

    float3 mainDirections[VISIBLE_LOBE_COUNT] = {};
    for (uint32_t k = 0; k < VISIBLE_LOBE_COUNT; ++k)
        if (mainNormalArea[k] > 0.0f)
            mainDirections[k] = safeNormalize(mainNormalSum[k]);

    // Fit one visible-area amplitude per lobe from the depth/owner maps that
    // the material bake already generates. No extra persistent voxel data.
    float fitNumerator[VISIBLE_LOBE_COUNT] = {};
    float fitDenominator[VISIBLE_LOBE_COUNT] = {};
    const float pixelArea = PROJ_PIXEL_SIZE * PROJ_PIXEL_SIZE;
    timer.stop();
    for (uint32_t k = 0; k < VISIBLE_LOBE_COUNT; ++k) {
        if (mainNormalArea[k] <= 0.0f)
            continue;

        timer.next(timings, BakeTiming::Task::VisibleRaster);
        float3 direction = mainDirections[k];
        float* depth = scratch.depthData(k);
        uint32_t* ids = scratch.idData(k);
        std::fill(depth, depth + VisibilityMaskBake::ProjectionMaps::PixelCount, std::numeric_limits<float>::max());
        std::fill(ids, ids + VisibilityMaskBake::ProjectionMaps::PixelCount, std::numeric_limits<uint32_t>::max());
        range.rasterizeDepth(nodePolys, direction, depth, ids);

        scratch.directions[k] = direction;
        scratch.bases[k] = orthonormal_basis(direction);

        timer.next(timings, BakeTiming::Task::VisibleAccumulate);
        float visibleByLobe[VISIBLE_LOBE_COUNT] = {};
        for (uint32_t pixel = 0; pixel < PROJ_RES * PROJ_RES; ++pixel) {
            uint32_t polygonIndex = ids[pixel];
            if (polygonIndex == std::numeric_limits<uint32_t>::max())
                continue;

            uint32_t packed = lobePairs[polygonIndex];
            uint32_t k0 = packed & 0xFu;
            uint32_t k1 = (packed >> 4) & 0xFu;
            uint32_t facingLobe = dot(materialSamples[polygonIndex].normal, direction) >= 0.0f ? k0 : k1;
            visibleByLobe[facingLobe] += pixelArea;
            float sign = 0.0f;
            if (k == k0) sign = 1.0f;
            else if (k == k1) sign = -1.0f;
            else continue;

            VisibleABSDFInput input = materialSamples[polygonIndex];
            if (sign < 0.0f)
                input.normal = -input.normal;
            input.projArea = pixelArea;
            material.accumulate(input);
        }
        for (uint32_t m = 0; m < VISIBLE_LOBE_COUNT; ++m) {
            if (mainNormalArea[m] <= 0.0f)
                continue;
            float cosine = std::max(0.0f, dot(mainDirections[m], direction));
            fitNumerator[m] += cosine * visibleByLobe[m];
            fitDenominator[m] += cosine * cosine;
        }
        material.lobes[k].normalizeSelf();
        timer.stop();
    }

    timer.next(timings, BakeTiming::Task::VisibleFinalize);
    for (uint32_t k = 0; k < VISIBLE_LOBE_COUNT; ++k) {
        if (!material.lobes[k].isValid())
            continue;
        material.lobes[k].weight = fitDenominator[k] > 1e-8f
            ? std::max(0.0f, fitNumerator[k] / fitDenominator[k]) : 0.0f;
    }
    material.normalizeWeightsOnly();
    timer.stop();

    if (visibilityMask) {
        // Finish material fitting before reusing its maps. Every octree level
        // samples its own clipped polygons; child bit identities cannot merge.
        const uint32_t seed = uint32_t(cellInt.x) * 0x8da6b343u ^ uint32_t(cellInt.y) * 0xd8163841u ^
                              uint32_t(cellInt.z) * 0xcb1ab31fu ^ level * 0x9e3779b9u;
        VisibilityMaskBake::Stats stats;
        *visibilityMask = VisibilityMaskBake::bake(material, range, nodePolys, lobePairs, scratch, seed, stats, timings);
        if (timings) {
            timings->maskCandidates += stats.candidates;
            timings->maskRetainedSamples += stats.acceptedSamples;
            timings->maskActiveLobes += stats.activeLobes;
            timings->maskInvalidLobes += stats.invalidLobes;
        }
    }
    timer.next(timings, BakeTiming::Task::VisibleFinalize);
    VisibleMaterialFile::Data result;
    result.baked = 1;
    result.area = material.area;
    for (uint32_t k = 0; k < VISIBLE_LOBE_COUNT; ++k) {
        const auto& in = material.lobes[k];
        auto& out = result.lobes[k];
        out.weight = in.weight;
        out.rough = in.rough;
        for (uint32_t j = 0; j < 3; ++j) {
            out.normal[j] = in.normal[j];
            out.diffuse[j] = in.diffuse[j];
            out.specular[j] = in.specular[j];
        }
    }
    VisibleMaterialFile::validate(result);
    return result;
}
