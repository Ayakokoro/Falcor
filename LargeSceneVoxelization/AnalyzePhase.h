#pragma once
#include "Types.h"
#include "Triangle.h"
#include "GridData.h"
#include "SceneLoader.h"
#include "VoxelData.h"
#include "ABSDF.h"
#include "Ellipsoid.h"
#include "Estimate.h"
#include "TextureSampler.h"
#include "PolygonSerializer.h"
#include "PolygonGenerator.h"
#include "OctreeBuilder.h"
#include "MergePhase.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <iostream>
#include <thread>
#include <atomic>
#include <algorithm>
#include <array>
#include <limits>

namespace AnalyzePhase {

// Context carrying all scene data needed to analyze one clipped node.
struct AnalyzeContext {
    const InstancedScene& scene;
    const GridData& grid;
    uint32_t maxDepth;
    uint32_t sampleFrequency;
    const std::vector<Texture2D>& baseColorTextures;
    const std::vector<Texture2D>& specularTextures;
    const std::vector<Texture2D>& metallicTextures;
    const std::vector<Texture2D>& normalMapTextures;
};

// The visible-material pass already renders one 64x64 depth/owner map for
// every valid lobe direction. Add four same-hemisphere perturbations around
// each lobe direction and use only direction pairs from the same lobe group to
// fit one same-side visibility-correlation scalar per voxel. All maps remain
// transient, and a scratch instance is reused by each worker.
constexpr uint32_t CORRELATION_SAMPLE_COUNT = 256;
constexpr float CORRELATION_ANGULAR_POWER = 4.0f;
constexpr uint32_t CORRELATION_PERTURBATION_COUNT = 4;
constexpr uint32_t CORRELATION_DIRECTIONS_PER_LOBE = 1 + CORRELATION_PERTURBATION_COUNT;
constexpr uint32_t CORRELATION_PROJECTION_COUNT =
    LOBE_COUNT * CORRELATION_DIRECTIONS_PER_LOBE;
constexpr float CORRELATION_PERTURBATION_COS = 0.9063077870f; // cos(25 degrees)
constexpr float CORRELATION_PERTURBATION_SIN = 0.4226182617f; // sin(25 degrees)

static constexpr uint32_t correlationProjectionIndex(uint32_t lobe, uint32_t localDirection) {
    return localDirection == 0
        ? lobe
        : LOBE_COUNT + lobe * CORRELATION_PERTURBATION_COUNT + (localDirection - 1);
}

struct AnalyzeScratch {
    static constexpr uint32_t PixelCount = PROJ_RES * PROJ_RES;

    std::vector<float> depths;
    std::vector<uint32_t> ids;
    std::array<float3, CORRELATION_PROJECTION_COUNT> directions;
    std::array<Basis2, CORRELATION_PROJECTION_COUNT> bases;
    std::array<uint32_t, CORRELATION_PROJECTION_COUNT> occupiedCounts{};
    std::array<uint32_t, CORRELATION_PROJECTION_COUNT> sampleCounts{};
    std::array<bool, CORRELATION_PROJECTION_COUNT> valid{};
    std::array<std::array<uint32_t, CORRELATION_SAMPLE_COUNT>, CORRELATION_PROJECTION_COUNT> samplePixels{};

    AnalyzeScratch()
        : depths(CORRELATION_PROJECTION_COUNT * PixelCount),
          ids(CORRELATION_PROJECTION_COUNT * PixelCount) {
        reset();
    }

    void reset() {
        valid.fill(false);
        occupiedCounts.fill(0);
        sampleCounts.fill(0);
    }

    float* depthData(uint32_t k) { return depths.data() + k * PixelCount; }
    uint32_t* idData(uint32_t k) { return ids.data() + k * PixelCount; }
    const float* depthData(uint32_t k) const { return depths.data() + k * PixelCount; }
    const uint32_t* idData(uint32_t k) const { return ids.data() + k * PixelCount; }

    // Select pixels uniformly by rank among occupied pixels. This retains the
    // projected-area measure while bounding pairwise reprojection work.
    void buildSamples(uint32_t k, uint32_t occupiedCount) {
        occupiedCounts[k] = occupiedCount;
        uint32_t desired = std::min(occupiedCount, CORRELATION_SAMPLE_COUNT);
        sampleCounts[k] = desired;
        if (desired == 0)
            return;

        const uint32_t* mapIds = idData(k);
        uint32_t occupiedRank = 0;
        uint32_t sampleIndex = 0;
        uint32_t targetRank = (uint32_t)(((uint64_t)occupiedCount) / (2u * desired));

        for (uint32_t pixel = 0; pixel < PixelCount && sampleIndex < desired; ++pixel) {
            if (mapIds[pixel] == std::numeric_limits<uint32_t>::max())
                continue;

            if (occupiedRank == targetRank) {
                samplePixels[k][sampleIndex++] = pixel;
                if (sampleIndex < desired) {
                    targetRank = (uint32_t)(((uint64_t)(2u * sampleIndex + 1u) * occupiedCount) /
                                            (2u * desired));
                }
            }
            ++occupiedRank;
        }
        sampleCounts[k] = sampleIndex;
    }
};

static inline float calcBakedLambda(const VoxelData& vd, const float3& direction) {
    float vpa = vd.polygonsProjAreaFunc.eval(direction);
    if (vpa <= 0.0f)
        return 1e6f;
    float tpa = vd.totalProjAreaFunc.eval(direction);
    return std::max(0.0f, (tpa - vpa) / vpa);
}

static inline bool visibleInProjection(const AnalyzeScratch& scratch,
                                       uint32_t targetIndex,
                                       const float3& pointFromNodeCenter,
                                       uint32_t sourcePolygon,
                                       const float3& sourceNormal) {
    const Basis2& basis = scratch.bases[targetIndex];
    float px = dot(pointFromNodeCenter, basis.u);
    float py = dot(pointFromNodeCenter, basis.w);
    int ix = (int)std::floor((px + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE);
    int iy = (int)std::floor((py + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE);
    if (ix < 0 || iy < 0 || ix >= PROJ_RES || iy >= PROJ_RES)
        return false;

    uint32_t pixel = (uint32_t)iy * PROJ_RES + (uint32_t)ix;
    const uint32_t* targetIds = scratch.idData(targetIndex);
    uint32_t targetPolygon = targetIds[pixel];
    if (targetPolygon == std::numeric_limits<uint32_t>::max())
        return false;
    if (targetPolygon == sourcePolygon)
        return true;

    const float3& targetDirection = scratch.directions[targetIndex];
    float pointDepth = -dot(pointFromNodeCenter, targetDirection);
    float targetDepth = scratch.depthData(targetIndex)[pixel];

    // The queried point and the target depth texel are generally at different
    // positions inside the same pixel. Allow the depth change expected from
    // the source plane, but cap it so a real foreground layer is not accepted.
    float3 n = safeNormalize(sourceNormal);
    float nd = std::abs(dot(n, targetDirection));
    float tangent = safeSqrt(1.0f - nd * nd) / std::max(nd, 0.05f);
    float depthEpsilon = 1e-4f + 0.75f * PROJ_PIXEL_SIZE * tangent;
    depthEpsilon = std::min(depthEpsilon, 0.05f);
    return pointDepth <= targetDepth + depthEpsilon;
}

static inline bool estimateConditionalVisibility(const AnalyzeScratch& scratch,
                                                 const std::vector<Polygon>& polygons,
                                                 uint32_t incidentIndex,
                                                 uint32_t outgoingIndex,
                                                 float& conditionalVisibility,
                                                 float& projectedMeasure) {
    uint32_t sampleCount = scratch.sampleCounts[outgoingIndex];
    if (sampleCount == 0)
        return false;

    const float3& wi = scratch.directions[incidentIndex];
    const float3& wo = scratch.directions[outgoingIndex];
    const Basis2& outgoingBasis = scratch.bases[outgoingIndex];
    const float* outgoingDepth = scratch.depthData(outgoingIndex);
    const uint32_t* outgoingIds = scratch.idData(outgoingIndex);

    double denominator = 0.0;
    double numerator = 0.0;
    for (uint32_t s = 0; s < sampleCount; ++s) {
        uint32_t pixel = scratch.samplePixels[outgoingIndex][s];
        uint32_t polygonIndex = outgoingIds[pixel];
        if (polygonIndex >= polygons.size())
            continue;

        const Polygon& polygon = polygons[polygonIndex];
        float ci = std::abs(dot(polygon.normal, wi));
        if (ci <= 1e-5f)
            continue;

        uint32_t ix = pixel % PROJ_RES;
        uint32_t iy = pixel / PROJ_RES;
        float x = -PROJ_HALF_SIZE + ((float)ix + 0.5f) * PROJ_PIXEL_SIZE;
        float y = -PROJ_HALF_SIZE + ((float)iy + 0.5f) * PROJ_PIXEL_SIZE;
        float3 pointFromNodeCenter = x * outgoingBasis.u + y * outgoingBasis.w
                                   - outgoingDepth[pixel] * wo;

        denominator += ci;
        if (visibleInProjection(scratch, incidentIndex, pointFromNodeCenter,
                                polygonIndex, polygon.normal)) {
            numerator += ci;
        }
    }

    if (denominator <= 1e-8)
        return false;

    conditionalVisibility = (float)std::clamp(numerator / denominator, 0.0, 1.0);
    projectedMeasure = (float)(denominator *
        ((double)scratch.occupiedCounts[outgoingIndex] / (double)sampleCount));
    return projectedMeasure > 0.0f;
}

static inline float fitVisibilityCorrelation(const VoxelData& vd,
                                             const std::vector<Polygon>& polygons,
                                             const AnalyzeScratch& scratch) {
    double fitNumerator = 0.0;
    double fitDenominator = 0.0;

    // c_voxel now describes correlation between distinct directions on the
    // same side of one lobe. Opposing lobe pairs are deliberately excluded;
    // runtime falls back to the Smith endpoint for opposite-side directions.
    for (uint32_t k = 0; k < LOBE_COUNT; ++k) {
        for (uint32_t localO = 0; localO < CORRELATION_DIRECTIONS_PER_LOBE; ++localO) {
            uint32_t o = correlationProjectionIndex(k, localO);
            if (!scratch.valid[o])
                continue;
            for (uint32_t localI = 0; localI < CORRELATION_DIRECTIONS_PER_LOBE; ++localI) {
                uint32_t i = correlationProjectionIndex(k, localI);
                if (i == o || !scratch.valid[i])
                    continue;

                float q = 0.0f;
                float projectedMeasure = 0.0f;
                if (!estimateConditionalVisibility(scratch, polygons, i, o, q, projectedMeasure))
                    continue;

                float Li = calcBakedLambda(vd, scratch.directions[i]);
                float Lo = calcBakedLambda(vd, scratch.directions[o]);
                float minLambda = std::min(Li, Lo);
                if (!std::isfinite(Li) || !std::isfinite(Lo) ||
                    minLambda <= 1e-3f || Li > 1e3f || Lo > 1e3f)
                    continue;

                // y = c_eff * min(Lambda_i, Lambda_o). Clamp to the range the
                // c in [0,1] model can represent; out-of-family negative overlap
                // therefore falls back to the Smith endpoint.
                float y = 0.0f;
                if (q > 1e-6f)
                    y = 1.0f + Li + Lo - (1.0f + Lo) / q;
                y = std::clamp(y, 0.0f, minLambda);

                float angular = std::clamp(0.5f *
                    (1.0f + dot(scratch.directions[i], scratch.directions[o])), 0.0f, 1.0f);
                float g = std::pow(angular, CORRELATION_ANGULAR_POWER);

                // c_eff = g + (1-g)c_voxel. Solve the single unknown with
                // weighted least squares over every ordered same-side pair.
                double x = (double)minLambda * (1.0 - (double)g);
                if (std::abs(x) <= 1e-8)
                    continue;
                double target = (double)y - (double)minLambda * (double)g;
                double weight = std::max((double)projectedMeasure, 1e-8);
                fitNumerator += weight * x * target;
                fitDenominator += weight * x * x;
            }
        }
    }

    if (fitDenominator <= 1e-12)
        return 0.0f;
    return (float)std::clamp(fitNumerator / fitDenominator, 0.0, 1.0);
}

// ---- Transform helpers (same as ClipPhase) ----

static inline float3 localToVoxel(const float3& localPos, const glm::mat4& worldM,
                                  const float3& gridMin, const float3& invVoxelSize) {
    glm::vec4 wp = worldM * glm::vec4(localPos, 1.0f);
    return float3(
        (wp.x - gridMin.x) * invVoxelSize.x,
        (wp.y - gridMin.y) * invVoxelSize.y,
        (wp.z - gridMin.z) * invVoxelSize.z
    );
}

static inline float3 transformNormal(const float3& localNormal, const glm::mat4& worldM) {
    glm::mat3 rot(worldM);
    glm::vec3 wn = rot * glm::vec3(localNormal.x, localNormal.y, localNormal.z);
    return safeNormalize(float3(wn.x, wn.y, wn.z));
}

// Reconstruct the material sample used by the renderer for one clipped
// polygon.  The polygon is stored in node coordinates, so rangeScale maps it
// back to the leaf-voxel coordinate system before UV/normal interpolation.
static inline ABSDFInput sampleMaterial(const Polygon& polygon,
                                        const AnalyzeContext& ctx,
                                        const float3& invVoxelSize,
                                        float rangeScale) {
    const uint32_t localTid = polygon.triRef.triangleID;
    const uint32_t meshID = polygon.triRef.meshID;
    const uint32_t instanceIdx = polygon.triRef.instanceIdx;
    const MeshGeometry& mesh = ctx.scene.meshes[meshID];
    const uint32_t matID = polygon.triRef.materialID;

    uint3 localIdx = mesh.triangles[localTid];
    const glm::mat4& worldM = ctx.scene.instances[instanceIdx].transform;

    Triangle origTri{};
    origTri.vertices[0] = localToVoxel(mesh.positions[localIdx.x], worldM, ctx.grid.gridMin, invVoxelSize);
    origTri.vertices[1] = localToVoxel(mesh.positions[localIdx.y], worldM, ctx.grid.gridMin, invVoxelSize);
    origTri.vertices[2] = localToVoxel(mesh.positions[localIdx.z], worldM, ctx.grid.gridMin, invVoxelSize);
    origTri.uvs[0] = mesh.texCoords[localIdx.x];
    origTri.uvs[1] = mesh.texCoords[localIdx.y];
    origTri.uvs[2] = mesh.texCoords[localIdx.z];
    origTri.normals[0] = transformNormal(mesh.normals[localIdx.x], worldM);
    origTri.normals[1] = transformNormal(mesh.normals[localIdx.y], worldM);
    origTri.normals[2] = transformNormal(mesh.normals[localIdx.z], worldM);
    origTri.buildTBN();

    float dummyArea = 0.0f;
    float3 centroid = polygon.calcCentroid(dummyArea) * rangeScale;

    float2 polygonUVs[MAX_VERTEX_COUNT];
    for (uint32_t vi = 0; vi < polygon.count; ++vi)
        polygonUVs[vi] = origTri.lerpUV(polygon.vertices[vi] * rangeScale);

    float2 uvCenter(0.0f);
    float uvArea = 0.0f;
    for (uint32_t vi = 0; vi < polygon.count; ++vi) {
        uvCenter += polygonUVs[vi];
        const float2& a = polygonUVs[vi];
        const float2& b = polygonUVs[(vi + 1) % polygon.count];
        uvArea += a.x * b.y - a.y * b.x;
    }
    uvCenter /= (float)polygon.count;
    uvArea = 0.5f * std::abs(uvArea);

    float3 bary = origTri.barycentricCoordinates(centroid);
    float3 interpolatedNormal = safeNormalize(
        origTri.normals[0] * bary.x + origTri.normals[1] * bary.y + origTri.normals[2] * bary.z);

    const MaterialData& mat = ctx.scene.materials[matID];
    float4 baseColorVal = float4(mat.baseColor, 1.0f);
    float roughnessVal = mat.specular.g;
    float metallicVal = mat.specular.b;

    if (mat.isSpecGloss) {
        // SpecGloss: RGB specular color + alpha gloss.
        float4 sgVal = mat.specular;
        if (matID < ctx.specularTextures.size() && ctx.specularTextures[matID].width > 0) {
            sgVal = sampleTextureArea(ctx.specularTextures[matID], uvCenter, uvArea,
                                      float4(mat.specular.x, 1.0f, 0.0f, sgVal.w));
        }
        float specLum = sgVal.x * 0.2126f + sgVal.y * 0.7152f + sgVal.z * 0.0722f;
        roughnessVal = 1.0f - sgVal.w;
        metallicVal = std::min(specLum * 2.0f, 1.0f);
    } else {
        // MetalRough: roughness from ORM G and metallic from ORM B unless a
        // separate metallic texture is available.
        if (matID < ctx.specularTextures.size() && ctx.specularTextures[matID].width > 0) {
            float4 r = sampleTextureArea(ctx.specularTextures[matID], uvCenter, uvArea,
                                         float4(0.0f, roughnessVal, metallicVal, 1.0f));
            roughnessVal = r.y;
            if (matID >= ctx.metallicTextures.size() || ctx.metallicTextures[matID].width == 0)
                metallicVal = r.z;
        }
        if (matID < ctx.metallicTextures.size() && ctx.metallicTextures[matID].width > 0) {
            float4 m = sampleTextureArea(ctx.metallicTextures[matID], uvCenter, uvArea,
                                         float4(metallicVal, 0.0f, 0.0f, 1.0f));
            metallicVal = m.x;
        }
    }

    if (matID < ctx.baseColorTextures.size() && ctx.baseColorTextures[matID].width > 0) {
        baseColorVal = sampleTextureArea(ctx.baseColorTextures[matID], uvCenter, uvArea,
                                         float4(mat.baseColor, 1.0f));
    }

    float3 shadingNormal = interpolatedNormal;
    if (matID < ctx.normalMapTextures.size() && ctx.normalMapTextures[matID].width > 0) {
        float4 nm = sampleTextureArea(ctx.normalMapTextures[matID], uvCenter, uvArea,
                                      float4(0.5f, 0.5f, 1.0f, 1.0f));
        float3 tangentNormal = safeNormalize(float3(nm.x * 2.0f - 1.0f,
                                                     nm.y * 2.0f - 1.0f,
                                                     nm.z * 2.0f - 1.0f));
        float3 T = float3(origTri.TBN[0].x, origTri.TBN[1].x, origTri.TBN[2].x);
        float3 B = float3(origTri.TBN[0].y, origTri.TBN[1].y, origTri.TBN[2].y);
        float3 N = float3(origTri.TBN[0].z, origTri.TBN[1].z, origTri.TBN[2].z);
        shadingNormal = safeNormalize(T * tangentNormal.x + B * tangentNormal.y + N * tangentNormal.z);
    }

    ABSDFInput input{};
    input.baseColor = float3(baseColorVal);
    input.specular = float4(mat.specular.x, roughnessVal, metallicVal, 1.0f);
    input.normal = shadingNormal;
    input.area = polygon.calcArea();
    input.projArea = 0.0f;
    return input;
}

// Analyze one node from its own clipped polygon block.  This is deliberately
// independent of the children: it mirrors AnalyzePolygon.cs.slang, where one
// dispatch thread reads one PolygonRange and writes one VoxelData.
inline void analyzeNodeData(VoxelData& vd,
                            const int3& cellInt,
                            uint32_t level,
                            const std::vector<Polygon>& nodePolys,
                            const AnalyzeContext& ctx,
                            AnalyzeScratch& scratch) {
    PolygonRange range;
    range.init(cellInt);
    range.nodeScale = (float)(1u << (ctx.maxDepth - level));
    range.count = (uint32_t)nodePolys.size();

    vd.init();
    scratch.reset();

    float3 invVoxelSize(1.0f / ctx.grid.voxelSize.x,
                        1.0f / ctx.grid.voxelSize.y,
                        1.0f / ctx.grid.voxelSize.z);

    std::vector<ABSDFInput> materialSamples;
    materialSamples.reserve(nodePolys.size());
    std::vector<uint32_t> lobePairs;
    lobePairs.reserve(nodePolys.size());

    float3 mainNormalSum[LOBE_COUNT] = {};
    float mainNormalArea[LOBE_COUNT] = {};

    for (const Polygon& polygon : nodePolys) {
        ABSDFInput input = sampleMaterial(polygon, ctx, invVoxelSize, range.nodeScale);
        uint32_t k0 = NormalIndex8(input.normal);
        uint32_t k1 = NormalIndex8(-input.normal);
        lobePairs.push_back(k0 | (k1 << 4));
        materialSamples.push_back(input);

        float area = input.area;
        vd.ABSDF.area += area;
        mainNormalSum[k0] += area * input.normal;
        mainNormalArea[k0] += area;
        mainNormalSum[k1] += area * (-input.normal);
        mainNormalArea[k1] += area;
    }

    float3 mainDirections[LOBE_COUNT] = {};
    for (uint32_t k = 0; k < LOBE_COUNT; ++k)
        if (mainNormalArea[k] > 0.0f)
            mainDirections[k] = safeNormalize(mainNormalSum[k]);

    // Fit one visible-area amplitude per lobe from the depth/owner maps that
    // the material bake already generates. No extra persistent voxel data.
    float fitNumerator[LOBE_COUNT] = {};
    float fitDenominator[LOBE_COUNT] = {};
    const float pixelArea = PROJ_PIXEL_SIZE * PROJ_PIXEL_SIZE;
    for (uint32_t k = 0; k < LOBE_COUNT; ++k) {
        if (mainNormalArea[k] <= 0.0f)
            continue;

        float3 direction = mainDirections[k];
        float* depth = scratch.depthData(k);
        uint32_t* ids = scratch.idData(k);
        std::fill(depth, depth + AnalyzeScratch::PixelCount, std::numeric_limits<float>::max());
        std::fill(ids, ids + AnalyzeScratch::PixelCount, std::numeric_limits<uint32_t>::max());
        range.rasterizeDepth(nodePolys, direction, depth, ids);

        scratch.valid[k] = true;
        scratch.directions[k] = direction;
        scratch.bases[k] = orthonormal_basis(direction);

        float visibleByLobe[LOBE_COUNT] = {};
        uint32_t occupiedPixelCount = 0;
        for (uint32_t pixel = 0; pixel < PROJ_RES * PROJ_RES; ++pixel) {
            uint32_t polygonIndex = ids[pixel];
            if (polygonIndex == std::numeric_limits<uint32_t>::max())
                continue;

            ++occupiedPixelCount;
            uint32_t packed = lobePairs[polygonIndex];
            uint32_t k0 = packed & 0xFu;
            uint32_t k1 = (packed >> 4) & 0xFu;
            uint32_t facingLobe = dot(materialSamples[polygonIndex].normal, direction) >= 0.0f ? k0 : k1;
            visibleByLobe[facingLobe] += pixelArea;
            float sign = 0.0f;
            if (k == k0) sign = 1.0f;
            else if (k == k1) sign = -1.0f;
            else continue;

            ABSDFInput input = materialSamples[polygonIndex];
            if (sign < 0.0f)
                input.normal = -input.normal;
            input.projArea = pixelArea;
            vd.ABSDF.accumulate(input);
        }
        for (uint32_t m = 0; m < LOBE_COUNT; ++m) {
            if (mainNormalArea[m] <= 0.0f)
                continue;
            float cosine = std::max(0.0f, dot(mainDirections[m], direction));
            fitNumerator[m] += cosine * visibleByLobe[m];
            fitDenominator[m] += cosine * cosine;
        }
        scratch.buildSamples(k, occupiedPixelCount);
        vd.ABSDF.lobes[k].normalizeSelf();
    }

    for (uint32_t k = 0; k < LOBE_COUNT; ++k) {
        if (!vd.ABSDF.lobes[k].isValid())
            continue;
        vd.ABSDF.lobes[k].weight = fitDenominator[k] > 1e-8f
            ? std::max(0.0f, fitNumerator[k] / fitDenominator[k]) : 0.0f;
    }
    vd.ABSDF.normalizeWeightsOnly();
    if (!vd.isSolid())
        return;

    // Add four 25-degree directions around every valid lobe. These extra
    // projections make same-side correlation observable even when a voxel
    // contains only one pair of opposite, perfectly parallel lobes.
    for (uint32_t k = 0; k < LOBE_COUNT; ++k) {
        if (!scratch.valid[k])
            continue;

        const float3 baseDirection = scratch.directions[k];
        const Basis2 baseBasis = scratch.bases[k];
        const float3 tangentDirections[CORRELATION_PERTURBATION_COUNT] = {
            baseBasis.u, -baseBasis.u, baseBasis.w, -baseBasis.w
        };

        for (uint32_t p = 0; p < CORRELATION_PERTURBATION_COUNT; ++p) {
            uint32_t projectionIndex = correlationProjectionIndex(k, p + 1);
            float3 direction = safeNormalize(
                CORRELATION_PERTURBATION_COS * baseDirection +
                CORRELATION_PERTURBATION_SIN * tangentDirections[p]);

            float* depth = scratch.depthData(projectionIndex);
            uint32_t* ids = scratch.idData(projectionIndex);
            std::fill(depth, depth + AnalyzeScratch::PixelCount, std::numeric_limits<float>::max());
            std::fill(ids, ids + AnalyzeScratch::PixelCount, std::numeric_limits<uint32_t>::max());
            range.rasterizeDepth(nodePolys, direction, depth, ids);

            uint32_t occupiedPixelCount = 0;
            for (uint32_t pixel = 0; pixel < AnalyzeScratch::PixelCount; ++pixel)
                occupiedPixelCount += ids[pixel] != std::numeric_limits<uint32_t>::max() ? 1u : 0u;

            scratch.valid[projectionIndex] = occupiedPixelCount > 0;
            scratch.directions[projectionIndex] = direction;
            scratch.bases[projectionIndex] = orthonormal_basis(direction);
            scratch.buildSamples(projectionIndex, occupiedPixelCount);
        }
    }

    vd.ellipsoid.fit(nodePolys, range);

    SphericalFunc polygonFunc = vd.polygonsProjAreaFunc;
    SphericalFunc primitiveFunc = vd.primitiveProjAreaFunc;
    SphericalFunc totalFunc = vd.totalProjAreaFunc;
    Estimate(vd.ellipsoid, range, polygonFunc, primitiveFunc, totalFunc,
             nodePolys, ctx.sampleFrequency);
    vd.polygonsProjAreaFunc = polygonFunc;
    vd.primitiveProjAreaFunc = primitiveFunc;
    vd.totalProjAreaFunc = totalFunc;
    vd.visibilityCorrelation = fitVisibilityCorrelation(vd, nodePolys, scratch);
}

// Convenience overload for the in-memory pipeline.
inline void analyzeNode(PolygonGenerator& generator,
                        uint32_t bfsIndex,
                        const AnalyzeContext& ctx) {
    const auto& node = generator.mBFSOrder[bfsIndex];
    AnalyzeScratch scratch;
    analyzeNodeData(generator.gBuffer[bfsIndex], node.cellInt, node.level,
                    generator.polygonArrays[bfsIndex], ctx, scratch);
}

// ---- Main entry point: stream through nodes.idx and analyze every node ----
// Each worker opens its own read-only handle on polygons.dat.  Since every BFS
// index is unique, gBuffer writes are disjoint.
inline void execute(const MergePhase::MergeResult& mergeResult,
                    const AnalyzeContext& ctx,
                    uint32_t numThreads = 0) {
    std::ifstream idxIn(mergeResult.nodesIdxPath, std::ios::binary);
    if (!idxIn) {
        std::cerr << "  [Analyze] ERROR: cannot open " << mergeResult.nodesIdxPath << std::endl;
        return;
    }

    PolygonSerializer::NodesIdxHeader idxHdr;
    if (!PolygonSerializer::readNodesIdxHeader(idxIn, idxHdr)) {
        std::cerr << "  [Analyze] ERROR: invalid nodes.idx header" << std::endl;
        return;
    }

    std::vector<PolygonSerializer::NodeIndex> nodeIndices;
    PolygonSerializer::readNodeIndices(idxIn, nodeIndices, idxHdr.nodeCount);
    idxIn.close();

    uint64_t totalNodes = nodeIndices.size();
    std::cout << "  [Analyze] " << totalNodes << " nodes to process" << std::endl;
    if (totalNodes == 0)
        return;

    if (numThreads == 0)
        numThreads = std::max(1u, std::thread::hardware_concurrency());
    numThreads = std::min(numThreads, (uint32_t)totalNodes);

    auto& octree = const_cast<OctreeBuilder::OctreeResult&>(mergeResult.octree);
    std::atomic<uint64_t> processed{0};
    std::atomic<uint64_t> skipped{0};

    auto worker = [&](uint64_t start, uint64_t end, uint32_t threadId) {
        std::ifstream polyIn(mergeResult.polygonsDatPath, std::ios::binary);
        if (!polyIn) {
            std::cerr << "  [Analyze] ERROR: thread " << threadId
                      << " cannot open " << mergeResult.polygonsDatPath << std::endl;
            return;
        }

        AnalyzeScratch scratch;

        uint64_t nextReport = 5000;
        for (uint64_t ni = start; ni < end; ++ni) {
            const auto& nodeIndex = nodeIndices[ni];
            auto bfsIt = octree.nodeKeyToBFSIndex.find(nodeIndex.nodeKey);
            if (bfsIt == octree.nodeKeyToBFSIndex.end()) {
                skipped.fetch_add(1, std::memory_order_relaxed);
                processed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            polyIn.clear();
            polyIn.seekg((std::streamoff)nodeIndex.dataOffset);
            if (!polyIn) {
                std::cerr << "  [Analyze] ERROR: seek to offset " << nodeIndex.dataOffset
                          << " failed (thread " << threadId << ")" << std::endl;
                skipped.fetch_add(1, std::memory_order_relaxed);
                processed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            std::vector<Polygon> nodePolys(nodeIndex.polyCount);
            for (Polygon& polygon : nodePolys) {
                polygon.init();
                PolygonSerializer::readPolygon(polyIn, polygon);
            }
            if (!polyIn) {
                skipped.fetch_add(1, std::memory_order_relaxed);
                processed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            uint32_t bfsIndex = bfsIt->second;
            const auto& node = octree.bfsOrder[bfsIndex];
            analyzeNodeData(octree.gBuffer[bfsIndex], node.cellInt, node.level,
                            nodePolys, ctx, scratch);

            uint64_t p = processed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (threadId == 0 && p >= nextReport) {
                std::cout << "\r  [Analyze] " << p << "/" << totalNodes
                          << " nodes (" << (p * 100 / totalNodes) << "%)" << std::flush;
                nextReport = p + 5000;
            }
        }
    };

    std::vector<std::thread> threads;
    uint64_t chunkSize = (totalNodes + numThreads - 1) / numThreads;
    for (uint32_t t = 0; t < numThreads; ++t) {
        uint64_t start = t * chunkSize;
        uint64_t end = std::min(start + chunkSize, totalNodes);
        if (start >= end) break;
        threads.emplace_back(worker, start, end, t);
    }
    for (auto& thread : threads)
        thread.join();

    std::cout << "\r  [Analyze] " << processed.load() << "/" << totalNodes
              << " nodes (100%)";
    if (skipped.load() > 0)
        std::cout << "  skipped=" << skipped.load();
    std::cout << std::endl;
}

} // namespace AnalyzePhase
