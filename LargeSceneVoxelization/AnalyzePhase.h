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
                            const AnalyzeContext& ctx) {
    PolygonRange range;
    range.init(cellInt);
    range.nodeScale = (float)(1u << (ctx.maxDepth - level));
    range.count = (uint32_t)nodePolys.size();

    vd.init();

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

    const float pixelArea = PROJ_PIXEL_SIZE * PROJ_PIXEL_SIZE;
    for (uint32_t k = 0; k < LOBE_COUNT; ++k) {
        if (mainNormalArea[k] <= 0.0f)
            continue;

        float3 direction = safeNormalize(mainNormalSum[k]);
        float depth[PROJ_RES * PROJ_RES];
        uint32_t ids[PROJ_RES * PROJ_RES];
        std::fill(std::begin(depth), std::end(depth), std::numeric_limits<float>::max());
        std::fill(std::begin(ids), std::end(ids), std::numeric_limits<uint32_t>::max());
        range.rasterizeDepth(nodePolys, direction, depth, ids);

        float visibleProjectedArea = 0.0f;
        for (uint32_t pixel = 0; pixel < PROJ_RES * PROJ_RES; ++pixel) {
            uint32_t polygonIndex = ids[pixel];
            if (polygonIndex == std::numeric_limits<uint32_t>::max())
                continue;

            visibleProjectedArea += pixelArea;
            uint32_t packed = lobePairs[polygonIndex];
            uint32_t k0 = packed & 0xFu;
            uint32_t k1 = (packed >> 4) & 0xFu;
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
        vd.ABSDF.lobes[k].normalizeSelf(visibleProjectedArea);
    }

    vd.ABSDF.normalizeWeightsOnly();
    if (!vd.isSolid())
        return;

    vd.ellipsoid.fit(nodePolys, range);

    SphericalFunc polygonFunc = vd.polygonsProjAreaFunc;
    SphericalFunc primitiveFunc = vd.primitiveProjAreaFunc;
    SphericalFunc totalFunc = vd.totalProjAreaFunc;
    Estimate(vd.ellipsoid, range, polygonFunc, primitiveFunc, totalFunc,
             nodePolys, ctx.sampleFrequency);
    vd.polygonsProjAreaFunc = polygonFunc;
    vd.primitiveProjAreaFunc = primitiveFunc;
    vd.totalProjAreaFunc = totalFunc;
}

// Convenience overload for the in-memory pipeline.
inline void analyzeNode(PolygonGenerator& generator,
                        uint32_t bfsIndex,
                        const AnalyzeContext& ctx) {
    const auto& node = generator.mBFSOrder[bfsIndex];
    analyzeNodeData(generator.gBuffer[bfsIndex], node.cellInt, node.level,
                    generator.polygonArrays[bfsIndex], ctx);
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
                            nodePolys, ctx);

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
