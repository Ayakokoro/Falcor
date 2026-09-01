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
#include <string>

namespace AnalyzePhase {

// Context carrying all scene data needed for per-node analysis.
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

// ---- Process a single node ----
// The same analysis is valid for a leaf or for an exact-clipped internal LOD
// node.  Polygon coordinates are local to the node, and rangeScale restores
// them to voxel space for UV/normal reconstruction.

inline void analyzeNode(
    OctreeBuilder::OctreeResult& octree,
    uint32_t bfsIndex,
    const std::vector<Polygon>& nodePolys,
    const AnalyzeContext& ctx)
{
    auto& bfsItem = octree.bfsOrder[bfsIndex];
    float rangeScale = (float)(1u << (ctx.maxDepth - bfsItem.level));

    // Build a temporary PolygonRange for Ellipsoid::fit and Estimate APIs.
    // localHead = 0 because nodePolys is our local contiguous buffer.
    PolygonRange range;
    range.init(bfsItem.cellInt);
    range.nodeScale = rangeScale;
    range.count = (uint32_t)nodePolys.size();

    VoxelData& vd = octree.gBuffer[bfsIndex];
    vd.init();

    float3 invVoxelSize(1.0f / ctx.grid.voxelSize.x,
                         1.0f / ctx.grid.voxelSize.y,
                         1.0f / ctx.grid.voxelSize.z);

    for (uint32_t pi = 0; pi < nodePolys.size(); pi++) {
        const Polygon& poly = nodePolys[pi];

        uint32_t localTid = poly.triRef.triangleID;
        uint32_t meshID   = poly.triRef.meshID;
        uint32_t instIdx  = poly.triRef.instanceIdx;
        const MeshGeometry& mesh = ctx.scene.meshes[meshID];
        uint32_t matID = mesh.materialID;

        // Reconstruct original triangle in voxel space
        uint3 localIdx = mesh.triangles[localTid];
        const glm::mat4& worldM = ctx.scene.instances[instIdx].transform;

        float3 tv0 = localToVoxel(mesh.positions[localIdx.x], worldM, ctx.grid.gridMin, invVoxelSize);
        float3 tv1 = localToVoxel(mesh.positions[localIdx.y], worldM, ctx.grid.gridMin, invVoxelSize);
        float3 tv2 = localToVoxel(mesh.positions[localIdx.z], worldM, ctx.grid.gridMin, invVoxelSize);

        Triangle origTri;
        origTri.vertices[0] = tv0; origTri.vertices[1] = tv1; origTri.vertices[2] = tv2;
        origTri.uvs[0] = mesh.texCoords[localIdx.x];
        origTri.uvs[1] = mesh.texCoords[localIdx.y];
        origTri.uvs[2] = mesh.texCoords[localIdx.z];
        origTri.buildTBN();

        // Polygon centroid in voxel space
        float dummy;
        float3 centroid = poly.calcCentroid(dummy);
        float3 leafCentroid = centroid * rangeScale;

        // Interpolate UV on polygon vertices
        float2 polyUVs[MAX_VERTEX_COUNT];
        for (uint32_t vi = 0; vi < poly.count; vi++) {
            float3 leafVertex = poly.vertices[vi] * rangeScale;
            polyUVs[vi] = origTri.lerpUV(leafVertex);
        }
        float2 uvCenter(0);
        for (uint32_t vi = 0; vi < poly.count; vi++)
            uvCenter += polyUVs[vi];
        uvCenter /= (float)poly.count;

        float uvArea = 0;
        for (uint32_t vi = 0; vi < poly.count; vi++) {
            const float2& a = polyUVs[vi];
            const float2& b = polyUVs[(vi + 1) % poly.count];
            uvArea += a.x * b.y - a.y * b.x;
        }
        uvArea = 0.5f * std::abs(uvArea);

        // Interpolate normal
        float3 bary = origTri.barycentricCoordinates(leafCentroid);
        float3 n0 = transformNormal(mesh.normals[localIdx.x], worldM);
        float3 n1 = transformNormal(mesh.normals[localIdx.y], worldM);
        float3 n2 = transformNormal(mesh.normals[localIdx.z], worldM);
        float3 interpolatedNormal = safeNormalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);

        // Sample textures
        const MaterialData& mat = ctx.scene.materials[matID];
        float4 baseColorVal = float4(mat.baseColor, 1.0f);

        // --- Roughness + Metallic ---
        float roughnessVal = mat.specular.g;
        float metallicVal  = mat.specular.b;

        if (mat.isSpecGloss)
        {
            // SpecGloss: specular texture is RGB spec color + A gloss
            float4 sgVal = mat.specular;
            if (matID < ctx.specularTextures.size() && ctx.specularTextures[matID].width > 0) {
                sgVal = sampleTextureArea(ctx.specularTextures[matID], uvCenter, uvArea,
                                           float4(mat.specular.x, 1.0f, 0.0f, sgVal.w));
            }
            float specLum = sgVal.x * 0.2126f + sgVal.y * 0.7152f + sgVal.z * 0.0722f;
            roughnessVal = 1.0f - sgVal.w;          // gloss → roughness
            metallicVal  = std::min(specLum * 2.0f, 1.0f);
        }
        else
        {
            // MetalRough: roughness from specular (ORM) texture G channel
            if (matID < ctx.specularTextures.size() && ctx.specularTextures[matID].width > 0) {
                float4 r = sampleTextureArea(ctx.specularTextures[matID], uvCenter, uvArea,
                                             float4(0.0f, roughnessVal, metallicVal, 1.0f));
                roughnessVal = r.y;
                // If no separate metallic texture, use ORM B channel
                if (matID >= ctx.metallicTextures.size() || ctx.metallicTextures[matID].width == 0)
                    metallicVal = r.z;
            }
            // Separate metallic texture (FBX Blender PBR) → R channel
            if (matID < ctx.metallicTextures.size() && ctx.metallicTextures[matID].width > 0) {
                float4 m = sampleTextureArea(ctx.metallicTextures[matID], uvCenter, uvArea,
                                             float4(metallicVal, 0.0f, 0.0f, 1.0f));
                metallicVal = m.x;
            }
        }

        float4 specVal = float4(mat.specular.x, roughnessVal, metallicVal, 1.0f);

        if (matID < ctx.baseColorTextures.size() &&
            ctx.baseColorTextures[matID].width > 0) {
            baseColorVal = sampleTextureArea(
                ctx.baseColorTextures[matID], uvCenter,
                uvArea, float4(mat.baseColor, 1.0f));
        }

        float3 shadingNormal = interpolatedNormal;
        if (matID < ctx.normalMapTextures.size() &&
            ctx.normalMapTextures[matID].width > 0) {
            float4 nm = sampleTextureArea(ctx.normalMapTextures[matID], uvCenter, uvArea,
                                          float4(0.5f, 0.5f, 1.0f, 1.0f));
            float3 tn = safeNormalize(float3(nm.x * 2.0f - 1.0f, nm.y * 2.0f - 1.0f, nm.z * 2.0f - 1.0f));
            float3 T = float3(origTri.TBN[0].x, origTri.TBN[1].x, origTri.TBN[2].x);
            float3 B = float3(origTri.TBN[0].y, origTri.TBN[1].y, origTri.TBN[2].y);
            float3 Ns = float3(origTri.TBN[0].z, origTri.TBN[1].z, origTri.TBN[2].z);
            shadingNormal = safeNormalize(T * tn.x + B * tn.y + Ns * tn.z);
        }

        ABSDFInput input = { float3(baseColorVal), specVal, shadingNormal, poly.calcArea() };
        vd.ABSDF.accumulate(input);
    }

    vd.ABSDF.normalizeSelf();

    if (vd.isSolid()) {
        // Ellipsoid::fit accesses polygons via range.localHead offset into the buffer.
        // Since nodePolys is a contiguous local buffer, localHead=0 works correctly.
        vd.ellipsoid.fit(nodePolys, range);

        SphericalFunc polyF  = vd.polygonsProjAreaFunc;
        SphericalFunc primF  = vd.primitiveProjAreaFunc;
        SphericalFunc totalF = vd.totalProjAreaFunc;

        Estimate(vd.ellipsoid, range, polyF, primF, totalF,
                 nodePolys, ctx.sampleFrequency);

        vd.polygonsProjAreaFunc  = polyF;
        vd.primitiveProjAreaFunc = primF;
        vd.totalProjAreaFunc     = totalF;
    }
}

// Analyze a generic node-index file. The leaf path and every exact LOD path
// share the same index/polygon layout; keyToBFSIndex resolves both leaves and
// internal nodes in the one octree built by the leaf merge.
inline void executeNodes(
    const std::filesystem::path& indexPath,
    const std::filesystem::path& polygonsDatPath,
    OctreeBuilder::OctreeResult& octree,
    const AnalyzeContext& ctx,
    uint32_t numThreads = 0,
    const std::string& label = "nodes")
{
    std::ifstream idxIn(indexPath, std::ios::binary);
    if (!idxIn) {
        std::cerr << "  [Analyze] ERROR: cannot open " << indexPath << std::endl;
        return;
    }

    PolygonSerializer::NodesIdxHeader idxHdr;
    if (!PolygonSerializer::readNodesIdxHeader(idxIn, idxHdr)) {
        std::cerr << "  [Analyze] ERROR: invalid node index " << indexPath << std::endl;
        return;
    }

    std::vector<PolygonSerializer::NodeIndex> nodeIndices;
    PolygonSerializer::readNodeIndices(idxIn, nodeIndices, idxHdr.leafCount);
    idxIn.close();

    uint64_t totalNodes = nodeIndices.size();
    std::cout << "  [Analyze] " << label << ": " << totalNodes
              << " nodes to process" << std::endl;

    if (totalNodes == 0) return;

    if (numThreads == 0)
        numThreads = std::max(1u, std::thread::hardware_concurrency());
    numThreads = std::min(numThreads, (uint32_t)totalNodes);

    std::atomic<uint64_t> processed{0};
    std::atomic<uint64_t> skipped{0};

    auto worker = [&](uint64_t start, uint64_t end, uint32_t threadId) {
        std::ifstream polyIn(polygonsDatPath, std::ios::binary);
        if (!polyIn) {
            std::cerr << "  [Analyze] ERROR: thread " << threadId
                      << " cannot open " << polygonsDatPath << std::endl;
            return;
        }

        uint64_t nextReport = 5000;

        for (uint64_t ni = start; ni < end; ni++) {
            const auto& nodeIndex = nodeIndices[ni];

            polyIn.seekg(nodeIndex.dataOffset);
            if (!polyIn) {
                std::cerr << "  [Analyze] ERROR: seek to offset "
                          << nodeIndex.dataOffset << " failed (thread "
                          << threadId << ")" << std::endl;
                skipped.fetch_add(1, std::memory_order_relaxed);
                processed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            std::vector<Polygon> nodePolys(nodeIndex.polyCount);
            for (uint32_t pi = 0; pi < nodeIndex.polyCount; pi++) {
                nodePolys[pi].init();
                PolygonSerializer::readPolygon(polyIn, nodePolys[pi]);
            }

            if (nodePolys.empty()) {
                skipped.fetch_add(1, std::memory_order_relaxed);
                processed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            auto bfsIt = octree.keyToBFSIndex.find(nodeIndex.nodeKey);
            if (bfsIt == octree.keyToBFSIndex.end()) {
                std::cerr << "  [Analyze] WARNING: nodeKey " << nodeIndex.nodeKey
                          << " is not present in the octree" << std::endl;
                skipped.fetch_add(1, std::memory_order_relaxed);
                processed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            analyzeNode(octree, bfsIt->second, nodePolys, ctx);

            uint64_t p = processed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (threadId == 0 && p >= nextReport) {
                std::cout << "\r  [Analyze] " << label << ": " << p << "/"
                          << totalNodes << " nodes ("
                          << (p * 100 / totalNodes) << "%)" << std::flush;
                nextReport = p + 5000;
            }
        }
    };

    std::vector<std::thread> threads;
    uint64_t chunkSize = (totalNodes + numThreads - 1) / numThreads;

    for (uint32_t t = 0; t < numThreads; t++) {
        uint64_t start = t * chunkSize;
        uint64_t end = std::min(start + chunkSize, totalNodes);
        if (start >= end) break;
        threads.emplace_back(worker, start, end, t);
    }

    for (auto& t : threads)
        t.join();

    std::cout << "\r  [Analyze] " << label << ": " << processed.load()
              << "/" << totalNodes << " nodes (100%)";
    if (skipped.load() > 0)
        std::cout << "  skipped=" << skipped.load();
    std::cout << std::endl;
}

// Backward-compatible leaf entry point.
inline void execute(
    const MergePhase::MergeResult& mergeResult,
    const AnalyzeContext& ctx,
    uint32_t numThreads = 0)
{
    if (!mergeResult.octree) {
        std::cerr << "  [Analyze] ERROR: merge result has no octree" << std::endl;
        return;
    }

    executeNodes(mergeResult.leavesIdxPath,
                 mergeResult.polygonsDatPath,
                 *mergeResult.octree,
                 ctx,
                 numThreads,
                 "leaves");
}

} // namespace AnalyzePhase
