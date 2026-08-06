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

namespace AnalyzePhase {

// Context carrying all scene data needed for leaf analysis
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

// ---- Process a single leaf node ----
// Extracted from SceneVoxelization::process() leaf analysis block (lines 108-217).

inline void analyzeLeaf(
    OctreeBuilder::OctreeResult& octree,
    uint32_t bfsIndex,
    const std::vector<Polygon>& leafPolys,
    const AnalyzeContext& ctx)
{
    auto& bfsItem = octree.bfsOrder[bfsIndex];
    float rangeScale = (float)(1u << (ctx.maxDepth - bfsItem.level));

    // Build a temporary PolygonRange for Ellipsoid::fit and Estimate APIs.
    // localHead = 0 because leafPolys is our local contiguous buffer.
    PolygonRange range;
    range.init(bfsItem.cellInt);
    range.nodeScale = rangeScale;
    range.count = (uint32_t)leafPolys.size();

    VoxelData& vd = octree.gBuffer[bfsIndex];
    vd.init();

    float3 invVoxelSize(1.0f / ctx.grid.voxelSize.x,
                         1.0f / ctx.grid.voxelSize.y,
                         1.0f / ctx.grid.voxelSize.z);

    for (uint32_t pi = 0; pi < leafPolys.size(); pi++) {
        const Polygon& poly = leafPolys[pi];

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
        // Since leafPolys is a contiguous local buffer, localHead=0 works correctly.
        vd.ellipsoid.fit(leafPolys, range);

        SphericalFunc polyF  = vd.polygonsProjAreaFunc;
        SphericalFunc primF  = vd.primitiveProjAreaFunc;
        SphericalFunc totalF = vd.totalProjAreaFunc;

        Estimate(vd.ellipsoid, range, polyF, primF, totalF,
                 leafPolys, ctx.sampleFrequency);

        vd.polygonsProjAreaFunc  = polyF;
        vd.primitiveProjAreaFunc = primF;
        vd.totalProjAreaFunc     = totalF;
    }
}

// ---- Main entry point: stream through leaves.idx, analyze each leaf ----
// Partitions leafIndices across numThreads workers. Each worker opens
// its own read-only ifstream on polygons.dat so seeks don't conflict.
// gBuffer writes are naturally disjoint (unique bfsIndex per leaf).

inline void execute(
    const MergePhase::MergeResult& mergeResult,
    const AnalyzeContext& ctx,
    uint32_t numThreads = 0)
{
    // Read leaves.idx (single-threaded, fast)
    std::ifstream idxIn(mergeResult.leavesIdxPath, std::ios::binary);
    if (!idxIn) {
        std::cerr << "  [Analyze] ERROR: cannot open " << mergeResult.leavesIdxPath << std::endl;
        return;
    }

    PolygonSerializer::LeavesIdxHeader idxHdr;
    if (!PolygonSerializer::readLeavesIdxHeader(idxIn, idxHdr)) {
        std::cerr << "  [Analyze] ERROR: invalid leaves.idx header" << std::endl;
        return;
    }

    std::vector<PolygonSerializer::LeafIndex> leafIndices;
    PolygonSerializer::readLeafIndices(idxIn, leafIndices, idxHdr.leafCount);
    idxIn.close();

    uint64_t totalLeaves = leafIndices.size();
    std::cout << "  [Analyze] " << totalLeaves << " leaves to process" << std::endl;

    if (totalLeaves == 0) return;

    // Determine thread count
    if (numThreads == 0)
        numThreads = std::max(1u, std::thread::hardware_concurrency());
    numThreads = std::min(numThreads, (uint32_t)totalLeaves);

    auto& octree = const_cast<OctreeBuilder::OctreeResult&>(mergeResult.octree);

    // Shared atomic counters
    std::atomic<uint64_t> processed{0};
    std::atomic<uint64_t> skipped{0};

    // Worker function: processes a contiguous slice of leafIndices
    auto worker = [&](uint64_t start, uint64_t end, uint32_t threadId) {
        // Each thread opens its own file handle for independent seeking
        std::ifstream polyIn(mergeResult.polygonsDatPath, std::ios::binary);
        if (!polyIn) {
            std::cerr << "  [Analyze] ERROR: thread " << threadId
                      << " cannot open " << mergeResult.polygonsDatPath << std::endl;
            return;
        }

        // Progress reporting: only thread 0 prints progress
        uint64_t nextReport = 5000;

        for (uint64_t li = start; li < end; li++) {
            const auto& leafIdx = leafIndices[li];

            // Seek to this leaf's polygon block
            polyIn.seekg(leafIdx.dataOffset);
            if (!polyIn) {
                std::cerr << "  [Analyze] ERROR: seek to offset " << leafIdx.dataOffset
                          << " failed (thread " << threadId << ")" << std::endl;
                skipped.fetch_add(1, std::memory_order_relaxed);
                processed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // Read polygon block
            std::vector<Polygon> leafPolys(leafIdx.polyCount);
            for (uint32_t pi = 0; pi < leafIdx.polyCount; pi++) {
                leafPolys[pi].init();
                PolygonSerializer::readPolygon(polyIn, leafPolys[pi]);
            }

            if (leafPolys.empty()) {
                skipped.fetch_add(1, std::memory_order_relaxed);
                processed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // Find BFS index for this leaf
            auto bfsIt = octree.leafKeyToBFSIndex.find(leafIdx.nodeKey);
            if (bfsIt == octree.leafKeyToBFSIndex.end()) {
                skipped.fetch_add(1, std::memory_order_relaxed);
                processed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            analyzeLeaf(octree, bfsIt->second, leafPolys, ctx);

            // Atomic progress update
            uint64_t p = processed.fetch_add(1, std::memory_order_relaxed) + 1;

            // Thread 0 handles progress reporting
            if (threadId == 0 && p >= nextReport) {
                std::cout << "\r  [Analyze] " << p << "/" << totalLeaves
                          << " leaves (" << (p * 100 / totalLeaves) << "%)" << std::flush;
                nextReport = p + 5000;
            }
        }
    };

    // Partition work and launch threads
    std::vector<std::thread> threads;
    uint64_t chunkSize = (totalLeaves + numThreads - 1) / numThreads;

    for (uint32_t t = 0; t < numThreads; t++) {
        uint64_t start = t * chunkSize;
        uint64_t end = std::min(start + chunkSize, totalLeaves);
        if (start >= end) break;
        threads.emplace_back(worker, start, end, t);
    }

    for (auto& t : threads)
        t.join();

    std::cout << "\r  [Analyze] " << processed.load() << "/" << totalLeaves
              << " leaves (100%)";
    if (skipped.load() > 0) std::cout << "  skipped=" << skipped.load();
    std::cout << std::endl;
}

} // namespace AnalyzePhase
