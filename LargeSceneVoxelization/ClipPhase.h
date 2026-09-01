#pragma once
#include "Types.h"
#include "Triangle.h"
#include "AABB.h"
#include "GridData.h"
#include "SceneLoader.h"
#include "VoxelizationUtility.h"
#include "PolygonSerializer.h"
#include "PolygonGenerator.h"
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>
#include <atomic>
#include <iostream>
#include <algorithm>
#include <string>

namespace ClipPhase {

struct ClipResult {
    std::vector<std::filesystem::path> shardFiles;
    std::filesystem::path shardDir;
    uint64_t totalPolygonsClipped = 0;
    uint32_t targetLevel = 0;
};

// ---- Transform helpers (mirror SceneVoxelization private statics) ----

static inline float3 localToVoxel(const float3& localPos, const glm::mat4& worldM,
                                   const float3& gridMin, const float3& invVoxelSize) {
    glm::vec4 wp = worldM * glm::vec4(localPos, 1.0f);
    return float3(
        (wp.x - gridMin.x) * invVoxelSize.x,
        (wp.y - gridMin.y) * invVoxelSize.y,
        (wp.z - gridMin.z) * invVoxelSize.z
    );
}

// TODO:Correct transformed normal
static inline float3 transformNormal(const float3& localNormal, const glm::mat4& worldM) {
    glm::mat3 rot(worldM);
    glm::vec3 wn = rot * glm::vec3(localNormal.x, localNormal.y, localNormal.z);
    return safeNormalize(float3(wn.x, wn.y, wn.z));
}

// ---- Recursive hierarchical clip, writes one selected tree level ----
// Identical algorithm to SceneVoxelization::clipHierarchical, but outputs
// to a binary stream instead of mNodePolygonMap.  The traversal still starts
// at the root for every triangle, but stops as soon as targetLevel is reached.
// This is what lets the caller run one independent pass per exact LOD level.
// Returns the number of node polygon entries written.

// TODO:Maybe cause stack overflow
static uint64_t clipHierarchicalToStream(
    uint32_t meshID, uint32_t materialID, uint32_t triangleID, uint32_t instanceIdx,
    Triangle& tri, const AABBInt& triAABB, const int3& nodeCell,
    uint32_t level, uint32_t maxDepth,
    uint32_t targetLevel,
    std::ostream& out)
{
    uint32_t scale = 1u << (maxDepth - level);
    float3 minPoint = float3(nodeCell) * (float)scale;
    float3 maxPoint = minPoint + float3((float)scale);

    Polygon polygon = VoxelizationUtility::BoxClipTriangle(minPoint, maxPoint, tri);
    if (polygon.count < 3 || polygon.calcArea() <= 1e-8f)
        return 0;

    float invScale = 1.0f / (float)scale;
    for (uint32_t vi = 0; vi < polygon.count; vi++)
        polygon.vertices[vi] *= invScale;

    polygon.normal = {tri.TBN[0].z, tri.TBN[1].z, tri.TBN[2].z};
    polygon.triRef.meshID      = meshID;
    polygon.triRef.triangleID  = triangleID;
    polygon.triRef.materialID  = materialID;
    polygon.triRef.instanceIdx = instanceIdx;

    uint64_t written = 0;

    if (level == targetLevel) {
        uint64_t nodeKey = PolygonGenerator::makeNodeKey(level, nodeCell);
        PolygonSerializer::writeShardEntry(out, nodeKey, polygon);
        written = 1;

        // This pass owns only one tree level.  Do not descend to finer
        // levels; the next exact LOD pass will rescan the scene separately.
        return written;
    }

    if (level >= maxDepth) return written;

    int childScale = (int)(scale >> 1);
    for (uint32_t ci = 0; ci < 8; ci++) {
        int3 childCell = nodeCell * 2 + int3(
            (int)(ci & 1), (int)((ci >> 1) & 1), (int)((ci >> 2) & 1));
        int3 childMin = childCell * childScale;
        int3 childMax = childMin + childScale - 1;

        if (triAABB.xMax < childMin.x || triAABB.xMin > childMax.x ||
            triAABB.yMax < childMin.y || triAABB.yMin > childMax.y ||
            triAABB.zMax < childMin.z || triAABB.zMin > childMax.z)
            continue;

        written += clipHierarchicalToStream(
            meshID, materialID, triangleID, instanceIdx,
            tri, triAABB, childCell, level + 1, maxDepth, targetLevel, out);
    }
    return written;
}

// ---- Per-thread worker ----

struct ThreadStats {
    uint64_t entriesWritten = 0;
    uint64_t trianglesProcessed = 0;
};

static void clipWorker(
    uint32_t threadId,
    const InstancedScene& scene,
    const GridData& grid,
    uint32_t maxDepth,
    uint32_t targetLevel,
    const std::filesystem::path& clipDir,
    const std::vector<uint32_t>& assignedInstances,
    std::atomic<uint64_t>& globalTriangleCounter,
    std::atomic<uint32_t>& completedCounter,
    ThreadStats& stats)
{
    std::filesystem::path shardPath = clipDir / ("thread_" + PolygonSerializer::pad4(threadId) + ".bin");
    std::ofstream out(shardPath, std::ios::binary);
    if (!out) {
        std::cerr << "  [Clip] ERROR: cannot open shard file " << shardPath << std::endl;
        completedCounter.fetch_add(1, std::memory_order_release);
        return;
    }

    // Write header with placeholder entryCount
    PolygonSerializer::ShardHeader hdr;
    hdr.threadId = threadId;
    hdr.entryCount = 0;
    PolygonSerializer::writeShardHeader(out, hdr);

    float3 invVoxelSize(1.0f / grid.voxelSize.x,
                         1.0f / grid.voxelSize.y,
                         1.0f / grid.voxelSize.z);

    uint64_t entryCount = 0;
    uint64_t triCount = 0;

    for (uint32_t instIdx : assignedInstances) {
        const MeshInstance& inst = scene.instances[instIdx];
        const MeshGeometry& mesh = scene.meshes[inst.meshID];
        const glm::mat4& worldM = inst.transform;
        uint32_t matID = mesh.materialID;

        for (uint32_t localTid = 0; localTid < (uint32_t)mesh.triangles.size(); localTid++) {
            uint3 idx = mesh.triangles[localTid];

            Triangle tri;
            tri.vertices[0] = localToVoxel(mesh.positions[idx.x], worldM, grid.gridMin, invVoxelSize);
            tri.vertices[1] = localToVoxel(mesh.positions[idx.y], worldM, grid.gridMin, invVoxelSize);
            tri.vertices[2] = localToVoxel(mesh.positions[idx.z], worldM, grid.gridMin, invVoxelSize);
            tri.uvs[0] = mesh.texCoords[idx.x];
            tri.uvs[1] = mesh.texCoords[idx.y];
            tri.uvs[2] = mesh.texCoords[idx.z];
            tri.normals[0] = transformNormal(mesh.normals[idx.x], worldM);
            tri.normals[1] = transformNormal(mesh.normals[idx.y], worldM);
            tri.normals[2] = transformNormal(mesh.normals[idx.z], worldM);
            tri.buildTBN();

            entryCount += clipHierarchicalToStream(
                inst.meshID, matID, localTid, instIdx,
                tri, tri.calcAABBInt(),
                int3(0, 0, 0), 0, maxDepth, targetLevel, out);

            triCount++;
            globalTriangleCounter.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Patch entryCount in header
    PolygonSerializer::patchShardEntryCount(out, entryCount);
    out.close();

    stats.entriesWritten = entryCount;
    stats.trianglesProcessed = triCount;
    completedCounter.fetch_add(1, std::memory_order_release);

    std::cout << "  [Clip] Thread " << threadId
              << " done: " << triCount << " triangles, "
              << entryCount << " level-" << targetLevel
              << " polygon entries" << std::endl;
}

// ---- Main entry point ----

inline ClipResult executeLevel(
    const InstancedScene& scene,
    const GridData& grid,
    uint32_t maxDepth,
    uint32_t targetLevel,
    const std::filesystem::path& tmpDir,
    uint32_t numThreads,
    const std::string& phaseName)
{
    namespace fs = std::filesystem;
    if (targetLevel > maxDepth) {
        std::cerr << "  [Clip] ERROR: target level " << targetLevel
                  << " exceeds maxDepth " << maxDepth << std::endl;
        return {};
    }

    fs::path clipDir = tmpDir / phaseName;
    fs::create_directories(clipDir);

    uint32_t numInstances = (uint32_t)scene.instances.size();
    if (numThreads == 0)
        numThreads = std::max(1u, std::thread::hardware_concurrency());
    numThreads = std::min(numThreads, numInstances);
    if (numThreads == 0) numThreads = 1;

    // Round-robin instance allocation across threads
    std::vector<std::vector<uint32_t>> assignments(numThreads);
    for (uint32_t i = 0; i < numInstances; i++)
        assignments[i % numThreads].push_back(i);

    // Compute total triangle count for progress reporting
    uint64_t totalTriangles = 0;
    for (const auto& mesh : scene.meshes) {
        uint32_t instanceCount = 0;
        for (const auto& inst : scene.instances)
            if (inst.meshID == mesh.meshID) instanceCount++;
        totalTriangles += (uint64_t)mesh.triangles.size() * instanceCount;
    }

    std::cout << "  [Clip] Spawning " << numThreads << " threads for "
              << numInstances << " instances, "
              << totalTriangles << " total triangles" << std::endl;

    std::atomic<uint64_t> globalCounter{0};
    std::atomic<uint32_t> completedCounter{0};
    std::vector<ThreadStats> stats(numThreads);
    std::vector<std::thread> threads;

    for (uint32_t t = 0; t < numThreads; t++) {
        threads.emplace_back(clipWorker, t,
        std::cref(scene), std::cref(grid), maxDepth,
            targetLevel,
            std::cref(clipDir),
            std::cref(assignments[t]),
            std::ref(globalCounter),
            std::ref(completedCounter),
            std::ref(stats[t]));
    }

    // Progress reporting loop — exits when all threads have completed
    while (completedCounter.load(std::memory_order_acquire) < numThreads) {
        uint64_t done = globalCounter.load(std::memory_order_relaxed);
        if (totalTriangles > 0) {
            std::cout << "\r  [Clip] " << done << "/" << totalTriangles
                      << " triangles ("
                      << (done * 100 / totalTriangles) << "%)" << std::flush;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    for (auto& th : threads) th.join();
    std::cout << std::endl;

    // Build result
    ClipResult result;
    result.shardDir = clipDir;
    result.targetLevel = targetLevel;
    for (uint32_t t = 0; t < numThreads; t++) {
        result.shardFiles.push_back(clipDir / ("thread_" + PolygonSerializer::pad4(t) + ".bin"));
        result.totalPolygonsClipped += stats[t].entriesWritten;
    }

    std::cout << "  [Clip] Level " << targetLevel << " phase complete: "
              << result.totalPolygonsClipped
              << " total polygon entries across "
              << result.shardFiles.size() << " shard files" << std::endl;

    return result;
}

// Backward-compatible leaf-only entry point used by existing callers.
inline ClipResult execute(
    const InstancedScene& scene,
    const GridData& grid,
    uint32_t maxDepth,
    const std::filesystem::path& tmpDir,
    uint32_t numThreads)
{
    return executeLevel(scene, grid, maxDepth, maxDepth, tmpDir,
                        numThreads, "clip");
}

} // namespace ClipPhase
