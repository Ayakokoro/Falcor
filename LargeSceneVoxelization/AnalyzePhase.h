#pragma once

#include "Types.h"
#include "VoxelizationCore.h"
#include "PolygonGenerator.h"
#include "OctreeBuilder.h"
#include "MergePhase.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace AnalyzePhase
{

// Context carrying scene resources used by the all-node voxelization pass.
// The actual node computation lives in VoxelizationCore so the inspector and
// the production pipeline cannot drift apart.
struct AnalyzeContext
{
    const InstancedScene& scene;
    const GridData& grid;
    uint32_t maxDepth;
    uint32_t sampleFrequency;
    const std::vector<Texture2D>& baseColorTextures;
    const std::vector<Texture2D>& specularTextures;
    const std::vector<Texture2D>& metallicTextures;
    const std::vector<Texture2D>& normalMapTextures;
};

inline void analyzeNode(
    OctreeBuilder::OctreeResult& octree,
    uint32_t bfsIndex,
    const std::vector<Polygon>& nodePolys,
    const AnalyzeContext& ctx)
{
    const auto& bfsItem = octree.bfsOrder[bfsIndex];

    VoxelizationCore::NodeData node;
    node.request.level = bfsItem.level;
    node.request.cell = bfsItem.cellInt;
    node.request.nodeKey = PolygonGenerator::makeNodeKey(
        bfsItem.level, bfsItem.cellInt);
    node.polygons = nodePolys;
    node.storedPolygonCount = static_cast<uint32_t>(node.polygons.size());

    VoxelizationCore::TextureSet textures{
        ctx.baseColorTextures,
        ctx.specularTextures,
        ctx.metallicTextures,
        ctx.normalMapTextures};
    VoxelizationCore::AnalysisContext analysisContext{
        ctx.scene, ctx.grid, ctx.maxDepth, textures};
    VoxelizationCore::AnalysisOptions options;
    options.sampleFrequency = ctx.sampleFrequency;

    const auto result = VoxelizationCore::analyzeNode(
        node, analysisContext, options);
    if (!result.success)
    {
        std::cerr << "  [Analyze] ERROR: " << result.error << std::endl;
        octree.gBuffer[bfsIndex].init();
        return;
    }
    octree.gBuffer[bfsIndex] = result.voxelData;
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
    if (!idxIn)
    {
        std::cerr << "  [Analyze] ERROR: cannot open " << indexPath << std::endl;
        return;
    }

    PolygonSerializer::NodesIdxHeader idxHdr;
    if (!PolygonSerializer::readNodesIdxHeader(idxIn, idxHdr))
    {
        std::cerr << "  [Analyze] ERROR: invalid node index " << indexPath << std::endl;
        return;
    }

    std::vector<PolygonSerializer::NodeIndex> nodeIndices;
    PolygonSerializer::readNodeIndices(idxIn, nodeIndices, idxHdr.leafCount);
    idxIn.close();

    const uint64_t totalNodes = nodeIndices.size();
    std::cout << "  [Analyze] " << label << ": " << totalNodes
              << " nodes to process" << std::endl;
    if (totalNodes == 0)
        return;

    if (numThreads == 0)
        numThreads = std::max(1u, std::thread::hardware_concurrency());
    numThreads = std::min(numThreads, static_cast<uint32_t>(totalNodes));

    std::atomic<uint64_t> processed{0};
    std::atomic<uint64_t> skipped{0};

    auto worker = [&](uint64_t start, uint64_t end, uint32_t threadId) {
        std::ifstream polyIn(polygonsDatPath, std::ios::binary);
        if (!polyIn)
        {
            std::cerr << "  [Analyze] ERROR: thread " << threadId
                      << " cannot open " << polygonsDatPath << std::endl;
            return;
        }

        uint64_t nextReport = 5000;
        for (uint64_t ni = start; ni < end; ++ni)
        {
            const auto& nodeIndex = nodeIndices[ni];
            polyIn.seekg(static_cast<std::streamoff>(nodeIndex.dataOffset));
            if (!polyIn)
            {
                std::cerr << "  [Analyze] ERROR: seek to offset "
                          << nodeIndex.dataOffset << " failed (thread "
                          << threadId << ")" << std::endl;
                skipped.fetch_add(1, std::memory_order_relaxed);
                processed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            std::vector<Polygon> nodePolys(nodeIndex.polyCount);
            for (uint32_t pi = 0; pi < nodeIndex.polyCount; ++pi)
            {
                nodePolys[pi].init();
                PolygonSerializer::readPolygon(polyIn, nodePolys[pi]);
            }

            if (nodePolys.empty())
            {
                skipped.fetch_add(1, std::memory_order_relaxed);
                processed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            const auto bfsIt = octree.keyToBFSIndex.find(nodeIndex.nodeKey);
            if (bfsIt == octree.keyToBFSIndex.end())
            {
                std::cerr << "  [Analyze] WARNING: nodeKey " << nodeIndex.nodeKey
                          << " is not present in the octree" << std::endl;
                skipped.fetch_add(1, std::memory_order_relaxed);
                processed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            analyzeNode(octree, bfsIt->second, nodePolys, ctx);

            const uint64_t progress =
                processed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (threadId == 0 && progress >= nextReport)
            {
                std::cout << "\r  [Analyze] " << label << ": " << progress
                          << "/" << totalNodes << " nodes ("
                          << (progress * 100 / totalNodes) << "%)" << std::flush;
                nextReport = progress + 5000;
            }
        }
    };

    std::vector<std::thread> threads;
    const uint64_t chunkSize =
        (totalNodes + numThreads - 1) / numThreads;
    for (uint32_t threadId = 0; threadId < numThreads; ++threadId)
    {
        const uint64_t start = uint64_t(threadId) * chunkSize;
        const uint64_t end = std::min(start + chunkSize, totalNodes);
        if (start >= end)
            break;
        threads.emplace_back(worker, start, end, threadId);
    }
    for (auto& thread : threads)
        thread.join();

    std::cout << "\r  [Analyze] " << label << ": " << processed.load()
              << "/" << totalNodes << " nodes (100%)";
    if (skipped.load() > 0)
        std::cout << "  skipped=" << skipped.load();
    std::cout << std::endl;
}

inline void execute(
    const MergePhase::MergeResult& mergeResult,
    const AnalyzeContext& ctx,
    uint32_t numThreads = 0)
{
    if (!mergeResult.octree)
    {
        std::cerr << "  [Analyze] ERROR: merge result has no octree" << std::endl;
        return;
    }

    executeNodes(
        mergeResult.leavesIdxPath,
        mergeResult.polygonsDatPath,
        *mergeResult.octree,
        ctx,
        numThreads,
        "leaves");
}

} // namespace AnalyzePhase

