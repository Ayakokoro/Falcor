#pragma once
#include "Types.h"
#include "VoxelData.h"
#include "PolygonGenerator.h"
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace OctreeBuilder {

// Output of building the octree from a set of occupied node keys.
// Mirrors the relevant outputs of PolygonGenerator after finalizeBFS().
struct OctreeResult {
    // BFS traversal order: one entry per occupied node
    std::vector<PolygonGenerator::BFSNodeInfo> bfsOrder;

    // Octree connectivity, parallel to bfsOrder
    std::vector<OctreeNode> octreeNodes;

    // Per-node voxel data, parallel to bfsOrder (zeroed on construction)
    // Leaf entries are filled by AnalyzePhase; non-leaf by aggregateParents.
    std::vector<VoxelData> gBuffer;

    // Per-level node counts, indexed by level [0..maxDepth]
    std::vector<uint32_t> levelNodeCounts;

    // Maps nodeKey -> bfsIndex for leaf nodes only.
    // Used by AnalyzePhase to know where to write analysis results.
    std::unordered_map<uint64_t, uint32_t> leafKeyToBFSIndex;

    // All occupied keys -> bfsIndex (for parent aggregation child lookups)
    std::unordered_map<uint64_t, uint32_t> keyToBFSIndex;

    uint32_t maxDepth = 0;

    uint32_t totalNodes() const { return (uint32_t)gBuffer.size(); }
};

// Build octree from a set of occupied leaf node keys.
// Automatically derives ancestor keys up to root.
inline OctreeResult buildFromLeafKeys(
    const std::unordered_set<uint64_t>& leafKeys,
    uint32_t maxDepth)
{
    // Step 1: Collect all occupied nodes (leaves + all ancestors)
    std::unordered_set<uint64_t> occupiedNodes;
    for (uint64_t leafKey : leafKeys) {
        uint64_t key = leafKey;
        while (true) {
            occupiedNodes.insert(key);
            uint32_t level = PolygonGenerator::levelFromKey(key);
            if (level == 0) break;
            int3 cell = PolygonGenerator::cellFromKey(key);
            int3 parentCell = cell / 2;
            key = PolygonGenerator::makeNodeKey(level - 1, parentCell);
        }
    }

    //
    OctreeResult result;
    result.maxDepth = maxDepth;
    result.levelNodeCounts.assign(maxDepth + 1, 0);

    struct BFSItem { int3 cellInt; uint32_t level; };
    std::vector<BFSItem> queue;

    // Start from root if occupied
    uint64_t rootKey = PolygonGenerator::makeNodeKey(0, int3(0, 0, 0));
    if (occupiedNodes.find(rootKey) != occupiedNodes.end())
        queue.push_back({int3(0, 0, 0), 0});

    size_t head = 0;
    std::unordered_map<uint64_t, uint32_t> allKeyToBFS;

    while (head < queue.size()) {
        BFSItem item = queue[head++];
        uint64_t nodeKey = PolygonGenerator::makeNodeKey(item.level, item.cellInt);

        if (occupiedNodes.find(nodeKey) == occupiedNodes.end())
            continue;

        uint32_t bfsIndex = (uint32_t)result.bfsOrder.size();
        result.bfsOrder.push_back({item.cellInt, item.level});
        allKeyToBFS[nodeKey] = bfsIndex;
        result.levelNodeCounts[item.level]++;

        if (item.level < maxDepth) {
            for (uint32_t ci = 0; ci < 8; ci++) {
                int3 childCell = item.cellInt * 2 + int3(
                    (int)(ci & 1), (int)((ci >> 1) & 1), (int)((ci >> 2) & 1));
                uint64_t childKey = PolygonGenerator::makeNodeKey(item.level + 1, childCell);
                if (occupiedNodes.find(childKey) != occupiedNodes.end())
                    queue.push_back({childCell, item.level + 1});
            }
        }
    }

    // Step 3: Allocate octree nodes and gBuffer
    uint32_t totalNodes = (uint32_t)result.bfsOrder.size();
    result.octreeNodes.resize(totalNodes);
    result.gBuffer.resize(totalNodes);
    for (auto& vd : result.gBuffer) vd.init();

    // Step 4: Build OctreeNode linkage (childBase, childMask, dataIndex)
    for (uint32_t bfsIdx = 0; bfsIdx < totalNodes; bfsIdx++) {
        auto& item = result.bfsOrder[bfsIdx];
        OctreeNode& oct = result.octreeNodes[bfsIdx];
        oct.dataIndex = bfsIdx;

        if (item.level < maxDepth) {
            oct.childMask = 0;
            uint32_t firstChildBFS = 0;
            bool hasFirst = false;
            for (uint32_t ci = 0; ci < 8; ci++) {
                int3 childCell = item.cellInt * 2 + int3(
                    (int)(ci & 1), (int)((ci >> 1) & 1), (int)((ci >> 2) & 1));
                uint64_t childKey = PolygonGenerator::makeNodeKey(item.level + 1, childCell);
                auto it = allKeyToBFS.find(childKey);
                if (it != allKeyToBFS.end()) {
                    oct.childMask |= (1u << ci);
                    if (!hasFirst) { firstChildBFS = it->second; hasFirst = true; }
                }
            }
            oct.childBase = hasFirst ? firstChildBFS : 0;
        }
    }

    // Step 5: Build leafKey -> bfsIndex mapping for AnalyzePhase
    for (uint32_t bfsIdx = 0; bfsIdx < totalNodes; bfsIdx++) {
        auto& item = result.bfsOrder[bfsIdx];
        if (item.level == maxDepth) {
            uint64_t key = PolygonGenerator::makeNodeKey(item.level, item.cellInt);
            if (leafKeys.find(key) != leafKeys.end())
                result.leafKeyToBFSIndex[key] = bfsIdx;
        }
    }

    result.keyToBFSIndex = std::move(allKeyToBFS);

    std::cout << "  [OctreeBuilder] " << totalNodes << " nodes (";
    for (uint32_t l = 0; l <= maxDepth; l++) {
        if (l > 0) std::cout << ", ";
        std::cout << "L" << l << "=" << result.levelNodeCounts[l];
    }
    std::cout << ")" << std::endl;

    return result;
}

} // namespace OctreeBuilder
