#pragma once
#include "Types.h"
#include "PolygonSerializer.h"
#include "PolygonGenerator.h"
#include "OctreeBuilder.h"
#include "ClipPhase.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <cstring>

namespace MergePhase {

struct MergeResult {
    std::filesystem::path nodesIdxPath;
    std::filesystem::path polygonsDatPath;
    OctreeBuilder::OctreeResult octree;
    uint64_t totalPolygons  = 0;
    uint64_t totalNodes     = 0;
};

// Lightweight reference to a polygon entry in a shard file.
// Used for in-memory sorting during merge (24 bytes each).
struct ShardEntryRef {
    uint64_t nodeKey;
    uint32_t shardIdx;       // which shard file
    uint64_t shardOffset;    // byte offset in shard file (points to dataSize field)
    uint32_t dataSize;       // size of serialized polygon bytes

    // Total bytes of the shard entry (nodeKey + dataSize + polygon)
    uint64_t entryByteSize() const { return sizeof(uint64_t) + sizeof(uint32_t) + dataSize; }
};

// Read all shard files, collect entry references, sort by nodeKey,
// merge into per-node contiguous polygons.dat + nodes.idx, build the BFS
// octree and keep the same node ordering as the renderer.
inline MergeResult execute(
    const ClipPhase::ClipResult& clipResult,
    const std::filesystem::path& tmpDir,
    uint32_t maxDepth)
{
    namespace fs = std::filesystem;
    fs::path mergeDir = tmpDir / "merge";
    fs::create_directories(mergeDir);

    fs::path nodesIdxPath   = mergeDir / "nodes.idx";
    fs::path polygonsDatPath = mergeDir / "polygons.dat";

    // ---- Step 1: Scan all shard files, collect entry references ----
    std::vector<ShardEntryRef> refs;
    uint64_t totalPolygons = 0;

    std::cout << "  [Merge] Scanning " << clipResult.shardFiles.size()
              << " shard files..." << std::endl;

    // Keep all shard files open for Step 3 data copy
    std::vector<std::ifstream> shardStreams;
    shardStreams.reserve(clipResult.shardFiles.size());

    for (uint32_t si = 0; si < (uint32_t)clipResult.shardFiles.size(); si++) {
        std::ifstream in(clipResult.shardFiles[si], std::ios::binary);
        if (!in) {
            std::cerr << "  [Merge] WARNING: cannot open shard "
                      << clipResult.shardFiles[si] << std::endl;
            // Push a dummy stream to keep indices aligned
            shardStreams.emplace_back();
            continue;
        }

        PolygonSerializer::ShardHeader hdr;
        if (!PolygonSerializer::readShardHeader(in, hdr)) {
            std::cerr << "  [Merge] WARNING: invalid shard header in "
                      << clipResult.shardFiles[si] << std::endl;
            shardStreams.emplace_back();
            continue;
        }

        // Read all entries from this shard, recording their positions
        uint64_t nodeKey;
        uint32_t dataSize;
        while (true) {
            std::streampos entryStart = in.tellg();
            if (!in.read(reinterpret_cast<char*>(&nodeKey), sizeof(uint64_t))) break;
            if (!in.read(reinterpret_cast<char*>(&dataSize), sizeof(uint32_t))) break;

            ShardEntryRef ref;
            ref.nodeKey     = nodeKey;
            ref.shardIdx    = si;
            ref.shardOffset = (uint64_t)entryStart + sizeof(uint64_t);  // points to dataSize
            ref.dataSize    = dataSize;
            refs.push_back(ref);
            totalPolygons++;

            // Skip the serialized polygon bytes
            in.seekg(dataSize, std::ios::cur);
        }

        // Re-open for data reading in Step 3
        in.close();
        shardStreams.emplace_back(clipResult.shardFiles[si], std::ios::binary);
        // Skip header in the reopened stream
        shardStreams.back().seekg(sizeof(PolygonSerializer::ShardHeader));
    }

    std::cout << "  [Merge] Collected " << refs.size()
              << " polygon entry references" << std::endl;

    // ---- Step 2: Sort by nodeKey ----
    std::cout << "  [Merge] Sorting by nodeKey..." << std::endl;
    std::sort(refs.begin(), refs.end(),
        [](const ShardEntryRef& a, const ShardEntryRef& b) {
            return a.nodeKey < b.nodeKey;
        });

    // ---- Step 3: Stream through sorted refs, group by nodeKey,
    //              copy polygon data from shards to polygons.dat ----
    std::cout << "  [Merge] Merging into " << polygonsDatPath.filename().string()
              << "..." << std::endl;

    std::ofstream polyOut(polygonsDatPath, std::ios::binary | std::ios::trunc);
    if (!polyOut) {
        std::cerr << "  [Merge] ERROR: cannot create " << polygonsDatPath << std::endl;
        return {};
    }

    std::vector<PolygonSerializer::NodeIndex> nodeIndices;
    std::unordered_set<uint64_t> nodeKeys;

    // Buffer for reading polygon data from shard files
    std::vector<char> readBuf;

    uint64_t currentKey = 0;
    uint64_t currentOffset = 0;
    uint32_t currentCount = 0;
    bool firstGroup = true;

    for (const auto& ref : refs) {
        if (firstGroup) {
            currentKey = ref.nodeKey;
            currentOffset = (uint64_t)polyOut.tellp();
            currentCount = 0;
            firstGroup = false;
        } else if (ref.nodeKey != currentKey) {
            // Finalize previous node.
            nodeIndices.push_back({currentKey, currentOffset, currentCount, 0});
            nodeKeys.insert(currentKey);

            // Start new node.
            currentKey = ref.nodeKey;
            currentOffset = (uint64_t)polyOut.tellp();
            currentCount = 0;
        }

        // Copy polygon data from the shard file to polygons.dat
        // The shard has: [nodeKey: u64][dataSize: u32][polygon data]
        // We need to read just the polygon data (dataSize bytes after dataSize field)
        auto& shardIn = shardStreams[ref.shardIdx];
        shardIn.seekg(ref.shardOffset);  // position at dataSize field
        uint32_t dataSize;
        shardIn.read(reinterpret_cast<char*>(&dataSize), sizeof(uint32_t));

        if (readBuf.size() < dataSize)
            readBuf.resize(dataSize);
        shardIn.read(readBuf.data(), dataSize);

        polyOut.write(readBuf.data(), dataSize);
        currentCount++;
    }

    // Flush last group
    if (currentCount > 0) {
        nodeIndices.push_back({currentKey, currentOffset, currentCount, 0});
        nodeKeys.insert(currentKey);
    }

    polyOut.close();

    // Close all shard streams
    for (auto& s : shardStreams)
        if (s.is_open()) s.close();

    // refs are already sorted by nodeKey, so nodeIndices are already ordered.

    {
        std::ofstream idxOut(nodesIdxPath, std::ios::binary | std::ios::trunc);
        PolygonSerializer::NodesIdxHeader idxHdr;
        idxHdr.nodeCount = nodeIndices.size();
        idxHdr.maxDepth  = maxDepth;
        PolygonSerializer::writeNodesIdxHeader(idxOut, idxHdr);
        PolygonSerializer::writeNodeIndices(idxOut, nodeIndices);
        idxOut.close();
    }

    std::cout << "  [Merge] Wrote " << nodeIndices.size() << " nodes, "
              << totalPolygons << " total polygons." << std::endl;

    // ---- Step 5: Build octree from all occupied node keys ----
    auto octree = OctreeBuilder::buildFromNodeKeys(nodeKeys, maxDepth);

    // ---- Step 6: Cleanup shard files (always deleted after merge) ----
    for (auto& shardPath : clipResult.shardFiles)
        fs::remove(shardPath);
    fs::path clipDir = tmpDir / "clip";
    std::error_code ec;
    fs::remove(clipDir, ec);  // will fail if not empty, which is fine

    return MergeResult{
        nodesIdxPath,
        polygonsDatPath,
        std::move(octree),
        totalPolygons,
        (uint64_t)nodeIndices.size()
    };
}

} // namespace MergePhase
