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
#include <chrono>

namespace MergePhase {

struct MergeResult {
    std::filesystem::path leavesIdxPath;
    std::filesystem::path polygonsDatPath;
    OctreeBuilder::OctreeResult octree;
    uint64_t totalPolygons  = 0;
    uint64_t totalLeaves    = 0;
};

// Lightweight reference to a polygon entry in the in-memory payload arena.
// Used for in-memory sorting during merge (24 bytes each).
struct ShardEntryRef {
    uint64_t nodeKey;
    uint64_t payloadOffset;  // offset in the in-memory payload arena
    uint32_t dataSize;       // size of serialized polygon bytes
};

// LSD radix sort for 64-bit node keys.
// The merge input contains a very large number of small fixed-shape refs;
// radix sort avoids the comparison and branch overhead of std::sort.
inline void radixSortRefs(std::vector<ShardEntryRef>& refs) {
    if (refs.size() < 2) return;

    constexpr uint32_t RADIX_BITS = 16;
    constexpr uint32_t RADIX_SIZE = 1u << RADIX_BITS;
    constexpr uint64_t RADIX_MASK = RADIX_SIZE - 1u;

    std::vector<ShardEntryRef> scratch(refs.size());
    std::vector<size_t> counts(RADIX_SIZE);

    for (uint32_t pass = 0; pass < 4; pass++) {
        std::fill(counts.begin(), counts.end(), size_t(0));
        uint32_t shift = pass * RADIX_BITS;

        for (const auto& ref : refs)
            counts[(ref.nodeKey >> shift) & RADIX_MASK]++;

        size_t offset = 0;
        for (size_t i = 0; i < counts.size(); i++) {
            size_t count = counts[i];
            counts[i] = offset;
            offset += count;
        }

        for (const auto& ref : refs) {
            size_t& dst = counts[(ref.nodeKey >> shift) & RADIX_MASK];
            scratch[dst++] = ref;
        }

        refs.swap(scratch);
    }
}

// Read all shard files, collect entry references, sort by nodeKey,
// merge into per-leaf contiguous polygons.dat + leaves.idx, build octree.
inline MergeResult execute(
    const ClipPhase::ClipResult& clipResult,
    const std::filesystem::path& tmpDir,
    uint32_t maxDepth)
{
    namespace fs = std::filesystem;
    fs::path mergeDir = tmpDir / "merge";
    fs::create_directories(mergeDir);

    fs::path leavesIdxPath  = mergeDir / "leaves.idx";
    fs::path polygonsDatPath = mergeDir / "polygons.dat";

    auto scanStart = std::chrono::steady_clock::now();

    // ---- Step 1: Scan all shard files and load polygon payloads ----
    // The old implementation kept file offsets and performed one seek/read
    // for every sorted reference. That turns a sequential shard scan into a
    // large random-I/O workload. Since merge memory is available, keep the
    // serialized polygon bytes in one arena and sort references to that arena.
    std::vector<ShardEntryRef> refs;
    std::vector<char> payloadArena;
    uint64_t totalPolygons = 0;

    if (clipResult.totalPolygonsClipped <= refs.max_size())
        refs.reserve((size_t)clipResult.totalPolygonsClipped);

    std::cout << "  [Merge] Scanning " << clipResult.shardFiles.size()
              << " shard files..." << std::endl;

    // Reserve an upper bound to avoid repeatedly moving a multi-GB arena.
    uint64_t payloadCapacity = 0;
    for (const auto& shardPath : clipResult.shardFiles) {
        std::error_code ec;
        uint64_t fileSize = fs::file_size(shardPath, ec);
        if (!ec && fileSize > sizeof(PolygonSerializer::ShardHeader))
            payloadCapacity += fileSize - sizeof(PolygonSerializer::ShardHeader);
    }
    if (payloadCapacity <= payloadArena.max_size())
        payloadArena.reserve((size_t)payloadCapacity);

    for (uint32_t si = 0; si < (uint32_t)clipResult.shardFiles.size(); si++) {
        std::ifstream in(clipResult.shardFiles[si], std::ios::binary);
        if (!in) {
            std::cerr << "  [Merge] WARNING: cannot open shard "
                      << clipResult.shardFiles[si] << std::endl;
            continue;
        }

        PolygonSerializer::ShardHeader hdr;
        if (!PolygonSerializer::readShardHeader(in, hdr)) {
            std::cerr << "  [Merge] WARNING: invalid shard header in "
                      << clipResult.shardFiles[si] << std::endl;
            continue;
        }

        // Read all entries from this shard sequentially.
        uint64_t nodeKey;
        uint32_t dataSize;
        while (true) {
            if (!in.read(reinterpret_cast<char*>(&nodeKey), sizeof(uint64_t))) break;
            if (!in.read(reinterpret_cast<char*>(&dataSize), sizeof(uint32_t))) break;

            uint64_t payloadOffset = payloadArena.size();
            size_t oldSize = payloadArena.size();
            payloadArena.resize(oldSize + dataSize);
            if (!in.read(payloadArena.data() + oldSize, dataSize)) {
                payloadArena.resize(oldSize);
                std::cerr << "  [Merge] WARNING: truncated polygon payload in "
                          << clipResult.shardFiles[si] << std::endl;
                break;
            }

            ShardEntryRef ref;
            ref.nodeKey      = nodeKey;
            ref.payloadOffset = payloadOffset;
            ref.dataSize      = dataSize;
            refs.push_back(ref);
            totalPolygons++;
        }
    }

    std::cout << "  [Merge] Collected " << refs.size()
              << " polygon entry references" << std::endl;
    std::cout << "  [Merge] Loaded " << payloadArena.size()
              << " serialized payload bytes into memory" << std::endl;
    auto scanEnd = std::chrono::steady_clock::now();
    std::cout << "  [Merge] Scan/load time: "
              << std::chrono::duration<double>(scanEnd - scanStart).count()
              << "s" << std::endl;

    // ---- Step 2: Sort by nodeKey ----
    std::cout << "  [Merge] Radix-sorting by nodeKey..." << std::endl;
    auto sortStart = std::chrono::steady_clock::now();
    radixSortRefs(refs);
    auto sortEnd = std::chrono::steady_clock::now();
    std::cout << "  [Merge] Sort time: "
              << std::chrono::duration<double>(sortEnd - sortStart).count()
              << "s" << std::endl;

    // ---- Step 3: Stream through sorted refs, group by nodeKey,
    //              copy polygon data to polygons.dat ----
    std::cout << "  [Merge] Merging into " << polygonsDatPath.filename().string()
              << "..." << std::endl;

    std::ofstream polyOut(polygonsDatPath, std::ios::binary | std::ios::trunc);
    if (!polyOut) {
        std::cerr << "  [Merge] ERROR: cannot create " << polygonsDatPath << std::endl;
        return {};
    }

    std::vector<PolygonSerializer::LeafIndex> leafIndices;
    std::unordered_set<uint64_t> leafKeys;

    // Large output buffer avoids one ostream write call per polygon.
    constexpr size_t OUTPUT_BUFFER_SIZE = 4u * 1024u * 1024u;
    std::vector<char> outputBuffer;
    outputBuffer.reserve(OUTPUT_BUFFER_SIZE);
    uint64_t outputOffset = 0;

    auto flushOutputBuffer = [&]() {
        if (outputBuffer.empty()) return;
        polyOut.write(outputBuffer.data(), (std::streamsize)outputBuffer.size());
        outputOffset += outputBuffer.size();
        outputBuffer.clear();
    };

    uint64_t currentKey = 0;
    uint64_t currentOffset = 0;
    uint32_t currentCount = 0;
    bool firstGroup = true;

    for (const auto& ref : refs) {
        if (firstGroup) {
            currentKey = ref.nodeKey;
            currentOffset = outputOffset + outputBuffer.size();
            currentCount = 0;
            firstGroup = false;
        } else if (ref.nodeKey != currentKey) {
            // Finalize previous leaf
            leafIndices.push_back({currentKey, currentOffset, currentCount, 0});
            leafKeys.insert(currentKey);

            // Start new leaf
            currentKey = ref.nodeKey;
            currentOffset = outputOffset + outputBuffer.size();
            currentCount = 0;
        }

        // Copy polygon data from the in-memory payload arena.
        const char* payload = payloadArena.data() + ref.payloadOffset;
        if (ref.dataSize > OUTPUT_BUFFER_SIZE) {
            flushOutputBuffer();
            polyOut.write(payload, ref.dataSize);
            outputOffset += ref.dataSize;
        } else {
            if (outputBuffer.size() + ref.dataSize > OUTPUT_BUFFER_SIZE)
                flushOutputBuffer();
            size_t oldSize = outputBuffer.size();
            outputBuffer.resize(oldSize + ref.dataSize);
            std::memcpy(outputBuffer.data() + oldSize, payload, ref.dataSize);
        }
        currentCount++;
    }

    // Flush last group
    if (currentCount > 0) {
        leafIndices.push_back({currentKey, currentOffset, currentCount, 0});
        leafKeys.insert(currentKey);
    }

    flushOutputBuffer();
    polyOut.close();

    auto writeEnd = std::chrono::steady_clock::now();
    std::cout << "  [Merge] Merge/write time: "
              << std::chrono::duration<double>(writeEnd - sortEnd).count()
              << "s" << std::endl;

    // No longer needed after polygons.dat has been written.
    std::vector<ShardEntryRef>().swap(refs);
    std::vector<char>().swap(payloadArena);

    // refs are already sorted by nodeKey, so leafIndices were emitted in
    // sorted order while grouping. Avoid a second O(leafCount log leafCount)
    // sort here.

    {
        std::ofstream idxOut(leavesIdxPath, std::ios::binary | std::ios::trunc);
        PolygonSerializer::LeavesIdxHeader idxHdr;
        idxHdr.leafCount = leafIndices.size();
        idxHdr.maxDepth  = maxDepth;
        PolygonSerializer::writeLeavesIdxHeader(idxOut, idxHdr);
        PolygonSerializer::writeLeafIndices(idxOut, leafIndices);
        idxOut.close();
    }

    std::cout << "  [Merge] Wrote " << leafIndices.size() << " leaves, "
              << totalPolygons << " total polygons." << std::endl;

    // ---- Step 5: Build octree from leaf keys ----
    auto octree = OctreeBuilder::buildFromLeafKeys(leafKeys, maxDepth);

    // ---- Step 6: Cleanup shard files (always deleted after merge) ----
    for (auto& shardPath : clipResult.shardFiles)
        fs::remove(shardPath);
    fs::path clipDir = tmpDir / "clip";
    std::error_code ec;
    fs::remove(clipDir, ec);  // will fail if not empty, which is fine

    return MergeResult{
        leavesIdxPath,
        polygonsDatPath,
        std::move(octree),
        totalPolygons,
        (uint64_t)leafIndices.size()
    };
}

} // namespace MergePhase
