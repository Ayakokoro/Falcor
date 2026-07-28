#pragma once
#include "Types.h"
#include "Triangle.h"
#include <istream>
#include <ostream>
#include <vector>
#include <cstdint>

namespace PolygonSerializer {

// --- Serialized polygon (compact, no padding) ---
// Layout: 5*u32 | float3(normal) | vertexCount*float3(vertices)
// Size for 6-vertex poly: 5*4 + 12 + 6*12 = 104 bytes
// Size for 3-vertex poly: 5*4 + 12 + 3*12 = 68 bytes
// (sizeof(Polygon) is always ~112 bytes with padding)

inline void writePolygon(std::ostream& os, const Polygon& poly) {
    uint32_t vc = poly.count;
    os.write(reinterpret_cast<const char*>(&poly.triRef.meshID),      sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&poly.triRef.triangleID),  sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&poly.triRef.materialID),  sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&poly.triRef.instanceIdx), sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&vc),                      sizeof(uint32_t));
    os.write(reinterpret_cast<const char*>(&poly.normal),             sizeof(float3));
    os.write(reinterpret_cast<const char*>(poly.vertices),            vc * sizeof(float3));
}

inline void readPolygon(std::istream& is, Polygon& poly) {
    uint32_t vc;
    is.read(reinterpret_cast<char*>(&poly.triRef.meshID),      sizeof(uint32_t));
    is.read(reinterpret_cast<char*>(&poly.triRef.triangleID),  sizeof(uint32_t));
    is.read(reinterpret_cast<char*>(&poly.triRef.materialID),  sizeof(uint32_t));
    is.read(reinterpret_cast<char*>(&poly.triRef.instanceIdx), sizeof(uint32_t));
    is.read(reinterpret_cast<char*>(&vc),                      sizeof(uint32_t));
    poly.count = vc;
    is.read(reinterpret_cast<char*>(&poly.normal),             sizeof(float3));
    is.read(reinterpret_cast<char*>(poly.vertices),            vc * sizeof(float3));
}

// Size of a serialized polygon in bytes (for buffer allocation)
inline uint32_t serializedPolygonSize(const Polygon& poly) {
    return 5 * sizeof(uint32_t) + sizeof(float3) + poly.count * sizeof(float3);
}

// --- Shard file entry ---
// Format: [nodeKey: u64][dataSize: u32][serializedPolygon bytes]
// dataSize = serializedPolygonSize (for validation during read)

inline void writeShardEntry(std::ostream& os, uint64_t nodeKey, const Polygon& poly) {
    os.write(reinterpret_cast<const char*>(&nodeKey), sizeof(uint64_t));
    uint32_t dataSize = serializedPolygonSize(poly);
    os.write(reinterpret_cast<const char*>(&dataSize), sizeof(uint32_t));
    writePolygon(os, poly);
}

// Returns true on success, false on EOF/read error
inline bool readShardEntry(std::istream& is, uint64_t& nodeKey, Polygon& poly) {
    if (!is.read(reinterpret_cast<char*>(&nodeKey), sizeof(uint64_t))) return false;
    uint32_t dataSize;
    if (!is.read(reinterpret_cast<char*>(&dataSize), sizeof(uint32_t))) return false;
    readPolygon(is, poly);
    return is.good();
}

// --- Shard file header (28 bytes) ---
struct ShardHeader {
    uint32_t magic    = 0x564F5843;   // "VOXC"
    uint32_t version  = 1;
    uint32_t threadId = 0;
    uint64_t entryCount = 0;          // patched after all writes
    uint64_t reserved = 0;
};

inline void writeShardHeader(std::ostream& os, const ShardHeader& hdr) {
    os.write(reinterpret_cast<const char*>(&hdr), sizeof(ShardHeader));
}

inline bool readShardHeader(std::istream& is, ShardHeader& hdr) {
    is.read(reinterpret_cast<char*>(&hdr), sizeof(ShardHeader));
    return is.good() && hdr.magic == 0x564F5843 && hdr.version == 1;
}

// Patch entryCount in an already-written shard header
inline void patchShardEntryCount(std::ostream& os, uint64_t count) {
    os.seekp(12, std::ios::beg);  // offset of entryCount in ShardHeader
    os.write(reinterpret_cast<const char*>(&count), sizeof(uint64_t));
}

// --- leaves.idx ---
// Header: magic|version|leafCount|maxDepth|reserved
// Followed by leafCount * LeafIndex (24 bytes each)

constexpr uint32_t LEAVES_IDX_MAGIC   = 0x49444C56;  // "VLDI"
constexpr uint32_t LEAVES_IDX_VERSION = 1;

struct LeafIndex {
    uint64_t nodeKey    = 0;
    uint64_t dataOffset = 0;    // offset into polygons.dat
    uint32_t polyCount  = 0;
    uint32_t padding    = 0;
};

struct LeavesIdxHeader {
    uint32_t magic     = LEAVES_IDX_MAGIC;
    uint32_t version   = LEAVES_IDX_VERSION;
    uint64_t leafCount = 0;
    uint32_t maxDepth  = 0;
    uint32_t reserved  = 0;
};

inline void writeLeavesIdxHeader(std::ostream& os, const LeavesIdxHeader& hdr) {
    os.write(reinterpret_cast<const char*>(&hdr), sizeof(LeavesIdxHeader));
}

inline bool readLeavesIdxHeader(std::istream& is, LeavesIdxHeader& hdr) {
    is.read(reinterpret_cast<char*>(&hdr), sizeof(LeavesIdxHeader));
    return is.good() && hdr.magic == LEAVES_IDX_MAGIC && hdr.version == LEAVES_IDX_VERSION;
}

inline void writeLeafIndices(std::ostream& os, const std::vector<LeafIndex>& indices) {
    os.write(reinterpret_cast<const char*>(indices.data()),
             indices.size() * sizeof(LeafIndex));
}

inline void readLeafIndices(std::istream& is, std::vector<LeafIndex>& indices, uint64_t count) {
    indices.resize(count);
    is.read(reinterpret_cast<char*>(indices.data()), count * sizeof(LeafIndex));
}

// --- polygons.dat ---
// Each leaf block: [polyCount: u32][serializedPolygon[polyCount]]
// Leaves are stored in bucket-hash order; leaves.idx provides the offset mapping.

inline void writeLeafBlock(std::ostream& os, const std::vector<Polygon>& polys) {
    uint32_t count = (uint32_t)polys.size();
    os.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
    for (const auto& poly : polys)
        writePolygon(os, poly);
}

inline uint32_t readLeafBlock(std::istream& is, std::vector<Polygon>& polys) {
    uint32_t count;
    is.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
    polys.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        polys[i].init();
        readPolygon(is, polys[i]);
    }
    return count;
}

// --- Utility: zero-padded thread/bucket filename ---
inline std::string pad4(uint32_t n) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%04u", n);
    return std::string(buf);
}

} // namespace PolygonSerializer
