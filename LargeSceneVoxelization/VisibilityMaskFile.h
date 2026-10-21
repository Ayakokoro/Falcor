#pragma once
#include "VisibilityMaskFormat.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace VisibilityMaskFile {

// FNV-1a over the exact companion .bin bytes (including its VPW3 tag).
// Detects stale/mismatched companions; this is not a cryptographic checksum.
struct Fingerprint {
    uint64_t bytes = 0;
    uint64_t hash = 14695981039346656037ull;
    void append(const void* data, size_t size) {
        const auto* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i) { hash ^= p[i]; hash *= 1099511628211ull; }
        bytes += size;
    }
};

struct Header {
    uint32_t magic = 0x314d4256u; // "VBM1" in little endian.
    uint32_t version = 1;
    uint32_t headerBytes = 48;
    uint32_t recordBytes = sizeof(VisibilityMask::Data);
    uint32_t resolution = VisibilityMask::Resolution;
    uint32_t sampleCount = VisibilityMask::SampleCount;
    uint32_t nodeCount = 0;
    uint32_t flags = 0;
    uint64_t voxelBytes = 0;
    uint64_t voxelHash = 0;
};
static_assert(sizeof(Header) == 48 && std::is_trivially_copyable_v<Header>);
static_assert(offsetof(Header, voxelBytes) == 32 && offsetof(Header, voxelHash) == 40);

inline void requireLittleEndian() {
    const uint32_t one = 1;
    if (*reinterpret_cast<const uint8_t*>(&one) != 1)
        throw std::runtime_error("VBM1 requires little-endian serialization.");
}

inline std::string companionPath(const std::string& voxelPath) { return voxelPath + ".vbm"; }

inline void write(const std::string& path, const std::vector<VisibilityMask::Data>& records,
                  const Fingerprint& voxel) {
    requireLittleEndian();
    if (records.size() > UINT32_MAX) throw std::runtime_error("Too many visibility-mask records.");
    Header header;
    header.nodeCount = uint32_t(records.size());
    header.voxelBytes = voxel.bytes;
    header.voxelHash = voxel.hash;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot open visibility-mask output: " + path);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(records.data()), std::streamsize(records.size() * sizeof(records[0])));
    out.close();
    if (!out) throw std::runtime_error("Cannot complete visibility-mask output: " + path);
}

// Shared reader for CPU validation and renderer loading. Require
// the companion identity and reject unsupported layouts before allocating.
inline std::vector<VisibilityMask::Data> read(const std::string& path, uint32_t expectedNodes,
                                             const Fingerprint& voxel) {
    requireLittleEndian();
    std::ifstream in(path, std::ios::binary);
    Header header;
    if (!in.read(reinterpret_cast<char*>(&header), sizeof(header)))
        throw std::runtime_error("Cannot read visibility-mask header: " + path);
    if (header.magic != 0x314d4256u || header.version != 1 || header.headerBytes != sizeof(Header) ||
        header.recordBytes != sizeof(VisibilityMask::Data) || header.resolution != VisibilityMask::Resolution ||
        header.sampleCount != VisibilityMask::SampleCount || header.flags != 0 || header.nodeCount != expectedNodes)
        throw std::runtime_error("Unsupported or mismatched visibility-mask layout.");
    if (header.voxelBytes != voxel.bytes || header.voxelHash != voxel.hash)
        throw std::runtime_error("Visibility-mask companion does not match the voxel bake.");
    const uint64_t expectedSize = sizeof(Header) + uint64_t(header.nodeCount) * header.recordBytes;
    if (std::filesystem::file_size(path) != expectedSize)
        throw std::runtime_error("Truncated or oversized visibility-mask file.");
    std::vector<VisibilityMask::Data> records(header.nodeCount);
    if (!in.read(reinterpret_cast<char*>(records.data()), std::streamsize(records.size() * sizeof(records[0]))))
        throw std::runtime_error("Cannot read visibility-mask payload.");
    for (const auto& record : records) {
        const uint32_t count = record.lobeCount();
        if ((record.metadata >> 16) || (record.validLobes() >> count))
            throw std::runtime_error("Invalid visibility-mask lobe metadata.");
        uint32_t validBits = 0;
        for (uint32_t k = 0; k < count; ++k)
            if (record.validLobes() & (1u << k)) validBits |= VisibilityMask::lobeBits(k, count);
        for (uint32_t mask : record.masks)
            if (mask & ~validBits) throw std::runtime_error("Visibility-mask bits refer to an invalid lobe.");
    }
    return records;
}

} // namespace VisibilityMaskFile
