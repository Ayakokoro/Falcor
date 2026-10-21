#pragma once
#include "VisibilityMaskFile.h"
#include <cmath>
#include <array>
#include <cstring>

// Math-library independent wire format. The legacy .bin/.meta ABI stays intact.
namespace VisibleMaterialFile {
constexpr uint32_t LobeCount = 8;
struct Lobe {
    float weight = 0;
    float normal[3]{};
    float rough = 0;
    float diffuse[3]{};
    float specular[3]{};
};
struct Data {
    Lobe lobes[LobeCount];
    float area = 0;
    uint32_t baked = 0; // Structural/unbaked nodes contain an all-zero record.
    uint32_t sourceLobes() const {
        uint32_t bits = 0;
        for (uint32_t i = 0; i < LobeCount; ++i) {
            const auto& l = lobes[i];
            if (l.weight > 0 && (l.normal[0]*l.normal[0] + l.normal[1]*l.normal[1] + l.normal[2]*l.normal[2]) > 0)
                bits |= 1u << i;
        }
        return bits;
    }
};
static_assert(sizeof(Lobe) == 44 && sizeof(Data) == 360);
static_assert(offsetof(Data, area) == 352 && std::is_trivially_copyable_v<Data>);

using Fingerprint = VisibilityMaskFile::Fingerprint;
inline Fingerprint fingerprint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot fingerprint: " + path);
    Fingerprint result;
    std::array<char, 65536> bytes;
    while (in) {
        in.read(bytes.data(), bytes.size());
        result.append(bytes.data(), size_t(in.gcount()));
    }
    if (!in.eof()) throw std::runtime_error("Cannot finish fingerprint: " + path);
    return result;
}

struct Header {
    uint32_t magic = 0x31544d56u; // VMT1
    uint32_t version = 1;
    uint32_t headerBytes = 80;
    uint32_t recordBytes = sizeof(Data);
    uint32_t nodeCount = 0;
    uint32_t lobeCount = LobeCount;
    uint32_t maxDepth = 0;
    uint32_t lodLevels = 0; // Contiguous independently baked levels, starting at leaves.
    uint64_t voxelBytes = 0;
    uint64_t voxelHash = 0;
    uint64_t maskBytes = 0; // Zero means no associated mask (ignore stale .vbm).
    uint64_t maskHash = 0;
    uint32_t reserved[4]{};
};
static_assert(sizeof(Header) == 80 && offsetof(Header, voxelBytes) == 32);

inline bool finite(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}

inline void validate(const Data& record) {
    if (record.baked > 1 || !finite(record.area) || record.area < 0)
        throw std::runtime_error("Invalid visible-material record.");
    for (const auto& l : record.lobes) {
        if (!finite(l.weight) || l.weight < 0 || !finite(l.rough) || l.rough < 0 || l.rough > 1)
            throw std::runtime_error("Invalid visible-material lobe.");
        for (uint32_t j = 0; j < 3; ++j)
            if (!finite(l.normal[j]) || !finite(l.diffuse[j]) || !finite(l.specular[j]))
                throw std::runtime_error("Non-finite visible material.");
        const double norm2 = double(l.normal[0])*l.normal[0] + double(l.normal[1])*l.normal[1] + double(l.normal[2])*l.normal[2];
        if (l.weight > 0 && (norm2 < 0.99 || norm2 > 1.01))
            throw std::runtime_error("Visible-material normal is not normalized.");
    }
    if (!record.baked && (record.area != 0 || record.sourceLobes() != 0))
        throw std::runtime_error("Unbaked node contains visible material.");
}

inline void write(const std::string& path, const std::vector<Data>& records,
                  uint32_t maxDepth, uint32_t lodLevels, const Fingerprint& voxel,
                  const Fingerprint* masks = nullptr) {
    VisibilityMaskFile::requireLittleEndian();
    if (records.size() > UINT32_MAX || lodLevels > maxDepth)
        throw std::runtime_error("Invalid visible-material dimensions.");
    Header h;
    h.nodeCount = uint32_t(records.size()); h.maxDepth = maxDepth; h.lodLevels = lodLevels;
    h.voxelBytes = voxel.bytes; h.voxelHash = voxel.hash;
    if (masks) { h.maskBytes = masks->bytes; h.maskHash = masks->hash; }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot open visible-material output: " + path);
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    out.write(reinterpret_cast<const char*>(records.data()), std::streamsize(records.size()*sizeof(Data)));
    out.close();
    if (!out) throw std::runtime_error("Cannot complete visible-material output: " + path);
}

// Validate identity and layout before allocating. Chunks are read directly into
// upload scratch, so the renderer does not retain another full scene in RAM.
class Reader {
public:
    Header header;
    Reader(const std::string& path, uint32_t nodes, uint32_t depth, const Fingerprint& voxel)
        : stream(path, std::ios::binary) {
        VisibilityMaskFile::requireLittleEndian();
        if (!stream.read(reinterpret_cast<char*>(&header), sizeof(header)))
            throw std::runtime_error("Cannot read visible-material header: " + path);
        const Header& h = header;
        if (h.magic != 0x31544d56u || h.version != 1 || h.headerBytes != sizeof(Header) ||
            h.recordBytes != sizeof(Data) || h.lobeCount != LobeCount || h.nodeCount != nodes ||
            h.maxDepth != depth || h.lodLevels > depth ||
            h.reserved[0] || h.reserved[1] || h.reserved[2] || h.reserved[3])
            throw std::runtime_error("Unsupported visible-material layout.");
        if (h.voxelBytes != voxel.bytes || h.voxelHash != voxel.hash)
            throw std::runtime_error("Visible material does not match the voxel file.");
        if (std::filesystem::file_size(path) != sizeof(Header) + uint64_t(nodes)*sizeof(Data))
            throw std::runtime_error("Truncated or oversized visible-material file.");
    }
    std::vector<Data> read(uint32_t first, uint32_t count) {
        if (uint64_t(first) + count > header.nodeCount) throw std::runtime_error("Visible-material read out of range.");
        stream.seekg(sizeof(Header) + uint64_t(first)*sizeof(Data));
        std::vector<Data> records(count);
        if (!stream.read(reinterpret_cast<char*>(records.data()), std::streamsize(count)*sizeof(Data)))
            throw std::runtime_error("Cannot read visible materials.");
        for (const auto& record : records) validate(record);
        return records;
    }
    void validateMasks(const std::string& path) const {
        if (!header.maskBytes) throw std::runtime_error("This material bake has no visibility masks.");
        const auto actual = fingerprint(path);
        if (actual.bytes != header.maskBytes || actual.hash != header.maskHash)
            throw std::runtime_error("Visibility mask does not match the visible material bake.");
    }
private:
    std::ifstream stream;
};
} // namespace VisibleMaterialFile
