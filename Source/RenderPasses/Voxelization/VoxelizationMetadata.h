#pragma once

// Text sidecar metadata shared by the CPU voxelizer, the GPU voxelization
// pass, and RayMarchingPass.  The binary voxel data layout is intentionally
// kept unchanged; metadata is stored next to <file>.bin as <file>.bin.meta.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

namespace VoxelizationMetadata
{

inline constexpr const char* kFormat = "FalcorVoxelization";
inline constexpr uint32_t kCurrentVersion = 1;
inline constexpr const char* kSidecarSuffix = ".meta";

struct Metadata
{
    uint32_t version = kCurrentVersion;
    uint32_t maxDepth = 0;
    uint32_t generatedLodLevels = 0;
    std::string lodMode = "unknown";
    std::string producer = "unknown";
    uint64_t totalNodes = 0;
    uint64_t binarySize = 0;
};

inline std::filesystem::path sidecarPath(const std::filesystem::path& binaryPath)
{
    return std::filesystem::path(binaryPath.string() + kSidecarSuffix);
}

inline std::string trim(const std::string& value)
{
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};

    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

inline bool parseUint64(const std::string& text, uint64_t& value)
{
    try
    {
        size_t consumed = 0;
        const unsigned long long parsed = std::stoull(text, &consumed, 10);
        if (consumed != text.size())
            return false;
        if (parsed > std::numeric_limits<uint64_t>::max())
            return false;
        value = static_cast<uint64_t>(parsed);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

inline bool parseUint32(const std::string& text, uint32_t& value)
{
    uint64_t parsed = 0;
    if (!parseUint64(text, parsed) || parsed > std::numeric_limits<uint32_t>::max())
        return false;
    value = static_cast<uint32_t>(parsed);
    return true;
}

inline bool write(const std::filesystem::path& binaryPath, Metadata metadata)
{
    std::error_code ec;
    metadata.binarySize = std::filesystem::file_size(binaryPath, ec);
    if (ec)
    {
        std::cerr << "[VoxelizationMetadata] Cannot stat binary file: "
                  << binaryPath << std::endl;
        return false;
    }

    const auto outputPath = sidecarPath(binaryPath);
    const auto temporaryPath = std::filesystem::path(outputPath.string() + ".tmp");

    std::ofstream out(temporaryPath, std::ios::trunc);
    if (!out)
    {
        std::cerr << "[VoxelizationMetadata] Cannot open metadata file: "
                  << temporaryPath << std::endl;
        return false;
    }

    out << "format=" << kFormat << '\n';
    out << "version=" << metadata.version << '\n';
    out << "maxDepth=" << metadata.maxDepth << '\n';
    out << "generatedLodLevels=" << metadata.generatedLodLevels << '\n';
    out << "lodMode=" << metadata.lodMode << '\n';
    out << "producer=" << metadata.producer << '\n';
    out << "totalNodes=" << metadata.totalNodes << '\n';
    out << "binarySize=" << metadata.binarySize << '\n';
    out.close();

    if (!out)
    {
        std::cerr << "[VoxelizationMetadata] Failed writing metadata file: "
                  << temporaryPath << std::endl;
        std::filesystem::remove(temporaryPath, ec);
        return false;
    }

    // Replace an old sidecar after the new content is complete. This avoids
    // leaving a partially-written metadata file if the process is interrupted
    // during the stream write.
    std::filesystem::remove(outputPath, ec);
    ec.clear();
    std::filesystem::rename(temporaryPath, outputPath, ec);
    if (ec)
    {
        std::cerr << "[VoxelizationMetadata] Cannot finalize metadata file: "
                  << outputPath << " (" << ec.message() << ")" << std::endl;
        std::filesystem::remove(temporaryPath, ec);
        return false;
    }

    return true;
}

inline bool read(const std::filesystem::path& binaryPath, Metadata& metadata)
{
    const auto inputPath = sidecarPath(binaryPath);
    std::ifstream in(inputPath);
    if (!in)
        return false;

    Metadata parsed;
    bool hasFormat = false;
    bool hasVersion = false;
    bool hasMaxDepth = false;
    bool hasGeneratedLodLevels = false;

    std::string line;
    while (std::getline(in, line))
    {
        line = trim(line);
        if (line.empty() || line[0] == '#')
            continue;

        const size_t separator = line.find('=');
        if (separator == std::string::npos)
            return false;

        const std::string key = trim(line.substr(0, separator));
        const std::string value = trim(line.substr(separator + 1));

        if (key == "format")
        {
            hasFormat = value == kFormat;
        }
        else if (key == "version")
        {
            hasVersion = parseUint32(value, parsed.version);
        }
        else if (key == "maxDepth")
        {
            hasMaxDepth = parseUint32(value, parsed.maxDepth);
        }
        else if (key == "generatedLodLevels")
        {
            hasGeneratedLodLevels = parseUint32(value, parsed.generatedLodLevels);
        }
        else if (key == "lodMode")
        {
            parsed.lodMode = value;
        }
        else if (key == "producer")
        {
            parsed.producer = value;
        }
        else if (key == "totalNodes")
        {
            if (!parseUint64(value, parsed.totalNodes))
                return false;
        }
        else if (key == "binarySize")
        {
            if (!parseUint64(value, parsed.binarySize))
                return false;
        }
        // Unknown keys are ignored so the sidecar can be extended without
        // breaking older readers.
    }

    if (!hasFormat || !hasVersion || !hasMaxDepth || !hasGeneratedLodLevels)
        return false;
    if (parsed.version == 0 || parsed.version > kCurrentVersion)
        return false;
    if (parsed.binarySize != 0)
    {
        std::error_code ec;
        const uint64_t actualSize = std::filesystem::file_size(binaryPath, ec);
        if (ec || actualSize != parsed.binarySize)
            return false;
    }

    metadata = std::move(parsed);
    return true;
}

} // namespace VoxelizationMetadata
