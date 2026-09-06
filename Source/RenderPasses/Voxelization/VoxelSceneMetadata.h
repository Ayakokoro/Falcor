#pragma once

// Scene composition metadata for the multi-bin voxel ray marcher.
//
// This file deliberately does not replace VoxelizationMetadata.h. The latter
// describes the contents of one voxel binary and its LOD sidecar, while this
// file describes how that binary is placed in a scene.
//
// Version 2 scene files have the following shape. Instance transforms are
// complete asset-local-to-world affine matrices serialized in row-major order:
// {
//   "format": "FalcorVoxelScene",
//   "version": 2,
//   "assets": [
//     { "id": "modelA", "voxelFile": "modelA.bin" },
//     { "id": "modelB", "voxelFile": "modelB.bin" }
//   ],
//   "instances": [{
//     "id": 0,
//     "asset": "modelA",
//     "transform": [
//       1, 0, 0, 0,
//       0, 1, 0, 0,
//       0, 0, 1, 0,
//       0, 0, 0, 1
//     ],
//     "enabled": true
//   }]
// }

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace VoxelSceneMetadata
{

inline constexpr const char* kFormat = "FalcorVoxelScene";
inline constexpr uint32_t kCurrentVersion = 2;
inline constexpr uint32_t kMaxAssets = 256;
inline constexpr uint32_t kMaxInstances = 4096;

using Transform16 = std::array<float, 16>;

// Transform16 is row-major in the JSON file and matches Falcor's host-side
// float4x4 storage. Matrix multiplication still uses the existing
// column-vector convention: world = transform * float4(local, 1).
inline constexpr Transform16 kIdentityTransform = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f,
};

struct Asset
{
    std::string assetId;
    std::filesystem::path voxelFile;
};

struct Instance
{
    uint32_t instanceId = 0;
    uint32_t assetIndex = 0;
    bool enabled = true;
    Transform16 transform = kIdentityTransform;
};

struct Scene
{
    std::filesystem::path sourcePath;
    std::vector<Asset> assets;
    // Compatibility aliases for callers written against the original
    // single-asset parser. They always identify assets[0].
    std::string assetId;
    std::filesystem::path voxelFile;
    std::vector<Instance> instances;
};

inline bool fail(std::string& error, std::string message)
{
    error = std::move(message);
    return false;
}

inline bool readUint32(
    const nlohmann::json& value,
    const std::string& fieldName,
    uint32_t& output,
    std::string& error
)
{
    if (!value.is_number_integer())
    {
        return fail(error, "Scene meta field '" + fieldName + "' must be a non-negative integer.");
    }

    try
    {
        if (value.is_number_unsigned())
        {
            const uint64_t parsed = value.get<uint64_t>();
            if (parsed > std::numeric_limits<uint32_t>::max())
                return fail(error, "Scene meta field '" + fieldName + "' is outside the uint32 range.");
            output = static_cast<uint32_t>(parsed);
        }
        else
        {
            const int64_t parsed = value.get<int64_t>();
            if (parsed < 0 || static_cast<uint64_t>(parsed) > std::numeric_limits<uint32_t>::max())
                return fail(error, "Scene meta field '" + fieldName + "' is outside the uint32 range.");
            output = static_cast<uint32_t>(parsed);
        }
    }
    catch (const std::exception&)
    {
        return fail(error, "Scene meta field '" + fieldName + "' could not be converted to uint32.");
    }
    return true;
}

inline bool readFloat16(
    const nlohmann::json& value,
    const std::string& fieldName,
    Transform16& output,
    std::string& error
)
{
    if (!value.is_array() || value.size() != 16)
    {
        return fail(error, "Scene meta field '" + fieldName + "' must be an array of 16 numbers.");
    }

    for (size_t i = 0; i < 16; ++i)
    {
        if (!value[i].is_number())
        {
            return fail(error, "Scene meta field '" + fieldName + "' contains a non-numeric component.");
        }

        double component = 0.0;
        try
        {
            component = value[i].get<double>();
        }
        catch (const std::exception&)
        {
            return fail(error, "Scene meta field '" + fieldName + "' contains an invalid numeric component.");
        }
        if (!std::isfinite(component) ||
            component < -static_cast<double>(std::numeric_limits<float>::max()) ||
            component > static_cast<double>(std::numeric_limits<float>::max()))
        {
            return fail(error, "Scene meta field '" + fieldName + "' contains a non-finite or out-of-range component.");
        }
        output[i] = static_cast<float>(component);
    }

    return true;
}

inline bool validateAffineTransform(
    const Transform16& transform,
    const std::string& fieldName,
    std::string& error
)
{
    constexpr double kAffineEpsilon = 1e-5;

    // An instance transform is affine. The final row must therefore be
    // [0, 0, 0, 1] in the row-major representation.
    if (std::abs(double(transform[12])) > kAffineEpsilon ||
        std::abs(double(transform[13])) > kAffineEpsilon ||
        std::abs(double(transform[14])) > kAffineEpsilon ||
        std::abs(double(transform[15]) - 1.0) > kAffineEpsilon)
    {
        return fail(error, "Scene meta field '" + fieldName + "' must be an affine matrix.");
    }

    // Reject singular transforms before RayMarchingPass computes the inverse.
    const double m00 = transform[0];
    const double m01 = transform[1];
    const double m02 = transform[2];
    const double m10 = transform[4];
    const double m11 = transform[5];
    const double m12 = transform[6];
    const double m20 = transform[8];
    const double m21 = transform[9];
    const double m22 = transform[10];
    const double determinant =
        m00 * (m11 * m22 - m12 * m21) -
        m01 * (m10 * m22 - m12 * m20) +
        m02 * (m10 * m21 - m11 * m20);

    if (!std::isfinite(determinant) || std::abs(determinant) <= 1e-12)
    {
        return fail(error, "Scene meta field '" + fieldName + "' must be invertible.");
    }

    return true;
}

inline std::filesystem::path resolveRelativePath(
    const std::filesystem::path& scenePath,
    const std::string& pathString
)
{
    std::filesystem::path path(pathString);
    if (path.is_relative())
        path = scenePath.parent_path() / path;
    return path.lexically_normal();
}

inline bool read(const std::filesystem::path& scenePath, Scene& scene, std::string& error)
{
    error.clear();

    std::ifstream input(scenePath);
    if (!input)
    {
        return fail(error, "Cannot open scene meta file: " + scenePath.string());
    }

    nlohmann::json document;
    try
    {
        // Allow comments so hand-authored scene files can contain notes.
        document = nlohmann::json::parse(
            input,
            nullptr /* callback */,
            true /* allow exceptions */,
            true /* ignore comments */
        );
    }
    catch (const std::exception& e)
    {
        return fail(error, "Failed to parse scene meta '" + scenePath.string() + "': " + e.what());
    }

    if (!document.is_object())
        return fail(error, "Scene meta root must be a JSON object.");

    if (!document.contains("format") || !document["format"].is_string() ||
        document["format"].get<std::string>() != kFormat)
    {
        return fail(error, "Scene meta has an invalid or missing 'format' field.");
    }

    uint32_t version = 0;
    if (!document.contains("version") ||
        !readUint32(document["version"], "version", version, error))
    {
        return false;
    }
    if (version != kCurrentVersion)
    {
        return fail(
            error,
            "Only scene meta version " + std::to_string(kCurrentVersion) + " is supported."
        );
    }

    if (!document.contains("assets") || !document["assets"].is_array() ||
        document["assets"].empty())
    {
        return fail(error, "Scene meta must contain a non-empty 'assets' array.");
    }
    if (document["assets"].size() > kMaxAssets)
    {
        return fail(error, "Scene meta contains more than " + std::to_string(kMaxAssets) + " assets.");
    }

    Scene parsed;
    parsed.sourcePath = scenePath;
    std::unordered_set<std::string> assetIds;
    for (size_t i = 0; i < document["assets"].size(); ++i)
    {
        const nlohmann::json& value = document["assets"][i];
        const std::string prefix = "assets[" + std::to_string(i) + "]";
        if (!value.is_object())
            return fail(error, prefix + " must be a JSON object.");
        if (!value.contains("id") || !value["id"].is_string() || value["id"].get<std::string>().empty())
            return fail(error, prefix + " must contain a non-empty string 'id'.");
        if (!value.contains("voxelFile") || !value["voxelFile"].is_string() ||
            value["voxelFile"].get<std::string>().empty())
        {
            return fail(error, prefix + " must contain a non-empty string 'voxelFile'.");
        }

        Asset asset;
        asset.assetId = value["id"].get<std::string>();
        if (!assetIds.insert(asset.assetId).second)
            return fail(error, prefix + ".id duplicates an earlier asset id.");

        asset.voxelFile = resolveRelativePath(scenePath, value["voxelFile"].get<std::string>());
        std::error_code fileEc;
        if (!std::filesystem::is_regular_file(asset.voxelFile, fileEc))
        {
            return fail(error, prefix + ".voxelFile does not exist or is not a regular file: " +
                                asset.voxelFile.string());
        }
        parsed.assets.push_back(std::move(asset));
    }
    parsed.assetId = parsed.assets.front().assetId;
    parsed.voxelFile = parsed.assets.front().voxelFile;

    if (!document.contains("instances") || !document["instances"].is_array() ||
        document["instances"].empty())
    {
        return fail(error, "Scene meta must contain a non-empty 'instances' array.");
    }
    if (document["instances"].size() > kMaxInstances)
    {
        return fail(error, "Scene meta contains more than " + std::to_string(kMaxInstances) + " instances.");
    }

    std::unordered_set<uint32_t> instanceIds;
    for (size_t i = 0; i < document["instances"].size(); ++i)
    {
        const nlohmann::json& value = document["instances"][i];
        const std::string prefix = "instances[" + std::to_string(i) + "]";
        if (!value.is_object())
            return fail(error, prefix + " must be a JSON object.");

        Instance instance;
        if (value.contains("id") &&
            !readUint32(value["id"], prefix + ".id", instance.instanceId, error))
        {
            return false;
        }
        else if (!value.contains("id"))
        {
            instance.instanceId = static_cast<uint32_t>(i);
        }

        if (!instanceIds.insert(instance.instanceId).second)
            return fail(error, prefix + " duplicates an earlier instance id.");

        if (!value.contains("asset"))
        {
            if (parsed.assets.size() != 1)
            {
                return fail(error, prefix + ".asset is required when the scene contains multiple assets.");
            }
            instance.assetIndex = 0;
        }
        else
        {
            if (!value["asset"].is_string())
                return fail(error, prefix + ".asset must be a string asset id.");

            const std::string assetId = value["asset"].get<std::string>();
            auto assetIt = std::find_if(
                parsed.assets.begin(),
                parsed.assets.end(),
                [&assetId](const Asset& asset) { return asset.assetId == assetId; }
            );
            if (assetIt == parsed.assets.end())
                return fail(error, prefix + ".asset references unknown asset '" + assetId + "'.");
            instance.assetIndex = static_cast<uint32_t>(std::distance(parsed.assets.begin(), assetIt));
        }

        if (value.contains("enabled"))
        {
            if (!value["enabled"].is_boolean())
                return fail(error, prefix + ".enabled must be a boolean.");
            instance.enabled = value["enabled"].get<bool>();
        }

        if (!value.contains("transform") ||
            !readFloat16(value["transform"], prefix + ".transform", instance.transform, error))
        {
            return false;
        }
        if (!validateAffineTransform(instance.transform, prefix + ".transform", error))
        {
            return false;
        }

        parsed.instances.push_back(instance);
    }

    scene = std::move(parsed);
    return true;
}

inline bool write(
    const std::filesystem::path& scenePath,
    const Scene& scene,
    std::string& error
)
{
    error.clear();

    if (scene.assets.empty())
        return fail(error, "Scene meta must contain at least one asset.");
    if (scene.assets.size() > kMaxAssets)
        return fail(error, "Scene meta contains more than " + std::to_string(kMaxAssets) + " assets.");
    if (scene.instances.empty())
        return fail(error, "Scene meta must contain at least one instance.");
    if (scene.instances.size() > kMaxInstances)
        return fail(error, "Scene meta contains more than " + std::to_string(kMaxInstances) + " instances.");

    nlohmann::json document = nlohmann::json::object();
    document["format"] = kFormat;
    document["version"] = kCurrentVersion;
    document["assets"] = nlohmann::json::array();
    document["instances"] = nlohmann::json::array();

    const std::filesystem::path sceneDirectory = scenePath.parent_path();
    for (size_t i = 0; i < scene.assets.size(); ++i)
    {
        const Asset& asset = scene.assets[i];
        if (asset.assetId.empty())
            return fail(error, "assets[" + std::to_string(i) + "] has an empty id.");
        if (asset.voxelFile.empty())
        {
            return fail(
                error,
                "assets[" + std::to_string(i) + "] has an empty voxelFile."
            );
        }

        std::filesystem::path voxelFile = asset.voxelFile;
        if (voxelFile.is_absolute())
        {
            std::error_code relativeEc;
            const std::filesystem::path relative =
                std::filesystem::relative(voxelFile, sceneDirectory, relativeEc);
            if (!relativeEc && !relative.empty())
                voxelFile = relative;
        }

        nlohmann::json assetValue = nlohmann::json::object();
        assetValue["id"] = asset.assetId;
        assetValue["voxelFile"] = voxelFile.generic_string();
        document["assets"].push_back(std::move(assetValue));
    }

    for (size_t i = 0; i < scene.instances.size(); ++i)
    {
        const Instance& instance = scene.instances[i];
        if (instance.assetIndex >= scene.assets.size())
        {
            return fail(
                error,
                "instances[" + std::to_string(i) + "].assetIndex is out of range."
            );
        }
        if (!validateAffineTransform(
                instance.transform,
                "instances[" + std::to_string(i) + "].transform",
                error))
        {
            return false;
        }

        nlohmann::json transform = nlohmann::json::array();
        for (float component : instance.transform)
            transform.push_back(component);

        nlohmann::json instanceValue = nlohmann::json::object();
        instanceValue["id"] = instance.instanceId;
        instanceValue["asset"] = scene.assets[instance.assetIndex].assetId;
        instanceValue["transform"] = std::move(transform);
        instanceValue["enabled"] = instance.enabled;
        document["instances"].push_back(std::move(instanceValue));
    }

    if (!sceneDirectory.empty())
    {
        std::error_code directoryEc;
        std::filesystem::create_directories(sceneDirectory, directoryEc);
        if (directoryEc)
        {
            return fail(
                error,
                "Cannot create scene meta directory '" + sceneDirectory.string() + "'."
            );
        }
    }

    std::ofstream output(scenePath, std::ios::trunc);
    if (!output)
        return fail(error, "Cannot create scene meta file: " + scenePath.string());

    output << document.dump(4) << '\n';
    if (!output)
        return fail(error, "Failed to write scene meta file: " + scenePath.string());

    return true;
}

} // namespace VoxelSceneMetadata
