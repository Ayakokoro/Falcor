#pragma once

// Scene composition metadata for the single-bin voxel ray marcher.
//
// This file deliberately does not replace VoxelizationMetadata.h. The latter
// describes the contents of one voxel binary and its LOD sidecar, while this
// file describes how that binary is placed in a scene.
//
// Version 1 scene files have the following shape:
// {
//   "format": "FalcorVoxelScene",
//   "version": 1,
//   "assets": [{ "id": "model", "voxelFile": "model.bin" }],
//   "instances": [{
//     "id": 0,
//     "asset": "model",
//     "translation": [0, 0, 0],
//     "rotationXYZDegrees": [0, 0, 0],
//     "scale": [1, 1, 1],
//     "pivot": "gridCenter",
//     "enabled": true
//   }]
// }

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <exception>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace VoxelSceneMetadata
{

inline constexpr const char* kFormat = "FalcorVoxelScene";
inline constexpr uint32_t kCurrentVersion = 1;
inline constexpr uint32_t kMaxInstances = 256;

struct Instance
{
    uint32_t instanceId = 0;
    bool enabled = true;
    std::array<float, 3> translation = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> rotationDegrees = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> scale = {1.0f, 1.0f, 1.0f};
};

struct Scene
{
    std::filesystem::path sourcePath;
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

inline bool readFloat3(
    const nlohmann::json& value,
    const std::string& fieldName,
    std::array<float, 3>& output,
    std::string& error
)
{
    if (!value.is_array() || value.size() != 3)
    {
        return fail(error, "Scene meta field '" + fieldName + "' must be an array of three numbers.");
    }

    float components[3] = {};
    for (size_t i = 0; i < 3; ++i)
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
        components[i] = static_cast<float>(component);
    }

    output = {components[0], components[1], components[2]};
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
    if (version == 0 || version > kCurrentVersion)
    {
        return fail(error, "Unsupported scene meta version " + std::to_string(version) + ".");
    }

    if (!document.contains("assets") || !document["assets"].is_array() ||
        document["assets"].empty())
    {
        return fail(error, "Scene meta must contain a non-empty 'assets' array.");
    }
    if (document["assets"].size() != 1)
    {
        return fail(error, "This RayMarchingPass version supports exactly one shared asset per scene meta file.");
    }

    const nlohmann::json& asset = document["assets"][0];
    if (!asset.is_object())
        return fail(error, "Each scene meta asset must be a JSON object.");
    if (!asset.contains("id") || !asset["id"].is_string() || asset["id"].get<std::string>().empty())
        return fail(error, "The scene meta asset must contain a non-empty string 'id'.");
    if (!asset.contains("voxelFile") || !asset["voxelFile"].is_string() ||
        asset["voxelFile"].get<std::string>().empty())
    {
        return fail(error, "The scene meta asset must contain a non-empty string 'voxelFile'.");
    }

    Scene parsed;
    parsed.sourcePath = scenePath;
    parsed.assetId = asset["id"].get<std::string>();
    parsed.voxelFile = resolveRelativePath(scenePath, asset["voxelFile"].get<std::string>());

    std::error_code fileEc;
    if (!std::filesystem::is_regular_file(parsed.voxelFile, fileEc))
    {
        return fail(error, "Scene meta voxel file does not exist or is not a regular file: " +
                            parsed.voxelFile.string());
    }

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

        if (value.contains("asset"))
        {
            if (!value["asset"].is_string() || value["asset"].get<std::string>() != parsed.assetId)
            {
                return fail(error, prefix + ".asset must reference the only asset '" + parsed.assetId + "'.");
            }
        }

        if (value.contains("enabled"))
        {
            if (!value["enabled"].is_boolean())
                return fail(error, prefix + ".enabled must be a boolean.");
            instance.enabled = value["enabled"].get<bool>();
        }

        if (value.contains("translation") &&
            !readFloat3(value["translation"], prefix + ".translation", instance.translation, error))
        {
            return false;
        }

        const char* rotationField = value.contains("rotationXYZDegrees")
            ? "rotationXYZDegrees"
            : (value.contains("rotationDegrees") ? "rotationDegrees" : nullptr);
        if (rotationField &&
            !readFloat3(value[rotationField], prefix + "." + rotationField, instance.rotationDegrees, error))
        {
            return false;
        }

        if (value.contains("scale") &&
            !readFloat3(value["scale"], prefix + ".scale", instance.scale, error))
        {
            return false;
        }
        if (instance.scale[0] <= 0.0f || instance.scale[1] <= 0.0f || instance.scale[2] <= 0.0f)
        {
            return fail(error, prefix + ".scale must contain only positive values.");
        }

        if (value.contains("pivot"))
        {
            if (!value["pivot"].is_string() || value["pivot"].get<std::string>() != "gridCenter")
            {
                return fail(error, prefix + ".pivot must be 'gridCenter' in this version.");
            }
        }

        parsed.instances.push_back(instance);
    }

    scene = std::move(parsed);
    return true;
}

} // namespace VoxelSceneMetadata
