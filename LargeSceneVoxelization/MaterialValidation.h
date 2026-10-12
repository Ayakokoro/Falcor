#pragma once
#include "AnalyzePhase.h"
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

// Read-only verification of a single baked node. The reference deliberately
// uses the bake's per-polygon texture filtering, not per-pixel texture samples.
namespace MaterialValidation {

struct Options {
    std::string voxelPath;
    std::string tmpDir = "./tmp";
    std::string reportPath;
    uint32_t lod = 0; // Renderer convention: finest leaf = 0.
    int3 cell = int3(0); // Coordinates at the selected LOD, not leaf-grid offsets.
    float3 view = float3(0); // Node -> observer, in world-space axes.
};

inline bool finite(const float3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

template<typename T>
inline void read(std::istream& in, T& value) {
    if (!in.read(reinterpret_cast<char*>(&value), sizeof(T)))
        throw std::runtime_error("Truncated or unreadable binary data.");
}

inline uint64_t fileSize(std::ifstream& in) {
    in.seekg(0, std::ios::end);
    auto size = in.tellg();
    if (size < 0) throw std::runtime_error("Cannot determine binary file size.");
    in.seekg(0);
    return static_cast<uint64_t>(size);
}

struct BakedNode {
    GridData grid;
    uint32_t maxDepth = 0;
    uint32_t level = 0;
    uint32_t dataIndex = 0;
    VoxelData data;
};

inline BakedNode readBakedNode(const Options& options) {
    std::ifstream in(options.voxelPath, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open voxel file: " + options.voxelPath);
    const uint64_t size = fileSize(in);
    BakedNode result;
    read(in, result.grid);
    read(in, result.maxDepth);
    if (result.maxDepth > PolygonGenerator::NODE_KEY_MAX_DEPTH || options.lod > result.maxDepth)
        throw std::runtime_error("Invalid bake depth or requested LOD.");
    const uint32_t resolution = 1u << result.maxDepth;
    if (result.grid.voxelCount != uint3(resolution) || !finite(result.grid.gridMin) ||
        !finite(result.grid.voxelSize) || glm::any(glm::lessThanEqual(result.grid.voxelSize, float3(0))))
        throw std::runtime_error("Invalid grid in voxel file.");
    result.level = result.maxDepth - options.lod;
    const int cells = 1 << result.level;
    if (glm::any(glm::lessThan(options.cell, int3(0))) || glm::any(glm::greaterThanEqual(options.cell, int3(cells))))
        throw std::runtime_error("--cell is outside the selected LOD grid.");

    uint64_t nodeCount = 0;
    std::vector<uint32_t> counts(result.maxDepth + 1);
    for (auto& count : counts) { read(in, count); nodeCount += count; }
    if (counts[0] != 1 || nodeCount != result.grid.solidVoxelCount || nodeCount > UINT32_MAX)
        throw std::runtime_error("Invalid node counts in voxel file.");
    const uint64_t octreeOffset = static_cast<uint64_t>(in.tellg());
    const uint64_t dataOffset = octreeOffset + nodeCount * sizeof(OctreeNode);
    const uint64_t tagOffset = dataOffset + nodeCount * sizeof(VoxelData);
    if (size != tagOffset + sizeof(uint32_t))
        throw std::runtime_error("Voxel layout mismatch; use a current VPW3 bake and the same build architecture.");
    in.seekg(static_cast<std::streamoff>(tagOffset));
    uint32_t tag;
    read(in, tag);
    if (tag != 0x33575056u) throw std::runtime_error("Unsupported voxel bake tag; expected VPW3.");

    // Traverse the stored tree, rather than assuming nodes.idx is in BFS order.
    uint32_t index = 0;
    uint64_t levelStart = 0;
    for (uint32_t level = 0; ; ++level) {
        if (index < levelStart || index >= levelStart + counts[level])
            throw std::runtime_error("Invalid octree child index.");
        in.seekg(static_cast<std::streamoff>(octreeOffset + uint64_t(index) * sizeof(OctreeNode)));
        OctreeNode node;
        read(in, node);
        if (node.childMask & ~255u) throw std::runtime_error("Invalid octree child mask.");
        if (level == result.level) {
            if (node.dataIndex >= nodeCount) throw std::runtime_error("Invalid voxel data index.");
            result.dataIndex = node.dataIndex;
            break;
        }
        const uint32_t shift = result.level - level - 1;
        const uint32_t child = ((options.cell.x >> shift) & 1) |
                               (((options.cell.y >> shift) & 1) << 1) |
                               (((options.cell.z >> shift) & 1) << 2);
        const uint32_t bit = 1u << child;
        if (!(node.childMask & bit)) throw std::runtime_error("Requested node is empty in the baked octree.");
        const uint64_t next = uint64_t(node.childBase) + countbits(node.childMask & (bit - 1));
        if (next >= nodeCount) throw std::runtime_error("Invalid octree child range.");
        index = static_cast<uint32_t>(next);
        levelStart += counts[level];
    }
    in.seekg(static_cast<std::streamoff>(dataOffset + uint64_t(result.dataIndex) * sizeof(VoxelData)));
    read(in, result.data);
    if (!result.data.isSolid()) throw std::runtime_error("Requested node has no solid material data.");
    return result;
}

inline std::vector<Polygon> readPolygons(const Options& options, const BakedNode& baked) {
    const auto mergeDir = std::filesystem::path(options.tmpDir) / "merge";
    std::ifstream indexIn(mergeDir / "nodes.idx", std::ios::binary);
    if (!indexIn) throw std::runtime_error("Cannot open nodes.idx under " + mergeDir.string());
    const uint64_t indexSize = fileSize(indexIn);
    PolygonSerializer::NodesIdxHeader header;
    if (!PolygonSerializer::readNodesIdxHeader(indexIn, header) || header.maxDepth != baked.maxDepth ||
        header.nodeCount != baked.grid.solidVoxelCount ||
        indexSize != sizeof(header) + header.nodeCount * sizeof(PolygonSerializer::NodeIndex))
        throw std::runtime_error("nodes.idx does not match the bake's depth/node count or has an invalid layout.");
    const uint64_t key = PolygonGenerator::makeNodeKey(baked.level, options.cell);
    PolygonSerializer::NodeIndex target;
    bool found = false;
    uint64_t totalPolygons = 0;
    uint64_t previousKey = 0;
    uint64_t previousOffset = 0;
    uint64_t blockEnd = 0;
    for (uint64_t i = 0; i < header.nodeCount; ++i) {
        PolygonSerializer::NodeIndex entry;
        read(indexIn, entry);
        if (i && entry.nodeKey <= previousKey) throw std::runtime_error("nodes.idx keys are not unique and sorted.");
        if (!entry.polyCount || (i && entry.dataOffset <= previousOffset) || (!i && entry.dataOffset != 0))
            throw std::runtime_error("Invalid nodes.idx polygon counts/offsets.");
        if (found && !blockEnd) blockEnd = entry.dataOffset;
        previousKey = entry.nodeKey;
        previousOffset = entry.dataOffset;
        totalPolygons += entry.polyCount;
        if (entry.nodeKey == key) { target = entry; found = true; }
    }
    if (static_cast<uint32_t>(totalPolygons) != baked.grid.totalPolygonCount)
        throw std::runtime_error("Polygon cache count does not match the voxel bake.");
    if (!found || !target.polyCount) throw std::runtime_error("Requested node has no cached polygons.");

    std::ifstream in(mergeDir / "polygons.dat", std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open polygons.dat under " + mergeDir.string());
    const uint64_t size = fileSize(in);
    if (!blockEnd) blockEnd = size;
    constexpr uint64_t minPolygonBytes = 5 * sizeof(uint32_t) + 4 * sizeof(float3);
    if (blockEnd > size || target.dataOffset > blockEnd ||
        target.polyCount > (blockEnd - target.dataOffset) / minPolygonBytes)
        throw std::runtime_error("Cached polygon offset/count is outside polygons.dat.");
    in.seekg(static_cast<std::streamoff>(target.dataOffset));
    std::vector<Polygon> polygons(target.polyCount);
    for (auto& p : polygons) {
        // Validate count before reading vertices: readPolygon() itself does not
        // bounds-check the fixed-size vertex array when given corrupt input.
        read(in, p.triRef.meshID); read(in, p.triRef.triangleID);
        read(in, p.triRef.materialID); read(in, p.triRef.instanceIdx);
        read(in, p.count); read(in, p.normal);
        if (p.count < 3 || p.count > MAX_VERTEX_COUNT || !finite(p.normal))
            throw std::runtime_error("Invalid cached polygon.");
        for (uint32_t v = 0; v < p.count; ++v) {
            read(in, p.vertices[v]);
            const float3 local = p.vertices[v] - float3(options.cell);
            if (!finite(local) || glm::any(glm::lessThan(local, float3(-1e-3f))) ||
                glm::any(glm::greaterThan(local, float3(1.001f))))
                throw std::runtime_error("Cached polygon vertices do not belong to the requested node.");
        }
        if (static_cast<uint64_t>(in.tellg()) > blockEnd)
            throw std::runtime_error("Cached polygon overlaps the next node block.");
    }
    if (static_cast<uint64_t>(in.tellg()) != blockEnd)
        throw std::runtime_error("Cached node block size does not match its polygon count.");
    return polygons;
}

struct MaterialMean {
    float3 diffuse = float3(0), specular = float3(0);
    float rough = 0;
    void add(const ABSDFLobe& lobe, float weight) {
        diffuse += weight * lobe.diffuse;
        specular += weight * lobe.specular;
        rough += weight * lobe.rough;
    }
};

struct Report {
    uint32_t visiblePixels = 0;
    uint32_t visiblePolygons = 0;
    bool aggregateValid = false;
    MaterialMean reference, aggregate;
    float referenceWeights[LOBE_COUNT] = {};
    float aggregateWeights[LOBE_COUNT] = {};
};

inline Report compare(const Options& options, const BakedNode& baked,
                      const std::vector<Polygon>& polygons, const InstancedScene& scene) {
    std::vector<bool> usedMaterials(scene.materials.size(), false);
    for (const auto& p : polygons) {
        const auto& r = p.triRef;
        if (r.meshID >= scene.meshes.size() || r.instanceIdx >= scene.instances.size() ||
            r.materialID >= scene.materials.size() || scene.instances[r.instanceIdx].meshID != r.meshID)
            throw std::runtime_error("Cached mesh/instance/material references do not match the source model.");
        const auto& mesh = scene.meshes[r.meshID];
        if (r.triangleID >= mesh.triangles.size() || mesh.materialID != r.materialID)
            throw std::runtime_error("Cached triangle references do not match the source model.");
        const uint3 triangle = mesh.triangles[r.triangleID];
        for (int v = 0; v < 3; ++v)
            if (triangle[v] >= mesh.positions.size() || triangle[v] >= mesh.normals.size() || triangle[v] >= mesh.texCoords.size())
                throw std::runtime_error("Source triangle has invalid vertex attributes.");
        usedMaterials[r.materialID] = true;
    }
    std::vector<Texture2D> base(scene.materials.size()), spec(scene.materials.size()),
                           metal(scene.materials.size()), normal(scene.materials.size());
    auto loadTexture = [](Texture2D& texture, const std::string& path, bool srgb = false) {
        if (!path.empty() && !texture.load(path, srgb))
            throw std::runtime_error("Cannot load reference texture: " + path);
    };
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        if (!usedMaterials[i]) continue;
        const auto& m = scene.materials[i];
        loadTexture(base[i], m.texBaseColor, true); loadTexture(spec[i], m.texSpecular);
        loadTexture(metal[i], m.texMetallic); loadTexture(normal[i], m.texNormalMap);
    }
    AnalyzePhase::AnalyzeContext context{scene, baked.grid, baked.maxDepth, 0, base, spec, metal, normal};
    const float scale = static_cast<float>(1u << options.lod);
    const float3 invVoxelSize = float3(1.0f) / baked.grid.voxelSize;
    const float3 view = safeNormalize(options.view);
    std::vector<ABSDFLobe> samples;
    samples.reserve(polygons.size());
    for (const auto& polygon : polygons) {
        ABSDFInput input = AnalyzePhase::sampleMaterial(polygon, context, invVoxelSize, scale);
        if (!finite(input.normal) || dot(input.normal, input.normal) <= 0 || !finite(input.baseColor) ||
            !std::isfinite(input.specular.g) || !std::isfinite(input.specular.b))
            throw std::runtime_error("Reference material contains invalid values.");
        // Match the bake's explicit front/back material lobes.
        if (dot(input.normal, view) < 0) input.normal = -input.normal;
        input.projArea = 1.0f;
        ABSDFLobe sample;
        sample.accumulate(input);
        sample.normalizeSelf();
        samples.push_back(sample);
    }

    PolygonRange range;
    range.cellInt = options.cell;
    range.nodeScale = scale;
    range.count = static_cast<uint32_t>(polygons.size());
    std::vector<float> depths(PROJ_RES * PROJ_RES, std::numeric_limits<float>::max());
    std::vector<uint32_t> ids(PROJ_RES * PROJ_RES, UINT32_MAX);
    range.rasterizeDepth(polygons, view, depths.data(), ids.data());
    Report result;
    std::vector<uint32_t> hits(polygons.size(), 0);
    for (uint32_t id : ids) {
        if (id == UINT32_MAX) continue;
        if (id >= samples.size()) throw std::runtime_error("Invalid raster owner index.");
        ++hits[id]; ++result.visiblePixels;
    }
    if (result.visiblePixels) {
        for (size_t p = 0; p < samples.size(); ++p) {
            if (!hits[p]) continue;
            ++result.visiblePolygons;
            const float weight = float(hits[p]) / float(result.visiblePixels);
            result.reference.add(samples[p], weight);
            result.referenceWeights[NormalIndex8(samples[p].normal)] += weight;
        }
    }
    float frontSum = 0;
    for (uint32_t k = 0; k < LOBE_COUNT; ++k) {
        const auto& lobe = baked.data.ABSDF.lobes[k];
        if (!std::isfinite(lobe.weight) || !finite(lobe.normal) || !finite(lobe.diffuse) ||
            !finite(lobe.specular) || !std::isfinite(lobe.rough))
            throw std::runtime_error("Baked lobe contains invalid values.");
        if (lobe.isValid()) {
            result.aggregateWeights[k] = lobe.weight * std::max(0.0f, dot(lobe.normal, view));
            frontSum += result.aggregateWeights[k];
        }
    }
    result.aggregateValid = frontSum > 0;
    if (result.aggregateValid) {
        for (uint32_t k = 0; k < LOBE_COUNT; ++k) {
            result.aggregateWeights[k] /= frontSum;
            result.aggregate.add(baked.data.ABSDF.lobes[k], result.aggregateWeights[k]);
        }
    }
    return result;
}

inline void writeReport(std::ostream& out, const std::string& sourcePath, const Options& options, const BakedNode& baked,
                        size_t polygonCount, const Report& report, bool csv) {
    out << std::setprecision(9);
    if (csv) out << "category,name,component,reference,aggregate,absolute_error\n";
    else out << "\n=== Directional Material Validation ===\n"
             << "Linear RGB; bake-filtered polygon reference; no Coverage/Visibility/shadows.\n";
    auto row = [&](const std::string& category, const std::string& name, const std::string& component,
                   const std::string& reference, const std::string& aggregate, const std::string& error) {
        auto field = [](const std::string& value) {
            if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
            std::string escaped = "\"";
            for (char c : value) { if (c == '"') escaped += '"'; escaped += c; }
            return escaped + '"';
        };
        if (csv) out << field(category) << ',' << field(name) << ',' << field(component) << ','
                     << field(reference) << ',' << field(aggregate) << ',' << field(error) << '\n';
        else out << category << " " << name << (component.empty() ? "" : "." + component)
                 << " | reference=" << (reference.empty() ? "N/A" : reference)
                 << " | aggregate=" << (aggregate.empty() ? "N/A" : aggregate)
                 << " | abs_error=" << (error.empty() ? "N/A" : error) << '\n';
    };
    auto number = [](double value) { std::ostringstream s; s << std::setprecision(9) << value; return s.str(); };
    auto metadata = [&](const std::string& name, double value) { row("metadata", name, "", number(value), "", ""); };
    row("metadata", "source_path", "", sourcePath, "", "");
    row("metadata", "voxel_path", "", options.voxelPath, "", "");
    row("metadata", "cache_directory", "", (std::filesystem::path(options.tmpDir) / "merge").string(), "", "");
    metadata("lod_leaf_zero", options.lod); metadata("level_root_zero", baked.level);
    metadata("cell_x", options.cell.x); metadata("cell_y", options.cell.y); metadata("cell_z", options.cell.z);
    metadata("gb_data_index", baked.dataIndex); metadata("polygon_count", static_cast<double>(polygonCount));
    const float3 view = safeNormalize(options.view);
    metadata("view_x", view.x); metadata("view_y", view.y); metadata("view_z", view.z);
    metadata("projection_resolution", PROJ_RES); metadata("visible_pixels", report.visiblePixels);
    metadata("visible_polygons", report.visiblePolygons);
    metadata("visible_projected_area_node_units", report.visiblePixels * PROJ_PIXEL_SIZE * PROJ_PIXEL_SIZE);
    metadata("reference_valid", report.visiblePixels != 0); metadata("aggregate_valid", report.aggregateValid);
    auto metric = [&](const std::string& name, const std::string& component, float reference, float aggregate) {
        const bool refValid = report.visiblePixels != 0;
        row("material", name, component, refValid ? number(reference) : "",
            report.aggregateValid ? number(aggregate) : "",
            refValid && report.aggregateValid ? number(std::abs(aggregate - reference)) : "");
    };
    for (int c = 0; c < 3; ++c) {
        const std::string component(1, "rgb"[c]);
        metric("diffuse", component, report.reference.diffuse[c], report.aggregate.diffuse[c]);
        metric("specular_F0", component, report.reference.specular[c], report.aggregate.specular[c]);
    }
    metric("roughness", "", report.reference.rough, report.aggregate.rough);
    for (uint32_t k = 0; k < LOBE_COUNT; ++k) {
        const auto& lobe = baked.data.ABSDF.lobes[k];
        const std::string name = "lobe_" + std::to_string(k);
        row("weight", name, "", report.visiblePixels ? number(report.referenceWeights[k]) : "",
            report.aggregateValid ? number(report.aggregateWeights[k]) : "",
            report.visiblePixels && report.aggregateValid ? number(std::abs(report.aggregateWeights[k] - report.referenceWeights[k])) : "");
        row("stored", name, "amplitude", "", number(lobe.weight), "");
        row("stored", name, "roughness", "", number(lobe.rough), "");
        for (int c = 0; c < 3; ++c) {
            row("stored", name, std::string("normal_") + "xyz"[c], "", number(lobe.normal[c]), "");
            row("stored", name, std::string("diffuse_") + "rgb"[c], "", number(lobe.diffuse[c]), "");
            row("stored", name, std::string("F0_") + "rgb"[c], "", number(lobe.specular[c]), "");
        }
    }
}

inline bool execute(const std::string& sourcePath, const Options& options) {
    try {
        if (!finite(options.view) || length(options.view) <= 0 || !std::isfinite(length(options.view)))
            throw std::runtime_error("--view must be a finite, nonzero node-to-observer direction.");
        if (!options.reportPath.empty() && std::filesystem::exists(options.reportPath))
            throw std::runtime_error("Report already exists; choose a new --report path (inputs are never overwritten).");
        const BakedNode baked = readBakedNode(options);
        const auto polygons = readPolygons(options, baked);
        SceneLoader loader;
        InstancedScene scene;
        if (!loader.loadMeshInstances(sourcePath, scene)) throw std::runtime_error(loader.getError());
        std::cout << "Source: " << sourcePath << "\nBake: " << options.voxelPath
                  << "\nCache: " << (std::filesystem::path(options.tmpDir) / "merge").string() << '\n'
                  << "WARNING: use the unchanged source and cache from this bake; legacy caches have no source fingerprint.\n";
        const Report report = compare(options, baked, polygons, scene);
        writeReport(std::cout, sourcePath, options, baked, polygons.size(), report, false);
        if (!options.reportPath.empty()) {
            std::ofstream out(options.reportPath);
            if (!out) throw std::runtime_error("Cannot create report: " + options.reportPath);
            writeReport(out, sourcePath, options, baked, polygons.size(), report, true);
            out.flush();
            if (!out) throw std::runtime_error("Failed to write report: " + options.reportPath);
            std::cout << "CSV report: " << options.reportPath << '\n';
        }
        if (!report.visiblePixels || !report.aggregateValid)
            std::cout << "No valid directional comparison; affected fields are N/A, not zero-valued materials.\n";
        return true;
    } catch (const std::exception& error) {
        std::cerr << "[MaterialValidation] " << error.what() << '\n';
        return false;
    }
}

} // namespace MaterialValidation
