#pragma once

#include "GridData.h"
#include "Polygon.h"
#include "PolygonSerializer.h"
#include "SceneLoader.h"
#include "TextureSampler.h"
#include "VoxelData.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Shared CPU-side data loading and single-node analysis used by both the
// voxelizer and VoxelizationInspector.  The library deliberately does not
// depend on OctreeBuilder or MergePhase: callers provide one node and its
// polygons, so the inspector never has to materialize the complete octree.
namespace VoxelizationCore
{

struct NodeRequest
{
    uint32_t level = 0;
    int3 cell = int3(0);
    uint64_t nodeKey = 0;
};

struct NodeData
{
    NodeRequest request;
    std::vector<Polygon> polygons;
    uint32_t storedPolygonCount = 0;
};

struct TextureSet
{
    const std::vector<Texture2D>& baseColor;
    const std::vector<Texture2D>& specular;
    const std::vector<Texture2D>& metallic;
    const std::vector<Texture2D>& normalMap;
};

struct AnalysisContext
{
    const InstancedScene& scene;
    const GridData& grid;
    uint32_t maxDepth = 0;
    const TextureSet& textures;
};

enum class SphericalFunctionType : uint32_t
{
    PrimitiveProjArea = 0,
    PolygonsProjArea = 1,
    TotalProjArea = 2,
};

struct AnalysisOptions
{
    uint32_t sampleFrequency = 1024;
    uint32_t samplesPerPolygon = 4;
    uint32_t sphericalResolution = 256;
    uint32_t splittingBlockCount = 8;
    uint32_t splittingBlockSize = 32;
    uint32_t ndfResolution = 256;
    uint32_t threadCount = 0; // 0 = hardware concurrency

    bool computeProjectionValidation = false;
    bool computeSphericalMaps = false;
    bool computeSplittingError = false;
    bool computeNdf = false;
};

struct ProjectionValidation
{
    float3 direction = float3(0);
    float exactVisibleArea = 0;
    float shVisibleArea = 0;
    float exactTotalArea = 0;
    float shTotalArea = 0;
};

struct FunctionMap
{
    // Spherical maps use an upper-hemisphere unit disk: width == height ==
    // spherical resolution, and pixels outside the disk remain zero.
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<float> exact;
    std::vector<float> approximation;
    std::vector<float> error;
};

struct Float4Map
{
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<float4> values;
};

struct AnalysisResult
{
    bool success = true;
    std::string error;

    VoxelData voxelData;
    std::vector<ProjectionValidation> projectionValidation;
    std::array<FunctionMap, 3> sphericalMaps;
    Float4Map splittingError;
    Float4Map ndf;
};

// Node-key helpers kept behind the shared library boundary.  The encoding is
// the same v2 encoding used by PolygonGenerator and the serialized index.
uint64_t makeNodeKey(uint32_t level, const int3& cell);
uint32_t levelFromNodeKey(uint64_t key);
int3 cellFromNodeKey(uint64_t key);

// Reads one selected level/node from nodes.idx/leaves.idx + polygons.dat.
// Index entries are sorted by nodeKey, so readNode() uses binary search and
// does not load the complete index into memory.
class TempNodeReader
{
public:
    bool open(const std::filesystem::path& indexPath,
              const std::filesystem::path& polygonsPath);

    bool readNode(uint32_t level, const int3& cell, NodeData& out);
    bool readNode(uint64_t nodeKey, NodeData& out);

    uint32_t maxDepth() const { return mHeader.maxDepth; }
    uint32_t targetLevel() const { return mHeader.reserved; }
    uint64_t nodeCount() const { return mHeader.leafCount; }
    const std::string& error() const { return mError; }

private:
    bool findNode(uint64_t nodeKey, PolygonSerializer::NodeIndex& out);

    std::filesystem::path mIndexPath;
    std::filesystem::path mPolygonsPath;
    PolygonSerializer::NodesIdxHeader mHeader{};
    bool mOpen = false;
    std::string mError;
};

// Reads only the GridData/maxDepth portion of a voxel output.  The complete
// VoxelData array is intentionally not loaded because this tool is designed
// for single-node diagnostics.
struct VoxelFileHeader
{
    GridData grid;
    uint32_t maxDepth = 0;
};

bool readVoxelFileHeader(const std::filesystem::path& binaryPath,
                         VoxelFileHeader& out,
                         std::string& error);

// Loads the same instanced scene representation and CPU texture set used by
// the voxelizer.  The returned textures remain owned by the caller.
bool loadSceneResources(const std::filesystem::path& scenePath,
                        InstancedScene& scene,
                        std::vector<Texture2D>& baseColorTextures,
                        std::vector<Texture2D>& specularTextures,
                        std::vector<Texture2D>& metallicTextures,
                        std::vector<Texture2D>& normalMapTextures,
                        std::string& error);

// Analyze exactly one clipped node.  VoxelData is always generated; the
// optional maps are controlled by AnalysisOptions.
AnalysisResult analyzeNode(const NodeData& node,
                           const AnalysisContext& context,
                           const AnalysisOptions& options = {});

// Directions used by VoxelizationPass's current SH validation UI.
std::vector<float3> defaultValidationDirections();

} // namespace VoxelizationCore
