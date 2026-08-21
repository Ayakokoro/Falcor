#pragma once
#include "Types.h"
#include "GridData.h"
#include "VoxelData.h"
#include "ABSDF.h"
#include "Ellipsoid.h"
#include "SphericalHarmonics.h"
#include "Estimate.h"
#include "TextureSampler.h"
#include "PolygonGenerator.h"
#include "SceneLoader.h"
#include "ClipPhase.h"
#include "MergePhase.h"
#include "AnalyzePhase.h"
#include "OctreeBuilder.h"
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <filesystem>

struct VoxelizationConfig {
    uint baseResolution = 512;     // N = 2^D
    uint sampleFrequency = 1024;   // rays per Lebedev direction
};

// CPU scene voxelization: instanced loading, single hierarchical clip pass,
// all-level polygon storage and independent per-node analysis.
// No tile subdivision: one full octree over the scene.
class SceneVoxelization {
public:
    SceneVoxelization(const VoxelizationConfig& cfg) : mConfig(cfg) {}

    // ---- Configuration setters for disk-backed pipeline ----
    void setTmpDir(const std::string& dir)   { mTmpDir = dir; }
    void setNumThreads(uint32_t n)          { mNumThreads = n; }
    void setKeepTemp(bool keep)             { mKeepTemp = keep; }

    // In-memory pipeline: hierarchical clip, per-node analysis, then BFS/output.
    bool process(const std::string& fbxPath, const std::string& outputPath) {
        // ---- Phase 0: Load scene (instanced mode: unique meshes + transforms) ----
        std::cout << "Loading (instanced): " << fbxPath << std::endl;
        SceneLoader loader;
        InstancedScene scene;
        if (!loader.loadMeshInstances(fbxPath, scene)) {
            std::cerr << "Failed to load: " << loader.getError() << std::endl;
            return false;
        }
        std::cout << "  Unique meshes: " << scene.meshes.size()
                  << "  Instances: " << scene.instances.size()
                  << "  Materials: " << scene.materials.size() << std::endl;

        // Compute grid from all instances' world-space bounds
        setupGrid(scene);

        uint resolution = std::max({mGrid.voxelCount.x, mGrid.voxelCount.y, mGrid.voxelCount.z});
        if (resolution > PolygonGenerator::NODE_KEY_MAX_RESOLUTION) {
            std::cerr << "Resolution " << resolution
                      << " exceeds the standalone node-key limit of "
                      << PolygonGenerator::NODE_KEY_MAX_RESOLUTION << "^3." << std::endl;
            return false;
        }
        mMaxDepth = 0;
        while ((1u << mMaxDepth) < resolution) mMaxDepth++;

        loadTextures(scene.materials);

        std::cout << "Grid: " << mGrid.voxelCount.x << "^3  voxelSize=" << mGrid.voxelSize.x
                  << "  maxDepth=" << mMaxDepth << std::endl;

        // ---- Phase 1: Hierarchical clip (all-level polygon storage) ----
        PolygonGenerator gen(mGrid);
        gen.reset();

        float3 invVoxelSize(1.0f / mGrid.voxelSize.x, 1.0f / mGrid.voxelSize.y, 1.0f / mGrid.voxelSize.z);

        for (uint instIdx = 0; instIdx < (uint)scene.instances.size(); instIdx++) {
            const MeshInstance& inst = scene.instances[instIdx];
            const MeshGeometry& mesh = scene.meshes[inst.meshID];
            const glm::mat4& worldM = inst.transform;
            uint matID = mesh.materialID;

            for (uint localTid = 0; localTid < (uint)mesh.triangles.size(); localTid++) {
                uint3 idx = mesh.triangles[localTid];

                Triangle tri;
                tri.vertices[0] = localToVoxel(mesh.positions[idx.x], worldM, mGrid.gridMin, invVoxelSize);
                tri.vertices[1] = localToVoxel(mesh.positions[idx.y], worldM, mGrid.gridMin, invVoxelSize);
                tri.vertices[2] = localToVoxel(mesh.positions[idx.z], worldM, mGrid.gridMin, invVoxelSize);
                tri.uvs[0] = mesh.texCoords[idx.x];
                tri.uvs[1] = mesh.texCoords[idx.y];
                tri.uvs[2] = mesh.texCoords[idx.z];
                tri.normals[0] = transformNormal(mesh.normals[idx.x], worldM);
                tri.normals[1] = transformNormal(mesh.normals[idx.y], worldM);
                tri.normals[2] = transformNormal(mesh.normals[idx.z], worldM);
                tri.buildTBN();

                clipHierarchical(inst.meshID, matID, localTid, instIdx,
                                tri, tri.calcAABBInt(),
                                int3(0, 0, 0), 0, mMaxDepth, gen);
            }
        }

        // ---- mem: after clip (before BFS, every occupied node has entries) ----
        {
            uint64_t totalPoly = 0;
            uint32_t maxPolyPerNode = 0;
            for (const auto& kv : gen.mNodePolygonMap) {
                uint n = (uint)kv.second.size();
                totalPoly += n;
                maxPolyPerNode = std::max(maxPolyPerNode, (uint32_t)n);
            }
            std::cout << "  [Mem] after-clip: nodes=" << gen.mNodePolygonMap.size()
                      << " totalPoly=" << totalPoly
                      << " maxPolyPerNode=" << maxPolyPerNode << std::endl;
        }

        gen.finalizeBFS(mMaxDepth, 0, int3(0, 0, 0));

        // ---- Phase 2: Analyze every occupied node from its own polygons ----
        AnalyzePhase::AnalyzeContext actx{
            scene, mGrid, mMaxDepth, mConfig.sampleFrequency,
            mBaseColorTextures, mSpecularTextures,
            mMetallicTextures, mNormalMapTextures
        };
        for (uint32_t nodeIdx = 0;
             nodeIdx < (uint32_t)gen.gBuffer.size(); ++nodeIdx) {
            AnalyzePhase::analyzeNode(gen, nodeIdx, actx);
        }

        // ---- Phase 3: Write output ----
        std::cout << "Writing output: " << outputPath << std::endl;
        writeOutput(outputPath, gen);

        gen.polygonArrays.clear();
        gen.polygonArrays.shrink_to_fit();
        gen.mNodePolygonMap.clear();
        gen.mOccupiedNodes.clear();

        std::cout << "Done." << std::endl;
        return true;
    }

    // ---- Disk-backed pipeline ----
    bool processDisk(const std::string& fbxPath, const std::string& outputPath);

private:
    VoxelizationConfig mConfig;
    GridData mGrid;
    uint mMaxDepth = 0;

    // Loaded textures: materialID -> texture
    std::vector<Texture2D> mBaseColorTextures;
    std::vector<Texture2D> mSpecularTextures;
    std::vector<Texture2D> mMetallicTextures;
    std::vector<Texture2D> mNormalMapTextures;

    // Disk-backed pipeline configuration
    std::string mTmpDir = "./tmp";
    uint32_t mNumThreads = 0;   // 0 = auto-detect
    bool mKeepTemp = true;      // default: keep temp files

    // ---- Transform helpers ----

    static inline float3 localToVoxel(const float3& localPos, const glm::mat4& worldM,
                                       const float3& gridMin, const float3& invVoxelSize) {
        glm::vec4 wp = worldM * glm::vec4(localPos, 1.0f);
        return float3(
            (wp.x - gridMin.x) * invVoxelSize.x,
            (wp.y - gridMin.y) * invVoxelSize.y,
            (wp.z - gridMin.z) * invVoxelSize.z
        );
    }

    static inline float3 transformNormal(const float3& localNormal, const glm::mat4& worldM) {
        glm::mat3 rot(worldM);
        glm::vec3 wn = rot * glm::vec3(localNormal.x, localNormal.y, localNormal.z);
        return safeNormalize(float3(wn.x, wn.y, wn.z));
    }

    // ---- Grid setup ----

    void setupGrid(const InstancedScene& scene) {
        float3 sceneMin(1e30f), sceneMax(-1e30f);

        for (const auto& inst : scene.instances) {
            const auto& mesh = scene.meshes[inst.meshID];
            float3 corners[8] = {
                float3(mesh.localMin.x, mesh.localMin.y, mesh.localMin.z),
                float3(mesh.localMin.x, mesh.localMin.y, mesh.localMax.z),
                float3(mesh.localMin.x, mesh.localMax.y, mesh.localMin.z),
                float3(mesh.localMin.x, mesh.localMax.y, mesh.localMax.z),
                float3(mesh.localMax.x, mesh.localMin.y, mesh.localMin.z),
                float3(mesh.localMax.x, mesh.localMin.y, mesh.localMax.z),
                float3(mesh.localMax.x, mesh.localMax.y, mesh.localMin.z),
                float3(mesh.localMax.x, mesh.localMax.y, mesh.localMax.z),
            };
            for (int c = 0; c < 8; c++) {
                glm::vec4 wp = inst.transform * glm::vec4(corners[c], 1.0f);
                sceneMin = glm::min(sceneMin, float3(wp.x, wp.y, wp.z));
                sceneMax = glm::max(sceneMax, float3(wp.x, wp.y, wp.z));
            }
        }

        float3 diag_scene = sceneMax - sceneMin;
        float3 diag = diag_scene * 1.02f;
        float3 center = (sceneMin + sceneMax) * 0.5f;

        std::cout << "  [Grid] bounds min=(" << sceneMin.x << "," << sceneMin.y << "," << sceneMin.z
                  << ") max=(" << sceneMax.x << "," << sceneMax.y << "," << sceneMax.z << ")" << std::endl;

        uint N = std::max(mConfig.baseResolution, 1u);
        N--; N |= N >> 1; N |= N >> 2; N |= N >> 4;
        N |= N >> 8; N |= N >> 16; N++;
        mGrid.voxelCount = uint3(N, N, N);

        float maxDim = std::max(diag.z, std::max(diag.x, diag.y));
        float s = maxDim / (float)N;
        mGrid.voxelSize = float3(s);
        mGrid.gridMin = center - 0.5f * s * float3((float)N);

        std::cout << "  [Grid] N=" << N << " maxDim=" << maxDim << " voxelSize=" << s << std::endl;
    }

    void loadTextures(const std::vector<MaterialData>& materials) {
        mBaseColorTextures.resize(materials.size());
        mSpecularTextures.resize(materials.size());
        mMetallicTextures.resize(materials.size());
        mNormalMapTextures.resize(materials.size());
        for (size_t i = 0; i < materials.size(); i++) {
            if (!materials[i].texBaseColor.empty())
                mBaseColorTextures[i].load(materials[i].texBaseColor, true);
            if (!materials[i].texSpecular.empty())
                mSpecularTextures[i].load(materials[i].texSpecular);
            if (!materials[i].texMetallic.empty())
                mMetallicTextures[i].load(materials[i].texMetallic);
            if (!materials[i].texNormalMap.empty())
                mNormalMapTextures[i].load(materials[i].texNormalMap);
        }
    }

    // ---- Hierarchical clip (all-level polygon storage) ----
    // Exact BoxClipTriangle at every level for correct AABB test.
    // Polygon vertices are normalized by the current node scale and retained
    // at every occupied level.

    static void clipHierarchical(
        uint meshID, uint materialID, uint triangleID, uint instanceIdx,
        Triangle& tri, const AABBInt& triAABB, const int3& nodeCell,
        uint level, uint maxDepth,
        PolygonGenerator& gen) {

        uint scale = 1u << (maxDepth - level);
        float3 minPoint = float3(nodeCell) * (float)scale;
        float3 maxPoint = minPoint + float3((float)scale);

        Polygon polygon = VoxelizationUtility::BoxClipTriangle(minPoint, maxPoint, tri);
        if (polygon.count < 3 || polygon.calcArea() <= 1e-8f) return;

        float invScale = 1.0f / (float)scale;
        for (uint vi = 0; vi < polygon.count; vi++)
            polygon.vertices[vi] *= invScale;

        polygon.normal = {tri.TBN[0].z, tri.TBN[1].z, tri.TBN[2].z};
        polygon.triRef.meshID = meshID;
        polygon.triRef.triangleID = triangleID;
        polygon.triRef.materialID = materialID;
        polygon.triRef.instanceIdx = instanceIdx;

        uint64_t nodeKey = PolygonGenerator::makeNodeKey(level, nodeCell);
        gen.mOccupiedNodes.insert(nodeKey);

        auto& polys = gen.mNodePolygonMap[nodeKey];
        if (polys.size() < SAFE_PER_NODE_POLYGON_LIMIT)
            polys.push_back(polygon);

        if (level >= maxDepth) return;

        int childScale = (int)(scale >> 1);
        for (uint ci = 0; ci < 8; ci++) {
            int3 childCell = nodeCell * 2 + int3((int)(ci & 1), (int)((ci >> 1) & 1), (int)((ci >> 2) & 1));
            int3 childMin = childCell * childScale;
            int3 childMax = childMin + childScale - 1;
            if (triAABB.xMax < childMin.x || triAABB.xMin > childMax.x ||
                triAABB.yMax < childMin.y || triAABB.yMin > childMax.y ||
                triAABB.zMax < childMin.z || triAABB.zMin > childMax.z)
                continue;
            clipHierarchical(meshID, materialID, triangleID, instanceIdx, tri, triAABB,
                            childCell, level + 1, maxDepth, gen);
        }
    }

    // ---- Output writer ----

    void writeOutput(const std::string& outputPath, const PolygonGenerator& gen) {
        std::ofstream f(outputPath, std::ios::binary);
        if (!f) {
            std::cerr << "Cannot open output: " << outputPath << std::endl;
            return;
        }

        uint totalNodes = (uint)gen.gBuffer.size();

        std::vector<uint32_t> nodeCounts(mMaxDepth + 1, 0);
        for (auto& bfn : gen.mBFSOrder)
            nodeCounts[bfn.level]++;

        uint totalPoly = 0, maxPolyPerNode = 0, nonEmptyNodes = 0;
        for (auto& pa : gen.polygonArrays) {
            uint pc = (uint)pa.size();
            totalPoly += pc;
            maxPolyPerNode = std::max(maxPolyPerNode, pc);
            if (pc > 0) nonEmptyNodes++;
        }
        std::cout << "  [Output] nodes=" << totalNodes << " totalPoly=" << totalPoly
                  << " maxPolyPerNode=" << maxPolyPerNode << std::endl;
        std::cout << "  [Output] per-level:";
        for (uint l = 0; l <= mMaxDepth; l++)
            std::cout << " L" << l << "=" << nodeCounts[l];
        std::cout << std::endl;

        // Consistency check
        {
            uint errors = 0;
            uint offset = 0;
            for (uint l = 0; l <= mMaxDepth; l++) {
                uint count = nodeCounts[l];
                uint nextOff = offset + count;
                for (uint i = offset; i < nextOff; i++) {
                    if (gen.mOctreeNodes[i].dataIndex != i) {
                        if (errors < 5) std::cerr << "  [CONSISTENCY] dataIndex mismatch at node[" << i << "]" << std::endl;
                        errors++;
                    }
                    if (gen.mOctreeNodes[i].childMask) {
                        uint childCount = countbits(gen.mOctreeNodes[i].childMask);
                        uint childStart = gen.mOctreeNodes[i].childBase;
                        uint childLevelOff = 0;
                        for (uint cl = 0; cl <= l; cl++) childLevelOff += nodeCounts[cl];
                        if (childStart < childLevelOff || childStart + childCount > childLevelOff + nodeCounts[l+1]) {
                            if (errors < 5) std::cerr << "  [CONSISTENCY] childBase OOB at node[" << i << "]" << std::endl;
                            errors++;
                        }
                    }
                }
                offset = nextOff;
            }
            if (errors == 0)
                std::cout << "  [CONSISTENCY] All checks passed." << std::endl;
            else
                std::cerr << "  [CONSISTENCY] " << errors << " errors found!" << std::endl;
        }

        GridData outGrid = mGrid;
        outGrid.solidVoxelCount = totalNodes;
        outGrid.maxPolygonCount = maxPolyPerNode;
        outGrid.totalPolygonCount = totalPoly;

        f.write((const char*)&outGrid, sizeof(GridData));
        f.write((const char*)&mMaxDepth, sizeof(uint32_t));
        f.write((const char*)nodeCounts.data(), (mMaxDepth + 1) * sizeof(uint32_t));
        f.write((const char*)gen.mOctreeNodes.data(), totalNodes * sizeof(OctreeNode));
        f.write((const char*)gen.gBuffer.data(), totalNodes * sizeof(VoxelData));

        f.close();
        std::cout << "Wrote " << totalNodes << " nodes." << std::endl;
    }

    // ---- Disk-backed output writer (works with OctreeResult) ----
    void writeOutputDisk(const std::string& outputPath,
                         const OctreeBuilder::OctreeResult& octree,
                         uint64_t totalPolygons,
                         uint32_t maxPolyPerNode) {
        std::ofstream f(outputPath, std::ios::binary);
        if (!f) {
            std::cerr << "Cannot open output: " << outputPath << std::endl;
            return;
        }

        uint32_t totalNodes = octree.totalNodes();

        std::cout << "  [Output] nodes=" << totalNodes << " totalPoly=" << totalPolygons
                  << " maxPolyPerNode=" << maxPolyPerNode << std::endl;
        std::cout << "  [Output] per-level:";
        for (uint32_t l = 0; l <= mMaxDepth; l++)
            std::cout << " L" << l << "=" << octree.levelNodeCounts[l];
        std::cout << std::endl;

        // Consistency check
        {
            uint32_t errors = 0;
            uint32_t offset = 0;
            for (uint32_t l = 0; l <= mMaxDepth; l++) {
                uint32_t count = octree.levelNodeCounts[l];
                uint32_t nextOff = offset + count;
                for (uint32_t i = offset; i < nextOff; i++) {
                    if (octree.octreeNodes[i].dataIndex != i) {
                        if (errors < 5) std::cerr << "  [CONSISTENCY] dataIndex mismatch at node[" << i << "]" << std::endl;
                        errors++;
                    }
                    if (octree.octreeNodes[i].childMask) {
                        uint32_t childCount = countbits(octree.octreeNodes[i].childMask);
                        uint32_t childStart = octree.octreeNodes[i].childBase;
                        uint32_t childLevelOff = 0;
                        for (uint32_t cl = 0; cl <= l; cl++) childLevelOff += octree.levelNodeCounts[cl];
                        if (childStart < childLevelOff || childStart + childCount > childLevelOff + octree.levelNodeCounts[l+1]) {
                            if (errors < 5) std::cerr << "  [CONSISTENCY] childBase OOB at node[" << i << "]" << std::endl;
                            errors++;
                        }
                    }
                }
                offset = nextOff;
            }
            if (errors == 0)
                std::cout << "  [CONSISTENCY] All checks passed." << std::endl;
            else
                std::cerr << "  [CONSISTENCY] " << errors << " errors found!" << std::endl;
        }

        GridData outGrid = mGrid;
        outGrid.solidVoxelCount = totalNodes;
        outGrid.maxPolygonCount = maxPolyPerNode;
        outGrid.totalPolygonCount = (uint32_t)totalPolygons;

        f.write((const char*)&outGrid, sizeof(GridData));
        f.write((const char*)&mMaxDepth, sizeof(uint32_t));
        f.write((const char*)octree.levelNodeCounts.data(), (mMaxDepth + 1) * sizeof(uint32_t));
        f.write((const char*)octree.octreeNodes.data(), totalNodes * sizeof(OctreeNode));
        f.write((const char*)octree.gBuffer.data(), totalNodes * sizeof(VoxelData));

        f.close();
        std::cout << "Wrote " << totalNodes << " nodes." << std::endl;
    }

};

// ---- processDisk: disk-backed pipeline implementation ----
inline bool SceneVoxelization::processDisk(
    const std::string& fbxPath, const std::string& outputPath)
{
    namespace fs = std::filesystem;

    // ---- Phase 0: Load scene (same as in-memory path) ----
    std::cout << "Loading (instanced): " << fbxPath << std::endl;
    SceneLoader loader;
    InstancedScene scene;
    if (!loader.loadMeshInstances(fbxPath, scene)) {
        std::cerr << "Failed to load: " << loader.getError() << std::endl;
        return false;
    }
    std::cout << "  Unique meshes: " << scene.meshes.size()
              << "  Instances: " << scene.instances.size()
              << "  Materials: " << scene.materials.size() << std::endl;

    setupGrid(scene);

    uint32_t resolution = std::max({mGrid.voxelCount.x, mGrid.voxelCount.y, mGrid.voxelCount.z});
    if (resolution > PolygonGenerator::NODE_KEY_MAX_RESOLUTION) {
        std::cerr << "Resolution " << resolution
                  << " exceeds the standalone node-key limit of "
                  << PolygonGenerator::NODE_KEY_MAX_RESOLUTION << "^3." << std::endl;
        return false;
    }
    mMaxDepth = 0;
    while ((1u << mMaxDepth) < resolution) mMaxDepth++;

    loadTextures(scene.materials);

    std::cout << "Grid: " << mGrid.voxelCount.x << "^3  voxelSize=" << mGrid.voxelSize.x
              << "  maxDepth=" << mMaxDepth << std::endl;

    // ---- Phase 1: Multi-threaded clip -> shard files ----
    std::cout << "\n=== Phase 1: Multi-threaded Clip ===" << std::endl;
    auto clipResult = ClipPhase::execute(scene, mGrid, mMaxDepth,
                                          mTmpDir, mNumThreads);
    if (clipResult.shardFiles.empty()) {
        std::cerr << "Clip phase produced no output." << std::endl;
        return false;
    }

    // ---- Phase 2: Merge shards -> nodes.idx + polygons.dat + octree ----
    std::cout << "\n=== Phase 2: Merge ===" << std::endl;
    auto mergeResult = MergePhase::execute(clipResult, mTmpDir, mMaxDepth);
    if (mergeResult.totalNodes == 0) {
        std::cerr << "Merge phase produced no occupied nodes." << std::endl;
        return false;
    }

    // ---- Phase 3: Stream-based per-node analysis ----
    std::cout << "\n=== Phase 3: Analyze ===" << std::endl;
    AnalyzePhase::AnalyzeContext actx{
        scene, mGrid, mMaxDepth, mConfig.sampleFrequency,
        mBaseColorTextures, mSpecularTextures, mMetallicTextures, mNormalMapTextures
    };
    AnalyzePhase::execute(mergeResult, actx, mNumThreads);

    // ---- Phase 4: Write output ----
    std::cout << "\n=== Phase 4: Write Output ===" << std::endl;
    const auto& octree = mergeResult.octree;
    // maxPolyPerNode: scan node indices for the max
    // (we could track this during merge, but scanning is fast)
    uint32_t maxPolyPerNode = 0;
    {
        std::ifstream idxIn(mergeResult.nodesIdxPath, std::ios::binary);
        PolygonSerializer::NodesIdxHeader idxHdr;
        if (PolygonSerializer::readNodesIdxHeader(idxIn, idxHdr)) {
            std::vector<PolygonSerializer::NodeIndex> nodes;
            PolygonSerializer::readNodeIndices(idxIn, nodes, idxHdr.nodeCount);
            for (auto& ni : nodes)
                maxPolyPerNode = std::max(maxPolyPerNode, ni.polyCount);
        }
    }
    writeOutputDisk(outputPath, octree, mergeResult.totalPolygons, maxPolyPerNode);

    // Clip shards already cleaned by MergePhase; merge data always kept.
    std::cout << "Merge data kept in: " << mTmpDir << "/merge" << std::endl;

    std::cout << "Done." << std::endl;
    return true;
}
