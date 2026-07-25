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
#include <vector>
#include <memory>
#include <map>
#include <thread>
#include <atomic>
#include <fstream>
#include <iostream>

struct TiledVoxelizationConfig {
    uint baseResolution = 512;     // N = 2^D
    uint tileLevel = 3;            // L: grid split into 2^L × 2^L × 2^L tiles
    uint sampleFrequency = 1024;   // rays per Lebedev direction
    uint maxThreads = 0;           // 0 = auto-detect
};

struct TileTask {
    int3 tileCoord;
    uint tileIndex;
    std::vector<uint> triangleIndices;  // global triangle indices assigned to this tile
};

// CPU tiled voxelization orchestrator
class TiledVoxelization {
public:
    TiledVoxelization(const TiledVoxelizationConfig& cfg) : mConfig(cfg) {}

    // Main entry point: load FBX, voxelize, write binary output
    bool process(const std::string& fbxPath, const std::string& outputPath) {
        // ---- Load scene ----
        std::cout << "Loading: " << fbxPath << std::endl;
        SceneLoader loader;
        LoadedScene scene;
        if (!loader.load(fbxPath, scene)) {
            std::cerr << "Failed to load: " << loader.getError() << std::endl;
            return false;
        }
        std::cout << "  Vertices: " << scene.totalVertices()
                  << "  Triangles: " << scene.totalTriangles()
                  << "  Meshes: " << scene.meshes.size() << std::endl;
        std::cout << "  Bounds: " << scene.sceneMin.x << "," << scene.sceneMin.y << "," << scene.sceneMin.z
                  << "  to  " << scene.sceneMax.x << "," << scene.sceneMax.y << "," << scene.sceneMax.z << std::endl;

        // ---- Setup grid ----
        setupGrid(scene);

        // ---- Compute max octree depth ----
        uint resolution = std::max({mGrid.voxelCount.x, mGrid.voxelCount.y, mGrid.voxelCount.z});
        mMaxDepth = 0;
        while ((1u << mMaxDepth) < resolution) mMaxDepth++;

        // ---- Load textures ----
        loadTextures(scene);

        std::cout << "Grid: " << mGrid.voxelCount.x << "^3  voxelSize=" << mGrid.voxelSize.x
                  << "  tileLevel=" << mConfig.tileLevel
                  << "  tiles=" << (1u << mConfig.tileLevel) * (1u << mConfig.tileLevel) * (1u << mConfig.tileLevel)
                  << "  maxDepth=" << mMaxDepth << std::endl;

        // ---- Phase 1: Coarse tile assignment ----
        auto tiles = assignTiles(scene);

        // ---- Phase 2: Per-tile process (parallel) ----
        uint numThreads = mConfig.maxThreads > 0 ? mConfig.maxThreads
                                                   : std::max(1u, std::thread::hardware_concurrency());
        std::cout << "Processing " << tiles.size() << " tiles with " << numThreads << " threads..." << std::endl;

        mTileResults.resize(tiles.size());
        std::atomic<uint> completedTiles{0};

        auto worker = [&](uint startTile, uint endTile) {
            for (uint t = startTile; t < endTile; t++) {
                processTile(tiles[t], t, scene);
                uint done = ++completedTiles;
                if (done % 10 == 0 || done == tiles.size())
                    std::cout << "  [" << done << "/" << tiles.size() << "] tiles done" << std::endl;
            }
        };

        uint tilesPerThread = ((uint)tiles.size() + numThreads - 1) / numThreads;
        std::vector<std::thread> threads;
        for (uint w = 0; w < numThreads && w * tilesPerThread < tiles.size(); w++) {
            uint start = w * tilesPerThread;
            uint end = std::min(start + tilesPerThread, (uint)tiles.size());
            threads.emplace_back(worker, start, end);
        }
        for (auto& th : threads) th.join();

        // ---- Phase 3: Merge + write ----
        std::cout << "Merging and writing output..." << std::endl;
        mergeAndWrite(outputPath);

        std::cout << "Done. Output: " << outputPath << std::endl;
        return true;
    }

private:
    TiledVoxelizationConfig mConfig;
    GridData mGrid;
    uint mMaxDepth = 0;

    // Loaded textures: materialID → texture
    std::vector<Texture2D> mBaseColorTextures;
    std::vector<Texture2D> mSpecularTextures;
    std::vector<Texture2D> mNormalMapTextures;

    // Per-tile results
    struct TileResult {
        std::unique_ptr<PolygonGenerator> generator;
    };
    std::vector<TileResult> mTileResults;

    void setupGrid(const LoadedScene& scene) {
        float3 diag_scene = scene.sceneMax - scene.sceneMin;
        float3 diag = diag_scene * 1.02f;
        float3 center = (scene.sceneMin + scene.sceneMax) * 0.5f;

        std::cout << "  [GridDebug] sceneDiag: " << diag_scene.x << "," << diag_scene.y << "," << diag_scene.z << std::endl;
        std::cout << "  [GridDebug] diag*1.02: " << diag.x << "," << diag.y << "," << diag.z << std::endl;
        std::cout << "  [GridDebug] center: " << center.x << "," << center.y << "," << center.z << std::endl;

        uint N = mConfig.baseResolution;
        // Round up to power of two
        N--;
        N |= N >> 1; N |= N >> 2; N |= N >> 4;
        N |= N >> 8; N |= N >> 16; N++;
        mGrid.voxelCount = uint3(N, N, N);

        float maxDim = std::max(diag.z, std::max(diag.x, diag.y));
        float s = maxDim / (float)N;
        mGrid.voxelSize = float3(s);
        mGrid.gridMin = center - 0.5f * s * float3(N);

        std::cout << "  [GridDebug] N=" << N << " maxDim=" << maxDim << " s=" << s << std::endl;
        std::cout << "  [GridDebug] gridMin: " << mGrid.gridMin.x << "," << mGrid.gridMin.y << "," << mGrid.gridMin.z << std::endl;
        std::cout << "  [GridDebug] gridMax: " << (mGrid.gridMin.x + s*N)
                  << "," << (mGrid.gridMin.y + s*N) << "," << (mGrid.gridMin.z + s*N) << std::endl;
    }

    void loadTextures(const LoadedScene& scene) {
        mBaseColorTextures.resize(scene.materials.size());
        mSpecularTextures.resize(scene.materials.size());
        mNormalMapTextures.resize(scene.materials.size());
        for (size_t i = 0; i < scene.materials.size(); i++) {
            if (!scene.materials[i].texBaseColor.empty()) {
                if (mBaseColorTextures[i].load(scene.materials[i].texBaseColor, true))
                    std::cout << "  [Tex] baseColor: " << scene.materials[i].texBaseColor << std::endl;
            }
            if (!scene.materials[i].texSpecular.empty()) {
                if (mSpecularTextures[i].load(scene.materials[i].texSpecular))
                    std::cout << "  [Tex] specular: " << scene.materials[i].texSpecular << std::endl;
            }
            if (!scene.materials[i].texNormalMap.empty()) {
                if (mNormalMapTextures[i].load(scene.materials[i].texNormalMap))
                    std::cout << "  [Tex] normal: " << scene.materials[i].texNormalMap << std::endl;
            }
        }
    }

    std::vector<TileTask> assignTiles(const LoadedScene& scene) {
        uint tileCount = 1u << mConfig.tileLevel;
        uint tilesPerAxis = 1u << mConfig.tileLevel;
        uint maxDepth = mMaxDepth;

        std::vector<TileTask> allTiles;
        for (uint i = 0; i < tileCount * tileCount * tileCount; i++) {
            TileTask task;
            task.tileIndex = i;
            task.tileCoord = int3(
                (int)(i % tilesPerAxis),
                (int)((i / tilesPerAxis) % tilesPerAxis),
                (int)(i / (tilesPerAxis * tilesPerAxis))
            );
            allTiles.push_back(task);
        }

        // For each triangle, find overlapping tiles
        // Triangle vertices are already in world-space → convert to voxel space
        for (uint tid = 0; tid < scene.totalTriangles(); tid++) {
            uint3 indices = scene.triangles[tid];
            float3 v0 = scene.positions[indices.x];
            float3 v1 = scene.positions[indices.y];
            float3 v2 = scene.positions[indices.z];

            // Convert to voxel coordinates
            auto toVoxel = [&](const float3& w) -> float3 {
                return (w - mGrid.gridMin) / mGrid.voxelSize;
            };
            float3 vv0 = toVoxel(v0), vv1 = toVoxel(v1), vv2 = toVoxel(v2);

            int3 vMin = floorToInt3(glm::min(vv0, glm::min(vv1, vv2)));
            int3 vMax = floorToInt3(glm::max(vv0, glm::max(vv1, vv2)));

            // Clamp to grid
            int maxCell = (int)mGrid.voxelCount.x - 1;
            vMin = glm::clamp(vMin, int3(0), int3(maxCell));
            vMax = glm::clamp(vMax, int3(0), int3(maxCell));

            // Map to tile coordinates
            int3 tileMin = vMin >> (int)(maxDepth - mConfig.tileLevel);
            int3 tileMax = vMax >> (int)(maxDepth - mConfig.tileLevel);

            for (int tz = tileMin.z; tz <= tileMax.z; tz++)
            for (int ty = tileMin.y; ty <= tileMax.y; ty++)
            for (int tx = tileMin.x; tx <= tileMax.x; tx++) {
                uint tIdx = tx + ty * tilesPerAxis + tz * tilesPerAxis * tilesPerAxis;
                if (tIdx < allTiles.size())
                    allTiles[tIdx].triangleIndices.push_back(tid);
            }
        }

        // Remove empty tiles
        std::vector<TileTask> nonEmpty;
        for (auto& t : allTiles)
            if (!t.triangleIndices.empty())
                nonEmpty.push_back(std::move(t));
        return nonEmpty;
    }

    void processTile(const TileTask& tile, uint resultIdx, const LoadedScene& scene) {
        TileResult& result = mTileResults[resultIdx];
        result.generator = std::make_unique<PolygonGenerator>(mGrid);
        PolygonGenerator& gen = *result.generator;
        gen.reset();

        uint maxDepth = mMaxDepth;

        // ---- Phase 2a: Hierarchical clip for all triangles in this tile ----
        uint tileRootLevel = mConfig.tileLevel;
        using PendingClip = std::pair<uint64_t, Polygon>;
        std::vector<PendingClip> pending;
        pending.reserve(65536);

        for (uint globalTid : tile.triangleIndices) {
            uint3 indices = scene.triangles[globalTid];
            const TriangleRef& tr = scene.triRefs[globalTid];

            Triangle tri;
            tri.vertices[0] = (scene.positions[indices.x] - mGrid.gridMin) / mGrid.voxelSize;
            tri.vertices[1] = (scene.positions[indices.y] - mGrid.gridMin) / mGrid.voxelSize;
            tri.vertices[2] = (scene.positions[indices.z] - mGrid.gridMin) / mGrid.voxelSize;
            tri.uvs[0] = scene.texCoords[indices.x];
            tri.uvs[1] = scene.texCoords[indices.y];
            tri.uvs[2] = scene.texCoords[indices.z];
            tri.normals[0] = scene.normals[indices.x];
            tri.normals[1] = scene.normals[indices.y];
            tri.normals[2] = scene.normals[indices.z];
            tri.buildTBN();

            clipHierarchicalForTile(tr.meshID, tr.materialID, globalTid, tri, tri.calcAABBInt(),
                                    tile.tileCoord, tileRootLevel, maxDepth, pending, gen);
        }
        for (auto& p : pending)
            gen.mNodePolygonMap[p.first].push_back(p.second);

        gen.finalizeBFS(maxDepth, tileRootLevel, tile.tileCoord);

        // // ---- Debug: dump all polygons of a specific voxel node ----
        // {
        //     uint debugNodeIdx = 33;  // <-- set to the node index you want to inspect (e.g. 0)
        //     if (debugNodeIdx < gen.polygonArrays.size()) {
        //         const PolygonRange& range = gen.polygonRangeBuffer[debugNodeIdx];
        //         auto& polys = gen.polygonArrays[debugNodeIdx];

        //         std::cout << "========== Node Debug (nodeIdx=" << debugNodeIdx << ") ==========" << std::endl;
        //         std::cout << "  PolygonRange:" << std::endl;
        //         std::cout << "    cellInt    = " << range.cellInt << std::endl;
        //         std::cout << "    nodeScale  = " << range.nodeScale << std::endl;
        //         std::cout << "    localHead  = " << range.localHead << std::endl;
        //         std::cout << "    count      = " << range.count << std::endl;
        //         std::cout << "  polygons (" << polys.size() << "):" << std::endl;

        //         for (uint pi = 0; pi < polys.size(); pi++) {
        //             const Polygon& poly = polys[pi];
        //             float polyArea = poly.calcArea();
        //             float centroidArea;
        //             float3 centroid = poly.calcCentroid(centroidArea);

        //             std::cout << "  --- poly[" << pi << "] ---" << std::endl;
        //             std::cout << "    triRef     = { meshID=" << poly.triRef.meshID
        //                       << ", triangleID=" << poly.triRef.triangleID
        //                       << ", materialID=" << poly.triRef.materialID << " }" << std::endl;
        //             std::cout << "    count      = " << poly.count << std::endl;
        //             std::cout << "    area       = " << polyArea << std::endl;
        //             std::cout << "    normal     = " << poly.normal << std::endl;
        //             std::cout << "    centroid   = " << centroid << " (area=" << centroidArea << ")" << std::endl;
        //             std::cout << "    vertices (" << poly.count << "):" << std::endl;
        //             for (uint vi = 0; vi < poly.count; vi++)
        //                 std::cout << "      [" << vi << "] " << poly.vertices[vi] << std::endl;
        //         }
        //         std::cout << "================================================================" << std::endl;
        //     }
        // }

        // ---- Phase 2b: CPU analysis per node ----
        for (uint nodeIdx = 0; nodeIdx < gen.gBuffer.size(); nodeIdx++) {
            VoxelData& vd = gen.gBuffer[nodeIdx];
            PolygonRange& range = gen.polygonRangeBuffer[nodeIdx];

            for (uint pi = 0; pi < range.count; pi++) {
                const Polygon& poly = gen.polygonArrays[nodeIdx][pi];
                uint globalTid = poly.triRef.triangleID;
                uint3 origVtxIndices = scene.triangles[globalTid];
                uint matID = scene.triRefs[globalTid].materialID;
                const MaterialData& mat = scene.materials[matID];

                float dummy;
                float3 centroid = poly.calcCentroid(dummy);
                float3 leafCentroid = centroid * range.nodeScale;

                // Build original triangle in voxel space (for barycentric / UV interpolation)
                float3 tv0 = (scene.positions[origVtxIndices.x] - mGrid.gridMin) / mGrid.voxelSize;
                float3 tv1 = (scene.positions[origVtxIndices.y] - mGrid.gridMin) / mGrid.voxelSize;
                float3 tv2 = (scene.positions[origVtxIndices.z] - mGrid.gridMin) / mGrid.voxelSize;
                Triangle origTri;
                origTri.vertices[0] = tv0; origTri.vertices[1] = tv1; origTri.vertices[2] = tv2;
                origTri.uvs[0] = scene.texCoords[origVtxIndices.x];
                origTri.uvs[1] = scene.texCoords[origVtxIndices.y];
                origTri.uvs[2] = scene.texCoords[origVtxIndices.z];

                // GPU-style: interpolate UV at every polygon vertex, compute UV-space centroid + area
                float2 polyUVs[MAX_VERTEX_COUNT];
                for (uint vi = 0; vi < poly.count; vi++) {
                    float3 leafVertex = poly.vertices[vi] * range.nodeScale;
                    polyUVs[vi] = origTri.lerpUV(leafVertex);
                }
                float2 uvCenter(0);
                for (uint vi = 0; vi < poly.count; vi++)
                    uvCenter += polyUVs[vi];
                uvCenter /= (float)poly.count;

                float uvArea = 0;
                for (uint vi = 0; vi < poly.count; vi++) {
                    const float2& a = polyUVs[vi];
                    const float2& b = polyUVs[(vi + 1) % poly.count];
                    uvArea += a.x * b.y - a.y * b.x;
                }
                uvArea = 0.5f * std::abs(uvArea);

                // Interpolate normal at centroid (matches GPU LERP_NORMAL path)
                float3 bary = origTri.barycentricCoordinates(leafCentroid);
                float3 n0 = scene.normals[origVtxIndices.x];
                float3 n1 = scene.normals[origVtxIndices.y];
                float3 n2 = scene.normals[origVtxIndices.z];
                float3 interpolatedNormal = safeNormalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);

                // Sample textures
                float4 baseColorVal = float4(mat.baseColor, 1.0f);
                float4 specVal = mat.specular;

                if (matID < mBaseColorTextures.size() &&
                    mBaseColorTextures[matID].width > 0) {
                    baseColorVal = sampleTextureArea(
                        mBaseColorTextures[matID], uvCenter,
                        uvArea, float4(mat.baseColor, 1.0f));
                }
                if (matID < mSpecularTextures.size() &&
                    mSpecularTextures[matID].width > 0) {
                    specVal = sampleTextureArea(
                        mSpecularTextures[matID], uvCenter,
                        uvArea, float4(mat.specular.x, mat.specular.y, mat.specular.z, 1.0f));
                }

                ABSDFInput input = { float3(baseColorVal), specVal, interpolatedNormal, poly.calcArea() };
                vd.ABSDF.accumulate(input);
            }
            vd.ABSDF.normalizeSelf();

            if (!vd.isSolid()) continue;

            vd.ellipsoid.fit(gen.polygonArrays[nodeIdx], range);

            SphericalFunc polyF = vd.polygonsProjAreaFunc;
            SphericalFunc primF = vd.primitiveProjAreaFunc;
            SphericalFunc totalF = vd.totalProjAreaFunc;

            Estimate(vd.ellipsoid, range, polyF, primF, totalF,
                     gen.polygonArrays[nodeIdx], mConfig.sampleFrequency);

            vd.polygonsProjAreaFunc = polyF;
            vd.primitiveProjAreaFunc = primF;
            vd.totalProjAreaFunc = totalF;
        }

        // Per-tile polygon stats
        uint tileTotalPoly = 0, tileMaxPoly = 0;
        for (uint ni = 0; ni < gen.polygonArrays.size(); ni++) {
            uint pc = (uint)gen.polygonArrays[ni].size();
            tileTotalPoly += pc;
            tileMaxPoly = std::max(tileMaxPoly, pc);
        }
        std::cout << "  [TileDebug] tile=" << tile.tileIndex << " nodes=" << gen.gBuffer.size()
                  << " totalPoly=" << tileTotalPoly << " maxPolyPerNode=" << tileMaxPoly << std::endl;
    }


    // Clipping variant that starts from a non-root tile node
    static void clipHierarchicalForTile(
        uint meshID, uint materialID, uint triangleID, Triangle& tri,
        const AABBInt& triAABB, const int3& nodeCell, uint level, uint maxDepth,
        std::vector<std::pair<uint64_t, Polygon>>& pending,
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

        pending.push_back({PolygonGenerator::makeNodeKey(level, nodeCell), polygon});

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
            clipHierarchicalForTile(meshID, materialID, triangleID, tri, triAABB, childCell, level + 1, maxDepth, pending, gen);
        }
    }

    void mergeAndWrite(const std::string& outputPath) {
        std::ofstream f(outputPath, std::ios::binary);
        if (!f) {
            std::cerr << "Cannot open output: " << outputPath << std::endl;
            return;
        }

        // ---- Step 1: Merge tiles level-by-level (true BFS order) ----
        uint totalNodes = 0;
        for (auto& tr : mTileResults) totalNodes += (uint)tr.generator->gBuffer.size();

        std::vector<uint32_t> mergedCounts(mMaxDepth + 1, 0);
        for (auto& tr : mTileResults) {
            auto& gen = *tr.generator;
            for (uint l = 0; l <= mMaxDepth && l < gen.mOctreeNodeCounts.size(); l++)
                mergedCounts[l] += gen.mOctreeNodeCounts[l];
        }

        // Per-tile level ranges: for each level, list of (tileIdx, localStart, count)
        struct TileLevelRange { uint tileIdx; uint start; uint count; };
        std::vector<std::vector<TileLevelRange>> levelRanges(mMaxDepth + 1);
        for (uint ti = 0; ti < mTileResults.size(); ti++) {
            auto& gen = *mTileResults[ti].generator;
            uint off = 0;
            for (uint l = 0; l <= mMaxDepth && l < gen.mOctreeNodeCounts.size(); l++) {
                uint c = gen.mOctreeNodeCounts[l];
                if (c > 0) levelRanges[l].push_back({ti, off, c});
                off += c;
            }
        }

        // Remap table: (tileIdx, localBfsIdx) -> mergedBfsIdx
        std::vector<std::vector<uint>> remap(mTileResults.size());
        for (uint ti = 0; ti < mTileResults.size(); ti++)
            remap[ti].resize(mTileResults[ti].generator->gBuffer.size(), ~0u);

        std::vector<OctreeNode> mergedNodes;
        std::vector<VoxelData> mergedVoxels;
        std::vector<int3> mergedCells;
        mergedNodes.reserve(totalNodes + 1024);
        mergedVoxels.reserve(totalNodes + 1024);
        mergedCells.reserve(totalNodes + 1024);

        uint bfsCursor = 0;
        for (uint l = 0; l <= mMaxDepth; l++) {
            for (auto& rng : levelRanges[l]) {
                auto& gen = *mTileResults[rng.tileIdx].generator;
                for (uint i = 0; i < rng.count; i++) {
                    uint li = rng.start + i;
                    OctreeNode cn = gen.mOctreeNodes[li];
                    cn.dataIndex = bfsCursor;
                    mergedNodes.push_back(cn);
                    mergedVoxels.push_back(gen.gBuffer[li]);
                    mergedCells.push_back(gen.mBFSOrder[li].cellInt);
                    remap[rng.tileIdx][li] = bfsCursor;
                    bfsCursor++;
                }
            }
        }

        // Sort tile-root level by (parentCell, octant) so siblings are contiguous
        {
            uint l = mConfig.tileLevel;
            uint levelStart = 0;
            for (uint pl = 0; pl < l; pl++) levelStart += mergedCounts[pl];
            uint levelCount = mergedCounts[l];

            auto pack = [](const int3& c) -> uint64_t {
                return (uint64_t)(uint32_t)c.x
                    | ((uint64_t)(uint32_t)c.y << 21)
                    | ((uint64_t)(uint32_t)c.z << 42);
            };
            auto octant = [](const int3& c) -> uint {
                return (uint)(c.x & 1) | ((uint)(c.y & 1) << 1) | ((uint)(c.z & 1) << 2);
            };

            std::vector<uint> order(levelCount);
            for (uint i = 0; i < levelCount; i++) order[i] = i;
            std::sort(order.begin(), order.end(), [&](uint a, uint b) {
                uint64_t pa = pack(mergedCells[levelStart + a] / 2);
                uint64_t pb = pack(mergedCells[levelStart + b] / 2);
                if (pa != pb) return pa < pb;
                return octant(mergedCells[levelStart + a]) < octant(mergedCells[levelStart + b]);
            });

            bool sorted = true;
            for (uint i = 0; i < levelCount; i++) {
                if (order[i] != i) { sorted = false; break; }
            }
            if (!sorted) {
                std::vector<uint> oldToNew(levelCount);
                for (uint i = 0; i < levelCount; i++) oldToNew[order[i]] = i;

                std::vector<OctreeNode> sn(levelCount);
                std::vector<VoxelData>   sv(levelCount);
                std::vector<int3>        sc(levelCount);
                for (uint i = 0; i < levelCount; i++) {
                    sn[i] = mergedNodes[levelStart + order[i]];
                    sv[i] = mergedVoxels[levelStart + order[i]];
                    sc[i] = mergedCells[levelStart + order[i]];
                }
                for (uint i = 0; i < levelCount; i++) {
                    mergedNodes[levelStart + i]  = sn[i];
                    mergedVoxels[levelStart + i] = sv[i];
                    mergedCells[levelStart + i]  = sc[i];
                }

                // Update remap for this level so childBase fixup uses new positions
                for (uint ti = 0; ti < mTileResults.size(); ti++) {
                    auto& gen = *mTileResults[ti].generator;
                    uint localOff = 0;
                    for (uint pl = 0; pl < l; pl++) localOff += gen.mOctreeNodeCounts[pl];
                    for (uint i = 0; i < gen.mOctreeNodeCounts[l]; i++) {
                        uint li = localOff + i;
                        if (remap[ti][li] != ~0u) {
                            uint oldPos = remap[ti][li] - levelStart;
                            remap[ti][li] = levelStart + oldToNew[oldPos];
                        }
                    }
                }
            }
        }

        // Fixup childBase from local BFS to merged BFS
        for (uint ti = 0; ti < mTileResults.size(); ti++) {
            auto& gen = *mTileResults[ti].generator;
            for (uint li = 0; li < gen.gBuffer.size(); li++) {
                uint mi = remap[ti][li];
                if (mi == ~0u) continue;
                if (mergedNodes[mi].childMask) {
                    uint localChild = gen.mOctreeNodes[li].childBase;
                    mergedNodes[mi].childBase = remap[ti][localChild];
                }
            }
        }

        // ---- Step 2: Build upper octree levels (0 .. tileRootLevel-1) ----
        uint tileRootLevel = mConfig.tileLevel;
        if (tileRootLevel > 0) {
            for (int buildLvl = (int)tileRootLevel - 1; buildLvl >= 0; buildLvl--) {
                uint childLvl = buildLvl + 1;

                uint childStart = 0;
                for (uint l = 0; l < childLvl; l++) childStart += mergedCounts[l];
                uint childCount = mergedCounts[childLvl];
                if (childCount == 0) continue;

                // Group children by parent cell (= childCell / 2)
                // Pack int3 into uint64 for map key (glm::ivec3 has no operator<)
                auto pack = [](const int3& c) -> uint64_t {
                    return (uint64_t)(uint32_t)c.x
                        | ((uint64_t)(uint32_t)c.y << 21)
                        | ((uint64_t)(uint32_t)c.z << 42);
                };
                auto unpack = [](uint64_t k) -> int3 {
                    return int3((int)(k & 0x1FFFFF),
                                (int)((k >> 21) & 0x1FFFFF),
                                (int)((k >> 42) & 0x1FFFFF));
                };
                std::map<uint64_t, std::vector<uint>> parentGroups;
                for (uint i = 0; i < childCount; i++) {
                    uint idx = childStart + i;
                    parentGroups[pack(mergedCells[idx] / 2)].push_back(idx);
                }

                uint numParents = (uint)parentGroups.size();
                std::vector<OctreeNode> newParents;
                std::vector<VoxelData> newVoxels;
                std::vector<int3> newCells;
                newParents.reserve(numParents);
                newVoxels.reserve(numParents);
                newCells.reserve(numParents);

                for (auto& [packedKey, children] : parentGroups) {
                    int3 parentCell = unpack(packedKey);
                    OctreeNode pn;
                    pn.childMask = 0;
                    pn.childBase = children[0] + numParents;  // children shift right
                    pn.dataIndex = 0;

                    for (uint ci : children) {
                        int3 cc = mergedCells[ci];
                        uint octant = (uint)(cc.x & 1) | ((uint)(cc.y & 1) << 1) | ((uint)(cc.z & 1) << 2);
                        pn.childMask |= (1u << octant);
                    }

                    newParents.push_back(pn);
                    VoxelData emptyVD; emptyVD.init();
                    newVoxels.push_back(emptyVD);
                    newCells.push_back(parentCell);
                }

                // Insert parents into their BFS slot (before the children)
                mergedNodes.insert(mergedNodes.begin() + childStart, newParents.begin(), newParents.end());
                mergedVoxels.insert(mergedVoxels.begin() + childStart, newVoxels.begin(), newVoxels.end());
                mergedCells.insert(mergedCells.begin() + childStart, newCells.begin(), newCells.end());

                mergedCounts[buildLvl] = numParents;

                // Offset childBase in all nodes after the insertion point
                for (uint i = childStart + numParents; i < mergedNodes.size(); i++) {
                    if (mergedNodes[i].childMask)
                        mergedNodes[i].childBase += numParents;
                }
            }
        }

        totalNodes = (uint)mergedNodes.size();

        // ---- Debug: per-level node counts and polygon stats ----
        std::cout << "  [MergeDebug] totalNodes=" << totalNodes << " maxDepth=" << mMaxDepth << std::endl;
        uint grandTotalPoly = 0, maxPolyPerNode = 0, nonEmptyNodes = 0;
        for (auto& tr : mTileResults) {
            auto& gen = *tr.generator;
            for (uint ni = 0; ni < gen.polygonArrays.size(); ni++) {
                uint pc = (uint)gen.polygonArrays[ni].size();
                grandTotalPoly += pc;
                maxPolyPerNode = std::max(maxPolyPerNode, pc);
                if (pc > 0) nonEmptyNodes++;
            }
        }
        std::cout << "  [MergeDebug] totalPolygons=" << grandTotalPoly
                  << " maxPolyPerNode=" << maxPolyPerNode
                  << " nonEmptyNodes=" << nonEmptyNodes << std::endl;
        std::cout << "  [MergeDebug] per-level nodes:";
        for (uint l = 0; l <= mMaxDepth; l++)
            std::cout << " L" << l << "=" << mergedCounts[l];
        std::cout << std::endl;

        // ---- Step 3: Write flat format matching RayMarchingPass.cpp ----
        GridData outGrid = mGrid;
        outGrid.solidVoxelCount = totalNodes;
        outGrid.maxPolygonCount = maxPolyPerNode;
        outGrid.totalPolygonCount = grandTotalPoly;

        f.write((const char*)&outGrid, sizeof(GridData));
        f.write((const char*)&mMaxDepth, sizeof(uint32_t));
        f.write((const char*)mergedCounts.data(), (mMaxDepth + 1) * sizeof(uint32_t));
        f.write((const char*)mergedNodes.data(), totalNodes * sizeof(OctreeNode));
        f.write((const char*)mergedVoxels.data(), totalNodes * sizeof(VoxelData));

        f.close();
        std::cout << "Wrote " << totalNodes << " nodes." << std::endl;
    }
};
