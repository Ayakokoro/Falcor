#pragma once
#include "Types.h"
#include "AABB.h"
#include "Triangle.h"
#include "GridData.h"
#include "VoxelData.h"
#include "VoxelizationUtility.h"
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <thread>
#include <iostream>
#include <cassert>

class PolygonGenerator {
public:
    GridData& gridData;
    std::vector<VoxelData> gBuffer;
    std::vector<int> vBuffer;
    std::vector<std::vector<Polygon>> polygonArrays;
    std::vector<PolygonRange> polygonRangeBuffer;

    // Hierarchical clip: per-node polygon storage (leaf-only after modification)
    std::unordered_map<uint64_t, std::vector<Polygon>> mNodePolygonMap;

    // Occupied node set: marks which octree nodes exist (have geometry in subtree).
    // Used by finalizeBFS for structure traversal; mNodePolygonMap holds leaf-only polygon data.
    std::unordered_set<uint64_t> mOccupiedNodes;

    struct BFSNodeInfo { int3 cellInt; uint level; };
    std::vector<BFSNodeInfo> mBFSOrder;
    std::vector<OctreeNode> mOctreeNodes;
    std::vector<uint32_t> mOctreeNodeCounts;
    uint32_t mOctreeMaxDepth = 0;

    // Node key encoding v2:
    //   bits [0..18]   : cell x (19 bits)
    //   bits [19..37]  : cell y (19 bits)
    //   bits [38..56]  : cell z (19 bits)
    //   bits [57..61]  : level (5 bits)
    //   bits [62..63]  : reserved
    //
    // The previous encoding used only 10 bits per coordinate, which caused
    // collisions as soon as a leaf grid exceeded 1024^3 (maxDepth >= 11).
    static constexpr uint NODE_KEY_COORD_BITS = 19;
    static constexpr uint NODE_KEY_LEVEL_BITS = 5;
    static constexpr uint NODE_KEY_LEVEL_SHIFT = NODE_KEY_COORD_BITS * 3;
    static constexpr uint NODE_KEY_MAX_DEPTH = NODE_KEY_COORD_BITS;
    static constexpr uint64_t NODE_KEY_COORD_MASK =
        (uint64_t(1) << NODE_KEY_COORD_BITS) - 1;
    static constexpr uint64_t NODE_KEY_LEVEL_MASK =
        (uint64_t(1) << NODE_KEY_LEVEL_BITS) - 1;

    static_assert(NODE_KEY_LEVEL_SHIFT + NODE_KEY_LEVEL_BITS <= 64);

    static uint64_t makeNodeKey(uint level, const int3& cellInt) {
        assert(level <= NODE_KEY_MAX_DEPTH);
        assert(cellInt.x >= 0 && cellInt.y >= 0 && cellInt.z >= 0);
        const uint maxCoord = 1u << level;
        assert((uint)cellInt.x < maxCoord &&
               (uint)cellInt.y < maxCoord &&
               (uint)cellInt.z < maxCoord);

        uint64_t ck = (uint64_t)(uint32_t)cellInt.x
                    | ((uint64_t)(uint32_t)cellInt.y << NODE_KEY_COORD_BITS)
                    | ((uint64_t)(uint32_t)cellInt.z << (NODE_KEY_COORD_BITS * 2));
        return ((uint64_t)level << NODE_KEY_LEVEL_SHIFT) | ck;
    }

    static uint levelFromKey(uint64_t key) {
        return (uint)((key >> NODE_KEY_LEVEL_SHIFT) & NODE_KEY_LEVEL_MASK);
    }

    static int3 cellFromKey(uint64_t key) {
        return int3(
            (int)(key & NODE_KEY_COORD_MASK),
            (int)((key >> NODE_KEY_COORD_BITS) & NODE_KEY_COORD_MASK),
            (int)((key >> (NODE_KEY_COORD_BITS * 2)) & NODE_KEY_COORD_MASK)
        );
    }

    explicit PolygonGenerator(GridData& gd) : gridData(gd) {}

    void reset() {
        gBuffer.clear(); vBuffer.clear();
        polygonArrays.clear(); polygonRangeBuffer.clear();
        vBuffer.assign(gridData.totalVoxelCount(), -1);
        mNodePolygonMap.clear();
        mOccupiedNodes.clear();
        mBFSOrder.clear(); mOctreeNodes.clear(); mOctreeNodeCounts.clear();
        mOctreeMaxDepth = 0;
    }

public:
    // MeshData: lightweight per-mesh metadata (mirrors Falcor's MeshHeader)
    struct MeshData {
        uint meshID, materialID, triangleOffset;
        uint vertexCount, triangleCount;
    };

private:
    std::mutex mNodeMapMutex;
    static constexpr uint kFlushBatchSize = 65536;

    struct PendingHierClip { uint64_t nodeKey; Polygon poly; };

    void flushPending(std::vector<PendingHierClip>& pending) {
        if (pending.empty()) return;
        std::lock_guard<std::mutex> lock(mNodeMapMutex);
        for (auto& p : pending)
            mNodePolygonMap[p.nodeKey].push_back(p.poly);
        pending.clear();
    }

    void clipHierarchicalNoLock(const MeshData& mesh, uint triangleID, Triangle& tri,
                                const AABBInt& triAABB, const int3& nodeCell,
                                uint level, uint maxDepth, std::vector<PendingHierClip>& pending) {
        uint scale = 1u << (maxDepth - level);
        float3 minPoint = float3(nodeCell) * (float)scale;
        float3 maxPoint = minPoint + float3((float)scale);

        Polygon polygon = VoxelizationUtility::BoxClipTriangle(minPoint, maxPoint, tri);
        if (polygon.count < 3 || polygon.calcArea() <= 1e-8f)
            return;

        float invScale = 1.0f / (float)scale;
        for (uint vi = 0; vi < polygon.count; vi++)
            polygon.vertices[vi] *= invScale;

        polygon.normal = tri.TBN[2];  // col 2 = normal
        polygon.triRef.meshID = mesh.meshID;
        polygon.triRef.triangleID = triangleID;
        polygon.triRef.materialID = mesh.materialID;

        {
            uint64_t nodeKey = makeNodeKey(level, nodeCell);
            // Mark node as occupied (needed for finalizeBFS which checks mOccupiedNodes)
            mOccupiedNodes.insert(nodeKey);
            auto it = mNodePolygonMap.find(nodeKey);
            uint currentCount = (it != mNodePolygonMap.end()) ? (uint)it->second.size() : 0u;
            if (currentCount < SAFE_PER_NODE_POLYGON_LIMIT)
                pending.push_back({nodeKey, polygon});
        }

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
            clipHierarchicalNoLock(mesh, triangleID, tri, triAABB, childCell, level + 1, maxDepth, pending);
        }

        if (pending.size() >= kFlushBatchSize)
            flushPending(pending);
    }

public:
    // Clip a range of global triangle indices (supports multi-threaded work distribution)
    void clipTrianglesRange(uint triBegin, uint triEnd,
                            const std::vector<MeshData>& meshList,
                            float3* pPos, float3* pNormal, float2* pUV, uint3* pTri,
                            uint maxDepth) {
        std::vector<PendingHierClip> pending;
        pending.reserve(kFlushBatchSize);

        uint meshIdx = 0;
        uint meshTriEnd = meshList[0].triangleCount;

        for (uint g = triBegin; g < triEnd; g++) {
            while (g >= meshTriEnd) {
                meshIdx++;
                meshTriEnd += meshList[meshIdx].triangleCount;
            }
            const MeshData& mesh = meshList[meshIdx];
            uint localTid = g - (meshTriEnd - mesh.triangleCount);

            Triangle tri;
            uint3 indices = pTri[localTid + mesh.triangleOffset];
            tri.vertices[0] = pPos[indices.x];
            tri.vertices[1] = pPos[indices.y];
            tri.vertices[2] = pPos[indices.z];
            tri.uvs[0] = pUV[indices.x];
            tri.uvs[1] = pUV[indices.y];
            tri.uvs[2] = pUV[indices.z];
            tri.normals[0] = pNormal[indices.x];
            tri.normals[1] = pNormal[indices.y];
            tri.normals[2] = pNormal[indices.z];

            for (int i = 0; i < 3; i++)
                tri.vertices[i] = (tri.vertices[i] - gridData.gridMin) / gridData.voxelSize;
            tri.buildTBN();

            AABBInt triAABB = tri.calcAABBInt();
            clipHierarchicalNoLock(mesh, localTid, tri, triAABB, int3(0, 0, 0), 0, maxDepth, pending);
        }
        flushPending(pending);
    }

    void clipHierarchicalAll(const std::vector<MeshData>& meshList,
                             float3* pPos, float3* pNormal, float2* pUV, uint3* pTri,
                             uint maxDepth, uint numThreads = 0) {
        if (numThreads == 0)
            numThreads = std::max(1u, std::thread::hardware_concurrency());

        mNodePolygonMap.clear();

        uint totalTriangles = 0;
        for (auto& m : meshList) totalTriangles += m.triangleCount;
        if (totalTriangles == 0) { finalizeBFS(maxDepth); return; }

        if (numThreads <= 1 || totalTriangles <= 1) {
            numThreads = 1;
        }

        if (numThreads == 1) {
            clipTrianglesRange(0, totalTriangles, meshList, pPos, pNormal, pUV, pTri, maxDepth);
        } else {
            uint chunkSize = (totalTriangles + numThreads - 1) / numThreads;
            std::vector<std::thread> threads;
            for (uint t = 0; t < numThreads && t * chunkSize < totalTriangles; t++) {
                uint begin = t * chunkSize;
                uint end = std::min(begin + chunkSize, totalTriangles);
                threads.emplace_back([this, begin, end, &meshList, pPos, pNormal, pUV, pTri, maxDepth]() {
                    clipTrianglesRange(begin, end, meshList, pPos, pNormal, pUV, pTri, maxDepth);
                });
            }
            for (auto& th : threads) th.join();
        }

        finalizeBFS(maxDepth);
    }

    // BFS: build octree node array, gBuffer, polygonArrays from clip result
    void finalizeBFS(uint maxDepth, uint rootLevel = 0, const int3& rootCell_ = int3(0, 0, 0)) {
        mBFSOrder.clear(); mOctreeNodes.clear();
        mOctreeNodeCounts.assign(maxDepth + 1, 0);
        mOctreeMaxDepth = maxDepth;
        gBuffer.clear(); polygonArrays.clear(); polygonRangeBuffer.clear();

        struct BFSItem { int3 cellInt; uint level; };
        std::vector<BFSItem> queue;
        queue.push_back({rootCell_, rootLevel});
        size_t head = 0;

        std::vector<uint32_t> levelNodeCount(maxDepth + 1, 0);
        std::unordered_map<uint64_t, uint> keyToBFSIndex;

        while (head < queue.size()) {
            BFSItem item = queue[head++];
            uint64_t nodeKey = makeNodeKey(item.level, item.cellInt);
            // Use mOccupiedNodes (populated by clip) instead of mNodePolygonMap
            if (mOccupiedNodes.find(nodeKey) == mOccupiedNodes.end())
                continue;

            uint bfsIndex = (uint)mBFSOrder.size();
            mBFSOrder.push_back({item.cellInt, item.level});
            keyToBFSIndex[nodeKey] = bfsIndex;
            levelNodeCount[item.level]++;

            if (item.level < maxDepth) {
                for (uint ci = 0; ci < 8; ci++) {
                    int3 childCell = item.cellInt * 2 + int3((int)(ci & 1), (int)((ci >> 1) & 1), (int)((ci >> 2) & 1));
                    uint64_t childKey = makeNodeKey(item.level + 1, childCell);
                    if (mOccupiedNodes.find(childKey) != mOccupiedNodes.end())
                        queue.push_back({childCell, item.level + 1});
                }
            }
        }

        uint bfsIdx = 0;
        for (auto& bfsItem : mBFSOrder) {
            uint64_t nodeKey = makeNodeKey(bfsItem.level, bfsItem.cellInt);
            // Only leaf nodes have polygon data in mNodePolygonMap
            auto mapIt = mNodePolygonMap.find(nodeKey);
            if (mapIt != mNodePolygonMap.end() && !mapIt->second.empty())
                polygonArrays.push_back(std::move(mapIt->second));
            else
                polygonArrays.emplace_back();  // empty for non-leaf nodes

            PolygonRange range;
            range.init(bfsItem.cellInt);
            range.nodeScale = (float)(1u << (maxDepth - bfsItem.level));
            range.count = (uint)polygonArrays.back().size();
            polygonRangeBuffer.push_back(range);

            VoxelData vd; vd.init();
            gBuffer.push_back(vd);

            OctreeNode octNode;
            octNode.dataIndex = bfsIdx;
            if (bfsItem.level < maxDepth) {
                octNode.childMask = 0;
                uint firstChildBFS = 0;
                bool hasFirst = false;
                for (uint ci = 0; ci < 8; ci++) {
                    int3 childCell = bfsItem.cellInt * 2 + int3((int)(ci & 1), (int)((ci >> 1) & 1), (int)((ci >> 2) & 1));
                    uint64_t childKey = makeNodeKey(bfsItem.level + 1, childCell);
                    auto childIt = keyToBFSIndex.find(childKey);
                    if (childIt != keyToBFSIndex.end()) {
                        octNode.childMask |= (1u << ci);
                        if (!hasFirst) { firstChildBFS = childIt->second; hasFirst = true; }
                    }
                }
                octNode.childBase = hasFirst ? firstChildBFS : 0;
            }
            mOctreeNodes.push_back(octNode);
            bfsIdx++;
        }

        for (uint l = 0; l <= maxDepth; l++)
            mOctreeNodeCounts[l] = levelNodeCount[l];
        gridData.solidVoxelCount = gBuffer.size();
    }
};
