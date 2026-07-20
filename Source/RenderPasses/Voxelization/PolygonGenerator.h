#pragma once
#include "VoxelizationBase.h"
#include "Profiler.h"
#include "Math/Polygon.slang"
#include "Math/Triangle.slang"
#include "Math/SphericalHarmonics.slang"
#include <unordered_map>
#include <mutex>
#include <thread>

using namespace Falcor;

class PolygonGenerator
{
private:
    GridData& gridData;
    std::mutex mtx;
    std::mutex mNodeMapMutex;

    struct PendingClip
    {
        int3 cellInt;
        Polygon poly;
    };

    struct PendingHierClip
    {
        uint64_t nodeKey;
        Polygon poly;
    };

    static constexpr uint kFlushBatchSize = 65536;

public:
    std::vector<VoxelData> gBuffer;
    std::vector<int> vBuffer;
    std::vector<std::vector<Polygon>> polygonArrays;
    std::vector<PolygonRange> polygonRangeBuffer;

    // ---- Hierarchical clip: per-node polygon accumulation ----
    std::unordered_map<uint64_t, std::vector<Polygon>> mNodePolygonMap;

    // BFS result
    struct BFSNodeInfo
    {
        int3 cellInt;
        uint level;
    };
    std::vector<BFSNodeInfo> mBFSOrder;          // nodes in BFS order
    std::vector<OctreeNode> mOctreeNodes;        // OctreeNode array (BFS order)
    std::vector<uint32_t> mOctreeNodeCounts;     // node count per level
    uint32_t mOctreeMaxDepth = 0;

    // ---- Node key encoding ----
    // level (8 bits) | cellInt per dimension (10 bits each)
    static uint64_t makeNodeKey(uint level, int3 cellInt)
    {
        uint64_t ck = (uint64_t)(uint32_t)cellInt.x
                    | ((uint64_t)(uint32_t)cellInt.y << 10)
                    | ((uint64_t)(uint32_t)cellInt.z << 20);
        return ((uint64_t)level << 32) | ck;
    }

    static uint levelFromKey(uint64_t key) { return (uint)(key >> 32); }
    static int3 cellFromKey(uint64_t key)
    {
        uint32_t ck = (uint32_t)(key & 0xFFFFFFFFull);
        return int3((int)(ck & 0x3FF), (int)((ck >> 10) & 0x3FF), (int)((ck >> 20) & 0x3FF));
    }

    PolygonGenerator() : gridData(VoxelizationBase::GlobalGridData) {}

    void reset()
    {
        gBuffer.clear();
        vBuffer.clear();
        polygonArrays.clear();
        polygonRangeBuffer.clear();
        vBuffer.assign(gridData.totalVoxelCount(), -1);
        mNodePolygonMap.clear();
        mBFSOrder.clear();
        mOctreeNodes.clear();
        mOctreeNodeCounts.clear();
        mOctreeMaxDepth = 0;
    }

    // ---- Single-level clip (keep for backward compat) ----

    int tryGetOffset(int3 cellInt)
    {
        int index = CellToIndex(cellInt, gridData.voxelCount);
        if (vBuffer[index] == -1)
        {
            int offset = gBuffer.size();
            vBuffer[index] = offset;
            gBuffer.emplace_back();
            gBuffer[offset].init();
            polygonArrays.emplace_back();
            polygonRangeBuffer.emplace_back();
            polygonRangeBuffer[offset].init(cellInt);
        }
        return vBuffer[index];
    }

    int getOffset(int3 cellInt)
    {
        int index = CellToIndex(cellInt, gridData.voxelCount);
        return vBuffer[index];
    }

    void clip(const MeshHeader& mesh, uint triangleID, Triangle& tri)
    {
        AABBInt aabb = tri.calcAABBInt();
        for (int i = 0; i < aabb.count(); i++)
        {
            int3 cellInt = aabb.indexToCell(i);
            float3 minPoint = float3(cellInt);
            Polygon polygon = VoxelizationUtility::BoxClipTriangle(minPoint, minPoint + 1.f, tri);
            polygon.normal = tri.TBN.getCol(2);
            if (polygon.count >= 3 && polygon.calcArea() > 1e-8f)
            {
                polygon.triRef.meshID = mesh.meshID;
                polygon.triRef.triangleID = triangleID;
                polygon.triRef.materialID = mesh.materialID;
                int offset = tryGetOffset(cellInt);
                polygonArrays[offset].push_back(polygon);
            }
        }
    }

    void clipMesh(const MeshHeader& mesh, float3* pPos, float3* pNormal, float2* pUV, uint3* pIndex)
    {
        for (uint tid = 0; tid < mesh.triangleCount; tid++)
        {
            Triangle tri = {};
            uint3 indices = pIndex[tid + mesh.triangleOffset];
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
            clip(mesh, tid, tri);
        }
    }

    // ---- Multi-threaded single-level clip ----

    void clipNoLock(const MeshHeader& mesh, uint triangleID, Triangle& tri, std::vector<PendingClip>& pending)
    {
        AABBInt aabb = tri.calcAABBInt();
        for (int i = 0; i < aabb.count(); i++)
        {
            int3 cellInt = aabb.indexToCell(i);
            float3 minPoint = float3(cellInt);
            Polygon polygon = VoxelizationUtility::BoxClipTriangle(minPoint, minPoint + 1.f, tri);
            polygon.normal = tri.TBN.getCol(2);
            if (polygon.count >= 3)
            {
                polygon.triRef.meshID = mesh.meshID;
                polygon.triRef.triangleID = triangleID;
                polygon.triRef.materialID = mesh.materialID;
                pending.push_back({cellInt, polygon});
            }
        }
    }

    void flushPending(std::vector<PendingClip>& pending)
    {
        if (pending.empty())
            return;
        std::lock_guard<std::mutex> lock(mtx);
        for (auto& p : pending)
        {
            int offset = tryGetOffset(p.cellInt);
            polygonArrays[offset].push_back(p.poly);
        }
        pending.clear();
    }

    void clipMeshThreaded(const MeshHeader& mesh, float3* pPos, float3* pNormal, float2* pUV, uint3* pIndex)
    {
        std::vector<PendingClip> pending;

        for (uint tid = 0; tid < mesh.triangleCount; tid++)
        {
            Triangle tri = {};
            uint3 indices = pIndex[tid + mesh.triangleOffset];
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
            clipNoLock(mesh, tid, tri, pending);

            if (pending.size() >= kFlushBatchSize)
                flushPending(pending);
        }
        flushPending(pending);
    }

    void clipAll(SceneHeader scene, std::vector<MeshHeader> meshList, float3* pPos, float3* pNormal, float2* pUV, uint3* pTri,
                 uint numThreads = 0)
    {
        if (numThreads == 0)
            numThreads = std::max(1u, std::thread::hardware_concurrency());

        if (numThreads <= 1 || meshList.size() <= 1)
        {
            for (size_t i = 0; i < meshList.size(); i++)
                clipMesh(meshList[i], pPos, pNormal, pUV, pTri);
        }
        else
        {
            std::vector<std::thread> threads;
            size_t chunkSize = (meshList.size() + numThreads - 1) / numThreads;

            for (uint t = 0; t < numThreads && t * chunkSize < meshList.size(); t++)
            {
                size_t begin = t * chunkSize;
                size_t end = std::min(begin + chunkSize, meshList.size());
                threads.emplace_back([this](size_t begin_, size_t end_,
                                            const std::vector<MeshHeader>* pMeshList,
                                            float3* pPos_, float3* pNormal_, float2* pUV_, uint3* pTri_) {
                    for (size_t i = begin_; i < end_; i++)
                        clipMeshThreaded((*pMeshList)[i], pPos_, pNormal_, pUV_, pTri_);
                }, begin, end, &meshList, pPos, pNormal, pUV, pTri);
            }

            for (auto& th : threads)
                th.join();
        }

        gridData.solidVoxelCount = gBuffer.size();
    }

    // ========================================================================
    // Hierarchical clip: coarse-to-fine, per-node independent polygon storage
    // ========================================================================

    void flushHierPending(std::vector<PendingHierClip>& pending)
    {
        if (pending.empty())
            return;
        std::lock_guard<std::mutex> lock(mNodeMapMutex);
        for (auto& p : pending)
        {
            mNodePolygonMap[p.nodeKey].push_back(p.poly);
        }
        pending.clear();
    }

    void clipHierarchicalNoLock(
        const MeshHeader& mesh,
        uint triangleID,
        Triangle& tri,
        const AABBInt& triAABB,
        int3 nodeCell,
        uint level,
        uint maxDepth,
        std::vector<PendingHierClip>& pending)
    {
        uint scale = 1u << (maxDepth - level);
        float3 minPoint = float3(nodeCell) * (float)scale;
        float3 maxPoint = minPoint + float3((float)scale);

        Polygon polygon = VoxelizationUtility::BoxClipTriangle(minPoint, maxPoint, tri);
        if (polygon.count < 3 || polygon.calcArea() <= 1e-8f)
            return; // Prune: no intersection at this level

        // Normalize vertices to node-local space [0, 1]^3
        // vertex is in leaf-voxel space ∈ [minPoint, minPoint+scale]
        float invScale = 1.0f / (float)scale;
        for (uint vi = 0; vi < polygon.count; vi++)
            polygon.vertices[vi] *= invScale;

        polygon.normal = tri.TBN.getCol(2);
        polygon.triRef.meshID = mesh.meshID;
        polygon.triRef.triangleID = triangleID;
        polygon.triRef.materialID = mesh.materialID;

        uint64_t nodeKey = makeNodeKey(level, nodeCell);
        pending.push_back({nodeKey, polygon});

        if (level >= maxDepth)
            return; // Leaf reached

        // AABB pre-culling: only recurse into children that the triangle overlaps
        int childScale = (int)(scale >> 1);
        for (uint ci = 0; ci < 8; ci++)
        {
            int3 childCell = nodeCell * 2 + int3((int)(ci & 1), (int)((ci >> 1) & 1), (int)((ci >> 2) & 1));

            // Child's integer leaf-voxel extents [min, max] inclusive
            int3 childMin = childCell * childScale;
            int3 childMax = childMin + childScale - 1;

            if (triAABB.xMax < childMin.x || triAABB.xMin > childMax.x ||
                triAABB.yMax < childMin.y || triAABB.yMin > childMax.y ||
                triAABB.zMax < childMin.z || triAABB.zMin > childMax.z)
                continue;

            clipHierarchicalNoLock(mesh, triangleID, tri, triAABB, childCell, level + 1, maxDepth, pending);
        }

        if (pending.size() >= kFlushBatchSize)
            flushHierPending(pending);
    }

    void clipHierarchicalMeshThreaded(
        const MeshHeader& mesh,
        float3* pPos, float3* pNormal, float2* pUV, uint3* pIndex,
        uint maxDepth)
    {
        std::vector<PendingHierClip> pending;

        for (uint tid = 0; tid < mesh.triangleCount; tid++)
        {
            Triangle tri = {};
            uint3 indices = pIndex[tid + mesh.triangleOffset];
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
            // Start from root: level 0, cell (0,0,0)
            clipHierarchicalNoLock(mesh, tid, tri, triAABB, int3(0, 0, 0), 0, maxDepth, pending);
        }
        flushHierPending(pending);
    }

    void clipHierarchicalAll(
        SceneHeader scene,
        std::vector<MeshHeader> meshList,
        float3* pPos, float3* pNormal, float2* pUV, uint3* pTri,
        uint maxDepth,
        uint numThreads = 0)
    {
        if (numThreads == 0)
        {
            numThreads = std::max(1u, std::thread::hardware_concurrency());
            std::cout << "Availiable Thread Num = " << numThreads << "\n";
        }

        mNodePolygonMap.clear();

        uint totalTriangles = 0;
        for (auto& m : meshList)
            totalTriangles += m.triangleCount;

        if (totalTriangles == 0)
        {
            finalizeBFS(maxDepth);
            return;
        }

        if (numThreads <= 1 || totalTriangles <= 1)
            numThreads = 1;

        if (numThreads == 1)
        {
            clipTrianglesRange(0, totalTriangles, meshList, pPos, pNormal, pUV, pTri, maxDepth);
        }
        else
        {
            uint chunkSize = (totalTriangles + numThreads - 1) / numThreads;
            std::vector<std::thread> threads;

            for (uint t = 0; t < numThreads && t * chunkSize < totalTriangles; t++)
            {
                uint begin = t * chunkSize;
                uint end = std::min(begin + chunkSize, totalTriangles);
                threads.emplace_back([this](uint begin_, uint end_,
                                            const std::vector<MeshHeader>* pMeshList,
                                            float3* pPos_, float3* pNormal_, float2* pUV_, uint3* pTri_,
                                            uint maxDepth_) {
                    clipTrianglesRange(begin_, end_, *pMeshList, pPos_, pNormal_, pUV_, pTri_, maxDepth_);
                }, begin, end, &meshList, pPos, pNormal, pUV, pTri, maxDepth);
            }

            for (auto& th : threads)
                th.join();
        }

        finalizeBFS(maxDepth);
    }

    void clipTrianglesRange(
        uint triBegin, uint triEnd,
        const std::vector<MeshHeader>& meshList,
        float3* pPos, float3* pNormal, float2* pUV, uint3* pTri,
        uint maxDepth)
    {
        std::vector<PendingHierClip> pending;
        pending.reserve(kFlushBatchSize);

        uint meshIdx = 0;
        uint meshTriEnd = meshList[0].triangleCount;

        for (uint g = triBegin; g < triEnd; g++)
        {
            // Advance to the mesh containing global triangle index g
            while (g >= meshTriEnd)
            {
                meshIdx++;
                meshTriEnd += meshList[meshIdx].triangleCount;
            }

            const MeshHeader& mesh = meshList[meshIdx];
            uint localTid = g - (meshTriEnd - mesh.triangleCount);

            Triangle tri = {};
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
        flushHierPending(pending);
    }

    // ========================================================================
    // BFS: build OctreeNode array, gBuffer, polygonArrays, polygonRangeBuffer
    // from the hierarchical clip result
    // ========================================================================

    void finalizeBFS(uint maxDepth)
    {
        mBFSOrder.clear();
        mOctreeNodes.clear();
        mOctreeNodeCounts.assign(maxDepth + 1, 0);
        mOctreeMaxDepth = maxDepth;

        gBuffer.clear();
        polygonArrays.clear();
        polygonRangeBuffer.clear();

        // BFS from root
        struct BFSItem
        {
            int3 cellInt;
            uint level;
        };
        std::vector<BFSItem> queue;
        queue.push_back({int3(0, 0, 0), 0});
        size_t head = 0;

        // Per-level tracking for node counts
        std::vector<uint32_t> levelNodeCount(maxDepth + 1, 0);

        // Temporary: map nodeKey → BFS index for childBase calculation
        std::unordered_map<uint64_t, uint> keyToBFSIndex;

        while (head < queue.size())
        {
            BFSItem item = queue[head];
            head++;

            uint64_t nodeKey = makeNodeKey(item.level, item.cellInt);
            auto it = mNodePolygonMap.find(nodeKey);
            if (it == mNodePolygonMap.end() || it->second.empty())
                continue; // No polygons for this node, skip

            uint bfsIndex = (uint)mBFSOrder.size();
            mBFSOrder.push_back({item.cellInt, item.level});
            keyToBFSIndex[nodeKey] = bfsIndex;
            levelNodeCount[item.level]++;

            if (item.level < maxDepth)
            {
                // Enqueue existing children
                for (uint ci = 0; ci < 8; ci++)
                {
                    int3 childCell = item.cellInt * 2 + int3((int)(ci & 1), (int)((ci >> 1) & 1), (int)((ci >> 2) & 1));
                    uint64_t childKey = makeNodeKey(item.level + 1, childCell);
                    if (mNodePolygonMap.find(childKey) != mNodePolygonMap.end())
                    {
                        queue.push_back({childCell, item.level + 1});
                    }
                }
            }
        }

        // Build OctreeNode array and output buffers in BFS order
        // Second pass: compute childBase and childMask using keyToBFSIndex
        uint bfsIdx = 0;
        for (auto& bfsItem : mBFSOrder)
        {
            uint64_t nodeKey = makeNodeKey(bfsItem.level, bfsItem.cellInt);
            auto& polys = mNodePolygonMap[nodeKey];

            // Fill polygonArrays
            polygonArrays.push_back(std::move(polys));

            // Fill polygonRangeBuffer
            PolygonRange range;
            range.init(bfsItem.cellInt);
            range.nodeScale = (float)(1u << (maxDepth - bfsItem.level));
            range.count = (uint)polygonArrays.back().size();
            polygonRangeBuffer.push_back(range);

            // Fill gBuffer placeholder
            VoxelData vd;
            vd.init();
            gBuffer.push_back(vd);

            // Build OctreeNode
            OctreeNode octNode;
            octNode.dataIndex = bfsIdx;

            if (bfsItem.level < maxDepth)
            {
                octNode.childMask = 0;
                uint firstChildBFS = 0;
                bool hasFirst = false;

                for (uint ci = 0; ci < 8; ci++)
                {
                    int3 childCell = bfsItem.cellInt * 2 + int3((int)(ci & 1), (int)((ci >> 1) & 1), (int)((ci >> 2) & 1));
                    uint64_t childKey = makeNodeKey(bfsItem.level + 1, childCell);
                    auto childIt = keyToBFSIndex.find(childKey);
                    if (childIt != keyToBFSIndex.end())
                    {
                        octNode.childMask |= (1u << ci);
                        if (!hasFirst)
                        {
                            firstChildBFS = childIt->second;
                            hasFirst = true;
                        }
                    }
                }

                octNode.childBase = hasFirst ? firstChildBFS : 0;
            }
            else
            {
                octNode.childBase = 0;
                octNode.childMask = 0;
            }

            mOctreeNodes.push_back(octNode);
            bfsIdx++;
        }

        // Build level node counts
        for (uint l = 0; l <= maxDepth; l++)
            mOctreeNodeCounts[l] = levelNodeCount[l];

        // Update gridData
        gridData.solidVoxelCount = gBuffer.size();
    }
};
