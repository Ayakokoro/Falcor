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

    struct PendingClip
    {
        int3 cellInt;
        Polygon poly;
    };

    static constexpr uint kFlushBatchSize = 4096;

public:
    std::vector<VoxelData> gBuffer;
    std::vector<int> vBuffer;
    std::vector<std::vector<Polygon>> polygonArrays;
    std::vector<PolygonRange> polygonRangeBuffer;

    PolygonGenerator() : gridData(VoxelizationBase::GlobalGridData) {}

    void reset()
    {
        gBuffer.clear();
        vBuffer.clear();
        polygonArrays.clear();
        polygonRangeBuffer.clear();
        vBuffer.assign(gridData.totalVoxelCount(), -1);
    }

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

    // ---- 单线程路径 ----

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

    // ---- 多线程路径 ----

    // 无锁裁剪：结果写入线程本地 pending buffer
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

    // 批量写入全局容器（加锁一次）
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
};
