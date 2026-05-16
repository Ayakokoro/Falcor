#pragma once
#include "VoxelizationBase.h"
#include "Profiler.h"
#include "Math/Polygon.slang"
#include "Math/Triangle.slang"
#include "Math/SphericalHarmonics.slang"
#include <unordered_map>

using namespace Falcor;

class PolygonGenerator
{
private:
    GridData& gridData;

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

    void clip(const MeshHeader& mesh, uint triangleID, Triangle& tri)
    {
        AABBInt aabb = tri.calcAABBInt();
        for (int i = 0; i < aabb.count(); i++)
        {
            int3 cellInt = aabb.indexToCell(i);
            float3 minPoint = float3(cellInt);
            Polygon polygon = VoxelizationUtility::BoxClipTriangle(minPoint, minPoint + 1.f, tri); // 多边形与三角形顶点顺序一致
            polygon.normal = tri.TBN.getCol(2);                                                    // 几何法线
            if (polygon.count >= 3)
            {
                // sampleArea(tri, polygon, cellInt);
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

            // 世界坐标处理成网格坐标
            for (int i = 0; i < 3; i++)
            {
                tri.vertices[i] = (tri.vertices[i] - gridData.gridMin) / gridData.voxelSize;
            }
            tri.buildTBN();
            clip(mesh, tid, tri);
        }
    }

    void clipAll(SceneHeader scene, std::vector<MeshHeader> meshList, float3* pPos, float3* pNormal, float2* pUV, uint3* pTri)
    {
        for (size_t i = 0; i < meshList.size(); i++)
        {
            clipMesh(meshList[i], pPos, pNormal, pUV, pTri);
        }
        gridData.solidVoxelCount = gBuffer.size();
    }
};
