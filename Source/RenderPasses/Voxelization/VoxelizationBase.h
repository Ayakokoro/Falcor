#pragma once
#include "Voxel/VoxelGrid.slang"
#include "Voxel/ABSDF.slang"
#include "Math/Ellipsoid.slang"
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Math/VoxelizationUtility.h"
#include "Math/Random.h"
#include "VoxelizationShared.slang"
#include <random>
using namespace Falcor;

inline std::string ToString(float3 v)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return oss.str();
}

inline uint nextPowerOfTwo(uint v)
{
    if (v <= 1) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}
inline std::string ToString(int2 v)
{
    std::ostringstream oss;
    oss << "(" << v.x << ", " << v.y << ")";
    return oss.str();
}
inline std::string ToString(int3 v)
{
    std::ostringstream oss;
    oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return oss.str();
}

inline std::string kGBuffer = "gBuffer";
inline std::string kPBuffer = "pBuffer";
inline std::string kOctreeBuffer = "octreeBuffer";
inline std::string kPolygonBuffer = "polygonBuffer";
inline std::string kPolygonRangeBuffer = "polygonRangeBuffer";

class VoxelizationBase
{
public:
    static const int NDFLobeCount = 8;
    static GridData GlobalGridData;
    static uint3 MinFactor; // 网格的分辨率必须是此值的整数倍
    static bool FileUpdated;
    static std::string ResourceFolder;
    static bool LightChanged;

    static ref<Buffer> GBuffer;
    static ref<Buffer> PBuffer;

    // Octree data
    static ref<Buffer> OctreeBuffer;
    static uint32_t OctreeMaxDepth;
    static std::vector<uint32_t> OctreeNodeCounts;

    static void UpdateVoxelGrid(ref<Scene> scene, uint baseVoxelResolution)
    {
        float3 diag;
        float3 center;
        if (scene)
        {
            AABB aabb = scene->getSceneBounds();
            // 获取膨胀后的场景尺寸与中心
            diag = (aabb.maxPoint - aabb.minPoint) * 1.02f; // 引入 2% 的安全边距
            center = aabb.center();
        }
        else
        {
            diag = float3(1.02f);
            center = float3(0.f);
        }

        // 强制每个轴的体素数都为同一个 2 的幂次方
        // 将基础分辨率对齐到最近的 2 的幂
        uint N = nextPowerOfTwo(baseVoxelResolution);
        GlobalGridData.voxelCount = uint3(N, N, N);

        // 找到场景膨胀后的最大边长（作为正方体网格的物理边长）
        float maxDim = std::max(diag.z, std::max(diag.x, diag.y));

        // 计算单个正方体体素的物理边长
        float s = maxDim / (float)N;
        GlobalGridData.voxelSize = float3(s);

        // 以场景中心为原点，计算体素网格的最小角起点（gridMin）
        GlobalGridData.gridMin = center - 0.5f * s * float3(N);
        GlobalGridData.solidVoxelCount = 0;
    }
};

struct SceneHeader
{
    uint meshCount;
    uint vertexCount;
    uint triangleCount;
};

struct MeshHeader
{
    uint meshID;
    uint materialID;
    uint vertexCount;
    uint triangleCount;
    uint triangleOffset;
};

inline ref<Buffer> copyToCpu(ref<Device> pDevice, RenderContext* pRenderContext, ref<Buffer> gpuBuffer)
{
    ref<Buffer> cpuBuffer = pDevice->createBuffer(gpuBuffer->getSize(), ResourceBindFlags::None, MemoryType::ReadBack);
    pRenderContext->copyResource(cpuBuffer.get(), gpuBuffer.get());
    return cpuBuffer;
}

class PolygonBufferGroup
{
private:
    GridData& gridData;
    ref<Device> mpDevice;
    std::vector<ref<Buffer>> mBuffers;
    std::vector<uint> voxelCount;     // 各组中的体素个数
    std::vector<uint> gBufferOffsets; // 每一组内的第一个体素在gBuffer中的偏移量
    std::vector<uint> polygonCount;   // 各组中的多边形个数

    std::vector<Polygon> currentPolygons; // 正在处理的Polygon
    uint currentVoxelCount = 0;
    uint currentPolygonCount = 0;

    void flushCurrent()
    {
        if (currentPolygonCount == 0)
            return;

        if (size() == 0)
            gBufferOffsets.push_back(0);
        else
            gBufferOffsets.push_back(voxelCount.back() + gBufferOffsets.back());

        ref<Buffer> buffer = mpDevice->createStructuredBuffer(
            sizeof(Polygon), currentPolygonCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );

        if (currentPolygons.size() > 0)
            buffer->setBlob(currentPolygons.data(), 0, size_t(currentPolygonCount) * sizeof(Polygon));

        mBuffers.push_back(buffer);
        voxelCount.push_back(currentVoxelCount);
        polygonCount.push_back(currentPolygonCount);

        currentPolygons.clear();
        currentVoxelCount = 0;
        currentPolygonCount = 0;
    }

public:
    uint maxPolygonCount = 256000;
    static constexpr uint kSafePerNodePolygonLimit = 65536; // per-thread GPU safety cap to avoid TDR
    PolygonBufferGroup(ref<Device> device, GridData& gridData) : gridData(gridData), mpDevice(device) {}

    uint getVoxelOffset(uint index) const
    {
        FALCOR_ASSERT(index < mBuffers.size());
        return gBufferOffsets[index];
    }

    uint getVoxelCount(uint index) const
    {
        FALCOR_ASSERT(index < mBuffers.size());
        return voxelCount[index];
    }

    uint getPolygonCount(uint index) const
    {
        FALCOR_ASSERT(index < mBuffers.size());
        return polygonCount[index];
    }

    uint size() const { return mBuffers.size(); }

    ref<Buffer> get(uint index)
    {
        FALCOR_ASSERT(index < mBuffers.size());
        return mBuffers[index];
    }

    void reset()
    {
        mBuffers.clear();
        voxelCount.clear();
        polygonCount.clear();
        gBufferOffsets.clear();
        currentVoxelCount = 0;
        currentPolygonCount = 0;
        currentPolygons.reserve(maxPolygonCount);
        gridData.maxPolygonCount = 0;
        gridData.totalPolygonCount = 0;
    }

    // 用于CPU上已经裁剪完成的情况
    void setBlob(const std::vector<std::vector<Polygon>>& polygonArrays, std::vector<PolygonRange>& polygonRangeBuffer)
    {
        FALCOR_ASSERT(polygonRangeBuffer.size() == polygonArrays.size());
        reset();

        for (size_t v = 0; v < polygonArrays.size(); ++v)
        {
            const std::vector<Polygon>& polys = polygonArrays[v];
            uint n = (uint)polys.size();

            if (n == 0)
                continue;

            uint effectiveLimit = std::min(maxPolygonCount, kSafePerNodePolygonLimit);
            if (n > effectiveLimit)
            {
                logWarning("setBlob: node has " + std::to_string(n) + " polygons, capping to " +
                           std::to_string(effectiveLimit));
                n = effectiveLimit;
            }

            if (currentPolygonCount + n > maxPolygonCount)
            {
                flushCurrent();
            }

            polygonRangeBuffer[v].count = n;
            polygonRangeBuffer[v].localHead = currentPolygonCount;

            gridData.maxPolygonCount = max(gridData.maxPolygonCount, n);
            gridData.totalPolygonCount += n;

            currentPolygons.insert(currentPolygons.end(), polys.begin(), polys.begin() + n);
            currentPolygonCount += n;
            currentVoxelCount++;
        }

        flushCurrent();
    }

    // From GPU ClipTriangle Pass 2 buffer initialization (no repacking)
    // polygonRanges filled by caller after Pass 1: localHead/count/cellInt
    void adoptBuffer(ref<Buffer> polygonBufferGPU, std::vector<PolygonRange>& polygonRanges, uint solidVoxelCount)
    {
        reset();
        gBufferOffsets.push_back(0);
        voxelCount.push_back(solidVoxelCount);
        polygonCount.push_back(0);
        mBuffers.push_back(polygonBufferGPU);
    }

    // Set localHead/count for each solid voxel based on polygonCounts
    // polygonRanges[voxelOffset].localHead = local polygon start index
    // polygonRanges[voxelOffset].count = number of polygons in this voxel
    void reserve(const std::vector<uint>& polygonCounts, std::vector<PolygonRange>& polygonRanges)
    {
        uint localHead = 0;
        for (size_t i = 0; i < polygonCounts.size(); ++i)
        {
            polygonRanges[i].localHead = localHead;
            polygonRanges[i].count = polygonCounts[i];
            localHead += polygonCounts[i];
        }
    }
};
