#include "RayMarchingPass.h"
#include "VoxelizationMetadata.h"
#include "VoxelSceneMetadata.h"
#include "Shading.slang"
#include "Math/SphericalHarmonics.slang"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Math/FalcorMath.h"
#include <algorithm>
#include <cmath>
#include <execution>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <limits>

namespace
{
const std::string kShaderFile = "RenderPasses/Voxelization/RayMarching.ps.slang";
const std::string kDisplayShaderFile = "RenderPasses/Voxelization/DisplayNDF.ps.slang";
const std::string kOutputColor = "color";
const std::string kBufferCountDefine = "RAY_MARCHING_BUFFER_COUNT";
constexpr size_t kRayMarchingBufferByteLimit = size_t(1) << 30; // 1 GiB per resource.
constexpr size_t kVoxelConversionChunkElements = size_t(64) * 1024;
constexpr uint32_t kMaxRayMarchingOctreeDepth = 23;

bool isVoxelSceneMetaFile(const std::filesystem::path& path)
{
    const std::string filename = path.filename().string();
    constexpr const char* kJsonSuffix = ".voxscene.json";
    const size_t suffixLength = std::char_traits<char>::length(kJsonSuffix);
    return path.extension() == ".voxscene" ||
        (filename.size() >= suffixLength &&
         filename.compare(filename.size() - suffixLength, suffixLength, kJsonSuffix) == 0);
}

// Host-side conversion replicating TEBSDF::init + SurfaceBRDF::init + LobeBRDF::init
// (Shading.slang) so the raw VoxelData never has to be staged on the GPU.
inline void buildTBNHost(float3 n, float3& t, float3& b)
{
    float3 up = (std::abs(n.z) < 0.999f) ? float3(0, 0, 1) : float3(1, 0, 0);
    t = normalize(cross(up, n));
    b = cross(n, t);
}

TEBSDF convertVoxelData(const VoxelData& data)
{
    const float kMinRough = 0.01f;  // matches Shading.slang MinRough
    TEBSDF gb{};
    gb.area = data.ABSDF.area;
    gb.coverage = 12345.0f;  // sentinel, matches PrepareShadingData.cs.slang

    if (gb.area > 0)
    {
        gb.primitiveProjAreaFunc = data.primitiveProjAreaFunc;
        gb.polygonsProjAreaFunc = data.polygonsProjAreaFunc;
        gb.totalProjAreaFunc = data.totalProjAreaFunc;

        SurfaceBRDF& surface = gb.surface;
        uint lobeCount = 0;
        for (uint i = 0; i < LOBE_COUNT; i++)
        {
            const ABSDFLobe& lobe = data.ABSDF.lobes[i];
            if (lobe.weight > 0 && dot(lobe.normal, lobe.normal) > 0)
            {
                LobeBRDF& l = surface.lobes[lobeCount];
                l.n = lobe.normal;
                buildTBNHost(l.n, l.t, l.b);
                float rough = max(lobe.rough, kMinRough);
                l.frostBite.diffuse = lobe.diffuse;
                l.frostBite.rough = rough;
                l.cookTorrence.specular = lobe.specular;
                l.cookTorrence.alpha = rough * rough;
                float3 spec = l.cookTorrence.specular;
                l.specularWeight = saturate((20 * (spec.x + spec.y + spec.z) / 3 + 1) / 21.0f);
                surface.weights[lobeCount] = lobe.weight;
                lobeCount++;
            }
        }
        if (lobeCount == 0)
        {
            LobeBRDF& l = surface.lobes[0];
            l.n = float3(0, 0, 1);
            buildTBNHost(l.n, l.t, l.b);
            l.frostBite.diffuse = float3(0);
            l.frostBite.rough = kMinRough;
            l.cookTorrence.specular = float3(0);
            l.cookTorrence.alpha = kMinRough * kMinRough;
            l.specularWeight = saturate(1.0f / 21.0f);
            surface.weights[0] = 1.0f;
            lobeCount = 1;
        }
        surface.lobeCount = lobeCount;
    }
    return gb;
}
} // namespace

void RayMarchingPass::updateInstanceTransforms()
{
    for (size_t i = 0; i < mInstances.size(); ++i)
    {
        VoxelInstance& instance = mInstances[i];
        if (instance.assetIndex >= mVoxelAssets.size())
        {
            instance.worldBoundsMin = float3(0.0f);
            instance.worldBoundsMax = float3(0.0f);
            continue;
        }

        const GridData& assetGrid = mVoxelAssets[instance.assetIndex].gridData;
        const float3 gridExtent(
            assetGrid.voxelSize.x * float(assetGrid.voxelCount.x),
            assetGrid.voxelSize.y * float(assetGrid.voxelCount.y),
            assetGrid.voxelSize.z * float(assetGrid.voxelCount.z)
        );

        // Include half a voxel around the grid for a conservative broad-phase
        // bound. The local traversal still clips against the exact grid bounds.
        const float3 boundsMinLocal = assetGrid.gridMin - 0.5f * assetGrid.voxelSize;
        const float3 boundsMaxLocal = assetGrid.gridMin + gridExtent + 0.5f * assetGrid.voxelSize;
        const float3 corners[8] = {
            float3(boundsMinLocal.x, boundsMinLocal.y, boundsMinLocal.z),
            float3(boundsMaxLocal.x, boundsMinLocal.y, boundsMinLocal.z),
            float3(boundsMinLocal.x, boundsMaxLocal.y, boundsMinLocal.z),
            float3(boundsMaxLocal.x, boundsMaxLocal.y, boundsMinLocal.z),
            float3(boundsMinLocal.x, boundsMinLocal.y, boundsMaxLocal.z),
            float3(boundsMaxLocal.x, boundsMinLocal.y, boundsMaxLocal.z),
            float3(boundsMinLocal.x, boundsMaxLocal.y, boundsMaxLocal.z),
            float3(boundsMaxLocal.x, boundsMaxLocal.y, boundsMaxLocal.z),
        };

        // Scene metadata stores the complete asset-local-to-world affine
        // transform. Do not apply the old grid-center pivot transform here.
        instance.localToWorld = instance.assetToWorld;

        const float determinant = math::determinant(instance.localToWorld);
        if (!std::isfinite(determinant) || std::abs(determinant) <= 1e-12f)
        {
            instance.worldBoundsMin = float3(0.0f);
            instance.worldBoundsMax = float3(0.0f);
            continue;
        }

        instance.worldToLocal = math::inverse(instance.localToWorld);
        instance.normalTransform = math::transpose(instance.worldToLocal);

        float3 boundsMinW(std::numeric_limits<float>::max());
        float3 boundsMaxW(-std::numeric_limits<float>::max());
        for (const float3& corner : corners)
        {
            const float3 cornerW = math::mul(instance.localToWorld, float4(corner, 1.0f)).xyz();
            boundsMinW = min(boundsMinW, cornerW);
            boundsMaxW = max(boundsMaxW, cornerW);
        }
        instance.worldBoundsMin = boundsMinW;
        instance.worldBoundsMax = boundsMaxW;
    }
}

void RayMarchingPass::updateScreenSpaceLOD(
    const float4x4& viewProj,
    VoxelInstance& instance,
    bool updateDebugStats
)
{
    constexpr float kTargetVoxelSizePixels = 0.5f;
    constexpr float kProjectionEpsilon = 1e-6f;

    instance.screenLOD = 0;
    if (updateDebugStats)
    {
        mGridProjectionValid = false;
        mGridProjectedWidthPixels = 0.0f;
        mGridProjectedHeightPixels = 0.0f;
        mGridProjectedAreaPixels = 0.0f;
        mLeafProjectedSizePixels = 0.0f;
        mScreenLOD = 0;
    }

    if (instance.assetIndex >= mVoxelAssets.size())
        return;

    const VoxelAsset& asset = mVoxelAssets[instance.assetIndex];
    const GridData& assetGrid = asset.gridData;
    const uint32_t maxLOD = std::min(asset.octreeMaxDepth, asset.availableLODLevels);
    if (mOutputResolution.x == 0 || mOutputResolution.y == 0 || assetGrid.voxelCount.x == 0)
        return;

    const float3 gridExtent(
        assetGrid.voxelSize.x * float(assetGrid.voxelCount.x),
        assetGrid.voxelSize.y * float(assetGrid.voxelCount.y),
        assetGrid.voxelSize.z * float(assetGrid.voxelCount.z)
    );
    const float3 gridMax = assetGrid.gridMin + gridExtent;

    const float3 corners[8] = {
        float3(assetGrid.gridMin.x, assetGrid.gridMin.y, assetGrid.gridMin.z),
        float3(gridMax.x,           assetGrid.gridMin.y, assetGrid.gridMin.z),
        float3(assetGrid.gridMin.x, gridMax.y,           assetGrid.gridMin.z),
        float3(gridMax.x,           gridMax.y,           assetGrid.gridMin.z),
        float3(assetGrid.gridMin.x, assetGrid.gridMin.y, gridMax.z),
        float3(gridMax.x,           assetGrid.gridMin.y, gridMax.z),
        float3(assetGrid.gridMin.x, gridMax.y,           gridMax.z),
        float3(gridMax.x,          gridMax.y,          gridMax.z),
    };

    const float maxFloat = std::numeric_limits<float>::max();
    float2 minPixels(maxFloat);
    float2 maxPixels(-maxFloat);

    for (const float3& corner : corners)
    {
        const float3 worldCorner = math::mul(instance.localToWorld, float4(corner, 1.0f)).xyz();
        const float4 clip = math::mul(viewProj, float4(worldCorner, 1.0f));

        // A grid crossing the near plane cannot be represented by a single
        // finite screen-space rectangle. Selecting the finest level is the
        // conservative choice for image quality in this case.
        if (!(clip.w > kProjectionEpsilon) ||
            !std::isfinite(clip.x) || !std::isfinite(clip.y) || !std::isfinite(clip.w))
        {
            return;
        }

        const float2 ndc = float2(clip.x, clip.y) / clip.w;
        const float2 pixels(
            (ndc.x * 0.5f + 0.5f) * float(mOutputResolution.x),
            (1.0f - (ndc.y * 0.5f + 0.5f)) * float(mOutputResolution.y)
        );

        if (!std::isfinite(pixels.x) || !std::isfinite(pixels.y))
            return;

        minPixels = min(minPixels, pixels);
        maxPixels = max(maxPixels, pixels);
    }

    const float2 projectedSize = maxPixels - minPixels;
    const float projectedArea = projectedSize.x * projectedSize.y;
    if (!(projectedSize.x > 0.0f) || !(projectedSize.y > 0.0f) ||
        !std::isfinite(projectedArea))
    {
        return;
    }

    const float leafProjectedSizePixels = std::sqrt(projectedArea) / float(assetGrid.voxelCount.x);
    int screenLOD = 0;

    // The projected grid area is divided by N^2 cells. Using the square root
    // gives an equivalent edge length in pixels, which is also well-defined
    // when the grid is viewed obliquely. A node at LOD L spans 2^L leaf cells.
    float nodeProjectedSize = leafProjectedSizePixels;
    while (screenLOD < int(maxLOD) && nodeProjectedSize <= kTargetVoxelSizePixels)
    {
        ++screenLOD;
        nodeProjectedSize *= 2.0f;
    }

    if (mMaxLODLevel >= 0)
        screenLOD = std::min(screenLOD, mMaxLODLevel);

    instance.screenLOD = screenLOD;
    if (updateDebugStats)
    {
        mGridProjectionValid = true;
        mGridProjectedWidthPixels = projectedSize.x;
        mGridProjectedHeightPixels = projectedSize.y;
        mGridProjectedAreaPixels = projectedArea;
        mLeafProjectedSizePixels = leafProjectedSizePixels;
        mScreenLOD = screenLOD;
    }
}

void RayMarchingPass::updateInstanceBuffer()
{
    if (mInstances.empty())
    {
        mInstanceBuffer = nullptr;
        return;
    }

    std::vector<VoxelInstanceGPU> gpuInstances;
    gpuInstances.reserve(mInstances.size());
    for (const VoxelInstance& instance : mInstances)
    {
        VoxelInstanceGPU gpu{};
        gpu.localToWorld = instance.localToWorld;
        gpu.worldToLocal = instance.worldToLocal;
        gpu.normalTransform = instance.normalTransform;
        gpu.worldBoundsMin = float4(instance.worldBoundsMin, 0.0f);
        gpu.worldBoundsMax = float4(instance.worldBoundsMax, 0.0f);
        gpu.instanceId = instance.instanceId;
        gpu.screenLOD = instance.screenLOD;
        gpu.enabled = instance.enabled ? 1u : 0u;
        gpu.assetIndex = instance.assetIndex;
        gpuInstances.push_back(gpu);
    }

    const uint32_t instanceCount = static_cast<uint32_t>(gpuInstances.size());
    if (!mInstanceBuffer || mInstanceBuffer->getElementCount() != instanceCount)
    {
        mInstanceBuffer = mpDevice->createStructuredBuffer(
            sizeof(VoxelInstanceGPU),
            instanceCount,
            ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            gpuInstances.data(),
            false
        );
    }
    else
    {
        mInstanceBuffer->setBlob(
            gpuInstances.data(),
            0,
            gpuInstances.size() * sizeof(VoxelInstanceGPU)
        );
    }
}

// Recursively build a balanced BVH. The recursion permutes instancePositions,
// whose entries are *actual indices into the GPU instance buffer* (== indices
// into mInstances). The parallel boundsMin/BoundsMax vectors are indexed by
// that same buffer position. Interior nodes are emitted depth-first so the
// left child of a node always follows it at node+1; the right child is placed
// at node + rightChildOffset. Leaves hold exactly one instance position.
static uint32_t buildBVHRecursive(
    std::vector<uint32_t>& instancePositions,
    const std::vector<float3>& boundsMinByPosition,
    const std::vector<float3>& boundsMaxByPosition,
    uint32_t first,
    uint32_t last,
    uint32_t nextFree,
    std::vector<VoxelInstanceBVHNodeGPU>& nodes
)
{
    const uint32_t nodeIndex = nextFree++;
    const uint32_t count = last - first;

    if (count == 1)
    {
        const uint32_t position = instancePositions[first];
        VoxelInstanceBVHNodeGPU node{};
        node.boundsMin = float4(boundsMinByPosition[position], 0.0f);
        node.boundsMax = float4(boundsMaxByPosition[position], 0.0f);
        node.info = uint4(1u, 0u, position, 0u);
        nodes[nodeIndex] = node;
        return nextFree;
    }

    // Union AABB of this range.
    float3 rangeMin(std::numeric_limits<float>::max());
    float3 rangeMax(-std::numeric_limits<float>::max());
    for (uint32_t i = first; i < last; ++i)
    {
        rangeMin = min(rangeMin, boundsMinByPosition[instancePositions[i]]);
        rangeMax = max(rangeMax, boundsMaxByPosition[instancePositions[i]]);
    }

    VoxelInstanceBVHNodeGPU node{};
    node.boundsMin = float4(rangeMin, 0.0f);
    node.boundsMax = float4(rangeMax, 0.0f);
    node.info = uint4(0u, 0u, 0u, 0u);

    // Median split on the longest axis.
    const float3 extent = rangeMax - rangeMin;
    const uint32_t axis = (extent.x >= extent.y && extent.x >= extent.z) ? 0u :
                          (extent.y >= extent.z) ? 1u : 2u;
    const uint32_t median = first + count / 2;
    std::nth_element(
        instancePositions.begin() + first,
        instancePositions.begin() + median,
        instancePositions.begin() + last,
        [axis, &boundsMinByPosition](uint32_t a, uint32_t b)
        {
            const float3& ca = boundsMinByPosition[a];
            const float3& cb = boundsMinByPosition[b];
            return axis == 0 ? ca.x < cb.x :
                   axis == 1 ? ca.y < cb.y : ca.z < cb.z;
        }
    );

    // Left child emitted first (node+1); right child follows the whole subtree.
    nextFree = buildBVHRecursive(instancePositions, boundsMinByPosition, boundsMaxByPosition, first, median, nextFree, nodes);
    const uint32_t rightStart = nextFree;
    nextFree = buildBVHRecursive(instancePositions, boundsMinByPosition, boundsMaxByPosition, median, last, nextFree, nodes);

    node.info.y = rightStart - nodeIndex;
    nodes[nodeIndex] = node;
    return nextFree;
}

void RayMarchingPass::buildInstanceBVH()
{
    mInstanceBVH.clear();

    // Gather the *mInstances positions* of instances that participate in ray
    // marching. The GPU instance buffer preserves mInstances order, so BVH
    // leaves must reference those same positions for traceInstance().
    std::vector<uint32_t> instancePositions;
    std::vector<float3> boundsMinByPosition(mInstances.size(), float3(0.0f));
    std::vector<float3> boundsMaxByPosition(mInstances.size(), float3(0.0f));
    for (size_t i = 0; i < mInstances.size(); ++i)
    {
        const VoxelInstance& instance = mInstances[i];
        if (!instance.enabled ||
            instance.assetIndex >= mVoxelAssets.size() ||
            !all(instance.worldBoundsMax - instance.worldBoundsMin > float3(0.0f)))
            continue;
        boundsMinByPosition[i] = instance.worldBoundsMin;
        boundsMaxByPosition[i] = instance.worldBoundsMax;
        instancePositions.push_back(static_cast<uint32_t>(i));
    }

    if (instancePositions.empty())
    {
        updateInstanceBVHBuffer();
        return;
    }

    // Two nodes per instance in the worst case (one leaf plus shared interiors).
    mInstanceBVH.resize(instancePositions.size() * 2u);
    const uint32_t nodeCount = buildBVHRecursive(
        instancePositions,
        boundsMinByPosition,
        boundsMaxByPosition,
        0u,
        static_cast<uint32_t>(instancePositions.size()),
        0u,
        mInstanceBVH
    );
    mInstanceBVH.resize(nodeCount);

    updateInstanceBVHBuffer();
}

void RayMarchingPass::updateInstanceBVHBuffer()
{
    if (mInstanceBVH.empty())
    {
        mInstanceBVHBuffer = nullptr;
        return;
    }

    const uint32_t nodeCount = static_cast<uint32_t>(mInstanceBVH.size());
    if (!mInstanceBVHBuffer || mInstanceBVHBuffer->getElementCount() != nodeCount)
    {
        mInstanceBVHBuffer = mpDevice->createStructuredBuffer(
            sizeof(VoxelInstanceBVHNodeGPU),
            nodeCount,
            ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            mInstanceBVH.data(),
            false
        );
    }
    else
    {
        mInstanceBVHBuffer->setBlob(
            mInstanceBVH.data(),
            0,
            mInstanceBVH.size() * sizeof(VoxelInstanceBVHNodeGPU)
        );
    }
}

void RayMarchingPass::rebuildInstanceBVHIfDirty()
{
    if (!mInstanceBVHDirty)
        return;
    buildInstanceBVH();
    mInstanceBVHDirty = false;
}

void RayMarchingPass::resetVoxelResources()
{
    // The shader array size is compiled from mBufferCount. These resources and
    // both fullscreen passes must be discarded together when a new asset set
    // is selected, otherwise a scene with a different split count can retain the
    // previous shader layout for one frame.
    mGBuffers.clear();
    mPBuffers.clear();
    mGBufferSplits.clear();
    mOctreeBuffer = nullptr;
    mAssetBuffer = nullptr;
    mVoxelAssets.clear();
    mBufferCount = 1;
    mInstanceBuffer = nullptr;
    mInstanceBVH.clear();
    mInstanceBVHBuffer = nullptr;
    mInstanceBVHDirty = true;
    mSelectedVoxel = nullptr;
    mpSelectedVoxelStaging = nullptr;
    mSelectedHit = false;
    mSelectedGbOffset = 0xFFFFFFFF;
    mSelectedCellInt = int3(-1);
    mSelectedInstanceId = 0xFFFFFFFF;
    mScreenLOD = 0;
    mGridProjectionValid = false;
    mpFullScreenPass = nullptr;
    mpDisplayNDFPass = nullptr;
}

void RayMarchingPass::resetInstancesToIdentity()
{
    mInstances.clear();
    mInstances.emplace_back();
    mInstanceEditIndex = 0;
    mInstanceBuffer = nullptr;
    mInstanceBVH.clear();
    mInstanceBVHBuffer = nullptr;
    mInstanceBVHDirty = true;
    mSelectedHit = false;
    mSelectedGbOffset = 0xFFFFFFFF;
    mSelectedCellInt = int3(-1);
    mSelectedInstanceId = 0xFFFFFFFF;
}

bool RayMarchingPass::loadSceneMeta(const std::filesystem::path& path)
{
    VoxelSceneMetadata::Scene scene;
    std::string error;
    if (!VoxelSceneMetadata::read(path, scene, error))
    {
        mSceneMetaError = std::move(error);
        std::cerr << "Failed to load voxel scene meta: " << mSceneMetaError << std::endl;
        return false;
    }

    std::vector<VoxelInstance> instances;
    instances.reserve(scene.instances.size());
    for (const VoxelSceneMetadata::Instance& source : scene.instances)
    {
        VoxelInstance instance;
        instance.instanceId = source.instanceId;
        instance.assetIndex = source.assetIndex;
        instance.enabled = source.enabled;
        instance.assetToWorld = math::matrixFromCoefficients<float, 4, 4>(
            source.transform.data()
        );
        instances.push_back(instance);
    }

    std::vector<VoxelAsset> assets;
    assets.reserve(scene.assets.size());
    for (const VoxelSceneMetadata::Asset& source : scene.assets)
    {
        VoxelAsset asset;
        asset.assetId = source.assetId;
        asset.voxelFile = source.voxelFile;
        assets.push_back(std::move(asset));
    }

    resetVoxelResources();
    mVoxelAssets = std::move(assets);
    mInstances = std::move(instances);
    mInstanceEditIndex = 0;
    mVoxelFilePath = mVoxelAssets.front().voxelFile;
    mSceneMetaPath = scene.sourcePath;
    mSceneMetaLoaded = true;
    mSceneMetaError.clear();

    requestRecompile();
    mComplete = false;
    mOptionsChanged = true;
    mInstanceBVHDirty = true;
    return true;
}

bool RayMarchingPass::loadVoxelResources(RenderContext* pRenderContext)
{
    auto failLoad = [this](const std::string& message)
    {
        mSceneMetaError = message;
        std::cerr << "Failed to load voxel resources: " << message << std::endl;
        return false;
    };

    std::vector<VoxelAsset> loadedAssets = mVoxelAssets;
    if (loadedAssets.empty())
    {
        if (mVoxelFilePath.empty() && selectedFile < filePaths.size())
            mVoxelFilePath = filePaths[selectedFile];
        if (mVoxelFilePath.empty())
            return false;

        VoxelAsset asset;
        asset.assetId = mVoxelFilePath.stem().string();
        asset.voxelFile = mVoxelFilePath;
        loadedAssets.push_back(std::move(asset));
    }

    std::vector<std::vector<OctreeNode>> assetOctrees(loadedAssets.size());
    uint64_t totalNodeCount64 = 0;
    uint64_t totalVoxelCount64 = 0;

    // Pass 1 reads the lightweight headers and octrees, validates all local
    // indices, and relocates them into one global node/data index space.
    for (size_t assetIndex = 0; assetIndex < loadedAssets.size(); ++assetIndex)
    {
        VoxelAsset& asset = loadedAssets[assetIndex];
        std::ifstream input(asset.voxelFile, std::ios::binary | std::ios::ate);
        if (!input.is_open())
            return failLoad("Cannot open voxel asset '" + asset.assetId + "': " + asset.voxelFile.string());

        std::error_code sizeEc;
        const std::uintmax_t fileSizeWide = std::filesystem::file_size(asset.voxelFile, sizeEc);
        if (sizeEc || fileSizeWide > std::numeric_limits<size_t>::max())
            return failLoad("Cannot determine a supported file size for voxel asset '" + asset.assetId + "'.");
        const size_t fileSize = static_cast<size_t>(fileSizeWide);
        size_t offset = 0;

        if (!tryRead(input, offset, sizeof(GridData), &asset.gridData, fileSize))
            return failLoad("Voxel asset '" + asset.assetId + "' has a truncated GridData header.");
        if (!tryRead(input, offset, sizeof(uint32_t), &asset.octreeMaxDepth, fileSize))
            return failLoad("Voxel asset '" + asset.assetId + "' has a truncated octree header.");

        if (asset.octreeMaxDepth > kMaxRayMarchingOctreeDepth)
        {
            return failLoad(
                "Voxel asset '" + asset.assetId + "' has octree depth " +
                std::to_string(asset.octreeMaxDepth) + ", but the traversal supports at most " +
                std::to_string(kMaxRayMarchingOctreeDepth) + "."
            );
        }
        if (asset.gridData.voxelCount.x == 0 || asset.gridData.voxelCount.y == 0 ||
            asset.gridData.voxelCount.z == 0 ||
            !(asset.gridData.voxelSize.x > 0.0f) || !(asset.gridData.voxelSize.y > 0.0f) ||
            !(asset.gridData.voxelSize.z > 0.0f) ||
            !std::isfinite(asset.gridData.voxelSize.x) ||
            !std::isfinite(asset.gridData.voxelSize.y) ||
            !std::isfinite(asset.gridData.voxelSize.z))
        {
            return failLoad("Voxel asset '" + asset.assetId + "' has invalid grid dimensions or voxel size.");
        }
        const uint32_t rootSize = 1u << asset.octreeMaxDepth;
        const uint64_t gridVoxelCountXY =
            uint64_t(asset.gridData.voxelCount.x) * asset.gridData.voxelCount.y;
        if (asset.gridData.voxelCount.x > rootSize || asset.gridData.voxelCount.y > rootSize ||
            asset.gridData.voxelCount.z > rootSize ||
            gridVoxelCountXY > std::numeric_limits<size_t>::max() / asset.gridData.voxelCount.z ||
            !std::isfinite(asset.gridData.gridMin.x) ||
            !std::isfinite(asset.gridData.gridMin.y) ||
            !std::isfinite(asset.gridData.gridMin.z))
        {
            return failLoad("Voxel asset '" + asset.assetId + "' is incompatible with its octree root extent.");
        }
        if (asset.gridData.solidVoxelCount == 0 ||
            asset.gridData.solidVoxelCount > std::numeric_limits<uint32_t>::max())
        {
            return failLoad("Voxel asset '" + asset.assetId + "' has an unsupported voxel payload size.");
        }

        asset.octreeNodeCounts.resize(size_t(asset.octreeMaxDepth) + 1);
        const size_t nodeCountBytes = asset.octreeNodeCounts.size() * sizeof(uint32_t);
        if (!tryRead(input, offset, nodeCountBytes, asset.octreeNodeCounts.data(), fileSize))
            return failLoad("Voxel asset '" + asset.assetId + "' has truncated octree level counts.");
        if (asset.octreeNodeCounts.empty() || asset.octreeNodeCounts[0] != 1)
            return failLoad("Voxel asset '" + asset.assetId + "' must contain exactly one octree root.");

        uint64_t localNodeCount64 = 0;
        for (uint32_t count : asset.octreeNodeCounts)
            localNodeCount64 += count;
        if (localNodeCount64 == 0 || localNodeCount64 > std::numeric_limits<uint32_t>::max())
            return failLoad("Voxel asset '" + asset.assetId + "' has an unsupported octree node count.");
        if (totalNodeCount64 + localNodeCount64 > std::numeric_limits<uint32_t>::max())
            return failLoad("The concatenated scene octree exceeds the uint32 index range.");
        if (totalVoxelCount64 + asset.gridData.solidVoxelCount > std::numeric_limits<uint32_t>::max())
            return failLoad("The concatenated voxel payload exceeds the uint32 index range.");

        const uint32_t localNodeCount = static_cast<uint32_t>(localNodeCount64);
        std::vector<OctreeNode>& nodes = assetOctrees[assetIndex];
        nodes.resize(localNodeCount);
        if (!tryRead(input, offset, size_t(localNodeCount) * sizeof(OctreeNode), nodes.data(), fileSize))
            return failLoad("Voxel asset '" + asset.assetId + "' has a truncated octree payload.");

        asset.octreeRoot = static_cast<uint32_t>(totalNodeCount64);
        asset.voxelDataOffset = static_cast<uint32_t>(totalVoxelCount64);
        asset.voxelDataFileOffset = offset;

        const uint32_t localVoxelCount = static_cast<uint32_t>(asset.gridData.solidVoxelCount);
        std::vector<uint32_t> levelStarts(asset.octreeNodeCounts.size() + 1, 0);
        for (size_t level = 0; level < asset.octreeNodeCounts.size(); ++level)
            levelStarts[level + 1] = levelStarts[level] + asset.octreeNodeCounts[level];

        for (uint32_t level = 0; level <= asset.octreeMaxDepth; ++level)
        {
            for (uint32_t nodeIndex = levelStarts[level]; nodeIndex < levelStarts[level + 1]; ++nodeIndex)
            {
                OctreeNode& node = nodes[nodeIndex];
                if ((node.childMask & ~0xffu) != 0)
                    return failLoad("Voxel asset '" + asset.assetId + "' contains an invalid octree child mask.");
                if (node.dataIndex >= localVoxelCount)
                {
                    return failLoad(
                        "Voxel asset '" + asset.assetId + "' contains an out-of-range dataIndex at node " +
                        std::to_string(nodeIndex) + "."
                    );
                }

                node.dataIndex += asset.voxelDataOffset;
                if (node.childMask != 0)
                {
                    uint32_t childCount = 0;
                    for (uint32_t bits = node.childMask & 0xffu; bits != 0; bits >>= 1)
                        childCount += bits & 1u;
                    const bool hasNextLevel = level < asset.octreeMaxDepth;
                    const uint32_t nextLevelStart = hasNextLevel ? levelStarts[level + 1] : localNodeCount;
                    const uint32_t nextLevelEnd = hasNextLevel ? levelStarts[level + 2] : localNodeCount;
                    if (!hasNextLevel || node.childBase < nextLevelStart ||
                        uint64_t(node.childBase) + childCount > nextLevelEnd)
                    {
                        return failLoad(
                            "Voxel asset '" + asset.assetId + "' contains an out-of-range childBase at node " +
                            std::to_string(nodeIndex) + "."
                        );
                    }
                    node.childBase += asset.octreeRoot;
                }
            }
        }

        const uint64_t payloadBytes = uint64_t(localVoxelCount) * sizeof(VoxelData);
        if (payloadBytes > fileSize || asset.voxelDataFileOffset > fileSize - size_t(payloadBytes))
            return failLoad("Voxel asset '" + asset.assetId + "' has a truncated VoxelData payload.");

        VoxelizationMetadata::Metadata metadata;
        std::error_code metadataEc;
        const bool hasSidecar = std::filesystem::exists(
            VoxelizationMetadata::sidecarPath(asset.voxelFile), metadataEc
        );
        if (VoxelizationMetadata::read(asset.voxelFile, metadata) &&
            metadata.maxDepth == asset.octreeMaxDepth &&
            (metadata.totalNodes == 0 || metadata.totalNodes == localNodeCount))
        {
            asset.hasVoxelMetadata = true;
            asset.voxelFormatVersion = metadata.version;
            asset.voxelLodMode = metadata.lodMode;
            asset.voxelProducer = metadata.producer;
            asset.availableLODLevels = std::min(metadata.generatedLodLevels, asset.octreeMaxDepth);
        }
        else if (hasSidecar)
        {
            asset.hasVoxelMetadata = false;
            asset.voxelFormatVersion = 0;
            asset.voxelLodMode.clear();
            asset.voxelProducer.clear();
            asset.availableLODLevels = 0;
            std::cerr << "Voxel metadata for '" << asset.assetId
                      << "' is invalid or does not match its binary; using leaf-only LOD." << std::endl;
        }
        else
        {
            asset.hasVoxelMetadata = false;
            asset.voxelFormatVersion = 0;
            asset.voxelLodMode = "legacy";
            asset.voxelProducer = "legacy";
            asset.availableLODLevels = asset.octreeMaxDepth;
        }

        totalNodeCount64 += localNodeCount64;
        totalVoxelCount64 += localVoxelCount;

        std::cout << "Voxel asset[" << assetIndex << "] '" << asset.assetId
                  << "': file=" << asset.voxelFile
                  << ", root=" << asset.octreeRoot
                  << ", dataBase=" << asset.voxelDataOffset
                  << ", nodes=" << localNodeCount
                  << ", voxels=" << localVoxelCount
                  << ", maxDepth=" << asset.octreeMaxDepth
                  << ", availableLODLevels=" << asset.availableLODLevels << std::endl;
    }

    const uint32_t totalNodeCount = static_cast<uint32_t>(totalNodeCount64);
    const uint32_t totalVoxelCount = static_cast<uint32_t>(totalVoxelCount64);
    std::vector<OctreeNode> sceneOctree;
    sceneOctree.reserve(totalNodeCount);
    for (std::vector<OctreeNode>& nodes : assetOctrees)
    {
        sceneOctree.insert(
            sceneOctree.end(),
            std::make_move_iterator(nodes.begin()),
            std::make_move_iterator(nodes.end())
        );
    }

    ref<Buffer> octreeBuffer = mpDevice->createStructuredBuffer(
        sizeof(OctreeNode), totalNodeCount, ResourceBindFlags::ShaderResource
    );
    octreeBuffer->setBlob(sceneOctree.data(), 0, sceneOctree.size() * sizeof(OctreeNode));
    assetOctrees.clear();
    sceneOctree.clear();
    sceneOctree.shrink_to_fit();

    const size_t maxElementsPerBuffer = kRayMarchingBufferByteLimit / sizeof(TEBSDF);
    FALCOR_CHECK(maxElementsPerBuffer > 0, "TEBSDF is larger than the ray-marching buffer limit.");
    const size_t splitCount64 =
        (size_t(totalVoxelCount) + maxElementsPerBuffer - 1) / maxElementsPerBuffer;
    FALCOR_CHECK(splitCount64 > 0 && splitCount64 <= std::numeric_limits<uint32_t>::max(),
                 "Unsupported ray-marching buffer split count.");
    const uint32_t splitCount = static_cast<uint32_t>(splitCount64);

    std::vector<uint32_t> counts(splitCount, 0);
    std::vector<uint32_t> splits(splitCount, 0);
    size_t globalBase = 0;
    for (uint32_t bufferIndex = 0; bufferIndex < splitCount; ++bufferIndex)
    {
        const size_t count = std::min(maxElementsPerBuffer, size_t(totalVoxelCount) - globalBase);
        counts[bufferIndex] = static_cast<uint32_t>(count);
        globalBase += count;
        splits[bufferIndex] = static_cast<uint32_t>(globalBase);
    }

    std::vector<ref<Buffer>> gBuffers(splitCount);
    std::vector<ref<Buffer>> pBuffers(splitCount);
    for (uint32_t bufferIndex = 0; bufferIndex < splitCount; ++bufferIndex)
    {
        gBuffers[bufferIndex] = mpDevice->createStructuredBuffer(
            sizeof(TEBSDF), counts[bufferIndex],
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal
        );
        pBuffers[bufferIndex] = mpDevice->createStructuredBuffer(
            sizeof(Ellipsoid), counts[bufferIndex],
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal
        );
    }

    // Pass 2 streams each VoxelData payload in bounded chunks. A chunk never
    // crosses a GPU split, so it can be uploaded directly at its global index.
    for (const VoxelAsset& asset : loadedAssets)
    {
        std::ifstream input(asset.voxelFile, std::ios::binary);
        if (!input.is_open())
            return failLoad("Cannot reopen voxel asset '" + asset.assetId + "' for payload streaming.");

        const uint32_t assetVoxelCount = static_cast<uint32_t>(asset.gridData.solidVoxelCount);
        uint32_t localBase = 0;
        while (localBase < assetVoxelCount)
        {
            const uint32_t globalIndex = asset.voxelDataOffset + localBase;
            const auto splitIt = std::upper_bound(splits.begin(), splits.end(), globalIndex);
            if (splitIt == splits.end())
                return failLoad("Internal voxel split lookup failed for asset '" + asset.assetId + "'.");

            const uint32_t bufferIndex = static_cast<uint32_t>(std::distance(splits.begin(), splitIt));
            const uint32_t bufferBase = bufferIndex == 0 ? 0 : splits[bufferIndex - 1];
            const uint32_t count = static_cast<uint32_t>(std::min<size_t>({
                size_t(assetVoxelCount - localBase),
                size_t(splits[bufferIndex] - globalIndex),
                kVoxelConversionChunkElements
            }));

            std::vector<VoxelData> source(count);
            const uint64_t sourceOffset64 = uint64_t(asset.voxelDataFileOffset) +
                uint64_t(localBase) * sizeof(VoxelData);
            if (sourceOffset64 > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()))
                return failLoad("Voxel payload offset is too large for asset '" + asset.assetId + "'.");
            input.seekg(static_cast<std::streamoff>(sourceOffset64), std::ios::beg);
            input.read(
                reinterpret_cast<char*>(source.data()),
                static_cast<std::streamsize>(size_t(count) * sizeof(VoxelData))
            );
            if (!input)
                return failLoad("Failed while streaming VoxelData for asset '" + asset.assetId + "'.");

            std::vector<TEBSDF> gbChunk(count);
            std::vector<Ellipsoid> pChunk(count);
            std::transform(
                std::execution::par_unseq,
                source.begin(),
                source.end(),
                gbChunk.begin(),
                [](const VoxelData& data) { return convertVoxelData(data); }
            );
            std::transform(
                std::execution::par_unseq,
                source.begin(),
                source.end(),
                pChunk.begin(),
                [](const VoxelData& data) { return data.ellipsoid; }
            );

            const size_t destinationElement = size_t(globalIndex - bufferBase);
            gBuffers[bufferIndex]->setBlob(
                gbChunk.data(), destinationElement * sizeof(TEBSDF), size_t(count) * sizeof(TEBSDF)
            );
            pBuffers[bufferIndex]->setBlob(
                pChunk.data(), destinationElement * sizeof(Ellipsoid), size_t(count) * sizeof(Ellipsoid)
            );
            localBase += count;
        }
    }

    std::vector<VoxelAssetGPU> gpuAssets;
    gpuAssets.reserve(loadedAssets.size());
    for (const VoxelAsset& asset : loadedAssets)
    {
        VoxelAssetGPU gpu{};
        gpu.gridMin = float4(asset.gridData.gridMin, 0.0f);
        gpu.voxelSize = float4(asset.gridData.voxelSize, 0.0f);
        gpu.voxelCountAndDepth = uint4(
            asset.gridData.voxelCount.x,
            asset.gridData.voxelCount.y,
            asset.gridData.voxelCount.z,
            asset.octreeMaxDepth
        );
        gpu.resourceInfo = uint4(
            asset.octreeRoot,
            asset.availableLODLevels,
            asset.voxelDataOffset,
            static_cast<uint32_t>(asset.gridData.solidVoxelCount)
        );
        gpuAssets.push_back(gpu);
    }
    ref<Buffer> assetBuffer = mpDevice->createStructuredBuffer(
        sizeof(VoxelAssetGPU),
        static_cast<uint32_t>(gpuAssets.size()),
        ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal,
        gpuAssets.data(),
        false
    );

    pRenderContext->submit(true);

    mVoxelAssets = std::move(loadedAssets);
    mGBuffers = std::move(gBuffers);
    mPBuffers = std::move(pBuffers);
    mGBufferSplits = std::move(splits);
    mBufferCount = splitCount;
    mOctreeBuffer = std::move(octreeBuffer);
    mAssetBuffer = std::move(assetBuffer);

    if (!mInstances.empty())
        mInstanceEditIndex = std::min<uint32_t>(mInstanceEditIndex, uint32_t(mInstances.size() - 1));
    if (!mInstances.empty() && mInstances[mInstanceEditIndex].assetIndex < mVoxelAssets.size())
        gridData = mVoxelAssets[mInstances[mInstanceEditIndex].assetIndex].gridData;
    else
        gridData = mVoxelAssets.front().gridData;

    mSceneMetaError.clear();
    mComplete = true;
    std::cout << "Loaded voxel scene: assets=" << mVoxelAssets.size()
              << ", instances=" << mInstances.size()
              << ", nodes=" << totalNodeCount
              << ", data=" << totalVoxelCount
              << ", buffers=" << mBufferCount << std::endl;
    return true;
}

RayMarchingPass::RayMarchingPass(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice), gridData(VoxelizationBase::GlobalGridData)
{
    mpDevice = pDevice;
    mInstances.emplace_back();
    mComplete = true;
    mShadowBias100 = 0.01f;
    mMinPdf100 = 0.1f;
    mTransmittanceThreshold100 = 5.f;
    mUseEmissiveLight = false;
    mDebug = false;
    mCheckEllipsoid = true;
    mCheckVisibility = true;
    mCheckCoverage = true;
    mDrawMode = 9;
    mMaxBounce = 0;
    mRenderBackGround = true;
    mClearColor = float3(0);
    mCoverageBlend = 0.0f;
    mSelectedResolution = 0;
    mOutputResolution = uint2(1920, 1080);

    mDisplayNDF = false;
    mSelectedUV = float2(0);
    mSelectedPixel = uint2(0);

    mOptionsChanged = false;
    mFrameIndex = 0;
    selectedFile = 0;
    mInstanceBVHDirty = true;

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point)
        .setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
    mpPointSampler = mpDevice->createSampler(samplerDesc);

    mpFbo = Fbo::create(mpDevice);
}

RenderPassReflection RayMarchingPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    reflector.addOutput(kOutputColor, "Color")
        .bindFlags(ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(mOutputResolution.x, mOutputResolution.y, 1, 1);
    return reflector;
}

void RayMarchingPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene)
        return;

    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // Load and concatenate every scene-meta asset before tracing.
    if (!mComplete && !loadVoxelResources(pRenderContext))
        return;
    if (mVoxelAssets.empty() || !mAssetBuffer || !mOctreeBuffer || mGBuffers.empty())
        return;

    // ---- Step 2: Ray marching ----
    FALCOR_PROFILE(pRenderContext, "RayMarching");
    ref<Camera> pCamera = mpScene->getCamera();
    ref<Texture> pOutputColor = renderData.getTexture(kOutputColor);
    const float4x4 viewProjNoJitter = pCamera->getViewProjMatrixNoJitter();
    if (mInstances.empty())
        mInstances.emplace_back();
    mInstanceEditIndex = std::min<uint32_t>(mInstanceEditIndex, static_cast<uint32_t>(mInstances.size() - 1));
    updateInstanceTransforms();
    for (size_t i = 0; i < mInstances.size(); ++i)
    {
        updateScreenSpaceLOD(viewProjNoJitter, mInstances[i], i == mInstanceEditIndex);
    }
    mScreenLOD = mInstances[mInstanceEditIndex].screenLOD;
    if (mInstances[mInstanceEditIndex].assetIndex < mVoxelAssets.size())
        gridData = mVoxelAssets[mInstances[mInstanceEditIndex].assetIndex].gridData;
    updateInstanceBuffer();
    rebuildInstanceBVHIfDirty();
    if (!mSelectedVoxel)
    {
        mSelectedVoxel =
            mpDevice->createStructuredBuffer(sizeof(float4), 2, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    }

    pRenderContext->clearRtv(pOutputColor->getRTV().get(), float4(0));

    mSelectedPixel = uint2(mSelectedUV.x * pOutputColor->getWidth(), mSelectedUV.y * pOutputColor->getHeight());

    if (!mDisplayNDF)
    {
        if (!mpFullScreenPass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(kShaderFile).psEntry("main");
            desc.setShaderModel(ShaderModel::SM6_5);
            desc.addTypeConformances(mpScene->getTypeConformances());
            DefineList defines = mpScene->getSceneDefines();
            defines.add(kBufferCountDefine, std::to_string(mBufferCount));
            mpFullScreenPass = FullScreenPass::create(mpDevice, desc, defines);
        }
        pRenderContext->clearUAV(mSelectedVoxel->getUAV().get(), float4(-1));

        mpFullScreenPass->addDefine("CHECK_ELLIPSOID", mCheckEllipsoid ? "1" : "0");
        mpFullScreenPass->addDefine("CHECK_VISIBILITY", mCheckVisibility ? "1" : "0");
        mpFullScreenPass->addDefine("CHECK_COVERAGE", mCheckCoverage ? "1" : "0");
        mpFullScreenPass->addDefine("DEBUG", mDebug ? "1" : "0");
        mpFullScreenPass->addDefine("MAX_BOUNCE", std::to_string(mMaxBounce));
        mpFullScreenPass->addDefine("USE_INSTANCE_BVH", mUseInstanceBVH ? "1" : "0");

        ref<EnvMap> pEnvMap = mpScene->getEnvMap();
        mpFullScreenPass->addDefine("USE_ENV_MAP", pEnvMap ? "1" : "0");
        if (pEnvMap)
        {
            if (!mpEnvMapSampler || mpEnvMapSampler->getEnvMap() != pEnvMap)
                mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, pEnvMap);
        }
        if (mUseEmissiveLight)
        {
            if (VoxelizationBase::LightChanged)
            {
                mpScene->getILightCollection(pRenderContext);
                mpFullScreenPass->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
                VoxelizationBase::LightChanged = false;
                pRenderContext->submit(true);
                return;
            }
        }
        else
        {
            mpFullScreenPass->addDefine("USE_EMISSIVE_LIGHTS", "0");
        }

        auto var = mpFullScreenPass->getRootVar();
        mpScene->bindShaderData(var["gScene"]);
        if (pEnvMap)
            mpEnvMapSampler->bindShaderData(var["gEnvMapSampler"]);

        for (size_t i = 0; i < mGBuffers.size(); ++i)
        {
            var["gBuffer"][i] = mGBuffers[i];
            var["pBuffer"][i] = mPBuffers[i];
        }
        var["octreeBuffer"] = mOctreeBuffer;
        var["voxelAssets"] = mAssetBuffer;
        var["instances"] = mInstanceBuffer;
        // USE_INSTANCE_BVH is compiled out when the checkbox is off, so the
        // buffer and count are only bound while the macro is active (mirrors
        // how USE_ENV_MAP gates the env-map sampler binding).
        if (mUseInstanceBVH)
            var["instanceBVH"] = mInstanceBVHBuffer;
        var["selectedVoxel"] = mSelectedVoxel;

        auto cb = var["CB"];
        cb["pixelCount"] = mOutputResolution;
        cb["invVP"] = math::inverse(viewProjNoJitter);
        cb["instanceCount"] = static_cast<uint32_t>(mInstances.size());
        cb["assetCount"] = static_cast<uint32_t>(mVoxelAssets.size());
        if (mUseInstanceBVH)
            cb["instanceBVHNodeCount"] = static_cast<uint32_t>(mInstanceBVH.size());
        cb["shadowBiasWorld"] = mShadowBias100 / 100;
        cb["drawMode"] = mDrawMode;
        cb["frameIndex"] = mFrameIndex;
        cb["minPdf"] = mMinPdf100 / 100;
        cb["transmittanceThreshold"] = mTransmittanceThreshold100 / 100;
        cb["selectedPixel"] = mSelectedPixel;
        cb["renderBackGround"] = mRenderBackGround;
        cb["clearColor"] = float4(mClearColor, 0);
        cb["tanHalfFovY"] = std::tan(Falcor::focalLengthToFovY(pCamera->getFocalLength(), pCamera->getFrameHeight()) * 0.5f);
        cb["forcedLOD"] = mForcedLOD;
        cb["maxLODLevel"] = mMaxLODLevel;
        for (size_t i = 0; i < mGBufferSplits.size(); ++i)
            cb["gbSplits"][i] = mGBufferSplits[i];
        cb["coverageBlend"] = mCoverageBlend;
        mFrameIndex++;

        mpFbo->attachColorTarget(pOutputColor, 0);
        mpFullScreenPass->execute(pRenderContext, mpFbo);
    }
    else
    {
        if (!mpDisplayNDFPass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(kDisplayShaderFile).psEntry("main");
            desc.setShaderModel(ShaderModel::SM6_5);
            DefineList defines;
            defines.add(kBufferCountDefine, std::to_string(mBufferCount));
            mpDisplayNDFPass = FullScreenPass::create(mpDevice, desc, defines);
        }
        auto var = mpDisplayNDFPass->getRootVar();
        for (size_t i = 0; i < mGBuffers.size(); ++i)
            var["gBuffer"][i] = mGBuffers[i];
        var["selectedVoxel"] = mSelectedVoxel;

        auto cb = var["CB"];
        cb["clearColor"] = float4(mClearColor, 0);
        for (size_t i = 0; i < mGBufferSplits.size(); ++i)
            cb["gbSplits"][i] = mGBufferSplits[i];

        mpFbo->attachColorTarget(pOutputColor, 0);
        mpDisplayNDFPass->execute(pRenderContext, mpFbo);
    }

    // Readback selected voxel info for UI display
    if (mSelectedVoxel)
    {
        if (!mpSelectedVoxelStaging)
            mpSelectedVoxelStaging = mpDevice->createBuffer(mSelectedVoxel->getSize(), ResourceBindFlags::None, MemoryType::ReadBack);
        pRenderContext->copyResource(mpSelectedVoxelStaging.get(), mSelectedVoxel.get());
        pRenderContext->submit(true);
        float4* pData = reinterpret_cast<float4*>(mpSelectedVoxelStaging->map());
        float4 v0 = pData[0];
        float4 v1 = pData[1];
        mpSelectedVoxelStaging->unmap();

        if (v0.x != -1.0f || v0.y != -1.0f || v0.z != -1.0f || v0.w != -1.0f)
        {
            mSelectedHit = true;
            mSelectedGbOffset = (uint)v0.w;
            mSelectedCellInt = int3((int)v1.x, (int)v1.y, (int)v1.z);
            mSelectedInstanceId = (uint)v1.w;
        }
        else
        {
            mSelectedHit = false;
            mSelectedGbOffset = 0xFFFFFFFF;
            mSelectedInstanceId = 0xFFFFFFFF;
            mSelectedCellInt = int3(-1);
        }
    }
}

void RayMarchingPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mUseEmissiveLight = false;
    VoxelizationBase::LightChanged = true;
}

void RayMarchingPass::renderUI(Gui::Widgets& widget)
{
    // ---- Voxel bin and scene-meta selection ----
    // Keep probing while no scene meta is present so a hand-authored file
    // created while the application is running becomes visible without
    // needing a voxel re-generation event.
    if (VoxelizationBase::FileUpdated || !mFileListInitialized || sceneMetaPaths.empty())
    {
        filePaths.clear();
        sceneMetaPaths.clear();
        for (const auto& entry : std::filesystem::directory_iterator(VoxelizationBase::ResourceFolder))
        {
            // Metadata is a sidecar (<file>.bin.meta), not a selectable voxel
            // data file. Keep the dropdown restricted to binary voxel files.
            if (std::filesystem::is_regular_file(entry) &&
                entry.path().extension() == ".bin")
            {
                filePaths.push_back(entry.path());
            }
            else if (std::filesystem::is_regular_file(entry) && isVoxelSceneMetaFile(entry.path()))
            {
                sceneMetaPaths.push_back(entry.path());
            }
        }
        std::sort(filePaths.begin(), filePaths.end());
        std::sort(sceneMetaPaths.begin(), sceneMetaPaths.end());
        mFileListInitialized = true;
        VoxelizationBase::FileUpdated = false;
    }

    Gui::DropdownList list;
    for (uint i = 0; i < filePaths.size(); i++)
    {
        list.push_back({i, filePaths[i].filename().string()});
    }
    if (!filePaths.empty())
    {
        selectedFile = std::min<uint>(selectedFile, static_cast<uint>(filePaths.size() - 1));
        widget.dropdown("Voxel Bin", list, selectedFile);

        if (mpScene && widget.button("Read Voxel Bin"))
        {
            const std::filesystem::path binaryPath = filePaths[selectedFile];
            std::ifstream f(binaryPath, std::ios::binary | std::ios::ate);
            if (f.is_open())
            {
                const size_t fileSize = std::filesystem::file_size(binaryPath);
                size_t offset = 0;
                GridData selectedGrid{};
                if (!tryRead(f, offset, sizeof(GridData), &selectedGrid, fileSize))
                    return;
                f.close();

                mVoxelFilePath = binaryPath;
                mSceneMetaPath.clear();
                mSceneMetaError.clear();
                mSceneMetaLoaded = false;
                resetInstancesToIdentity();
                resetVoxelResources();

                VoxelAsset asset;
                asset.assetId = binaryPath.stem().string();
                asset.voxelFile = binaryPath;
                asset.gridData = selectedGrid;
                mVoxelAssets.push_back(std::move(asset));
                gridData = selectedGrid;

                requestRecompile();
                mComplete = false;
                mOptionsChanged = true;
            }
        }
    }
    else
    {
        widget.text("No voxel .bin files found.");
    }

    Gui::DropdownList sceneList;
    for (uint i = 0; i < sceneMetaPaths.size(); i++)
    {
        sceneList.push_back({i, sceneMetaPaths[i].filename().string()});
    }
    if (!sceneMetaPaths.empty())
    {
        selectedSceneMeta = std::min<uint>(
            selectedSceneMeta,
            static_cast<uint>(sceneMetaPaths.size() - 1)
        );
        widget.dropdown("Voxel Scene Meta", sceneList, selectedSceneMeta);
        if (mpScene && widget.button("Read Voxel Scene Meta"))
            loadSceneMeta(sceneMetaPaths[selectedSceneMeta]);
    }
    else
    {
        widget.text("No .voxscene scene meta files found.");
    }

    if (mSceneMetaLoaded)
    {
        widget.text("Scene Meta: " + mSceneMetaPath.filename().string());
        widget.text("Scene Assets: " + std::to_string(mVoxelAssets.size()));
    }
    if (!mSceneMetaError.empty())
        widget.text("Voxel Scene Error: " + mSceneMetaError);

    if (mInstances.empty())
        mInstances.emplace_back();
    widget.text("Assets: " + std::to_string(mVoxelAssets.size()));
    widget.text("Instances: " + std::to_string(mInstances.size()));
    widget.text("Ray-Marching Buffer Count: " + std::to_string(mBufferCount));
    if (widget.checkbox("Use Instance BVH Traversal", mUseInstanceBVH))
        mOptionsChanged = true;
    if (widget.button("Add Instance") && mInstances.size() < 256)
    {
        VoxelInstance instance;
        uint32_t nextInstanceId = 0;
        while (std::any_of(
            mInstances.begin(),
            mInstances.end(),
            [nextInstanceId](const VoxelInstance& existing) { return existing.instanceId == nextInstanceId; }
        ))
        {
            ++nextInstanceId;
        }
        instance.instanceId = nextInstanceId;
        if (mInstanceEditIndex < mInstances.size())
            instance.assetIndex = mInstances[mInstanceEditIndex].assetIndex;
        mInstances.push_back(instance);
        mInstanceEditIndex = static_cast<uint32_t>(mInstances.size() - 1);
        mOptionsChanged = true;
        mInstanceBVHDirty = true;
    }
    if (mInstances.size() > 1 && widget.button("Remove Selected Instance"))
    {
        mInstances.erase(mInstances.begin() + mInstanceEditIndex);
        mInstanceEditIndex = std::min<uint32_t>(
            mInstanceEditIndex,
            static_cast<uint32_t>(mInstances.size() - 1)
        );
        mOptionsChanged = true;
        mInstanceBVHDirty = true;
    }

    mInstanceEditIndex = std::min<uint32_t>(
        mInstanceEditIndex,
        static_cast<uint32_t>(mInstances.size() - 1)
    );
    if (mInstances.size() > 1 &&
        widget.var(
            "Edit Instance",
            mInstanceEditIndex,
            0u,
            static_cast<uint32_t>(mInstances.size() - 1)
        ))
    {
        mOptionsChanged = true;
    }

    VoxelInstance& instance = mInstances[mInstanceEditIndex];
    if (!mVoxelAssets.empty())
    {
        instance.assetIndex = std::min<uint32_t>(
            instance.assetIndex,
            static_cast<uint32_t>(mVoxelAssets.size() - 1)
        );
        Gui::DropdownList assetList;
        for (uint32_t assetIndex = 0; assetIndex < mVoxelAssets.size(); ++assetIndex)
        {
            const VoxelAsset& asset = mVoxelAssets[assetIndex];
            assetList.push_back({
                assetIndex,
                asset.assetId + " (" + asset.voxelFile.filename().string() + ")"
            });
        }
        uint32_t editedAssetIndex = instance.assetIndex;
        if (widget.dropdown("Instance Asset", assetList, editedAssetIndex))
        {
            instance.assetIndex = editedAssetIndex;
            mOptionsChanged = true;
            mInstanceBVHDirty = true;
        }
    }

    const VoxelAsset* editedAsset = instance.assetIndex < mVoxelAssets.size()
        ? &mVoxelAssets[instance.assetIndex]
        : nullptr;
    const GridData& displayGrid = editedAsset ? editedAsset->gridData : gridData;
    gridData = displayGrid;
    widget.text("Voxel Size: " + ToString(displayGrid.voxelSize));
    widget.text("Voxel Count: " + ToString((int3)displayGrid.voxelCount));
    widget.text("Grid Min: " + ToString(displayGrid.gridMin));
    widget.text("Solid Voxel Count: " + std::to_string(displayGrid.solidVoxelCount));
    const size_t totalGridVoxels = size_t(displayGrid.voxelCount.x) *
        size_t(displayGrid.voxelCount.y) * size_t(displayGrid.voxelCount.z);
    const float solidRate = totalGridVoxels > 0
        ? displayGrid.solidVoxelCount / float(totalGridVoxels)
        : 0.0f;
    widget.text("Solid Rate: " + std::to_string(solidRate));
    widget.text("Max Polygon Count: " + std::to_string(displayGrid.maxPolygonCount));
    widget.text("Total Polygon Count: " + std::to_string(displayGrid.totalPolygonCount));
    if (widget.checkbox("Instance Enabled", instance.enabled))
    {
        mOptionsChanged = true;
        mInstanceBVHDirty = true;
    }
    if (widget.matrix("Instance Transform (row-major)", instance.assetToWorld, -1000000.0f, 1000000.0f))
    {
        mOptionsChanged = true;
        mInstanceBVHDirty = true;
    }
    widget.text("Editing Instance ID: " + std::to_string(instance.instanceId));
    if (editedAsset)
        widget.text("Editing Asset: " + editedAsset->assetId);
    if (mGridProjectionValid)
    {
        widget.text("Grid Projection (px): " + std::to_string(mGridProjectedWidthPixels) + " x " +
                    std::to_string(mGridProjectedHeightPixels));
        widget.text("Grid Projection Area (px^2): " + std::to_string(mGridProjectedAreaPixels));
        widget.text("Leaf Voxel Projection (eq. px): " + std::to_string(mLeafProjectedSizePixels));
        widget.text("Screen-space LOD: " + std::to_string(mScreenLOD) +
                    " (node size: " + std::to_string(1 << mScreenLOD) + " leaf cells)");
    }
    else
    {
        widget.text("Grid Projection: invalid/near-plane, using leaf LOD");
    }

    if (editedAsset && editedAsset->octreeMaxDepth > 0)
    {
        widget.text("Octree Root Offset: " + std::to_string(editedAsset->octreeRoot));
        widget.text("Voxel Data Offset: " + std::to_string(editedAsset->voxelDataOffset));
        widget.text("Octree Max Depth: " + std::to_string(editedAsset->octreeMaxDepth));
        widget.text("Available LOD Levels: " + std::to_string(editedAsset->availableLODLevels));
        if (editedAsset->hasVoxelMetadata)
        {
            widget.text("Voxel Format Version: " + std::to_string(editedAsset->voxelFormatVersion));
            widget.text("Voxel Producer: " + editedAsset->voxelProducer);
            widget.text("LOD Build Mode: " + editedAsset->voxelLodMode);
        }
        uint32_t totalNodes = 0;
        for (auto c : editedAsset->octreeNodeCounts)
            totalNodes += c;
        widget.text("Octree Total Nodes: " + std::to_string(totalNodes));
    }

    // ---- Ray marching controls (original) ----
    if (widget.checkbox("Debug", mDebug))
        mOptionsChanged = true;
    if (mDebug)
    {
        int maxLOD = editedAsset
            ? int(std::min(editedAsset->octreeMaxDepth, editedAsset->availableLODLevels))
            : 0;
        mForcedLOD = std::min(mForcedLOD, maxLOD);
        mMaxLODLevel = std::min(mMaxLODLevel, maxLOD);
        if (widget.slider("Forced LOD", mForcedLOD, -1, maxLOD))
            mOptionsChanged = true;
        if (mForcedLOD >= 0)
            widget.text("LOD " + std::to_string(mForcedLOD) + ": node size = " +
                        std::to_string(1 << mForcedLOD) + " leaf voxels");

        // Optional cap for the screen-space selected LOD.
        if (widget.slider("Max LOD Level", mMaxLODLevel, -1, maxLOD))
            mOptionsChanged = true;
        if (mMaxLODLevel >= 0)
            widget.text("LOD cap at level " + std::to_string(mMaxLODLevel)
                        + ": node size = " + std::to_string(1 << mMaxLODLevel) + " leaf cells");
    }
    if (widget.checkbox("Use Emissive Light", mUseEmissiveLight))
        mOptionsChanged = true;
    if (widget.checkbox("Check Ellipsoid", mCheckEllipsoid))
        mOptionsChanged = true;
    if (widget.checkbox("Check Visibility", mCheckVisibility))
        mOptionsChanged = true;
    if (widget.checkbox("Check Coverage", mCheckCoverage))
        mOptionsChanged = true;
    if (mCheckCoverage)
    {
        if (widget.slider("Coverage Blend", mCoverageBlend, 0.0f, 1.0f))
            mOptionsChanged = true;
    }
    if (widget.slider("Shadow Bias World(x100)", mShadowBias100, 0.0f, 0.2f))
        mOptionsChanged = true;
    if (widget.slider("Min Pdf(x100)", mMinPdf100, 0.0f, 0.2f))
        mOptionsChanged = true;
    if (widget.slider("T Threshold(x100)", mTransmittanceThreshold100, 0.0f, 10.0f))
        mOptionsChanged = true;
    if (widget.dropdown("Draw Mode", reinterpret_cast<ABSDFDrawMode&>(mDrawMode)))
        mOptionsChanged = true;
    if (widget.slider("Max Bounce", mMaxBounce, 0u, 10u))
        mOptionsChanged = true;
    if (widget.checkbox("Display NDF", mDisplayNDF))
        mOptionsChanged = true;
    if (widget.rgbColor("Clear Color", mClearColor))
        mOptionsChanged = true;
    if (widget.checkbox("Render Background", mRenderBackGround))
        mOptionsChanged = true;

    static const uint resolutions[] = {0, 32, 64, 128, 256, 512, 1024};
    {
        Gui::DropdownList list_res;
        for (uint32_t i = 0; i < sizeof(resolutions) / sizeof(uint); i++)
        {
            list_res.push_back({resolutions[i], std::to_string(resolutions[i])});
        }
        if (widget.dropdown("Output Resolution", list_res, mSelectedResolution))
        {
            if (mSelectedResolution == 0)
                mOutputResolution = uint2(1920, 1080);
            else
                mOutputResolution = uint2(mSelectedResolution, mSelectedResolution);
            ref<Camera> camera = mpScene->getCamera();
            if (camera)
                camera->setAspectRatio(mOutputResolution.x / (float)mOutputResolution.y);
            requestRecompile();
        }
    }

    widget.text("Selected Pixel: " + ToString(mSelectedPixel));
    if (mSelectedHit)
    {
        widget.text("Selected Voxel cellInt: " + ToString(mSelectedCellInt));
        widget.text("Selected Voxel gbOffset: " + std::to_string(mSelectedGbOffset));
        widget.text("Selected Instance ID: " + std::to_string(mSelectedInstanceId));
    }
    else
    {
        widget.text("Selected Voxel: none (no hit)");
    }
}

void RayMarchingPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mDebug = false;
    mUseEmissiveLight = false;
    mVoxelFilePath.clear();
    mSceneMetaPath.clear();
    mSceneMetaError.clear();
    mSceneMetaLoaded = false;
    resetInstancesToIdentity();
    resetVoxelResources();
    mComplete = false;
}

bool RayMarchingPass::onMouseEvent(const MouseEvent& mouseEvent)
{
    if (mouseEvent.type == MouseEvent::Type::ButtonDown && mouseEvent.button == Input::MouseButton::Left)
    {
        mSelectedUV = mouseEvent.pos;
        return true;
    }
    return false;
}

bool RayMarchingPass::tryRead(std::ifstream& f, size_t& offset, size_t bytes, void* dst, size_t fileSize)
{
    if (bytes > fileSize || offset > fileSize - bytes ||
        offset > size_t(std::numeric_limits<std::streamoff>::max()) ||
        bytes > size_t(std::numeric_limits<std::streamsize>::max()))
        return false;
    if (dst)
    {
        f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
        if (!f)
            return false;
    }
    offset += bytes;
    return true;
}
