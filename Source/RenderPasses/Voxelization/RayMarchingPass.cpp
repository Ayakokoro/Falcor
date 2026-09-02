#include "RayMarchingPass.h"
#include "VoxelizationMetadata.h"
#include "VoxelSceneMetadata.h"
#include "Shading.slang"
#include "Math/SphericalHarmonics.slang"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Math/FalcorMath.h"
#include <algorithm>
#include <execution>
#include <fstream>
#include <filesystem>
#include <limits>

namespace
{
const std::string kShaderFile = "RenderPasses/Voxelization/RayMarching.ps.slang";
const std::string kDisplayShaderFile = "RenderPasses/Voxelization/DisplayNDF.ps.slang";
const std::string kOutputColor = "color";
const std::string kBufferCountDefine = "RAY_MARCHING_BUFFER_COUNT";
constexpr size_t kRayMarchingBufferByteLimit = size_t(1) << 30; // 1 GiB per resource.

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
    const float3 gridExtent(
        gridData.voxelSize.x * float(gridData.voxelCount.x),
        gridData.voxelSize.y * float(gridData.voxelCount.y),
        gridData.voxelSize.z * float(gridData.voxelCount.z)
    );
    const float3 pivot = gridData.gridMin + 0.5f * gridExtent;

    // Include half a voxel around the grid for a conservative broad-phase
    // bound. The local traversal still clips against the exact grid bounds.
    const float3 boundsMinLocal = gridData.gridMin - 0.5f * gridData.voxelSize;
    const float3 boundsMaxLocal = gridData.gridMin + gridExtent + 0.5f * gridData.voxelSize;
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

    for (size_t i = 0; i < mInstances.size(); ++i)
    {
        VoxelInstance& instance = mInstances[i];

        // Keep every scale component strictly positive so the inverse
        // transform and the local/world distance conversion remain valid.
        instance.scale.x = std::max(instance.scale.x, 1e-4f);
        instance.scale.y = std::max(instance.scale.y, 1e-4f);
        instance.scale.z = std::max(instance.scale.z, 1e-4f);

        const float3 rotationRadians = math::radians(instance.rotationDegrees);
        const float4x4 rotation = math::matrixFromRotationXYZ(
            rotationRadians.x, rotationRadians.y, rotationRadians.z
        );
        const float4x4 scaling = math::matrixFromScaling(instance.scale);
        const float4x4 toPivot = math::matrixFromTranslation(-pivot);
        const float4x4 fromPivot = math::matrixFromTranslation(pivot);
        const float4x4 translation = math::matrixFromTranslation(instance.translation);

        // Column-vector convention: local point is rotated around the pivot
        // first, scaled around the same pivot, then translated into world space.
        instance.localToWorld = math::mul(
            translation,
            math::mul(fromPivot, math::mul(rotation, math::mul(scaling, toPivot)))
        );
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

    const uint32_t maxLOD = std::min(mOctreeMaxDepth, mAvailableLODLevels);
    if (mOutputResolution.x == 0 || mOutputResolution.y == 0 || gridData.voxelCount.x == 0)
        return;

    const float3 gridExtent(
        gridData.voxelSize.x * float(gridData.voxelCount.x),
        gridData.voxelSize.y * float(gridData.voxelCount.y),
        gridData.voxelSize.z * float(gridData.voxelCount.z)
    );
    const float3 gridMax = gridData.gridMin + gridExtent;

    const float3 corners[8] = {
        float3(gridData.gridMin.x, gridData.gridMin.y, gridData.gridMin.z),
        float3(gridMax.x,          gridData.gridMin.y, gridData.gridMin.z),
        float3(gridData.gridMin.x, gridMax.y,          gridData.gridMin.z),
        float3(gridMax.x,          gridMax.y,          gridData.gridMin.z),
        float3(gridData.gridMin.x, gridData.gridMin.y, gridMax.z),
        float3(gridMax.x,          gridData.gridMin.y, gridMax.z),
        float3(gridData.gridMin.x, gridMax.y,          gridMax.z),
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

    const float leafProjectedSizePixels = std::sqrt(projectedArea) / float(gridData.voxelCount.x);
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
        gpu.padding = 0;
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

void RayMarchingPass::resetVoxelResources()
{
    // The shader array size is compiled from mBufferCount. These resources and
    // both fullscreen passes must be discarded together when a new bin is
    // selected, otherwise a scene with a different split count can retain the
    // previous shader layout for one frame.
    mGBuffers.clear();
    mPBuffers.clear();
    mGBufferSplits.clear();
    mOctreeBuffer = nullptr;
    mOctreeMaxDepth = 0;
    mOctreeNodeCounts.clear();
    mAvailableLODLevels = 0;
    mVoxelFormatVersion = 0;
    mVoxelLodMode.clear();
    mVoxelProducer.clear();
    mHasVoxelMetadata = false;
    mBufferCount = 1;
    mInstanceBuffer = nullptr;
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
        instance.enabled = source.enabled;
        instance.translation = float3(
            source.translation[0], source.translation[1], source.translation[2]
        );
        instance.rotationDegrees = float3(
            source.rotationDegrees[0], source.rotationDegrees[1], source.rotationDegrees[2]
        );
        instance.scale = float3(source.scale[0], source.scale[1], source.scale[2]);
        instances.push_back(instance);
    }

    mInstances = std::move(instances);
    mInstanceEditIndex = 0;
    mVoxelFilePath = scene.voxelFile;
    mSceneMetaPath = scene.sourcePath;
    mSceneMetaAssetId = scene.assetId;
    mSceneMetaLoaded = true;
    mSceneMetaError.clear();

    resetVoxelResources();
    requestRecompile();
    mComplete = false;
    mOptionsChanged = true;
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

    // ---- Step 1: Load voxel data from file if needed (from ReadVoxelPass) ----
    if (!mComplete)
    {
        // Direct bin loading and scene-meta loading share the same binary
        // reader. A fallback keeps the old UI state usable if a pass is
        // restored with mComplete=false before a file has been selected.
        if (mVoxelFilePath.empty() && selectedFile < filePaths.size())
            mVoxelFilePath = filePaths[selectedFile];
        if (mVoxelFilePath.empty())
            return;

        const std::filesystem::path binaryPath = mVoxelFilePath;
        GridData& gd = VoxelizationBase::GlobalGridData;

        std::ifstream f;
        f.open(binaryPath, std::ios::binary | std::ios::ate);
        if (!f.is_open())
            return;

        std::cout << "Reading voxel data from: " << binaryPath << std::endl;
        size_t fileSize = std::filesystem::file_size(binaryPath);
        size_t offset = 0;

        // Read GridData header
        tryRead(f, offset, sizeof(GridData), &gd, fileSize);

        // Read octree header
        uint32_t maxDepth = 0;
        tryRead(f, offset, sizeof(uint32_t), &maxDepth, fileSize);

        std::vector<uint32_t> nodeCounts(maxDepth + 1);
        tryRead(f, offset, (maxDepth + 1) * sizeof(uint32_t), nodeCounts.data(), fileSize);

        uint32_t totalNodes = 0;
        for (uint32_t i = 0; i <= maxDepth; i++)
            totalNodes += nodeCounts[i];

        // The binary layout is shared with the original voxelization pass.
        // The optional text sidecar carries the producer/version and, for
        // CPU-generated files, how many parent LOD levels contain data.
        VoxelizationMetadata::Metadata metadata;
        std::error_code metadataEc;
        const bool hasSidecar = std::filesystem::exists(
            VoxelizationMetadata::sidecarPath(binaryPath), metadataEc);
        if (VoxelizationMetadata::read(binaryPath, metadata) &&
            metadata.maxDepth == maxDepth &&
            (metadata.totalNodes == 0 || metadata.totalNodes == totalNodes))
        {
            mHasVoxelMetadata = true;
            mVoxelFormatVersion = metadata.version;
            mVoxelLodMode = metadata.lodMode;
            mVoxelProducer = metadata.producer;
            mAvailableLODLevels = std::min(metadata.generatedLodLevels, maxDepth);
            std::cout << "Voxel metadata: version=" << mVoxelFormatVersion
                      << ", producer=" << mVoxelProducer
                      << ", lodMode=" << mVoxelLodMode
                      << ", availableLODLevels=" << mAvailableLODLevels << std::endl;
        }
        else if (hasSidecar)
        {
            // A present but invalid sidecar is not treated as a legacy file:
            // conservatively expose leaves only instead of stopping at a
            // possibly unavailable parent node.
            mHasVoxelMetadata = false;
            mVoxelFormatVersion = 0;
            mVoxelLodMode.clear();
            mVoxelProducer.clear();
            mAvailableLODLevels = 0;
            std::cerr << "Voxel metadata is missing, invalid, or does not "
                      << "match the binary: "
                      << VoxelizationMetadata::sidecarPath(binaryPath)
                      << ". Falling back to leaf-only LOD." << std::endl;
        }
        else
        {
            // Files written before sidecar support generated data for every
            // tree level, so preserve their historical behavior.
            mHasVoxelMetadata = false;
            mVoxelFormatVersion = 0;
            mVoxelLodMode = "legacy";
            mVoxelProducer = "legacy";
            mAvailableLODLevels = maxDepth;
            std::cout << "No voxel metadata sidecar; treating file as legacy "
                      << "with all LOD levels available." << std::endl;
        }

        std::cout << "Octree: maxDepth=" << maxDepth << ", totalNodes=" << totalNodes;

        // Read all octree nodes
        std::vector<OctreeNode> octreeNodes(totalNodes);
        tryRead(f, offset, totalNodes * sizeof(OctreeNode), octreeNodes.data(), fileSize);

        // Debug: verify octree dataIndex bounds
        {
            uint32_t badIdx = 0, maxDI = 0, minDI = 0xFFFFFFFF;
            uint32_t leafCount = 0, internalCount = 0;
            // Calculate per-level node offsets
            std::vector<uint32_t> levelStart(maxDepth + 2);
            levelStart[0] = 0;
            for (uint32_t i = 0; i <= maxDepth; i++)
                levelStart[i + 1] = levelStart[i] + nodeCounts[i];

            for (uint32_t lvl = 0; lvl <= maxDepth; lvl++)
            {
                uint32_t lvlBad = 0;
                for (uint32_t n = levelStart[lvl]; n < levelStart[lvl + 1]; n++)
                {
                    auto& node = octreeNodes[n];
                    bool isLeaf = (lvl == maxDepth) || (node.childMask == 0);
                    if (isLeaf)
                    {
                        leafCount++;
                        if (node.dataIndex > maxDI) maxDI = node.dataIndex;
                        if (node.dataIndex < minDI) minDI = node.dataIndex;
                        if (node.dataIndex >= gd.solidVoxelCount)
                        {
                            lvlBad++;
                            if (badIdx < 5)
                                std::cout << "  OOB leaf at lvl=" << lvl << " node=" << n
                                          << " dataIndex=" << node.dataIndex
                                          << " (max=" << gd.solidVoxelCount - 1 << ")" << std::endl;
                        }
                    }
                    else
                    {
                        internalCount++;
                    }
                }
                badIdx += lvlBad;
                if (lvlBad > 0)
                    std::cout << "  Level " << lvl << ": " << lvlBad << " OOB leaves" << std::endl;
            }
            std::cout << "Octree: leaves=" << leafCount << " internal=" << internalCount
                      << " dataIndex range=[" << minDI << ", " << maxDI << "]"
                      << " validRange=[0, " << gd.solidVoxelCount - 1 << "]"
                      << " totalOOB=" << badIdx << std::endl;
        }

        // Read VoxelData
        std::vector<VoxelData> voxelData(gd.solidVoxelCount);
        tryRead(f, offset, gd.solidVoxelCount * sizeof(VoxelData), voxelData.data(), fileSize);

        float maxArea = 0, minArea = FLT_MAX;
        uint zeroAreaCount = 0;
        for (auto& vd : voxelData)
        {
            float a = vd.ABSDF.area;
            maxArea = max(maxArea, a);
            minArea = min(minArea, a);
            if (a <= 0) zeroAreaCount++;
        }
        std::cout << "VoxelData area: min=" << minArea << " max=" << maxArea
                  << " zeroCount=" << zeroAreaCount << "/" << voxelData.size() << std::endl;

        f.close();

        std::cout << ", solidVoxels=" << gd.solidVoxelCount << std::endl;
        for (uint32_t i = 0; i <= maxDepth; i++)
            std::cout << "  Level " << i << ": " << nodeCounts[i] << " nodes" << std::endl;

        // Create GPU buffer for octree nodes
        auto pOctreeBuffer = mpDevice->createStructuredBuffer(
            sizeof(OctreeNode), totalNodes, ResourceBindFlags::ShaderResource
        );
        pOctreeBuffer->setBlob(octreeNodes.data(), 0, totalNodes * sizeof(OctreeNode));

        // Split data across buffers to avoid D3D12 i32 offset overflow:
        // structured buffer byte offset = elementIndex * stride.
        // For 388-byte TEBSDF, max safe index per buffer = (2^31-1)/388 ≈ 5,534,751.
        const uint32_t totalVoxels = gd.solidVoxelCount;
        const size_t maxElemsPerBuffer = kRayMarchingBufferByteLimit / sizeof(TEBSDF);
        FALCOR_CHECK(maxElemsPerBuffer > 0, "TEBSDF is larger than the ray-marching buffer limit.");

        const size_t splitCount64 = totalVoxels == 0
            ? 1
            : (size_t(totalVoxels) + maxElemsPerBuffer - 1) / maxElemsPerBuffer;
        FALCOR_CHECK(splitCount64 <= std::numeric_limits<uint32_t>::max(), "Too many ray-marching buffer splits.");
        const uint32_t splitCount = static_cast<uint32_t>(splitCount64);

        std::vector<uint32_t> counts(splitCount, 0);
        std::vector<uint32_t> bases(splitCount, 0);
        std::vector<uint32_t> splits(splitCount, 0);
        size_t globalBase = 0;
        for (uint32_t b = 0; b < splitCount; ++b)
        {
            bases[b] = static_cast<uint32_t>(globalBase);
            const size_t count = std::min(maxElemsPerBuffer, size_t(totalVoxels) - globalBase);
            counts[b] = static_cast<uint32_t>(count);
            globalBase += count;
            splits[b] = static_cast<uint32_t>(globalBase);
        }

        // Debug: print struct sizes
        std::cout << "sizeof(VoxelData)=" << sizeof(VoxelData)
                  << " sizeof(TEBSDF)=" << sizeof(TEBSDF)
                  << " sizeof(Ellipsoid)=" << sizeof(Ellipsoid) << std::endl;
        std::cout << "RayMarching buffers=" << splitCount
                  << ", maxElementsPerBuffer=" << maxElemsPerBuffer
                  << ", maxGBufferBytes=" << kRayMarchingBufferByteLimit << std::endl;
        for (uint32_t b = 0; b < splitCount; ++b)
        {
            std::cout << "count[" << b << "]=" << counts[b]
                      << " (" << (size_t(counts[b]) * sizeof(TEBSDF) / (1024.0 * 1024.0))
                      << " MB gBuffer)" << std::endl;
        }

        // Convert VoxelData -> TEBSDF on the CPU and upload only the final
        // gBuffer (TEBSDF) + pBuffer (Ellipsoid), split into chunks to avoid
        // the D3D12 2^31-1 byte-offset limit. The raw VoxelData is never staged
        // on the GPU, keeping VRAM at gBuffer + pBuffer only.
        std::vector<ref<Buffer>> gBuffers(splitCount);
        std::vector<ref<Buffer>> pBuffers(splitCount);

        for (uint32_t b = 0; b < splitCount; ++b)
        {
            if (counts[b] == 0)
                continue;

            std::vector<TEBSDF> gbChunk(counts[b]);
            std::vector<Ellipsoid> pChunk(counts[b]);

            // Conversion is pure/read-only with respect to voxelData, so let
            // the standard parallel algorithm use the CPU worker threads.
            // GPU resource creation and setBlob() stay on this render thread.
            const auto sourceBegin = voxelData.begin() + bases[b];
            const auto sourceEnd = sourceBegin + counts[b];
            std::transform(
                std::execution::par_unseq,
                sourceBegin,
                sourceEnd,
                gbChunk.begin(),
                [](const VoxelData& data) { return convertVoxelData(data); }
            );
            std::transform(
                std::execution::par_unseq,
                sourceBegin,
                sourceEnd,
                pChunk.begin(),
                [](const VoxelData& data) { return data.ellipsoid; }
            );

            gBuffers[b] = mpDevice->createStructuredBuffer(
                sizeof(TEBSDF), counts[b],
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal
            );
            pBuffers[b] = mpDevice->createStructuredBuffer(
                sizeof(Ellipsoid), counts[b],
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal
            );
            std::cout << "gBuffer[" << b << "]: " << (gBuffers[b] ? "OK" : "NULL!")
                      << " (" << (size_t(counts[b]) * sizeof(TEBSDF) / (1024.0 * 1024.0)) << " MB)"
                      << std::endl;
            std::cout << "pBuffer[" << b << "]: " << (pBuffers[b] ? "OK" : "NULL!")
                      << " (" << (size_t(counts[b]) * sizeof(Ellipsoid) / (1024.0 * 1024.0)) << " MB)"
                      << std::endl;
            if (gBuffers[b])
                gBuffers[b]->setBlob(gbChunk.data(), 0, size_t(counts[b]) * sizeof(TEBSDF));
            if (pBuffers[b])
                pBuffers[b]->setBlob(pChunk.data(), 0, size_t(counts[b]) * sizeof(Ellipsoid));
        }

        // Flush the uploads before Step 2 reads the buffers.
        pRenderContext->submit(true);

        mGBuffers = std::move(gBuffers);
        mPBuffers = std::move(pBuffers);
        mGBufferSplits = std::move(splits);
        mBufferCount = splitCount;
        mOctreeBuffer = pOctreeBuffer;
        mOctreeMaxDepth = maxDepth;
        mOctreeNodeCounts = std::move(nodeCounts);

        mComplete = true;
    }

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
    updateInstanceBuffer();
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
        var["instances"] = mInstanceBuffer;
        var["selectedVoxel"] = mSelectedVoxel;

        auto cb_GridData = var["GridData"];
        cb_GridData["gridMin"] = gridData.gridMin;
        cb_GridData["voxelSize"] = gridData.voxelSize;
        cb_GridData["voxelCount"] = gridData.voxelCount;
        cb_GridData["octreeMaxDepth"] = mOctreeMaxDepth;

        auto cb = var["CB"];
        cb["pixelCount"] = mOutputResolution;
        cb["invVP"] = math::inverse(viewProjNoJitter);
        const VoxelInstance& debugInstance = mInstances[mInstanceEditIndex];
        cb["instanceCount"] = static_cast<uint32_t>(mInstances.size());
        // These fields remain as a compatibility/fallback context for local
        // helper paths. Primary tracing uses the per-instance structured buffer.
        cb["instanceTransform"] = debugInstance.localToWorld;
        cb["inverseInstanceTransform"] = debugInstance.worldToLocal;
        cb["normalTransform"] = debugInstance.normalTransform;
        cb["instanceScale"] = debugInstance.scale;
        // The shader traces in local cell units. The bias is asset-local and
        // therefore scales with the instance together with the voxel grid.
        cb["shadowBias"] = mShadowBias100 / 100 / gridData.voxelSize.x;
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
        cb["screenLOD"] = mScreenLOD;
        cb["availableLODLevels"] = mAvailableLODLevels;
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
                tryRead(f, offset, sizeof(GridData), &gridData, fileSize);
                f.close();

                mVoxelFilePath = binaryPath;
                mSceneMetaPath.clear();
                mSceneMetaAssetId.clear();
                mSceneMetaError.clear();
                mSceneMetaLoaded = false;
                resetInstancesToIdentity();
                resetVoxelResources();

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
        widget.text("Scene Asset: " + mSceneMetaAssetId);
        widget.text("Scene Voxel Bin: " + mVoxelFilePath.filename().string());
    }
    if (!mSceneMetaError.empty())
        widget.text("Scene Meta Error: " + mSceneMetaError);

    widget.text("Voxel Size: " + ToString(gridData.voxelSize));
    widget.text("Voxel Count: " + ToString((int3)gridData.voxelCount));
    widget.text("Grid Min: " + ToString(gridData.gridMin));
    widget.text("Solid Voxel Count: " + std::to_string(gridData.solidVoxelCount));
    widget.text("Solid Rate: " + std::to_string(gridData.solidVoxelCount / (float)gridData.totalVoxelCount()));
    widget.text("Ray-Marching Buffer Count: " + std::to_string(mBufferCount));
    widget.text("Max Polygon Count: " + std::to_string(gridData.maxPolygonCount));
    widget.text("Total Polygon Count: " + std::to_string(gridData.totalPolygonCount));
    if (mInstances.empty())
        mInstances.emplace_back();
    widget.text("Instances: " + std::to_string(mInstances.size()) + " (shared single bin)");
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
        mInstances.push_back(instance);
        mInstanceEditIndex = static_cast<uint32_t>(mInstances.size() - 1);
        mOptionsChanged = true;
    }
    if (mInstances.size() > 1 && widget.button("Remove Selected Instance"))
    {
        mInstances.erase(mInstances.begin() + mInstanceEditIndex);
        mInstanceEditIndex = std::min<uint32_t>(
            mInstanceEditIndex,
            static_cast<uint32_t>(mInstances.size() - 1)
        );
        mOptionsChanged = true;
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
    if (widget.checkbox("Instance Enabled", instance.enabled))
        mOptionsChanged = true;
    if (widget.var("Instance Translation", instance.translation, -1000.0f, 1000.0f, 0.01f))
        mOptionsChanged = true;
    if (widget.var("Instance Rotation XYZ (deg)", instance.rotationDegrees, -360.0f, 360.0f, 0.5f))
        mOptionsChanged = true;
    if (widget.var("Instance Scale XYZ", instance.scale, 0.01f, 100.0f, 0.01f))
        mOptionsChanged = true;
    widget.text("Editing Instance ID: " + std::to_string(instance.instanceId));
    widget.text("Instance Rotation/Scale Pivot: Grid Center");
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

    if (mOctreeMaxDepth > 0)
    {
        widget.text("Octree Max Depth: " + std::to_string(mOctreeMaxDepth));
        widget.text("Available LOD Levels: " + std::to_string(mAvailableLODLevels));
        if (mHasVoxelMetadata)
        {
            widget.text("Voxel Format Version: " + std::to_string(mVoxelFormatVersion));
            widget.text("Voxel Producer: " + mVoxelProducer);
            widget.text("LOD Build Mode: " + mVoxelLodMode);
        }
        uint32_t totalNodes = 0;
        for (auto c : mOctreeNodeCounts)
            totalNodes += c;
        widget.text("Octree Total Nodes: " + std::to_string(totalNodes));
    }

    // ---- Ray marching controls (original) ----
    if (widget.checkbox("Debug", mDebug))
        mOptionsChanged = true;
    if (mDebug)
    {
        int maxLOD = (int)std::min(mOctreeMaxDepth, mAvailableLODLevels);
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
    if (widget.slider("Shadow Bias(x100)", mShadowBias100, 0.0f, 0.2f))
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
    mSceneMetaAssetId.clear();
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
    if (offset + bytes > fileSize)
        return false;
    if (dst)
    {
        f.seekg(offset, std::ios::beg);
        f.read(reinterpret_cast<char*>(dst), bytes);
    }
    offset += bytes;
    return true;
}
