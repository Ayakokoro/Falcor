#include "RayMarchingPass.h"
#include "VoxelizationMetadata.h"
#include "Shading.slang"
#include "VisibleShading.slang"
#include "../../../LargeSceneVoxelization/VisibleMaterialFile.h"
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

VisibleTEBSDF convertVisibleMaterial(const VoxelData& geometry, const VisibleMaterialFile::Data& material)
{
    VisibleTEBSDF result{};
    result.area = material.area;
    result.coverage = 12345.0f;
    result.primitiveProjAreaFunc = geometry.primitiveProjAreaFunc;
    result.polygonsProjAreaFunc = geometry.polygonsProjAreaFunc;
    result.totalProjAreaFunc = geometry.totalProjAreaFunc;
    for (uint32_t i = 0; i < VisibleMaterialFile::LobeCount; ++i)
    {
        if (!(material.sourceLobes() & (1u << i))) continue;
        const auto& source = material.lobes[i];
        const uint32_t compact = result.surface.lobeCount++;
        auto& lobe = result.surface.lobes[compact];
        lobe.n = float3(source.normal[0], source.normal[1], source.normal[2]);
        buildTBNHost(lobe.n, lobe.t, lobe.b);
        const float rough = std::max(source.rough, 0.01f);
        lobe.frostBite.diffuse = float3(source.diffuse[0], source.diffuse[1], source.diffuse[2]);
        lobe.frostBite.rough = rough;
        lobe.cookTorrence.specular = float3(source.specular[0], source.specular[1], source.specular[2]);
        lobe.cookTorrence.alpha = rough * rough;
        const auto spec = lobe.cookTorrence.specular;
        lobe.specularWeight = saturate((20 * (spec.x + spec.y + spec.z) / 3 + 1) / 21.0f);
        result.surface.weights[compact] = source.weight;
    }
    return result;
}
static_assert(LOBE_COUNT == 4 && VISIBLE_LOBE_COUNT == 8);
static_assert(sizeof(VoxelData) == 420 && sizeof(TEBSDF) == 508 && sizeof(VisibleTEBSDF) == 812);
static_assert(sizeof(VisibilityMask::Data) == 17 * sizeof(uint32_t));
} // namespace

RayMarchingPass::RayMarchingPass(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice), gridData(VoxelizationBase::GlobalGridData)
{
    mpDevice = pDevice;
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

    // A script may preselect a file without reading it. Unattended rendering
    // must explicitly opt in to a single load attempt with autoLoad=true.
    bool autoLoad = false;
    for (const auto& [key, value] : props)
    {
        if (key == "voxelFile")
        {
            const std::string path = value;
            filePaths = {std::filesystem::path(path)};
        }
        else if (key == "autoLoad") autoLoad = value;
        else if (key == "visibleMaterial") mUseVisibleMaterial = value;
        else if (key == "internalVisibilityMode") mInternalVisibilityMode = value;
        else if (key == "maxBounce") mMaxBounce = value;
        else if (key == "drawMode") mDrawMode = value;
    }
    FALCOR_CHECK(mInternalVisibilityMode <= 2, "internalVisibilityMode must be 0, 1 or 2.");
    mLoadRequested = autoLoad && !filePaths.empty();

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
    ref<Texture> pOutputColor = renderData.getTexture(kOutputColor);
    pRenderContext->clearRtv(pOutputColor->getRTV().get(), float4(0));
    if (!mpScene)
        return;

    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // ---- Step 1: Consume one explicit load request. Failures wait for Read. ----
    if (mLoadRequested)
    {
        mLoadRequested = false;
        try
        {
        GridData& gd = VoxelizationBase::GlobalGridData;

        std::ifstream f;
        f.open(filePaths[selectedFile], std::ios::binary | std::ios::ate);
        if (!f.is_open())
        {
            mExtensionStatus = "Cannot open voxel file: " + filePaths[selectedFile].string();
            std::cerr << mExtensionStatus << std::endl;
            return;
        }

        std::cout << "Reading voxel data from: " << filePaths[selectedFile] << std::endl;
        size_t fileSize = std::filesystem::file_size(filePaths[selectedFile]);
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

        // CPU files use the Octahedral projection-area layout. Do not fall
        // back to the old SH/legacy layout because VoxelData has changed.
        VoxelizationMetadata::Metadata metadata;
        const auto& binaryPath = filePaths[selectedFile];
        if (VoxelizationMetadata::read(binaryPath, metadata) &&
            metadata.version == VoxelizationMetadata::kCurrentVersion &&
            metadata.projectionEncoding == "octahedral" &&
            metadata.projectionResolution == OCTA_RES &&
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
        else
        {
            mHasVoxelMetadata = false;
            mVoxelFormatVersion = 0;
            mVoxelLodMode.clear();
            mVoxelProducer.clear();
            mAvailableLODLevels = 0;
            mExtensionStatus = "Voxel metadata is missing, not Octahedral, or does not match the binary: " +
                VoxelizationMetadata::sidecarPath(binaryPath).string() + ". Refusing to load the voxel file.";
            std::cerr << mExtensionStatus << std::endl;
            f.close();
            return;
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

        mVisibleMaterialActive = false;
        mVisibilityMasksAvailable = false;
        mVisibleLODLevels = 0;
        mExtensionStatus.clear();
        std::unique_ptr<VisibleMaterialFile::Reader> visibleReader;
        std::unique_ptr<VisibilityMaskFile::Reader> maskReader;
        std::vector<uint8_t> requiredVisibleNodes;
        if (mUseVisibleMaterial)
        {
            const auto fingerprint = VisibleMaterialFile::fingerprint(binaryPath.string());
            visibleReader = std::make_unique<VisibleMaterialFile::Reader>(
                binaryPath.string() + ".vmat", gd.solidVoxelCount, maxDepth, fingerprint);
            mVisibleLODLevels = std::min(mAvailableLODLevels, visibleReader->header.lodLevels);
            if (visibleReader->header.maskBytes)
            {
                visibleReader->validateMasks(binaryPath.string() + ".vbm");
                maskReader = std::make_unique<VisibilityMaskFile::Reader>(binaryPath.string() + ".vbm", gd.solidVoxelCount, fingerprint);
                mVisibilityMasksAvailable = true;
            }
            requiredVisibleNodes.resize(gd.solidVoxelCount, 0);
            uint32_t first = 0;
            for (uint32_t level = 0; level <= maxDepth; ++level)
            {
                for (uint32_t n = first; n < first + nodeCounts[level]; ++n)
                {
                    FALCOR_CHECK(octreeNodes[n].dataIndex < gd.solidVoxelCount, "Invalid extension node index.");
                    if (level >= maxDepth - mVisibleLODLevels)
                        requiredVisibleNodes[octreeNodes[n].dataIndex] = 1;
                }
                first += nodeCounts[level];
            }
            mExtensionStatus = mVisibilityMasksAvailable ? "Visible material + paired masks" : "Visible material; no masks (Q=1)";
        }

        // Create GPU buffer for octree nodes
        auto pOctreeBuffer = mpDevice->createStructuredBuffer(
            sizeof(OctreeNode), totalNodes, ResourceBindFlags::ShaderResource
        );
        pOctreeBuffer->setBlob(octreeNodes.data(), 0, totalNodes * sizeof(OctreeNode));

        // Split data across buffers to avoid D3D12 i32 offset overflow:
        // structured buffer byte offset = elementIndex * stride.
        // The maximum element count is derived from the current TEBSDF size.
        const uint32_t totalVoxels = gd.solidVoxelCount;
        const size_t materialStride = visibleReader ? sizeof(VisibleTEBSDF) : sizeof(TEBSDF);
        const size_t maxElemsPerBuffer = kRayMarchingBufferByteLimit / materialStride;
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
                  << " activeMaterialStride=" << materialStride
                  << " sizeof(Ellipsoid)=" << sizeof(Ellipsoid) << std::endl;
        std::cout << "RayMarching buffers=" << splitCount
                  << ", maxElementsPerBuffer=" << maxElemsPerBuffer
                  << ", maxGBufferBytes=" << kRayMarchingBufferByteLimit << std::endl;
        for (uint32_t b = 0; b < splitCount; ++b)
        {
            std::cout << "count[" << b << "]=" << counts[b]
                      << " (" << (size_t(counts[b]) * materialStride / (1024.0 * 1024.0))
                      << " MB gBuffer)" << std::endl;
        }

        // Convert VoxelData -> TEBSDF on the CPU and upload only the final
        // gBuffer (TEBSDF) + pBuffer (Ellipsoid), split into chunks to avoid
        // the D3D12 2^31-1 byte-offset limit. The raw VoxelData is never staged
        // on the GPU, keeping VRAM at gBuffer + pBuffer only.
        std::vector<ref<Buffer>> gBuffers(splitCount);
        std::vector<ref<Buffer>> pBuffers(splitCount);
        std::vector<ref<Buffer>> maskBuffers(visibleReader ? splitCount : 0);

        for (uint32_t b = 0; b < splitCount; ++b)
        {
            if (counts[b] == 0)
                continue;

            std::vector<TEBSDF> gbChunk(visibleReader ? 0 : counts[b]);
            std::vector<VisibleTEBSDF> visibleChunk;
            std::vector<Ellipsoid> pChunk(counts[b]);

            // Conversion is pure/read-only with respect to voxelData, so let
            // the standard parallel algorithm use the CPU worker threads.
            // GPU resource creation and setBlob() stay on this render thread.
            const auto sourceBegin = voxelData.begin() + bases[b];
            const auto sourceEnd = sourceBegin + counts[b];
            if (visibleReader)
            {
                const auto records = visibleReader->read(bases[b], counts[b]);
                const auto maskChunk = maskReader ? maskReader->read(bases[b], counts[b]) : std::vector<VisibilityMask::Data>{};
                visibleChunk.resize(counts[b]);
                for (uint32_t j = 0; j < counts[b]; ++j)
                {
                    const uint32_t index = bases[b] + j;
                    FALCOR_CHECK(!requiredVisibleNodes[index] || records[j].baked, "Visible material missing for a rendered LOD.");
                    if (mVisibilityMasksAvailable)
                        FALCOR_CHECK(maskChunk[j].sourceLobes() == records[j].sourceLobes(), "Mask/material lobe order mismatch.");
                    visibleChunk[j] = convertVisibleMaterial(voxelData[index], records[j]);
                }
                // Keep an explicit zero record bound when no mask was baked.
                maskBuffers[b] = mpDevice->createStructuredBuffer(sizeof(uint32_t),
                    mVisibilityMasksAvailable ? counts[b] * VisibilityMask::WordsPerRecord : VisibilityMask::WordsPerRecord,
                    ResourceBindFlags::ShaderResource);
                const VisibilityMask::Data empty{};
                maskBuffers[b]->setBlob(mVisibilityMasksAvailable ? maskChunk.data() : &empty, 0,
                    (mVisibilityMasksAvailable ? size_t(counts[b]) : 1) * sizeof(empty));
            }
            else std::transform(
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
                materialStride, counts[b],
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal
            );
            pBuffers[b] = mpDevice->createStructuredBuffer(
                sizeof(Ellipsoid), counts[b],
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal
            );
            std::cout << "gBuffer[" << b << "]: " << (gBuffers[b] ? "OK" : "NULL!")
                      << " (" << (size_t(counts[b]) * materialStride / (1024.0 * 1024.0)) << " MB)"
                      << std::endl;
            std::cout << "pBuffer[" << b << "]: " << (pBuffers[b] ? "OK" : "NULL!")
                      << " (" << (size_t(counts[b]) * sizeof(Ellipsoid) / (1024.0 * 1024.0)) << " MB)"
                      << std::endl;
            if (gBuffers[b])
                gBuffers[b]->setBlob(visibleReader ? static_cast<const void*>(visibleChunk.data()) : gbChunk.data(),
                    0, size_t(counts[b]) * materialStride);
            if (pBuffers[b])
                pBuffers[b]->setBlob(pChunk.data(), 0, size_t(counts[b]) * sizeof(Ellipsoid));
        }

        // Flush the uploads before Step 2 reads the buffers.
        pRenderContext->submit(true);

        mGBuffers = std::move(gBuffers);
        mPBuffers = std::move(pBuffers);
        mVisibilityMaskBuffers = std::move(maskBuffers);
        mVisibleMaterialActive = visibleReader != nullptr;
        mGBufferSplits = std::move(splits);
        mBufferCount = splitCount;
        mOctreeBuffer = pOctreeBuffer;
        mOctreeMaxDepth = maxDepth;
        mOctreeNodeCounts = std::move(nodeCounts);

        }
        catch (const std::exception& error)
        {
            mExtensionStatus = error.what();
            std::cerr << "Voxel loading failed: " << mExtensionStatus << std::endl;
            return;
        }
    }

    if (mGBuffers.empty() || !mOctreeBuffer) return;

    // ---- Step 2: Ray marching ----
    FALCOR_PROFILE(pRenderContext, "RayMarching");
    ref<Camera> pCamera = mpScene->getCamera();
    if (!mSelectedVoxel)
    {
        mSelectedVoxel =
            mpDevice->createStructuredBuffer(sizeof(float4), 2, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    }

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
            defines.add("USE_VISIBLE_MATERIAL", mVisibleMaterialActive ? "1" : "0");
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
            if (mVisibleMaterialActive) var["visibilityMaskBuffer"][i] = mVisibilityMaskBuffers[i];
        }
        var["octreeBuffer"] = mOctreeBuffer;
        var["selectedVoxel"] = mSelectedVoxel;

        auto cb_GridData = var["GridData"];
        cb_GridData["gridMin"] = gridData.gridMin;
        cb_GridData["voxelSize"] = gridData.voxelSize;
        cb_GridData["voxelCount"] = gridData.voxelCount;
        cb_GridData["octreeMaxDepth"] = mOctreeMaxDepth;

        auto cb = var["CB"];
        cb["pixelCount"] = mOutputResolution;
        cb["invVP"] = math::inverse(pCamera->getViewProjMatrixNoJitter());
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
        cb["availableLODLevels"] = mVisibleMaterialActive ? mVisibleLODLevels : mAvailableLODLevels;
        for (size_t i = 0; i < mGBufferSplits.size(); ++i)
            cb["gbSplits"][i] = mGBufferSplits[i];
        cb["coverageBlend"] = mCoverageBlend;
        if (mVisibleMaterialActive)
        {
            cb["internalVisibilityMode"] = mInternalVisibilityMode;
            cb["visibilityMasksAvailable"] = mVisibilityMasksAvailable;
            cb["maxTransmission"] = mMaxTransmission;
            cb["maskIncidentDirection"] = mMaskIncidentDirection;
            cb["maskLobe"] = mMaskLobe;
        }
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
            defines.add("USE_VISIBLE_MATERIAL", mVisibleMaterialActive ? "1" : "0");
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
        }
        else
        {
            mSelectedHit = false;
            mSelectedGbOffset = 0xFFFFFFFF;
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
    // ---- File selection (from ReadVoxelPass) ----
    const bool refreshFiles = widget.button("Refresh files");
    if (!mFileListInitialized || VoxelizationBase::FileUpdated || refreshFiles)
    {
        const auto previousFile = selectedFile < filePaths.size() ? filePaths[selectedFile] : std::filesystem::path{};
        filePaths.clear();
        selectedFile = 0;
        mFileListError.clear();
        // A missing/inaccessible directory must not unwind out of the UI callback.
        std::error_code ec;
        for (std::filesystem::directory_iterator it(VoxelizationBase::ResourceFolder, ec), end;
             !ec && it != end; it.increment(ec))
        {
            // Metadata is a sidecar (<file>.bin.meta), not a selectable voxel
            // data file. Keep the dropdown restricted to binary voxel files.
            if (it->is_regular_file(ec) && it->path().extension() == ".bin")
            {
                filePaths.push_back(it->path());
            }
            if (ec) break;
        }
        if (ec)
        {
            mFileListError = "Cannot list voxel files in " + VoxelizationBase::ResourceFolder + ": " + ec.message();
            logWarning("RayMarchingPass: {}", mFileListError);
        }
        // Preserve a script-specified file outside ResourceFolder as well.
        if (!previousFile.empty())
        {
            auto previous = std::find(filePaths.begin(), filePaths.end(), previousFile);
            if (previous == filePaths.end()) { filePaths.push_back(previousFile); previous = filePaths.end() - 1; }
            selectedFile = uint32_t(previous - filePaths.begin());
        }
        mFileListInitialized = true;
        VoxelizationBase::FileUpdated = false;
    }
    if (!mFileListError.empty()) widget.text(mFileListError);
    Gui::DropdownList list;
    for (uint i = 0; i < filePaths.size(); i++)
    {
        list.push_back({i, filePaths[i].filename().string()});
    }
    if (list.empty())
        widget.text("No voxel .bin files found in " + VoxelizationBase::ResourceFolder);
    else
        widget.dropdown("File", list, selectedFile);
    widget.checkbox("Visible material (apply with Read)", mUseVisibleMaterial);

    if (mpScene && selectedFile < filePaths.size() && widget.button("Read"))
    {
        // The buffer count is part of the shader resource layout. Drop the
        // old resources and passes before execute() computes the new count.
        // All file I/O is deferred to that single load attempt.
        mGBuffers.clear();
        mPBuffers.clear();
        mVisibilityMaskBuffers.clear();
        mVisibleMaterialActive = false;
        mVisibilityMasksAvailable = false;
        mVisibleLODLevels = 0;
        mExtensionStatus.clear();
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
        mpFullScreenPass = nullptr;
        mpDisplayNDFPass = nullptr;

        requestRecompile();
        mLoadRequested = true;
        mOptionsChanged = true;
    }
    if (mGBuffers.empty()) widget.text("Select a voxel file and press Read to load it.");

    widget.text("Voxel Size: " + ToString(gridData.voxelSize));
    widget.text("Voxel Count: " + ToString((int3)gridData.voxelCount));
    widget.text("Grid Min: " + ToString(gridData.gridMin));
    widget.text("Solid Voxel Count: " + std::to_string(gridData.solidVoxelCount));
    widget.text("Solid Rate: " + std::to_string(gridData.solidVoxelCount / (float)gridData.totalVoxelCount()));
    widget.text("Ray-Marching Buffer Count: " + std::to_string(mBufferCount));
    if (!mExtensionStatus.empty()) widget.text(mExtensionStatus);
    if (mVisibleMaterialActive)
    {
        widget.text("Visible-material coarse LODs: " + std::to_string(mVisibleLODLevels));
        Gui::DropdownList modes = {{0, "Legacy VPA/TPA"}, {1, "Bitmask"}, {2, "Off"}};
        if (widget.dropdown("Internal visibility", modes, mInternalVisibilityMode)) mOptionsChanged = true;
        if (widget.slider("Max null transmissions", mMaxTransmission, 0u, 1024u)) mOptionsChanged = true;
        if (widget.var("Mask incident direction", mMaskIncidentDirection)) mOptionsChanged = true;
        if (widget.slider("Mask lobe", mMaskLobe, 0u, 7u)) mOptionsChanged = true;
    }
    widget.text("Max Polygon Count: " + std::to_string(gridData.maxPolygonCount));
    widget.text("Total Polygon Count: " + std::to_string(gridData.totalPolygonCount));

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
        int maxLOD = (int)std::min(mOctreeMaxDepth, mVisibleMaterialActive ? mVisibleLODLevels : mAvailableLODLevels);
        mForcedLOD = std::min(mForcedLOD, maxLOD);
        mMaxLODLevel = std::min(mMaxLODLevel, maxLOD);
        if (widget.slider("Forced LOD", mForcedLOD, -1, maxLOD))
            mOptionsChanged = true;
        if (mForcedLOD >= 0)
            widget.text("LOD " + std::to_string(mForcedLOD) + ": node size = " +
                        std::to_string(1 << (maxLOD - mForcedLOD)) + " leaf voxels");

        // Max LOD level cap for dynamic LOD
        if (widget.slider("Max LOD Level", mMaxLODLevel, -1, maxLOD))
            mOptionsChanged = true;
        if (mMaxLODLevel >= 0)
            widget.text("LOD cap at level " + std::to_string(mMaxLODLevel)
                        + ": node size = " + std::to_string(1 << (maxLOD - mMaxLODLevel)) + " leaf cells");
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
            ref<Camera> camera = mpScene ? mpScene->getCamera() : nullptr;
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
    }
    else
    {
        widget.text("Selected Voxel: none (no hit)");
    }
}

void RayMarchingPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpFullScreenPass = nullptr;
    mpDisplayNDFPass = nullptr;
    mDebug = false;
    mUseEmissiveLight = false;
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
