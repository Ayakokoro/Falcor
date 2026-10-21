#include "RayMarchingPass.h"
#include "Shading.slang"
#include "Math/SphericalHarmonics.slang"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Math/FalcorMath.h"
#include "../../../LargeSceneVoxelization/VisibilityMaskFile.h"
#include <fstream>
#include <filesystem>

namespace
{
const std::string kShaderFile = "RenderPasses/Voxelization/RayMarching.ps.slang";
#if VOXELIZATION_ENABLE_PIXEL_INSPECTION
const std::string kDisplayShaderFile = "RenderPasses/Voxelization/DisplayNDF.ps.slang";
#endif
const std::string kPrepareProgramFile = "RenderPasses/Voxelization/PrepareShadingData.cs.slang";
const std::string kOutputColor = "color";
static_assert(kVisibilityMaskResolution == VisibilityMask::Resolution);
static_assert(kVisibilityMaskDirectionCount == VisibilityMask::DirectionCount);
static_assert(kVisibilityMaskSampleCount == VisibilityMask::SampleCount);
static_assert(kVisibilityMaskMaxLobes == VisibilityMask::MaxLobes && LOBE_COUNT == VisibilityMask::MaxLobes);
static_assert(kVisibilityMaskWordsPerRecord == VisibilityMask::WordsPerRecord);
} // namespace

RayMarchingPass::RayMarchingPass(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice), gridData(VoxelizationBase::GlobalGridData)
{
    mpDevice = pDevice;
    mComplete = true;
    mShadowBias100 = 0.01f;
    mMinPdf100 = 0.1f;
    mTransmittanceThreshold100 = 5.f;
    mUseEmissiveLight = false;
    mDebug = false;
    mCheckEllipsoid = true;
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
        mFrameIndex = 0;
    }

    // ---- Step 1: Load voxel data from file if needed (from ReadVoxelPass) ----
    if (!mComplete)
    {
        // Consume this load request once. Commit all buffers and GridData only
        // after success, so a failed file switch cannot mix old/new node data.
        mComplete = true;
        if (selectedFile >= filePaths.size())
        {
            mVoxelLoadStatus = "Voxel load failed: no file selected.";
            return;
        }
        mVoxelLoadStatus = "Voxel load failed: " + filePaths[selectedFile].filename().string();
        if (!mPreparePass)
        {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kPrepareProgramFile).csEntry("main");
            desc.addTypeConformances(mpScene->getTypeConformances());

            DefineList defines;
            defines.add(mpScene->getSceneDefines());
            mPreparePass = ComputePass::create(mpDevice, desc, defines, true);
        }

        GridData gd{};

        std::ifstream f;
        f.open(filePaths[selectedFile], std::ios::binary | std::ios::ate);
        if (!f.is_open())
            return;

        std::cout << "Reading voxel data from: " << filePaths[selectedFile] << std::endl;
        size_t fileSize = std::filesystem::file_size(filePaths[selectedFile]);
        size_t offset = 0;
        const std::string maskPath = VisibilityMaskFile::companionPath(filePaths[selectedFile].string());
        std::error_code maskError;
        const bool hasMaskFile = std::filesystem::exists(maskPath, maskError);
        VisibilityMaskFile::Fingerprint fingerprint;
        auto readVoxelBytes = [&](size_t bytes, void* dst)
        {
            if (!tryRead(f, offset, bytes, dst, fileSize)) return false;
            if (hasMaskFile) fingerprint.append(dst, bytes);
            return true;
        };

        // Read GridData header
        if (!readVoxelBytes(sizeof(GridData), &gd)) return;

        // Read octree header
        uint32_t maxDepth = 0;
        if (!readVoxelBytes(sizeof(uint32_t), &maxDepth)) return;
        // The traversal uses 23 mantissa bits (S_MAX in RayMarching.ps.slang).
        if (maxDepth > 23 || size_t(maxDepth + 1) * sizeof(uint32_t) > fileSize - offset)
            return;

        std::vector<uint32_t> nodeCounts(maxDepth + 1);
        if (!readVoxelBytes(nodeCounts.size() * sizeof(uint32_t), nodeCounts.data())) return;

        uint64_t totalNodes64 = 0;
        for (auto count : nodeCounts) totalNodes64 += count;
        if (totalNodes64 == 0 || totalNodes64 > UINT32_MAX || gd.solidVoxelCount == 0 || gd.solidVoxelCount > UINT32_MAX ||
            totalNodes64 * sizeof(OctreeNode) > fileSize - offset)
            return;
        const uint32_t totalNodes = uint32_t(totalNodes64);

        std::cout << "Octree: maxDepth=" << maxDepth << ", totalNodes=" << totalNodes;

        // Read all octree nodes
        std::vector<OctreeNode> octreeNodes(totalNodes);
        if (!readVoxelBytes(size_t(totalNodes) * sizeof(OctreeNode), octreeNodes.data())) return;
        // Internal LOD nodes also carry material and visibility records.
        for (const auto& node : octreeNodes)
        {
            if (node.dataIndex >= gd.solidVoxelCount)
            {
                std::cerr << "Voxel dataIndex out of range: " << node.dataIndex << std::endl;
                return;
            }
        }

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

        // VoxelData is raw and its stride did not change when lobe.weight became
        // a visible-area amplitude. A trailing tag rejects old, same-size bakes.
        constexpr uint32_t kVoxelBakeFormatTag = 0x33575056u; // "VPW3": same-side visibility correlation
        size_t expectedVoxelBytes = (size_t)gd.solidVoxelCount * sizeof(VoxelData);
        size_t expectedRemainingBytes = expectedVoxelBytes + sizeof(kVoxelBakeFormatTag);
        if (offset > fileSize || fileSize - offset != expectedRemainingBytes)
        {
            std::cerr << "Voxel data layout mismatch in " << filePaths[selectedFile]
                      << " (remaining bytes=" << (offset <= fileSize ? fileSize - offset : 0)
                      << ", expected=" << expectedRemainingBytes
                      << "). Re-bake this voxel file with the current version." << std::endl;
            return;
        }

        // Read VoxelData
        std::vector<VoxelData> voxelData(gd.solidVoxelCount);
        if (!readVoxelBytes(expectedVoxelBytes, voxelData.data())) return;
        uint32_t fileTag = 0;
        if (!readVoxelBytes(sizeof(fileTag), &fileTag)) return;
        if (fileTag != kVoxelBakeFormatTag)
        {
            std::cerr << "Unsupported voxel bake format in " << filePaths[selectedFile]
                      << ". Re-bake this voxel file." << std::endl;
            return;
        }

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

        std::vector<VisibilityMask::Data> visibilityMasks;
        std::string maskStatus = maskError ? "Cannot inspect companion: " + maskError.message()
                                          : "No companion; Bitmask uses Q=1 (unsupported).";
        if (hasMaskFile)
        {
            try
            {
                visibilityMasks = VisibilityMaskFile::read(maskPath, uint32_t(gd.solidVoxelCount), fingerprint);
                for (size_t i = 0; i < voxelData.size(); ++i)
                {
                    uint32_t sourceLobes = 0;
                    for (uint32_t k = 0; k < LOBE_COUNT; ++k)
                        if (voxelData[i].ABSDF.lobes[k].IsValid()) sourceLobes |= 1u << k;
                    if (visibilityMasks[i].sourceLobes() != sourceLobes)
                        throw std::runtime_error("Visibility-mask material lobe order mismatch at node " + std::to_string(i));
                }
                maskStatus = "VBM1 loaded: " + std::to_string(visibilityMasks.size()) + " nodes, 68 bytes/node.";
            }
            catch (const std::exception& error)
            {
                visibilityMasks.clear();
                maskStatus = std::string("Companion rejected; Bitmask uses Q=1 (unsupported): ") + error.what();
                std::cerr << maskStatus << std::endl;
            }
        }

        std::cout << ", solidVoxels=" << gd.solidVoxelCount << std::endl;
        for (uint32_t i = 0; i <= maxDepth; i++)
            std::cout << "  Level " << i << ": " << nodeCounts[i] << " nodes" << std::endl;

        // The same node boundaries serve VoxelData, TEBSDF, ellipsoids and
        // 17-word mask records. Bound the largest stride, including upload data.
        const uint32_t totalVoxels = uint32_t(gd.solidVoxelCount);
        constexpr uint32_t kMaxSafeBytes = 0x7FFFFFFFu;
        const size_t maxStride = std::max({sizeof(VoxelData), sizeof(TEBSDF), sizeof(Ellipsoid), sizeof(VisibilityMask::Data)});
        const uint32_t maxElemsPerBuffer = uint32_t(kMaxSafeBytes / maxStride);
        const uint32_t numSplits = uint32_t((uint64_t(totalVoxels) + maxElemsPerBuffer - 1) / maxElemsPerBuffer);
        if (numSplits > kMaxVoxelBufferCount)
        {
            mVoxelLoadStatus = "Voxel load failed: requires " + std::to_string(numSplits) +
                " buffer segments; the renderer supports at most " + std::to_string(kMaxVoxelBufferCount) + ".";
            return;
        }
        if (uint64_t(totalNodes) * sizeof(OctreeNode) > kMaxSafeBytes)
        {
            mVoxelLoadStatus = "Voxel load failed: octree buffer exceeds 2 GiB.";
            return;
        }

        // Create GPU buffer for octree nodes
        auto pOctreeBuffer = mpDevice->createStructuredBuffer(
            sizeof(OctreeNode), totalNodes, ResourceBindFlags::ShaderResource
        );
        pOctreeBuffer->setBlob(octreeNodes.data(), 0, totalNodes * sizeof(OctreeNode));

        // Split data across buffers to avoid D3D12 i32 offset overflow:
        // structured buffer byte offset = elementIndex * stride.
        std::array<uint32_t, kMaxVoxelBufferCount + 1> boundaries{};
        std::array<ref<Buffer>, kMaxVoxelBufferCount> pVoxelDataBuffers{};
        std::array<ref<Buffer>, kMaxVoxelBufferCount> pGBuffers{};
        std::array<ref<Buffer>, kMaxVoxelBufferCount> pPBuffers{};
        std::array<ref<Buffer>, kMaxVoxelBufferCount> pVisibilityBuffers{};

        // Debug: print struct sizes
        std::cout << "sizeof(VoxelData)=" << sizeof(VoxelData)
                  << " sizeof(TEBSDF)=" << sizeof(TEBSDF)
                  << " sizeof(Ellipsoid)=" << sizeof(Ellipsoid) << std::endl;
        std::cout << "MaxElemsPerBuffer=" << maxElemsPerBuffer << " splits=" << numSplits << std::endl;
        for (uint32_t i = 0; i < kMaxVoxelBufferCount; ++i)
        {
            const uint32_t firstNode = boundaries[i];
            const uint32_t count = min(maxElemsPerBuffer, totalVoxels - firstNode);
            boundaries[i + 1] = firstNode + count;
            if (count == 0) continue;

            std::cout << "count" << i << "=" << count << " ("
                      << (count * sizeof(TEBSDF) / (1024.0 * 1024.0)) << " MB gBuffer" << i << ")" << std::endl;

            pVoxelDataBuffers[i] = mpDevice->createStructuredBuffer(
                sizeof(VoxelData), count, ResourceBindFlags::ShaderResource,
                MemoryType::DeviceLocal, voxelData.data() + firstNode
            );
            pGBuffers[i] = mpDevice->createStructuredBuffer(
                sizeof(TEBSDF), count,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal
            );
            pPBuffers[i] = mpDevice->createStructuredBuffer(
                sizeof(Ellipsoid), count,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal
            );
            if (!visibilityMasks.empty())
                pVisibilityBuffers[i] = mpDevice->createStructuredBuffer(
                    sizeof(uint32_t), count * VisibilityMask::WordsPerRecord, ResourceBindFlags::ShaderResource,
                    MemoryType::DeviceLocal, visibilityMasks.data() + firstNode
                );
        }

        // Prepare pass: bind buffers and dispatch
        ShaderVar var = mPreparePass->getRootVar();
        for (uint32_t i = 0; i < kMaxVoxelBufferCount; ++i)
        {
            const std::string suffix = std::to_string(i);
            var["voxelDataBuffer" + suffix] = pVoxelDataBuffers[i];
            var["gBuffer" + suffix] = pGBuffers[i];
            var["pBuffer" + suffix] = pPBuffers[i];
        }

        auto cb = var["CB"];
        cb["voxelCount"] = totalVoxels;
        for (uint32_t i = 0; i < kMaxVoxelBufferCount - 1; ++i)
            cb["gbSplit" + std::to_string(i)] = boundaries[i + 1];

        const uint32_t kMaxThreads = 65535u * 256u;
        for (uint32_t base = 0; base < totalVoxels; base += kMaxThreads)
        {
            uint32_t chunkVoxels = min(totalVoxels - base, kMaxThreads);
            cb["baseOffset"] = base;
            mPreparePass->execute(pRenderContext, uint3(chunkVoxels, 1, 1));
        }

        pRenderContext->submit(true);

        // Store as member variables
        mGBuffers = std::move(pGBuffers);
        mPBuffers = std::move(pPBuffers);
        mVisibilityBuffers = std::move(pVisibilityBuffers);
        mVisibilityMaskStatus = std::move(maskStatus);
        for (uint32_t i = 0; i < kMaxVoxelBufferCount - 1; ++i)
            mGBufferSplits[i] = boundaries[i + 1];
        mOctreeBuffer = pOctreeBuffer;
        mOctreeMaxDepth = maxDepth;
        mOctreeNodeCounts = std::move(nodeCounts);
        gridData = gd;
        mVoxelLoadStatus = "Loaded: " + filePaths[selectedFile].filename().string();
        mSelectedHit = false;

    }

    if (!mGBuffers[0])
    {
        pRenderContext->clearRtv(renderData.getTexture(kOutputColor)->getRTV().get(), float4(0));
        return;
    }

    // ---- Step 2: Ray marching ----
    FALCOR_PROFILE(pRenderContext, "RayMarching");
    ref<Camera> pCamera = mpScene->getCamera();
    ref<Texture> pOutputColor = renderData.getTexture(kOutputColor);
#if VOXELIZATION_ENABLE_PIXEL_INSPECTION
    if (!mSelectedVoxel)
    {
        mSelectedVoxel =
            mpDevice->createStructuredBuffer(sizeof(float4), 3, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    }
    mSelectedPixel = uint2(mSelectedUV.x * pOutputColor->getWidth(), mSelectedUV.y * pOutputColor->getHeight());
#endif
    if (!mEmptyVisibilityBuffer)
    {
        const uint32_t zero = 0;
        mEmptyVisibilityBuffer = mpDevice->createStructuredBuffer(
            sizeof(uint32_t), 1, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, &zero);
    }

    pRenderContext->clearRtv(pOutputColor->getRTV().get(), float4(0));

#if VOXELIZATION_ENABLE_PIXEL_INSPECTION
    if (!mDisplayNDF)
#endif
    {
        if (!mpFullScreenPass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(kShaderFile).psEntry("main");
            desc.setShaderModel(ShaderModel::SM6_5);
            desc.addTypeConformances(mpScene->getTypeConformances());
            mpFullScreenPass = FullScreenPass::create(mpDevice, desc, mpScene->getSceneDefines());
        }
#if VOXELIZATION_ENABLE_PIXEL_INSPECTION
        pRenderContext->clearUAV(mSelectedVoxel->getUAV().get(), float4(-1));
#endif

        mpFullScreenPass->addDefine("VOXELIZATION_ENABLE_PIXEL_INSPECTION", std::to_string(VOXELIZATION_ENABLE_PIXEL_INSPECTION));
        mpFullScreenPass->addDefine("CHECK_ELLIPSOID", mCheckEllipsoid ? "1" : "0");
        // Mode selection is uniform shader data; the legacy function itself
        // stays enabled for Legacy and its diagnostic. Bitmask/Off bypass it.
        mpFullScreenPass->addDefine("CHECK_VISIBILITY", "1");
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

        for (uint32_t i = 0; i < kMaxVoxelBufferCount; ++i)
        {
            const std::string suffix = std::to_string(i);
            var["gBuffer" + suffix] = mGBuffers[i];
            var["pBuffer" + suffix] = mPBuffers[i];
            var["visibilityBuffer" + suffix] = mVisibilityBuffers[i] ? mVisibilityBuffers[i] : mEmptyVisibilityBuffer;
        }
        var["octreeBuffer"] = mOctreeBuffer;
#if VOXELIZATION_ENABLE_PIXEL_INSPECTION
        var["selectedVoxel"] = mSelectedVoxel;
#endif

        auto cb_GridData = var["GridData"];
        cb_GridData["gridMin"] = gridData.gridMin;
        cb_GridData["voxelSize"] = gridData.voxelSize;
        cb_GridData["voxelCount"] = gridData.voxelCount;
        cb_GridData["octreeMaxDepth"] = mOctreeMaxDepth;

        auto cb = var["CB"];
        cb["pixelCount"] = mOutputResolution;
        cb["invVP"] = math::inverse(pCamera->getViewProjMatrixNoJitter());
        cb["shadowBias"] = mShadowBias100 / 100 / gridData.voxelSize.x;
        cb["shadowNormalOffsetVoxels"] = mShadowNormalOffsetVoxels;
        cb["drawMode"] = mDrawMode;
        cb["frameIndex"] = mFrameIndex;
        cb["minPdf"] = mMinPdf100 / 100;
        cb["transmittanceThreshold"] = mTransmittanceThreshold100 / 100;
#if VOXELIZATION_ENABLE_PIXEL_INSPECTION
        cb["selectedPixel"] = mSelectedPixel;
#endif
        cb["renderBackGround"] = mRenderBackGround;
        cb["clearColor"] = float4(mClearColor, 0);
        cb["tanHalfFovY"] = std::tan(Falcor::focalLengthToFovY(pCamera->getFocalLength(), pCamera->getFrameHeight()) * 0.5f);
        cb["forcedLOD"] = mForcedLOD;
        cb["maxLODLevel"] = mMaxLODLevel;
        for (uint32_t i = 0; i < kMaxVoxelBufferCount - 1; ++i)
            cb["gbSplit" + std::to_string(i)] = mGBufferSplits[i];
        cb["coverageBlend"] = mCoverageBlend;
        cb["coverageTolerance"] = mCoverageTolerance;
        cb["maxTransmission"] = mMaxTransmission;
        cb["internalVisibilityMode"] = uint32_t(mInternalVisibilityMode);
        cb["visibilityMasksAvailable"] = mVisibilityBuffers[0] != nullptr;
        cb["visibilityDebugIncident"] = mVisibilityDebugIncident;
        cb["visibilityDebugLobe"] = mVisibilityDebugLobe;
        mFrameIndex++;

        mpFbo->attachColorTarget(pOutputColor, 0);
        mpFullScreenPass->execute(pRenderContext, mpFbo);
    }
#if VOXELIZATION_ENABLE_PIXEL_INSPECTION
    else
    {
        if (!mpDisplayNDFPass)
        {
            ProgramDesc desc;
            desc.addShaderLibrary(kDisplayShaderFile).psEntry("main");
            desc.setShaderModel(ShaderModel::SM6_5);
            mpDisplayNDFPass = FullScreenPass::create(mpDevice, desc);
        }
        auto var = mpDisplayNDFPass->getRootVar();
        for (uint32_t i = 0; i < kMaxVoxelBufferCount; ++i)
            var["gBuffer" + std::to_string(i)] = mGBuffers[i];
        var["selectedVoxel"] = mSelectedVoxel;

        auto cb = var["CB"];
        cb["clearColor"] = float4(mClearColor, 0);
        for (uint32_t i = 0; i < kMaxVoxelBufferCount - 1; ++i)
            cb["gbSplit" + std::to_string(i)] = mGBufferSplits[i];

        mpFbo->attachColorTarget(pOutputColor, 0);
        mpDisplayNDFPass->execute(pRenderContext, mpFbo);
    }

    // Diagnostic builds only: this readback waits for the GPU every frame.
    if (mSelectedVoxel)
    {
        if (!mpSelectedVoxelStaging)
            mpSelectedVoxelStaging = mpDevice->createBuffer(mSelectedVoxel->getSize(), ResourceBindFlags::None, MemoryType::ReadBack);
        pRenderContext->copyResource(mpSelectedVoxelStaging.get(), mSelectedVoxel.get());
        pRenderContext->submit(true);
        float4* pData = reinterpret_cast<float4*>(mpSelectedVoxelStaging->map());
        float4 v0 = pData[0];
        float4 v1 = pData[1];
        float4 visibility = pData[2];
        mpSelectedVoxelStaging->unmap();

        if (v0.x != -1.0f || v0.y != -1.0f || v0.z != -1.0f || v0.w != -1.0f)
        {
            mSelectedHit = true;
            mSelectedGbOffset = (uint)v0.w;
            mSelectedCellInt = int3((int)v1.x, (int)v1.y, (int)v1.z);
            mSelectedLegacyVisibility = v1.w;
            mSelectedVisibilityQuery = visibility;
        }
        else
        {
            mSelectedHit = false;
            mSelectedGbOffset = 0xFFFFFFFF;
            mSelectedCellInt = int3(-1);
            mSelectedLegacyVisibility = -1.0f;
            mSelectedVisibilityQuery = float4(-1.0f);
        }
    }
#endif
}

void RayMarchingPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mUseEmissiveLight = false;
    VoxelizationBase::LightChanged = true;
}

void RayMarchingPass::renderUI(Gui::Widgets& widget)
{
    // ---- File selection (from ReadVoxelPass) ----
    if (VoxelizationBase::FileUpdated)
    {
        filePaths.clear();
        for (const auto& entry : std::filesystem::directory_iterator(VoxelizationBase::ResourceFolder))
        {
            if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".bin")
            {
                filePaths.push_back(entry.path());
            }
        }
        VoxelizationBase::FileUpdated = false;
        if (selectedFile >= filePaths.size()) selectedFile = 0;
    }
    Gui::DropdownList list;
    for (uint i = 0; i < filePaths.size(); i++)
    {
        list.push_back({i, filePaths[i].filename().string()});
    }
    widget.dropdown("File", list, selectedFile);

    if (mpScene && selectedFile < filePaths.size() && widget.button("Read"))
    {
        mComplete = false;
        mDisplayNDF = false; // Refresh the selected node before inspecting a new file.
        mOptionsChanged = true;
    }
    widget.text(mVoxelLoadStatus);

    widget.text("Voxel Size: " + ToString(gridData.voxelSize));
    widget.text("Voxel Count: " + ToString((int3)gridData.voxelCount));
    widget.text("Grid Min: " + ToString(gridData.gridMin));
    widget.text("Solid Voxel Count: " + std::to_string(gridData.solidVoxelCount));
    widget.text("Solid Rate: " + std::to_string(gridData.solidVoxelCount / (float)gridData.totalVoxelCount()));
    widget.text("Max Polygon Count: " + std::to_string(gridData.maxPolygonCount));
    widget.text("Total Polygon Count: " + std::to_string(gridData.totalPolygonCount));

    if (mOctreeMaxDepth > 0)
    {
        widget.text("Octree Max Depth: " + std::to_string(mOctreeMaxDepth));
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
        int maxLOD = (int)mOctreeMaxDepth;
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
    if (widget.dropdown("Internal Visibility", mInternalVisibilityMode))
        mOptionsChanged = true;
    widget.text(mVisibilityMaskStatus);
    if (mInternalVisibilityMode == InternalVisibilityMode::Bitmask && !mVisibilityBuffers[0])
        widget.text("Bitmask unavailable: Q=1 fallback. Legacy attenuation is not applied.");
    if (widget.checkbox("Check Coverage", mCheckCoverage))
        mOptionsChanged = true;
    if (mCheckCoverage)
    {
        if (widget.slider("Coverage Tolerance", mCoverageTolerance, 0.0f, 0.49f))
            mOptionsChanged = true;
        if (widget.slider("Coverage Blend", mCoverageBlend, 0.0f, 1.0f))
            mOptionsChanged = true;
    }
    if (widget.slider("Shadow Bias(x100)", mShadowBias100, 0.0f, 0.2f))
        mOptionsChanged = true;
    if (widget.slider("Shadow Normal Offset (voxels)", mShadowNormalOffsetVoxels, 0.0f, 8.0f))
        mOptionsChanged = true;
    if (widget.slider("Min Pdf(x100)", mMinPdf100, 0.0f, 0.2f))
        mOptionsChanged = true;
    if (widget.slider("T Threshold(x100)", mTransmittanceThreshold100, 0.0f, 10.0f))
        mOptionsChanged = true;
    if (widget.dropdown("Draw Mode", reinterpret_cast<ABSDFDrawMode&>(mDrawMode)))
        mOptionsChanged = true;
    if (mInternalVisibilityMode == InternalVisibilityMode::Bitmask || mDrawMode == uint(ABSDFDrawMode::InternalVisibility))
    {
        if (widget.var("Visibility Incident (world)", mVisibilityDebugIncident, -1.0f, 1.0f, 0.01f))
        {
            mDisplayNDF = false;
            mOptionsChanged = true;
        }
        if (mInternalVisibilityMode == InternalVisibilityMode::Bitmask)
        {
            if (widget.slider("Visibility Lobe (compact)", mVisibilityDebugLobe, 0u, kVisibilityMaskMaxLobes - 1))
            {
                mDisplayNDF = false;
                mOptionsChanged = true;
            }
            widget.text("Visibility debug: grayscale=Q, magenta=unsupported. Incident must be nonzero.");
        }
    }
    if (widget.slider("Max Bounce", mMaxBounce, 0u, 4u))
        mOptionsChanged = true;
    if (widget.var("Max Transmission", mMaxTransmission, 0u, 16384u))
        mOptionsChanged = true;
#if VOXELIZATION_ENABLE_PIXEL_INSPECTION
    if (widget.checkbox("Display NDF", mDisplayNDF))
        mOptionsChanged = true;
#endif
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

#if VOXELIZATION_ENABLE_PIXEL_INSPECTION
    widget.text("Selected Pixel: " + ToString(mSelectedPixel));
    if (mSelectedHit)
    {
        widget.text("Selected Voxel cellInt: " + ToString(mSelectedCellInt));
        widget.text("Selected Voxel gbOffset: " + std::to_string(mSelectedGbOffset));
        if (mInternalVisibilityMode == InternalVisibilityMode::Bitmask)
        {
            widget.text("Selected Q[" + std::to_string(mVisibilityDebugLobe) + "]: " + std::to_string(mSelectedVisibilityQuery.x));
            widget.text("Outgoing samples: " + std::to_string(uint(mSelectedVisibilityQuery.y)) +
                        ", material lobes: " + std::to_string(uint(mSelectedVisibilityQuery.w)));
            widget.text(mSelectedVisibilityQuery.z > 0.0f ? "Visibility supported" : "Visibility unsupported (Q=1 fallback)");
        }
        else if (mInternalVisibilityMode == InternalVisibilityMode::Legacy)
            widget.text("Selected legacy VPA/TPA(incident): " + std::to_string(mSelectedLegacyVisibility));
        else
            widget.text("Internal visibility: Off (factor 1)");
    }
    else
    {
        widget.text("Selected Voxel: none (no hit)");
    }
#endif
}

void RayMarchingPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpFullScreenPass = nullptr;
    mpDisplayNDFPass = nullptr;
    mPreparePass = nullptr;
    mDebug = false;
    mUseEmissiveLight = false;
}

bool RayMarchingPass::onMouseEvent(const MouseEvent& mouseEvent)
{
#if VOXELIZATION_ENABLE_PIXEL_INSPECTION
    if (mouseEvent.type == MouseEvent::Type::ButtonDown && mouseEvent.button == Input::MouseButton::Left)
    {
        mSelectedUV = mouseEvent.pos;
        return true;
    }
#endif
    return false;
}

bool RayMarchingPass::tryRead(std::ifstream& f, size_t& offset, size_t bytes, void* dst, size_t fileSize)
{
    if (offset > fileSize || bytes > fileSize - offset)
        return false;
    if (dst)
    {
        f.seekg(offset, std::ios::beg);
        f.read(reinterpret_cast<char*>(dst), bytes);
        if (!f) return false;
    }
    offset += bytes;
    return true;
}
