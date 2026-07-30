#include "RayMarchingPass.h"
#include "Shading.slang"
#include "Math/SphericalHarmonics.slang"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Math/FalcorMath.h"
#include <fstream>
#include <filesystem>

namespace
{
const std::string kShaderFile = "RenderPasses/Voxelization/RayMarching.ps.slang";
const std::string kDisplayShaderFile = "RenderPasses/Voxelization/DisplayNDF.ps.slang";
const std::string kPrepareProgramFile = "RenderPasses/Voxelization/PrepareShadingData.cs.slang";
const std::string kOutputColor = "color";
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
    mCheckVisibility = true;
    mCheckCoverage = true;
    mDrawMode = 9;
    mMaxBounce = 0;
    mRenderBackGround = true;
    mClearColor = float3(0);
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

        GridData& gd = VoxelizationBase::GlobalGridData;

        std::ifstream f;
        f.open(filePaths[selectedFile], std::ios::binary | std::ios::ate);
        if (!f.is_open())
            return;

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
        uint32_t totalVoxels = gd.solidVoxelCount;
        const uint32_t kMaxSafeBytes = 0x7FFFFFFFu;  // 2^31 - 1
        uint32_t maxElemsPerBuffer = kMaxSafeBytes / sizeof(TEBSDF);  // ~5.5M
        uint32_t numSplits = (totalVoxels + maxElemsPerBuffer - 1) / maxElemsPerBuffer;

        uint32_t count0 = min(maxElemsPerBuffer, totalVoxels);
        uint32_t count1 = (totalVoxels > maxElemsPerBuffer) ? min(maxElemsPerBuffer, totalVoxels - maxElemsPerBuffer) : 0;
        uint32_t count2 = (totalVoxels > maxElemsPerBuffer * 2) ? (totalVoxels - maxElemsPerBuffer * 2) : 0;
        uint32_t gbSplit0 = count0;
        uint32_t gbSplit1 = count0 + count1;

        // Debug: print struct sizes
        std::cout << "sizeof(VoxelData)=" << sizeof(VoxelData)
                  << " sizeof(TEBSDF)=" << sizeof(TEBSDF)
                  << " sizeof(Ellipsoid)=" << sizeof(Ellipsoid) << std::endl;
        std::cout << "MaxElemsPerBuffer=" << maxElemsPerBuffer << " splits=" << numSplits << std::endl;
        std::cout << "count0=" << count0 << " ("
                  << (count0 * sizeof(TEBSDF) / (1024.0 * 1024.0)) << " MB gBuffer0)"
                  << std::endl;
        if (count1 > 0)
            std::cout << "count1=" << count1 << " ("
                      << (count1 * sizeof(TEBSDF) / (1024.0 * 1024.0)) << " MB gBuffer1)"
                      << std::endl;
        if (count2 > 0)
            std::cout << "count2=" << count2 << " ("
                      << (count2 * sizeof(TEBSDF) / (1024.0 * 1024.0)) << " MB gBuffer2)"
                      << std::endl;

        auto pVoxelDataBuffer0 = mpDevice->createStructuredBuffer(
            sizeof(VoxelData), count0, ResourceBindFlags::ShaderResource
        );
        std::cout << "voxelDataBuffer0: " << (pVoxelDataBuffer0 ? "OK" : "NULL!") << std::endl;
        if (pVoxelDataBuffer0)
            pVoxelDataBuffer0->setBlob(voxelData.data(), 0, count0 * sizeof(VoxelData));

        ref<Buffer> pVoxelDataBuffer1 = nullptr, pVoxelDataBuffer2 = nullptr;
        if (count1 > 0)
        {
            pVoxelDataBuffer1 = mpDevice->createStructuredBuffer(
                sizeof(VoxelData), count1, ResourceBindFlags::ShaderResource
            );
            std::cout << "voxelDataBuffer1: " << (pVoxelDataBuffer1 ? "OK" : "NULL!") << std::endl;
            if (pVoxelDataBuffer1)
                pVoxelDataBuffer1->setBlob(voxelData.data() + count0, 0, count1 * sizeof(VoxelData));
        }
        if (count2 > 0)
        {
            pVoxelDataBuffer2 = mpDevice->createStructuredBuffer(
                sizeof(VoxelData), count2, ResourceBindFlags::ShaderResource
            );
            std::cout << "voxelDataBuffer2: " << (pVoxelDataBuffer2 ? "OK" : "NULL!") << std::endl;
            if (pVoxelDataBuffer2)
                pVoxelDataBuffer2->setBlob(voxelData.data() + count0 + count1, 0, count2 * sizeof(VoxelData));
        }

        auto pGBuffer0 = mpDevice->createStructuredBuffer(
            sizeof(TEBSDF), count0,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal
        );
        std::cout << "gBuffer0: " << (pGBuffer0 ? "OK" : "NULL!") << std::endl;

        ref<Buffer> pGBuffer1 = nullptr, pGBuffer2 = nullptr;
        if (count1 > 0)
        {
            pGBuffer1 = mpDevice->createStructuredBuffer(
                sizeof(TEBSDF), count1,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal
            );
            std::cout << "gBuffer1: " << (pGBuffer1 ? "OK" : "NULL!") << std::endl;
        }
        if (count2 > 0)
        {
            pGBuffer2 = mpDevice->createStructuredBuffer(
                sizeof(TEBSDF), count2,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal
            );
            std::cout << "gBuffer2: " << (pGBuffer2 ? "OK" : "NULL!") << std::endl;
        }

        auto pPBuffer0 = mpDevice->createStructuredBuffer(
            sizeof(Ellipsoid), count0,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal
        );
        std::cout << "pBuffer0: " << (pPBuffer0 ? "OK" : "NULL!") << std::endl;

        ref<Buffer> pPBuffer1 = nullptr, pPBuffer2 = nullptr;
        if (count1 > 0)
        {
            pPBuffer1 = mpDevice->createStructuredBuffer(
                sizeof(Ellipsoid), count1,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal
            );
            std::cout << "pBuffer1: " << (pPBuffer1 ? "OK" : "NULL!") << std::endl;
        }
        if (count2 > 0)
        {
            pPBuffer2 = mpDevice->createStructuredBuffer(
                sizeof(Ellipsoid), count2,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal
            );
            std::cout << "pBuffer2: " << (pPBuffer2 ? "OK" : "NULL!") << std::endl;
        }

        // Prepare pass: bind buffers and dispatch
        ShaderVar var = mPreparePass->getRootVar();
        var["voxelDataBuffer0"] = pVoxelDataBuffer0;
        if (count1 > 0)
            var["voxelDataBuffer1"] = pVoxelDataBuffer1;
        if (count2 > 0)
            var["voxelDataBuffer2"] = pVoxelDataBuffer2;
        var["gBuffer0"] = pGBuffer0;
        if (count1 > 0)
            var["gBuffer1"] = pGBuffer1;
        if (count2 > 0)
            var["gBuffer2"] = pGBuffer2;
        var["pBuffer0"] = pPBuffer0;
        if (count1 > 0)
            var["pBuffer1"] = pPBuffer1;
        if (count2 > 0)
            var["pBuffer2"] = pPBuffer2;

        auto cb = var["CB"];
        cb["voxelCount"] = totalVoxels;
        cb["gbSplit0"] = gbSplit0;
        cb["gbSplit1"] = gbSplit1;

        const uint32_t kMaxThreads = 65535u * 256u;
        for (uint32_t base = 0; base < totalVoxels; base += kMaxThreads)
        {
            uint32_t chunkVoxels = min(totalVoxels - base, kMaxThreads);
            cb["baseOffset"] = base;
            mPreparePass->execute(pRenderContext, uint3(chunkVoxels, 1, 1));
        }

        pRenderContext->submit(true);

        // Store as member variables
        mGBuffer0 = pGBuffer0;
        mGBuffer1 = pGBuffer1;
        mGBuffer2 = pGBuffer2;
        mPBuffer0 = pPBuffer0;
        mPBuffer1 = pPBuffer1;
        mPBuffer2 = pPBuffer2;
        mGBufferSplit0 = gbSplit0;
        mGBufferSplit1 = gbSplit1;
        mOctreeBuffer = pOctreeBuffer;
        mOctreeMaxDepth = maxDepth;
        mOctreeNodeCounts = std::move(nodeCounts);

        mComplete = true;
    }

    // ---- Step 2: Ray marching ----
    FALCOR_PROFILE(pRenderContext, "RayMarching");
    ref<Camera> pCamera = mpScene->getCamera();
    ref<Texture> pOutputColor = renderData.getTexture(kOutputColor);
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
            mpFullScreenPass = FullScreenPass::create(mpDevice, desc, mpScene->getSceneDefines());
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

        var["gBuffer0"] = mGBuffer0;
        var["gBuffer1"] = mGBuffer1;
        var["gBuffer2"] = mGBuffer2;
        var["pBuffer0"] = mPBuffer0;
        var["pBuffer1"] = mPBuffer1;
        var["pBuffer2"] = mPBuffer2;
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
        cb["gbSplit0"] = mGBufferSplit0;
        cb["gbSplit1"] = mGBufferSplit1;
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
            mpDisplayNDFPass = FullScreenPass::create(mpDevice, desc);
        }
        auto var = mpDisplayNDFPass->getRootVar();
        var["gBuffer0"] = mGBuffer0;
        var["gBuffer1"] = mGBuffer1;
        var["gBuffer2"] = mGBuffer2;
        var["selectedVoxel"] = mSelectedVoxel;

        auto cb = var["CB"];
        cb["clearColor"] = float4(mClearColor, 0);
        cb["gbSplit0"] = mGBufferSplit0;
        cb["gbSplit1"] = mGBufferSplit1;

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
    if (VoxelizationBase::FileUpdated)
    {
        filePaths.clear();
        for (const auto& entry : std::filesystem::directory_iterator(VoxelizationBase::ResourceFolder))
        {
            if (std::filesystem::is_regular_file(entry))
            {
                filePaths.push_back(entry.path());
            }
        }
        VoxelizationBase::FileUpdated = false;
    }
    Gui::DropdownList list;
    for (uint i = 0; i < filePaths.size(); i++)
    {
        list.push_back({i, filePaths[i].filename().string()});
    }
    widget.dropdown("File", list, selectedFile);

    if (mpScene && widget.button("Read"))
    {
        std::ifstream f;
        f.open(filePaths[selectedFile], std::ios::binary | std::ios::ate);
        if (f.is_open())
        {
            size_t fileSize = std::filesystem::file_size(filePaths[selectedFile]);
            size_t offset = 0;
            tryRead(f, offset, sizeof(GridData), &gridData, fileSize);
            f.close();

            requestRecompile();
            mComplete = false;
            mOptionsChanged = true;
        }
    }

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
    }
    if (widget.checkbox("Use Emissive Light", mUseEmissiveLight))
        mOptionsChanged = true;
    if (widget.checkbox("Check Ellipsoid", mCheckEllipsoid))
        mOptionsChanged = true;
    if (widget.checkbox("Check Visibility", mCheckVisibility))
        mOptionsChanged = true;
    if (widget.checkbox("Check Coverage", mCheckCoverage))
        mOptionsChanged = true;
    if (widget.slider("Shadow Bias(x100)", mShadowBias100, 0.0f, 0.2f))
        mOptionsChanged = true;
    if (widget.slider("Min Pdf(x100)", mMinPdf100, 0.0f, 0.2f))
        mOptionsChanged = true;
    if (widget.slider("T Threshold(x100)", mTransmittanceThreshold100, 0.0f, 10.0f))
        mOptionsChanged = true;
    if (widget.dropdown("Draw Mode", reinterpret_cast<ABSDFDrawMode&>(mDrawMode)))
        mOptionsChanged = true;
    if (widget.slider("Max Bounce", mMaxBounce, 0u, 4u))
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
    mPreparePass = nullptr;
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
