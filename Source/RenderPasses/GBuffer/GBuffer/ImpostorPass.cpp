/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "ImpostorPass.h"
namespace
{
const std::string kDepthPassProgramFile = "RenderPasses/GBuffer/GBuffer/DepthPass.3d.slang";
const std::string kGBufferPassProgramFile = "RenderPasses/GBuffer/GBuffer/ImpostorRaster.3d.slang";
const std::string kComputePassProgramFile = "RenderPasses/GBuffer/GBuffer/MipmapGenerator.cs.slang";
const RasterizerState::CullMode kDefaultCullMode = RasterizerState::CullMode::Back;
const ChannelList kImpostorChannels = {
    {"packedNDO", "gPackedNDO", "World normal(x,y), depth, opacity", false, ResourceFormat::RGBA32Float},
    {"packedMCR", "gPackedMCR", "Material id or counter(int), uv, extra roughness", false, ResourceFormat::RGBA32Float},
};

const std::string kDepthName = "depth";

const std::string kViewpointIndex = "viewpointIndex";
const std::string kOutputInvVP = "invVP";
} // namespace

ImpostorPass::ImpostorPass(ref<Device> pDevice, const Properties& props) : GBuffer(pDevice), mComplete(false)
{
    // Check for required features.
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_2))
        FALCOR_THROW("GBufferRaster requires Shader Model 6.2 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::Barycentrics))
        FALCOR_THROW("GBufferRaster requires pixel shader barycentrics support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RasterizerOrderedViews))
        FALCOR_THROW("GBufferRaster requires rasterizer ordered views (ROVs) support.");

    parseProperties(props);

    // Initialize graphics state
    mDepthPass.pState = GraphicsState::create(mpDevice);
    mGBufferPass.pState = GraphicsState::create(mpDevice);

    // Set depth function
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthFunc(ComparisonFunc::Equal).setDepthWriteMask(false);
    ref<DepthStencilState> pDsState = DepthStencilState::create(dsDesc);
    mGBufferPass.pState->setDepthStencilState(pDsState);

    mpFbo = Fbo::create(mpDevice);

    mViewDirections.emplace_back(0, 0, 0); // 0号为默认观察方向
    mViewDirections.emplace_back(1, 0, 0);
    mViewDirections.emplace_back(-1, 0, 0);
    mViewDirections.emplace_back(0, 1, 0);
    mViewDirections.emplace_back(0, -1, 0);
    mViewDirections.emplace_back(0, 0, 1);
    mViewDirections.emplace_back(0, 0, -1);
    aspectRatio = 1.f;
}

RenderPassReflection ImpostorPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);
    reflector.addInternal(kDepthName, "Depth buffer")
        .format(ResourceFormat::D32Float)
        .bindFlags(ResourceBindFlags::DepthStencil)
        .texture2D(RiLoDWidth, RiLoDHeight, 1, 1);

    for (const auto& it : kImpostorChannels)
    {
        auto& tex = reflector.addOutput(it.name, it.desc).texture2D(RiLoDWidth, RiLoDHeight, 1, RiLoDMipCount);
        tex.bindFlags(ResourceBindFlags::RenderTarget);
        if (it.format != ResourceFormat::Unknown)
            tex.format(it.format);
        if (it.optional)
            tex.flags(RenderPassReflection::Field::Flags::Optional);
    }

    reflector.addOutput(kOutputInvVP, "Inverse ViewProjMatrix of impostor")
        .format(ResourceFormat::Unknown)
        .bindFlags(ResourceBindFlags::Constant)
        .rawBuffer(sizeof(float4x4));

    return reflector;
}

void ImpostorPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    GBuffer::compile(pRenderContext, compileData);
}

void ImpostorPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene)
    {
        return;
    }

    ref<Buffer> matrixBuffer = renderData.getResource(kOutputInvVP)->asBuffer();
    ref<Camera> camera = mpScene->getCamera();
    if (mComplete)
    {
        pRenderContext->copyResource(renderData.getTexture("packedNDO").get(), cachedPackedNDO.get());
        pRenderContext->copyResource(renderData.getTexture("packedMCR").get(), cachedPackedMCR.get());
        matrixBuffer->setBlob(&invVP, 0, sizeof(float4x4));
        return;
    }
    GBuffer::execute(pRenderContext, renderData);
    auto pDepth = renderData.getTexture(kDepthName);
    FALCOR_ASSERT(pDepth);
    updateFrameDim(uint2(pDepth->getWidth(), pDepth->getHeight()));

    pRenderContext->clearDsv(pDepth->getDSV().get(), 1.f, 0);

    const RasterizerState::CullMode cullMode = mForceCullMode ? mCullMode : kDefaultCullMode;

    // Check for scene changes.
    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded))
    {
        recreatePrograms();
    }

    mpScene->selectViewpoint(mViewpointIndex);

    camera->setFocalLength(0.f);
    camera->setAspectRatio(aspectRatio);
    camera->setFrameHeight(mViewpoint.cameraSize * 1000);
    camera->setFarPlane(mViewpoint.cameraSize * 2);
    mpScene->update(pRenderContext, 0.);

    invVP = math::inverse(camera->getViewProjMatrixNoJitter());
    matrixBuffer->setBlob(&invVP, 0, sizeof(float4x4));

    // Depth pass.
    {
        // Create depth pass program.
        if (!mDepthPass.pProgram)
        {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kDepthPassProgramFile).vsEntry("vsMain").psEntry("psMain");
            desc.addTypeConformances(mpScene->getTypeConformances());

            mDepthPass.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
            mDepthPass.pState->setProgram(mDepthPass.pProgram);
        }

        // Set program defines.
        mDepthPass.pState->getProgram()->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");

        // Create program vars.
        if (!mDepthPass.pVars)
            mDepthPass.pVars = ProgramVars::create(mpDevice, mDepthPass.pProgram.get());

        mpFbo->attachDepthStencilTarget(pDepth);
        mDepthPass.pState->setFbo(mpFbo);

        mpScene->rasterize(pRenderContext, mDepthPass.pState.get(), mDepthPass.pVars.get(), cullMode);
    }

    // GBuffer pass.
    {
        // Create GBuffer pass program.
        if (!mGBufferPass.pProgram)
        {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kGBufferPassProgramFile).vsEntry("vsMain").psEntry("psMain");
            desc.addTypeConformances(mpScene->getTypeConformances());

            mGBufferPass.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
            mGBufferPass.pState->setProgram(mGBufferPass.pProgram);
        }

        // Set program defines.
        mGBufferPass.pProgram->addDefine("ADJUST_SHADING_NORMALS", mAdjustShadingNormals ? "1" : "0");
        mGBufferPass.pProgram->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");

        // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
        // mGBufferPass.pProgram->addDefines(getValidResourceDefines(kGBufferChannels, renderData));
        mGBufferPass.pProgram->addDefines(getValidResourceDefines(kImpostorChannels, renderData));

        // Create program vars.
        if (!mGBufferPass.pVars)
            mGBufferPass.pVars = ProgramVars::create(mpDevice, mGBufferPass.pProgram.get());

        auto var = mGBufferPass.pVars->getRootVar();
        var["PerFrameCB"]["gFrameDim"] = mFrameDim;
        cachedPackedNDO = mpDevice->createTexture2D(
            RiLoDWidth,
            RiLoDHeight,
            ResourceFormat::RGBA32Float,
            1,
            RiLoDMipCount,
            nullptr,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
        );
        cachedPackedMCR = mpDevice->createTexture2D(
            RiLoDWidth,
            RiLoDHeight,
            ResourceFormat::RGBA32Float,
            1,
            RiLoDMipCount,
            nullptr,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
        );

        mpFbo->attachColorTarget(cachedPackedNDO, 0);
        mpFbo->attachColorTarget(cachedPackedMCR, 1);

        mGBufferPass.pState->setFbo(mpFbo); // Sets the viewport

        // Rasterize the scene.
        mpScene->rasterize(pRenderContext, mGBufferPass.pState.get(), mGBufferPass.pVars.get(), cullMode);
    }

    // Generate Mipmap
    {
        if (!mpComputePass)
        {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kComputePassProgramFile).csEntry("main");

            DefineList defines;
            defines.add(mpScene->getSceneDefines());
            // defines.add(getShaderDefines(renderData));

            mpComputePass = ComputePass::create(mpDevice, desc, defines, true);
        }

        uint2 size = uint2(cachedPackedNDO->getWidth(), cachedPackedNDO->getHeight());
        for (size_t i = 1; i < RiLoDMipCount; i++) // 目前只能生成两层Mipmap
        {
            if (i == 1)
                mpComputePass->addDefine("FIRSTCOUNTER", "1", true);
            else
                mpComputePass->addDefine("FIRSTCOUNTER", "0", true);

            ShaderVar var = mpComputePass->getRootVar();
            var["gPrevNDO"].setSrv(cachedPackedNDO->getSRV(i - 1, 1, 0, 1));
            var["gCurrentNDO"].setUav(cachedPackedNDO->getUAV(i, 0, 1));
            var["gPrevMCR"].setSrv(cachedPackedMCR->getSRV(i - 1, 1, 0, 1));
            var["gCurrentMCR"].setUav(cachedPackedMCR->getUAV(i, 0, 1));
            mpComputePass->execute(pRenderContext, uint3(size.x >> i, size.y >> i, 1));
            pRenderContext->uavBarrier(cachedPackedNDO.get());
        }
    }

    pRenderContext->copyResource(renderData.getTexture("packedNDO").get(), cachedPackedNDO.get());
    pRenderContext->copyResource(renderData.getTexture("packedMCR").get(), cachedPackedMCR.get());
    mComplete = true;
}

void ImpostorPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    GBuffer::setScene(pRenderContext, pScene);
    mComplete = false;
    recreatePrograms();

    if (pScene)
    {
        if (pScene->getMeshVao() && pScene->getMeshVao()->getPrimitiveTopology() != Vao::Topology::TriangleList)
        {
            FALCOR_THROW("GBufferRaster: Requires triangle list geometry due to usage of SV_Barycentrics.");
        }
    }

    AABB aabb = pScene->getSceneBounds();
    calculateViewPoint(aabb.minPoint, aabb.maxPoint, mViewpointIndex);
    float3 right = math::normalize(math::cross(mViewpoint.target - mViewpoint.position, mViewpoint.up));
    float3 diag = aabb.maxPoint - aabb.minPoint;
    float size = math::max(math::max(diag.x, diag.y), diag.z);
    mViewpoint.cameraSize = size * 1.01f;
    pScene->addViewpoint(mViewpoint.position, mViewpoint.target, mViewpoint.up);
}

void ImpostorPass::parseProperties(const Properties& props)
{
    GBuffer::parseProperties(props);
    for (const auto& [key, value] : props)
    {
        if (key == kViewpointIndex)
            mViewpointIndex = value;
    }
}

void ImpostorPass::onSceneUpdates(RenderContext* pRenderContext, IScene::UpdateFlags sceneUpdates) {}

void ImpostorPass::recreatePrograms()
{
    mDepthPass.pProgram = nullptr;
    mDepthPass.pVars = nullptr;
    mGBufferPass.pProgram = nullptr;
    mGBufferPass.pVars = nullptr;
    mpComputePass = nullptr;
}

void ImpostorPass::calculateViewPoint(float3 min, float3 max, uint32_t index)
{
    float3 center = (min + max) / 2.f;
    float3 half = center - min;
    float3 direction = mViewDirections[index];
    float3 mid = center + math::abs(math::dot(half, direction)) * direction;
    mViewpoint.position = mid + 1.f * direction;
    mViewpoint.target = mid;
    if (mViewDirections[index].y != 0)
        mViewpoint.up = float3(0, 0, 1);
    else
        mViewpoint.up = float3(0, 1, 0);
}
