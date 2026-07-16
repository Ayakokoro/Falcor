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
#include "ForwardMappingPass.h"

namespace
{
const std::string kComputePassProgramFile = "RenderPasses/ForwardMappingPass/ForwardMapping.cs.slang";
const std::string kInputInvVP = "impostorInvVP";
const std::string kInputPackedNDO = "packedNDO";
const std::string kInputPackedMCR = "packedMCR";
const std::string kImpostorCount = "impostorCount";
const std::string kOutputMappedNDO = "mappedNDO";
const std::string kOutputMappedMCR = "mappedMCR";
const std::string kOutputTexelSource = "texelSource";
const std::string kTexelLock = "texelLock";
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ForwardMappingPass>();
}

ForwardMappingPass::ForwardMappingPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    mImpostorCount = 1;
    mEnableSuperSampling = true;
    mEnableLock = true;
    mEnableAdaptiveContinuityCheck = true;
    mEnableBarrier = true;
    mLoDLevel = mCurrentLoDLevel = 0;
    mForceLoDLevel = false;
    mDistanceThreshold_Times1000 = 1;
    mMask = 63;

    for (const auto& [key, value] : props)
    {
        if (key == kImpostorCount)
            mImpostorCount = value;
    }
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_DEFAULT);

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point)
        .setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    mpPointSampler = mpDevice->createSampler(samplerDesc);
}

RenderPassReflection ForwardMappingPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    for (size_t i = 0; i < mImpostorCount; i++)
    {
        std::string si = std::to_string(i);
        reflector.addInput(kInputPackedNDO + si, "Packed NDO data from Impostor" + si)
            .format(ResourceFormat::RGBA32Float)
            .bindFlags(ResourceBindFlags::ShaderResource)
            .texture2D(RiLoDWidth, RiLoDHeight, 1, RiLoDMipCount);
        reflector.addInput(kInputPackedMCR + si, "Packed MCR data from Impostor" + si)
            .format(ResourceFormat::RGBA32Float)
            .bindFlags(ResourceBindFlags::ShaderResource)
            .texture2D(RiLoDWidth, RiLoDHeight, 1, RiLoDMipCount);
        reflector.addInput(kInputInvVP + si, "Inverse viewProjectMatrix of Impostor" + si)
            .format(ResourceFormat::Unknown)
            .bindFlags(ResourceBindFlags::Constant)
            .rawBuffer(sizeof(float4x4));
    }

    reflector.addOutput(kOutputMappedNDO, "Mapped NDO data")
        .format(ResourceFormat::RGBA32Float)
        .bindFlags(ResourceBindFlags::UnorderedAccess)
        .texture2D(RiLoDOutputWidth, RiLoDOutputHeight, 1, 1);
    reflector.addOutput(kOutputMappedMCR, "Mapped MCR data")
        .format(ResourceFormat::RGBA32Float)
        .bindFlags(ResourceBindFlags::UnorderedAccess)
        .texture2D(RiLoDOutputWidth, RiLoDOutputHeight, 1, 1);

    reflector.addOutput(kOutputTexelSource, "The index of impostor that each texel comes from")
        .format(ResourceFormat::RGBA32Float)
        .bindFlags(ResourceBindFlags::UnorderedAccess)
        .texture2D(RiLoDOutputWidth, RiLoDOutputHeight, 1, 1);

    reflector.addInternal(kTexelLock, "Texel lock")
        .format(ResourceFormat::R32Uint)
        .bindFlags(ResourceBindFlags::UnorderedAccess)
        .texture2D(RiLoDOutputWidth, RiLoDOutputHeight, 1, 1);

    return reflector;
}

void ForwardMappingPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene)
        return;

    if (!mpComputePass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kComputePassProgramFile).csEntry("main");
        // desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        // defines.add(getShaderDefines(renderData));

        mpComputePass = ComputePass::create(mpDevice, desc, defines, true);
    }
    mpCamera->setFocalLength(21.f);
    mpCamera->setFrameHeight(24.f);
    mpCamera->setFarPlane(100.f);
    mpCamera->setAspectRatio(RiLoDOutputWidth / (float)RiLoDOutputHeight);
    mpScene->update(pRenderContext, 0.f);

    // 重建G-Buffer
    {
        ref<Texture> mappedNDO = renderData.getTexture(kOutputMappedNDO);
        ref<Texture> mappedMCR = renderData.getTexture(kOutputMappedMCR);
        ref<Texture> pTexelLock = renderData.getTexture(kTexelLock);
        ref<Texture> pTexelSource = renderData.getTexture(kOutputTexelSource);

        pRenderContext->clearUAV(mappedNDO->getUAV().get(), float4());
        pRenderContext->clearUAV(mappedMCR->getUAV().get(), float4());
        pRenderContext->clearUAV(pTexelLock->getUAV().get(), uint4());
        pRenderContext->clearUAV(pTexelSource->getUAV().get(), float4());

        mpComputePass->addDefine("ENABLE_SUPERSAMPLING", mEnableSuperSampling ? "1" : "0");
        mpComputePass->addDefine("ENABLE_LOCK", mEnableLock ? "1" : "0");
        mpComputePass->addDefine("ENABLE_ADAPTIVE_CONTINUITY_CHECK", mEnableAdaptiveContinuityCheck ? "1" : "0");
        ShaderVar var = mpComputePass->getRootVar();
        float4x4 GBufferVP = mpCamera->getViewProjMatrixNoJitter();

        mCurrentLoDLevel = CalculateLoD();
        if (mForceLoDLevel)
            mCurrentLoDLevel = mLoDLevel;
        mCurrentLoDLevel = math::clamp(mCurrentLoDLevel, 0.f, RiLoDMipCount - 1.000001f);
        uint lowerLevel = (uint)math::floor(mCurrentLoDLevel);
        float t = mCurrentLoDLevel - lowerLevel;
        var["CB"]["GBufferVP"] = GBufferVP;
        var["gPointSampler"] = mpPointSampler;
        var["gTexelLock"] = pTexelLock;
        var["gTexelSource"] = pTexelSource;
        for (uint i = 0; i < mImpostorCount; i++)
        {
            if (((1 << i) & mMask) == 0)
                continue;
            ref<Buffer> buffer = renderData.getResource(kInputInvVP + std::to_string(i))->asBuffer();
            float4x4 invVP;
            buffer->getBlob(&invVP, 0, sizeof(float4x4));
            ref<Texture> packedNDO = renderData.getTexture(kInputPackedNDO + std::to_string(i));
            ref<Texture> packedMCR = renderData.getTexture(kInputPackedMCR + std::to_string(i));

            var["CB"]["invOriginVP"] = invVP;
            int width = packedNDO->getWidth();
            int height = packedNDO->getHeight();
            width = (uint)math::ceil(math::lerp((float)(width >> lowerLevel), (float)(width >> (lowerLevel + 1)), t));
            height = (uint)math::ceil(math::lerp((float)(height >> lowerLevel), (float)(height >> (lowerLevel + 1)), t));

            var["gPackedNDO"] = packedNDO;
            var["gPackedMCR"] = packedMCR;
            var["gMappedNDO"] = mappedNDO;
            var["gMappedMCR"] = mappedMCR;
            var["CB"]["width"] = width;
            var["CB"]["height"] = height;
            var["CB"]["outputWidth"] = RiLoDOutputWidth;
            var["CB"]["outputHeight"] = RiLoDOutputHeight;
            var["CB"]["lowerLevel"] = lowerLevel;
            var["CB"]["t"] = t;
            var["CB"]["index"] = i;
            var["CB"]["distanceThreshold"] = mDistanceThreshold_Times1000 / 1000.f;

            mpComputePass->execute(pRenderContext, uint3(width, height, 1));
            if (mEnableBarrier)
            {
                pRenderContext->uavBarrier(mappedNDO.get()); // 不同方向不可并行重建
                pRenderContext->uavBarrier(mappedMCR.get());
                pRenderContext->uavBarrier(pTexelLock.get());
                pRenderContext->uavBarrier(pTexelSource.get());
            }
        }
    }
}

void ForwardMappingPass::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("EnableSuperSampling", mEnableSuperSampling);
    widget.var("CurrentLoDLevel", mCurrentLoDLevel);
    widget.checkbox("ForceLoDLevel", mForceLoDLevel);
    widget.var("LoDLevel", mLoDLevel);
    widget.checkbox("EnableLock", mEnableLock);
    widget.checkbox("EnableBarrier", mEnableBarrier);
    widget.var("DistanceThreshold(x1000)", mDistanceThreshold_Times1000);
    widget.checkbox("EnableAdaptiveContinuityCheck", mEnableAdaptiveContinuityCheck);
    widget.var("Mask", mMask);
}

void ForwardMappingPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpCamera = pScene->getCamera();
}

float ForwardMappingPass::CalculateLoD()
{
    AABB aabb = mpScene->getSceneBounds();
    float3 min = aabb.minPoint;
    float3 max = aabb.maxPoint;
    float r = math::length(max - min) * 0.5f;
    float d = math::length(mpCamera->getPosition() - aabb.center());
    float k = math::length(float2(RiLoDWidth, RiLoDHeight)) / math::length(float2(RiLoDOutputWidth, RiLoDOutputHeight));
    k *= mpCamera->getFrameHeight() / mpCamera->getFocalLength();
    return math::log2(k * d / r);
}

float2 ForwardMappingPass::WorldToTexCoord(float4 world, float4x4 VP)
{
    float4 clip = math::mul(VP, world);
    float4 NDC = clip / clip.w;
    float2 texCoord = float2(0.5f * NDC.x + 0.5f, 0.5f - 0.5f * NDC.y);
    return texCoord;
}

float ForwardMappingPass::CalculateHomographMatrix(
    float4x4 originVP,
    float4x4 targetVP,
    float4 point0,
    float4 point1,
    float3x3& homographMatrix
)
{
    float2 originTexCoord0 = WorldToTexCoord(point0, originVP);
    float2 originTexCoord1 = WorldToTexCoord(point1, originVP);
    float2 targetTexCoord0 = WorldToTexCoord(point0, targetVP);
    float2 targetTexCoord1 = WorldToTexCoord(point1, targetVP);

    float2 originMid = 0.5f * (originTexCoord0 + originTexCoord1);
    float2 targetMid = 0.5f * (targetTexCoord0 + targetTexCoord1);
    float3 origin = float3(originTexCoord1 - originTexCoord0, 0);
    float3 target = float3(targetTexCoord1 - targetTexCoord0, 0);
    // float scaleRate = math::dot(target, origin) / math::dot(origin, origin);
    float scaleRate = math::length(target) / math::length(origin);

    homographMatrix = float3x3::identity();
    float3x3 transform = float3x3::identity();
    transform.setCol(2, float3(-originMid.x, -originMid.y, 1));
    homographMatrix = math::mul(transform, homographMatrix); // 平移到原点
    float3x3 scale = float3x3::identity();
    scale.setRow(0, float3(scaleRate, 0, 0));
    scale.setRow(1, float3(0, scaleRate, 0));
    homographMatrix = math::mul(scale, homographMatrix); // 缩放
    transform = float3x3::identity();
    transform.setCol(2, float3(targetMid.x, targetMid.y, 1));
    homographMatrix = math::mul(transform, homographMatrix);
    return scaleRate;
}
