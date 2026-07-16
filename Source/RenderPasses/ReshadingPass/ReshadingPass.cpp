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
#include "ReshadingPass.h"

namespace
{
const std::string kInputFilteredNDO = "filteredNDO";
const std::string kInputFilteredMCR = "filteredMCR";
const std::string kOutputColor = "color";
const std::string kOutputPosW = "posW";
const std::string kOutputMaterialID = "MaterialID";
const std::string kOutputMaterialUV = "MaterialUV";
const std::string kShaderFile = "RenderPasses/ReshadingPass/Reshading.ps.slang";
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReshadingPass>();
}

ref<ReshadingPass> ReshadingPass::create(ref<Device> pDevice, const Properties& props)
{
    return make_ref<ReshadingPass>(pDevice, props);
}

ReshadingPass::ReshadingPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    mpDevice = pDevice;
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    mUpdateScene = false;
    mDebug = false;

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point)
        .setAddressingMode(TextureAddressingMode::Border, TextureAddressingMode::Border, TextureAddressingMode::Border)
        .setBorderColor(float4());
    mpPointSampler = mpDevice->createSampler(samplerDesc);
}

RenderPassReflection ReshadingPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputFilteredNDO, "Input normal, depth, opacity")
        .bindFlags(ResourceBindFlags::ShaderResource)
        .texture2D(RiLoDOutputWidth, RiLoDOutputHeight, 1, 1);
    reflector.addInput(kInputFilteredMCR, "Input material id, texCoord, roughness")
        .bindFlags(ResourceBindFlags::ShaderResource)
        .texture2D(RiLoDOutputWidth, RiLoDOutputHeight, 1, 1);

    reflector.addOutput(kOutputColor, "Output color")
        .bindFlags(ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(RiLoDOutputWidth, RiLoDOutputHeight, 1, 1);
    reflector.addOutput(kOutputPosW, "Output world position")
        .bindFlags(ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(RiLoDOutputWidth, RiLoDOutputHeight, 1, 1);
    reflector.addOutput(kOutputMaterialID, "Output Material ID")
        .bindFlags(ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RG32Float)
        .texture2D(RiLoDOutputWidth, RiLoDOutputHeight, 1, 1);
    reflector.addOutput(kOutputMaterialUV, "Output Material UV")
        .bindFlags(ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RG32Float)
        .texture2D(RiLoDOutputWidth, RiLoDOutputHeight, 1, 1);
    return reflector;
}

void ReshadingPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene)
        return;

    if (mUpdateScene)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kShaderFile).psEntry("main");
        desc.setShaderModel(ShaderModel::SM6_5);
        desc.addTypeConformances(mpScene->getTypeConformances());
        mpFullScreenPass = FullScreenPass::create(mpDevice, desc, mpScene->getSceneDefines());
        mUpdateScene = false;
    }

    const auto& pFilteredNDO = renderData.getTexture(kInputFilteredNDO);
    const auto& pFilteredMCR = renderData.getTexture(kInputFilteredMCR);
    const auto& pOutputColor = renderData.getTexture(kOutputColor);
    const auto& pOutputPosW = renderData.getTexture(kOutputPosW);
    const auto& pOutputMaterialID = renderData.getTexture(kOutputMaterialID);
    const auto& pOutputMaterialUV = renderData.getTexture(kOutputMaterialUV);

    pRenderContext->clearRtv(pOutputColor->getRTV().get(), float4(0, 0, 0, 0));
    pRenderContext->clearRtv(pOutputPosW->getRTV().get(), float4(0, 0, 0, 0));

    mpFullScreenPass->addDefine("DEBUG", mDebug ? "1" : "0");

    // Bind resources to the full-screen pass
    auto var = mpFullScreenPass->getRootVar();
    var["gFilteredNDO"] = pFilteredNDO;
    var["gFilteredMCR"] = pFilteredMCR;
    var["gPointSampler"] = mpPointSampler;

    const auto& lights = mpScene->getLights();
    for (size_t i = 0; i < lights.size(); i++)
    {
        if (lights[i]->getType() == LightType::Directional)
        {
            var["DirectionalLightCB"]["lightPosW"] = lights[i]->getData().posW;
            var["DirectionalLightCB"]["lightDirW"] = lights[i]->getData().dirW;
            var["DirectionalLightCB"]["lightColor"] = lights[i]->getData().intensity;
            break;
        }
    }

    ref<Camera> camera = mpScene->getCamera();

    var["CB"]["cameraPosW"] = camera->getPosition();
    var["CB"]["invVP"] = math::inverse(camera->getViewProjMatrixNoJitter());

    ref<Fbo> fbo = Fbo::create(mpDevice);
    fbo->attachColorTarget(pOutputColor, 0);
    fbo->attachColorTarget(pOutputPosW, 1);
    fbo->attachColorTarget(pOutputMaterialID, 2);
    fbo->attachColorTarget(pOutputMaterialUV, 3);
    mpFullScreenPass->execute(pRenderContext, fbo);
}

void ReshadingPass::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Debug", mDebug);
}

void ReshadingPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mUpdateScene = true;
}
