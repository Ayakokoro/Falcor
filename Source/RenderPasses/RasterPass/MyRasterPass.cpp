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
#include "MyRasterPass.h"
#include <sstream>

// 此处的字符串必须与py脚本中的字符串一致
namespace
{
const std::string kDrawMode = "mode";
const std::string kOutput = "output";
const std::string kInputDepth = "depth";
const std::string kInputNormal = "normal";
const std::string kInputPosW = "posW";
const std::string kInputDiffuseOpacity = "diffuseOpacity";
const std::string kInputSpecRough = "specRough";
const std::string kInputEmissive = "emissive";
const std::string kInputViewW = "viewW";
const std::string kInputMaterial = "mtlData";
const std::string kOutputColor = "color";
const std::string kShaderFile = "RenderPasses/RasterPass/MyRasterPass.ps.slang";
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, MyRasterPass>();
}

ref<MyRasterPass> MyRasterPass::create(ref<Device> pDevice, const Properties& props)
{
    return make_ref<MyRasterPass>(pDevice, props);
}

MyRasterPass::MyRasterPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    mpDevice = pDevice;
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point)
        .setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    mpSampler = mpDevice->createSampler(samplerDesc);

    mpFullScreenPass = FullScreenPass::create(mpDevice, kShaderFile);

    mMode = (uint)MyRasterMode::BlinnPhong;
}

RenderPassReflection MyRasterPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputDepth, "Depth buffer").bindFlags(ResourceBindFlags::DepthStencil);
    reflector.addInput(kInputNormal, "World space normal").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInputViewW, "World Space view").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInputPosW, "World space position").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInputMaterial, "Material").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInputDiffuseOpacity, "Diffuse albedo and opacity").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInputEmissive, "Emissive").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInputSpecRough, "Specular reflectance and roughness").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInputMaterial, "Material").bindFlags(ResourceBindFlags::ShaderResource);

    reflector.addOutput(kOutputColor, "Output color").bindFlags(ResourceBindFlags::RenderTarget).format(ResourceFormat::RGBA32Float);
    return reflector;
}

void MyRasterPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pOutput = renderData.getTexture(kOutputColor);
    pRenderContext->clearRtv(pOutput->getRTV().get(), float4(0, 0, 0, 1));
    if (!mpScene)
        return;

    const auto& pDepth = renderData.getTexture(kInputDepth);
    const auto& pNormal = renderData.getTexture(kInputNormal);
    const auto& pPosW = renderData.getTexture(kInputPosW);
    const auto& pViewW = renderData.getTexture(kInputViewW);
    const auto& pDiffuseOpacity = renderData.getTexture(kInputDiffuseOpacity);
    const auto& pSpecRough = renderData.getTexture(kInputSpecRough);
    const auto& pEmissive = renderData.getTexture(kInputEmissive);
    const auto& pMtlData = renderData.getTexture(kInputMaterial);

    if (!mpProgram)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kShaderFile).psEntry("main");
        desc.setShaderModel(ShaderModel::SM6_5);
        mpProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
    }

    // Bind resources to the full-screen pass
    auto var = mpFullScreenPass->getRootVar();
    var["gNormal"] = pNormal;
    var["gPosW"] = pPosW;
    var["gViewW"] = pViewW;
    var["gDiffuseOpacity"] = pDiffuseOpacity;
    var["gSpecRough"] = pSpecRough;
    var["gEmissive"] = pEmissive;
    var["gSampler"] = mpSampler;
    var["CB"]["gRasterMode"] = mMode;

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

    ref<Fbo> fbo = Fbo::create(mpDevice);
    fbo->attachColorTarget(pOutput, 0);
    fbo->attachDepthStencilTarget(pDepth);
    mpFullScreenPass->execute(pRenderContext, fbo);
}

void MyRasterPass::renderUI(Gui::Widgets& widget)
{
    widget.dropdown("Mode", reinterpret_cast<MyRasterMode&>(mMode));
}

void MyRasterPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}
