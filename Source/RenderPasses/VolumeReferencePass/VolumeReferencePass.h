#pragma once
#include "Falcor.h"
#include "Core/Pass/ComputePass.h"
#include "RenderGraph/RenderPass.h"
#include "../LatticeGPIS/DensityClipmap.h"

using namespace Falcor;

class VolumeReferencePass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(VolumeReferencePass, "VolumeReferencePass", "Render an OpenVDB density grid with Beer-Lambert extinction and single scattering.");
    static ref<VolumeReferencePass> create(ref<Device> device, const Properties& props) { return make_ref<VolumeReferencePass>(device, props); }
    VolumeReferencePass(ref<Device> device, const Properties& props);

    Properties getProperties() const override;
    void setProperties(const Properties& props) override;
    RenderPassReflection reflect(const CompileData& compileData) override;
    void compile(RenderContext* context, const CompileData& compileData) override;
    void setScene(RenderContext* context, const ref<Scene>& scene) override;
    void execute(RenderContext* context, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;

private:
    struct Options
    {
        std::filesystem::path file;
        std::string grid = "density";
        float extinction = 1.f;
        float worldScale = 1.f;
        float3 translation = float3(0.f);
        bool useClipmap = false;
        uint32_t resolution = 65;
        uint32_t levels = 8;
        uint32_t seed = 0;
    };

    void frameVolume();
    Options mOptions;
    std::unique_ptr<DensityClipmap> mpDensity;
    ref<ComputePass> mpIntegrate;
    ref<Scene> mpScene;
    ref<Camera> mpFallbackCamera = Camera::create();
    uint32_t mFrameIndex = 0;
    bool mReset = true;
    std::string mError;
};
