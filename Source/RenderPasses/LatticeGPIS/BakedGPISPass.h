#pragma once
#include "Falcor.h"
#include "Core/Pass/ComputePass.h"
#include "RenderGraph/RenderPass.h"
#include "DensityClipmap.h"

class BakedGPISPass : public Falcor::RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(BakedGPISPass, "BakedGPISPass", "Trace a baked GPIS mean grid under single-scattering illumination.");
    static Falcor::ref<BakedGPISPass> create(Falcor::ref<Falcor::Device> device, const Falcor::Properties& props)
    {
        return Falcor::make_ref<BakedGPISPass>(device, props);
    }
    BakedGPISPass(Falcor::ref<Falcor::Device> device, const Falcor::Properties& props);
    void setProperties(const Falcor::Properties& props) override;
    Falcor::Properties getProperties() const override;
    Falcor::RenderPassReflection reflect(const CompileData&) override;
    void compile(Falcor::RenderContext*, const CompileData& data) override;
    void execute(Falcor::RenderContext* context, const Falcor::RenderData& data) override;
    void setScene(Falcor::RenderContext*, const Falcor::ref<Falcor::Scene>& scene) override;
    void renderUI(Falcor::Gui::Widgets& widget) override;

private:
    std::filesystem::path mFile;
    Falcor::float3 mTranslation = Falcor::float3(0.f);
    uint32_t mSeed = 0;
    uint32_t mFrameIndex = 0;
    bool mReset = true;
    std::unique_ptr<DensityClipmap> mpMean;
    Falcor::ref<Falcor::Camera> mpFallbackCamera = Falcor::Camera::create();
    Falcor::ref<Falcor::Scene> mpScene;
    Falcor::ref<Falcor::ComputePass> mpTrace;
    Falcor::ref<Falcor::ComputePass> mpShadow;
    Falcor::ref<Falcor::Texture> mpHitDistance;
    std::string mError;
};
