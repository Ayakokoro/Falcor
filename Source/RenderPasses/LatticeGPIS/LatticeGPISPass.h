#pragma once
#include "Falcor.h"
#include "Core/Pass/ComputePass.h"
#include "RenderGraph/RenderPass.h"
#include "MeshClipmap.h"
#include "MeshDistanceField.h"
#include "SdfField.h"

using namespace Falcor;

class LatticeGPISPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(LatticeGPISPass, "LatticeGPISPass", "Render a Gaussian process implicit surface over a signed distance field.");

    static ref<LatticeGPISPass> create(ref<Device> device, const Properties& props) { return make_ref<LatticeGPISPass>(device, props); }
    LatticeGPISPass(ref<Device> device, const Properties& props);
    void setProperties(const Properties& props) override;
    Properties getProperties() const override;
    RenderPassReflection reflect(const CompileData& compileData) override;
    void compile(RenderContext* context, const CompileData& compileData) override;
    void execute(RenderContext* context, const RenderData& renderData) override;
    void setScene(RenderContext* context, const ref<Scene>& scene) override;
    bool onMouseEvent(const MouseEvent& event) override;
    bool onKeyEvent(const KeyboardEvent& event) override;
    void renderUI(Gui::Widgets& widget) override;

    enum class KernelMode : uint32_t
    {
        Stationary,
        Linear
    };
    FALCOR_ENUM_INFO(KernelMode, {{KernelMode::Stationary, "stationary"}, {KernelMode::Linear, "linear"}});

private:
    struct Options
    {
        std::filesystem::path sdfFile;
        std::filesystem::path meshFile;
        std::filesystem::path baseColorTexture;
        uint32_t clipmapResolution = 33;
        uint32_t clipmapLevels = 8;
        std::string gridName = "sdf";
        KernelMode kernelMode = KernelMode::Stationary;
        float correlationLength = 0.2f;
        float amplitude = 0.02f;
        float correlationLengthEnd = 0.032f;
        float amplitudeEnd = 0.0016f;
        float2 heightRange = {0.f, 0.6f};
        float latticeSpacingInEll = 0.5f;
        float kernelRadiusInEll = 3.f;
        uint32_t seed = 0;
        bool hasVaryingKernel() const
        {
            return kernelMode == KernelMode::Linear && (correlationLength != correlationLengthEnd || amplitude != amplitudeEnd);
        }
        void validate(const AABB& bounds) const;
    };

    void applyOptions(Options options, const AABB& bounds);
    void updateStationaryWeights(const Options& options);
    AABB getFieldBounds() const;
    void frameField();
    Options mOptions;
    std::unique_ptr<SdfField> mpField;
    std::shared_ptr<const MeshDistanceField> mpMesh;
    MeshDistanceField::SurfaceUVBuffers mSurfaceUV;
    ref<Texture> mpBaseColor;
    ref<Sampler> mpMeshSampler;
    std::unique_ptr<MeshClipmap> mpClipmap;
    ref<ComputePass> mpTrace;
    ref<Buffer> mpStationaryWeights;
    float mStationarySpacingInEll = 0.f;
    int32_t mStationaryRadius = 0;
    float2 mStationaryInvVariance = float2(0.f);
    ref<Scene> mpScene;
    ref<Camera> mpCamera = Camera::create();
    FirstPersonCameraController mController{mpCamera};
    uint32_t mFrameIndex = 0;
    bool mReset = true;
    float3 mSpeedSamplePosition = float3(0.f);
    std::string mFileInput, mGridInput, mTextureInput, mError;
};

FALCOR_ENUM_REGISTER(LatticeGPISPass::KernelMode);
