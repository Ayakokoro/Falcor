#pragma once
#include "Falcor.h"
#include "Core/Pass/ComputePass.h"
#include "RenderGraph/RenderPass.h"
#include "DensityClipmap.h"
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
        bool surfaceEnabled = true;
        bool cloudEnabled = false;
        float3 cloudCenter = {0.55f, 0.15f, 0.25f};
        float3 cloudRadii = {1.2f, 0.65f, 0.85f};
        float cloudCorrelationLength = 0.14f;
        float cloudAmplitude = 0.095f;
        float cloudMeanOffset = 0.095f;
        float3 cloudColor = {0.96f, 0.97f, 0.99f};
        std::filesystem::path cloudDensityFile;
        std::string cloudDensityGrid = "density";
        bool densityUseClipmap = true;
        uint32_t densityClipmapResolution = 65;
        uint32_t densityClipmapLevels = 8;
        float cloudExtinctionScale = 1.f;
        float cloudDensityWorldScale = 1.f;
        float3 cloudDensityTranslation = float3(0.f);
        uint32_t seed = 0;
        bool usesDualFieldShader() const { return cloudEnabled || !surfaceEnabled; }
        bool usesDensityShader() const { return cloudEnabled && !cloudDensityFile.empty(); }
        bool hasVaryingKernel() const
        {
            return kernelMode == KernelMode::Linear && (correlationLength != correlationLengthEnd || amplitude != amplitudeEnd);
        }
        AABB cloudBounds(const DensityClipmap* density) const;
        void validate(const AABB& bounds, const DensityClipmap* density) const;
    };

    void applyOptions(Options options, const AABB& bounds, const DensityClipmap* density);
    void renderDensitySourceUI(Gui::Widgets& widget);
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
    std::unique_ptr<DensityClipmap> mpDensityClipmap;
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
    std::string mDensityFileInput, mDensityGridInput;
};

FALCOR_ENUM_REGISTER(LatticeGPISPass::KernelMode);
