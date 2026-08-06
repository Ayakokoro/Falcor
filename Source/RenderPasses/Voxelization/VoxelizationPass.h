#pragma once
#include "VoxelizationBase.h"
#include "PolygonGenerator.h"
#include <Core/Pass/FullScreenPass.h>

using namespace Falcor;

class VoxelizationPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(VoxelizationPass, "VoxelizationPass", "Voxelization pass with CPU clipping + GPU BSDF analysis.");

    static ref<VoxelizationPass> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<VoxelizationPass>(pDevice, props);
    }

    VoxelizationPass(ref<Device> pDevice, const Properties& props);

    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;

    void runHierarchicalClip(RenderContext* pRenderContext);
    void uploadBuffers(RenderContext* pRenderContext);
    void analyzeAllNodes(RenderContext* pRenderContext);
    void readbackAndWrite(RenderContext* pRenderContext);
    void outputDebugInfo();
    std::string getFileName();

    // Find a node's gBuffer index by traversing the octree from root
    uint32_t findNodeByLODAndCell(int3 targetCellInt, uint32_t targetLOD);

    // Validate SH fitting: compare ground truth (rasterization) vs SH reconstruction
    // directions are in world space, normalized
    void validateProjection(RenderContext* pRenderContext, uint32_t lodLevel,
                            int3 cellInt, const std::vector<float3>& directions);

    // Spherical function visualization: compute exact map + display sphere
    void computeSphericalFuncMap(RenderContext* pRenderContext);
    void displaySphericalFunc(RenderContext* pRenderContext, const RenderData& renderData);

    static uint64_t morton3(uint32_t x, uint32_t y, uint32_t z);

protected:
    void write(std::string fileName, void* pGBuffer);

    ref<ComputePass> mAnalyzePolygonPass;
    ref<ComputePass> mLoadMeshPass;
    ref<ComputePass> mValidationPass;
    ref<ComputePass> mSphericalMapPass;
    ref<FullScreenPass> mDisplaySphericalFuncPass;

    ref<Device> mpDevice;
    ref<Scene> mpScene;
    ref<SampleGenerator> mpSampleGenerator;
    ref<Sampler> mpSampler;
    ref<Sampler> mpPointSampler;
    ref<Fbo> mpFbo;

    ref<Buffer> gBuffer;
    ref<Buffer> pBuffer;
    ref<Buffer> octreeBuffer;
    ref<Buffer> polygonRangeBuffer;

    ref<Texture> mSphericalFuncMap;   // precomputed exact-value texture

    PolygonBufferGroup polygonGroup;
    PolygonGenerator polygonGenerator;

    uint mSampleFrequency;
    uint mMaxVoxelResolution;
    GridData& gridData;

    bool mUseMultiThread = true;
    bool mEnableDebug = false;
    bool mVoxelizationDirty = false;

    // Validation state
    int3 mValidationCellInt = int3(0, 0, 0);
    uint32_t mValidationLOD = 0;
    bool mValidationRequested = false;

    bool mLerpNormal;

    // Spherical function visualization state
    bool mShowSphericalFunc = false;
    bool mSphericalFuncDirty = false;
    uint32_t mSphericalFuncType = 0;     // 0=primitiveProjArea, 1=polygonsProjArea, 2=totalProjArea
    uint32_t mVisualizationMode = 0;     // 0=SH Approx, 1=Exact, 2=Error
    uint32_t mMapResolution = 256;       // precomputed map resolution (width, height=res/2)
    uint32_t mDisplayResolution = 512;   // output render target size
    uint32_t mPrimSampleFreq = 64;       // sample frequency for primitive area exact computation
    uint32_t mTargetGBufferIndex = 0xFFFFFFFF;  // cached gBuffer index for selected voxel
    float mSphericalFuncValueMin = 0.0f;
    float mSphericalFuncValueMax = 1.0f;
};
