#pragma once
#include "VoxelizationBase.h"
#include "PolygonGenerator.h"

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

    static uint64_t morton3(uint32_t x, uint32_t y, uint32_t z);

protected:
    void write(std::string fileName, void* pGBuffer);

    ref<ComputePass> mAnalyzePolygonPass;
    ref<ComputePass> mLoadMeshPass;

    ref<Device> mpDevice;
    ref<Scene> mpScene;
    ref<SampleGenerator> mpSampleGenerator;
    ref<Sampler> mpSampler;

    ref<Buffer> gBuffer;
    ref<Buffer> pBuffer;
    ref<Buffer> octreeBuffer;
    ref<Buffer> polygonRangeBuffer;

    PolygonBufferGroup polygonGroup;
    PolygonGenerator polygonGenerator;

    uint mSampleFrequency;
    uint mMaxVoxelResolution;
    GridData& gridData;

    bool mUseMultiThread = true;
    bool mEnableDebug = false;
    bool mVoxelizationDirty = false;
};
