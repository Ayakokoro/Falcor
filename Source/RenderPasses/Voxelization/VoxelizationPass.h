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

    void clipPolygon(RenderContext* pRenderContext, const RenderData& renderData);
    void analyzePolygon(RenderContext* pRenderContext, const RenderData& renderData);
    std::string getFileName();
    void* getVBufferCPU() const { return pVBuffer_CPU; }

protected:
    void write(std::string fileName, void* gBuffer, void* vBuffer, void* pBlockMap, void* pHyperBlockMap);

    ref<ComputePass> mAnalyzePolygonPass;
    ref<ComputePass> mLoadMeshPass;

    ref<Device> mpDevice;
    ref<Scene> mpScene;
    ref<SampleGenerator> mpSampleGenerator;
    ref<Sampler> mpSampler;

    ref<Buffer> gBuffer;
    ref<Buffer> polygonRangeBuffer;
    PolygonBufferGroup polygonGroup;
    ref<Buffer> blockMap;
    ref<Buffer> hyperBlockMap;

    PolygonGenerator polygonGenerator;
    void* pVBuffer_CPU;

    uint mSampleFrequency;
    uint mVoxelResolution;
    GridData& gridData;

    bool mSamplingComplete;
    bool mVoxelizationComplete;
    uint mCompleteTimes;
};
