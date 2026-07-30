#pragma once
#include "VoxelizationBase.h"
#include <Rendering/Lights/EnvMapSampler.h>
#include <Core/Pass/FullScreenPass.h>
#include <fstream>
#include <filesystem>

using namespace Falcor;

class RayMarchingPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(RayMarchingPass, "RayMarchingPass", "Insert pass description here.");

    static ref<RayMarchingPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<RayMarchingPass>(pDevice, props); }

    RayMarchingPass(ref<Device> pDevice, const Properties& props);

    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;

private:
    bool tryRead(std::ifstream& f, size_t& offset, size_t bytes, void* dst, size_t fileSize);

    ref<Scene> mpScene;
    ref<Device> mpDevice;
    ref<FullScreenPass> mpFullScreenPass;
    ref<FullScreenPass> mpDisplayNDFPass;
    ref<ComputePass> mPreparePass;
    ref<Sampler> mpPointSampler;
    ref<Buffer> mSelectedVoxel;
    ref<Buffer> mpSelectedVoxelStaging;
    int3 mSelectedCellInt = int3(-1);
    uint mSelectedGbOffset = 0xFFFFFFFF;
    bool mSelectedHit = false;
    ref<Fbo> mpFbo;
    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;

    GridData& gridData;
    std::vector<std::filesystem::path> filePaths;

    uint mDrawMode;
    uint mMaxBounce;
    uint selectedFile;
    float mShadowBias100;
    float mMinPdf100;
    float mTransmittanceThreshold100;
    bool mCheckEllipsoid;
    bool mCheckVisibility;
    bool mCheckCoverage;
    bool mUseEmissiveLight;
    bool mDebug;
    bool mRenderBackGround;
    bool mComplete;
    int mForcedLOD = -1;  // -1=disabled, 0=finest leaf, 1..N=coarser levels
    float3 mClearColor;

    bool mDisplayNDF;
    float2 mSelectedUV;
    uint2 mSelectedPixel;
    uint mSelectedResolution;

    // Triple buffer to avoid i32 overflow (>2³¹ bytes per buffer with 388-byte stride)
    ref<Buffer> mGBuffer0;
    ref<Buffer> mGBuffer1;
    ref<Buffer> mGBuffer2;
    ref<Buffer> mPBuffer0;
    ref<Buffer> mPBuffer1;
    ref<Buffer> mPBuffer2;
    uint32_t mGBufferSplit0 = 0;
    uint32_t mGBufferSplit1 = 0;
    ref<Buffer> mOctreeBuffer;
    uint32_t mOctreeMaxDepth = 0;
    std::vector<uint32_t> mOctreeNodeCounts;

    bool mOptionsChanged;
    uint mFrameIndex;
    uint2 mOutputResolution;
};
