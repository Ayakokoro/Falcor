#pragma once
#include "VoxelizationBase.h"
#include <Rendering/Lights/EnvMapSampler.h>
#include <Core/Pass/FullScreenPass.h>
#include <array>
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
    float mSelectedLegacyVisibility = -1.0f;
    float4 mSelectedVisibilityQuery = float4(-1.0f); // Q, No, supported, material lobe count.
    bool mSelectedHit = false;
    ref<Fbo> mpFbo;
    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;

    GridData& gridData;
    std::vector<std::filesystem::path> filePaths;

    uint mDrawMode;
    uint mMaxBounce;
    uint mMaxTransmission = 4096;
    uint selectedFile;
    float mShadowBias100;
    float mShadowNormalOffsetVoxels = 0.0f;
    float mMinPdf100;
    float mTransmittanceThreshold100;
    bool mCheckEllipsoid;
    InternalVisibilityMode mInternalVisibilityMode = InternalVisibilityMode::Legacy;
    float3 mVisibilityDebugIncident = float3(0, 0, 1);
    uint mVisibilityDebugLobe = 0;
    bool mCheckCoverage;
    bool mUseEmissiveLight;
    bool mDebug;
    bool mRenderBackGround;
    bool mComplete;
    int mForcedLOD = -1;  // -1=disabled, 0=finest leaf, 1..N=coarser levels
    int mMaxLODLevel = -1; // -1=no cap, N=never LOD-stop above this level (leaf=0)
    float mCoverageBlend = 0.0f; // 0=raw coverage, 1=fill all empty cavity
    float mCoverageTolerance = 0.0f; // 0=disabled; smooth near-full coverage toward 1.
    float3 mClearColor;

    bool mDisplayNDF;
    float2 mSelectedUV;
    uint2 mSelectedPixel;
    uint mSelectedResolution;

    // Shared node splits keep every structured buffer below 2^31 bytes.
    static constexpr uint32_t kMaxVoxelBufferCount = 5;
    std::array<ref<Buffer>, kMaxVoxelBufferCount> mGBuffers{};
    std::array<ref<Buffer>, kMaxVoxelBufferCount> mPBuffers{};
    std::array<ref<Buffer>, kMaxVoxelBufferCount> mVisibilityBuffers{};
    ref<Buffer> mEmptyVisibilityBuffer;
    std::string mVoxelLoadStatus = "No voxel data loaded.";
    std::string mVisibilityMaskStatus = "No visibility companion loaded.";
    std::array<uint32_t, kMaxVoxelBufferCount - 1> mGBufferSplits{};
    ref<Buffer> mOctreeBuffer;
    uint32_t mOctreeMaxDepth = 0;
    std::vector<uint32_t> mOctreeNodeCounts;

    bool mOptionsChanged;
    uint mFrameIndex;
    uint2 mOutputResolution;
};
