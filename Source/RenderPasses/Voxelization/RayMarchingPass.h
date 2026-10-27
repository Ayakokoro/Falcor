#pragma once
#include "VoxelizationBase.h"
#include "VoxelInspector.h"
#include <Rendering/Lights/EnvMapSampler.h>
#include <Core/Pass/FullScreenPass.h>
#include <fstream>
#include <filesystem>
#include <string>

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
    ref<Sampler> mpPointSampler;
    ref<Buffer> mSelectedVoxel;
    ref<Buffer> mpSelectedVoxelStaging;
    int3 mSelectedCellInt = int3(-1);
    uint mSelectedGbOffset = 0xFFFFFFFF;
    bool mSelectedHit = false;
    VoxelSelection mSelection = {};
    std::unique_ptr<VoxelInspector> mpInspector;
    bool mInspectNode = false;
    ref<Fbo> mpFbo;
    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;

    GridData& gridData;
    std::vector<std::filesystem::path> filePaths;
    bool mFileListInitialized = false;
    std::string mFileListError;

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
    bool mLoadRequested = false; // One attempt per Read click (or explicit script autoLoad).
    int mForcedLOD = -1;  // -1=disabled, 0=finest leaf, 1..N=coarser levels
    int mMaxLODLevel = -1; // -1=no cap, N=never LOD-stop above this level (leaf=0)
    float mCoverageBlend = 0.0f; // 0=raw coverage, 1=fill all empty cavity
    float3 mClearColor;

    bool mDisplayNDF;
    float2 mSelectedUV;
    uint2 mSelectedPixel;
    uint mSelectedResolution;

    // Dynamically-sized split buffers keep each resource below the byte-offset limit.
    // The data index stored in the octree remains a global voxel index. The
    // buffers below contain consecutive ranges of that index space.
    std::vector<ref<Buffer>> mGBuffers;
    std::vector<ref<Buffer>> mPBuffers;
    std::vector<uint32_t> mGBufferSplits; // Exclusive global end index per buffer.
    uint32_t mBufferCount = 1;
    ref<Buffer> mOctreeBuffer;
    uint32_t mOctreeMaxDepth = 0;
    std::vector<uint32_t> mOctreeNodeCounts;
    uint32_t mAvailableLODLevels = 0;
    uint32_t mVoxelFormatVersion = 0;
    std::string mVoxelLodMode;
    std::string mVoxelProducer;
    bool mHasVoxelMetadata = false;
    bool mUseVisibleMaterial = false; // Opt-in; existing files/rendering stay on four lobes.
    bool mVisibleMaterialActive = false;
    bool mVisibilityMasksAvailable = false;
    uint32_t mVisibleLODLevels = 0;
    uint32_t mInternalVisibilityMode = 1; // Legacy=0, Bitmask=1, Off=2 (visible mode only).
    uint32_t mMaxTransmission = 128;
    float3 mMaskIncidentDirection = float3(0, 1, 0);
    uint32_t mMaskLobe = 0;
    std::vector<ref<Buffer>> mVisibilityMaskBuffers;
    std::string mExtensionStatus;

    bool mOptionsChanged;
    uint mFrameIndex;
    uint2 mOutputResolution;
};
