#pragma once
#include "VoxelizationBase.h"
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
    void updateInstanceTransform();
    void updateScreenSpaceLOD(const float4x4& viewProj, const float4x4& localToWorld);

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

    // The current pass contains exactly one implicit instance. Keeping its
    // transforms explicit makes the local-grid/world-space boundary clear and
    // lets the screen-space LOD code use the same path as future instances.
    float3 mInstanceTranslation = float3(0.0f);
    float3 mInstanceRotationDegrees = float3(0.0f);
    float4x4 mInstanceTransform = float4x4::identity();
    float4x4 mInverseInstanceTransform = float4x4::identity();
    float4x4 mNormalTransform = float4x4::identity();

    int mForcedLOD = -1;  // -1=disabled, 0=finest leaf, 1..N=coarser levels
    int mMaxLODLevel = -1; // -1=no cap, N=do not select a coarser level than N
    float mCoverageBlend = 0.0f; // 0=raw coverage, 1=fill all empty cavity
    float3 mClearColor;

    // One screen-space LOD is selected for the whole grid every frame. LOD 0
    // is the leaf level and larger values are coarser octree levels.
    int mScreenLOD = 0;
    bool mGridProjectionValid = false;
    float mGridProjectedWidthPixels = 0.0f;
    float mGridProjectedHeightPixels = 0.0f;
    float mGridProjectedAreaPixels = 0.0f;
    float mLeafProjectedSizePixels = 0.0f;

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

    bool mOptionsChanged;
    uint mFrameIndex;
    uint2 mOutputResolution;
};
