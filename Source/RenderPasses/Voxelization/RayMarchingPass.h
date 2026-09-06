#pragma once
#include "VoxelizationBase.h"
#include <Rendering/Lights/EnvMapSampler.h>
#include <Core/Pass/FullScreenPass.h>
#include <fstream>
#include <filesystem>
#include <string>
#include <cstdint>
#include <vector>

using namespace Falcor;

// CPU-side description of one voxel binary in the composed scene. All assets
// are concatenated into shared GPU buffers; these offsets identify each
// asset's ranges and octree root.
struct VoxelAsset
{
    std::string assetId;
    std::filesystem::path voxelFile;
    GridData gridData{};
    uint32_t octreeRoot = 0;
    uint32_t voxelDataOffset = 0;
    uint32_t octreeMaxDepth = 0;
    std::vector<uint32_t> octreeNodeCounts;
    uint32_t availableLODLevels = 0;
    uint32_t voxelFormatVersion = 0;
    std::string voxelLodMode;
    std::string voxelProducer;
    bool hasVoxelMetadata = false;
    size_t voxelDataFileOffset = 0;
};

// CPU-side description of one placement of a voxel asset.
struct VoxelInstance
{
    uint32_t instanceId = 0;
    uint32_t assetIndex = 0;
    bool enabled = true;

    // Complete transform from the voxel asset's local coordinate system to
    // world space. This is the direct representation loaded from scene meta.
    float4x4 assetToWorld = float4x4::identity();

    float4x4 localToWorld = float4x4::identity();
    float4x4 worldToLocal = float4x4::identity();
    float4x4 normalTransform = float4x4::identity();

    float3 worldBoundsMin = float3(0.0f);
    float3 worldBoundsMax = float3(0.0f);
    int32_t screenLOD = 0;
};

// GPU representations are defined in VoxelizationShared.slang, which is also
// included by the ray-marching shader.
using VoxelInstanceGPU = VoxelInstanceData;
using VoxelAssetGPU = VoxelAssetData;
using VoxelInstanceBVHNodeGPU = InstanceBVHNode;

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
    bool loadSceneMeta(const std::filesystem::path& path);
    bool loadVoxelResources(RenderContext* pRenderContext);
    void resetVoxelResources();
    void resetInstancesToIdentity();
    void updateInstanceTransforms();
    void updateScreenSpaceLOD(const float4x4& viewProj, VoxelInstance& instance, bool updateDebugStats);
    void updateInstanceBuffer();
    void buildInstanceBVH();
    void updateInstanceBVHBuffer();
    void rebuildInstanceBVHIfDirty();

    ref<Scene> mpScene;
    ref<Device> mpDevice;
    ref<FullScreenPass> mpFullScreenPass;
    ref<FullScreenPass> mpDisplayNDFPass;
    ref<Sampler> mpPointSampler;
    ref<Buffer> mSelectedVoxel;
    ref<Buffer> mpSelectedVoxelStaging;
    int3 mSelectedCellInt = int3(-1);
    uint mSelectedGbOffset = 0xFFFFFFFF;
    uint mSelectedInstanceId = 0xFFFFFFFF;
    bool mSelectedHit = false;
    ref<Fbo> mpFbo;
    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;

    GridData& gridData;
    std::vector<std::filesystem::path> filePaths;
    std::vector<std::filesystem::path> sceneMetaPaths;
    uint selectedSceneMeta = 0;
    std::filesystem::path mVoxelFilePath;
    std::filesystem::path mSceneMetaPath;
    std::string mSceneMetaError;
    bool mSceneMetaLoaded = false;
    bool mFileListInitialized = false;

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

    std::vector<VoxelInstance> mInstances;
    std::vector<VoxelAsset> mVoxelAssets;
    uint32_t mInstanceEditIndex = 0;
    ref<Buffer> mInstanceBuffer;
    ref<Buffer> mAssetBuffer;
    std::vector<VoxelInstanceBVHNodeGPU> mInstanceBVH;
    ref<Buffer> mInstanceBVHBuffer;
    bool mInstanceBVHDirty = false;
    bool mUseInstanceBVH = true;

    int mForcedLOD = -1;  // -1=disabled, 0=finest leaf, 1..N=coarser levels
    int mMaxLODLevel = -1; // -1=no cap, N=do not select a coarser level than N
    float mCoverageBlend = 0.0f; // 0=raw coverage, 1=fill all empty cavity
    float3 mClearColor;

    // These are debug/UI values for the currently edited instance. LOD 0 is
    // the leaf level and larger values are coarser octree levels.
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

    bool mOptionsChanged;
    uint mFrameIndex;
    uint2 mOutputResolution;
};
