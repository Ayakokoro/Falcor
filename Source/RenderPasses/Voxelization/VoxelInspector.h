#pragma once
#include "Falcor.h"
#include "VoxelSelection.slang"
#include "Core/Pass/ComputePass.h"
#include "Core/Pass/FullScreenPass.h"

using namespace Falcor;

// Owned by RayMarchingPass; no global state and no duplicate voxel uploads.
class VoxelInspector
{
public:
    struct Resources
    {
        ref<Buffer> material, primitive, masks;
        uint32_t localIndex = 0;
        bool visible = false;
        bool masksAvailable = false;
        uint32_t visibilityMode = 1;
    };

    explicit VoxelInspector(ref<Device> device) : mpDevice(device) {}
    void reset();
    void select(const VoxelSelection& selection);
    void execute(RenderContext* context, const ref<Scene>& scene, const ref<Texture>& output,
        float3 gridMin, float3 voxelSize, const Resources& resources);
    bool renderUI(Gui::Widgets& widget);
    bool onMouseEvent(const MouseEvent& event);

private:
    void clip(RenderContext* context, const ref<Scene>& scene, float3 gridMin, float3 voxelSize);
    static constexpr uint32_t kCapacity = 4096;
    ref<Device> mpDevice;
    ref<ComputePass> mpClip;
    ref<FullScreenPass> mpRender;
    ref<Buffer> mpTriangles, mpStats, mpStatsReadback;
    ref<Fbo> mpFbo;
    VoxelSelection mSelection = {};
    bool mDirty = true;
    bool mVisible = false;
    bool mDragging = false;
    bool mCoverage = true;
    bool mVisibility = true;
    float mCoverageBlend = 0.f;
    float mYaw = 0.f, mPitch = 15.f, mDistance = 2.8f;
    float mLightIntensity = 3.14159265f;
    float3 mLightDirection = float3(-1, 1, 1);
    float mTextureLod = 0.f;
    float2 mLastMouse = float2(0);
    uint32_t mViewMode = 0;
    uint32_t mFrame = 0;
    uint32_t mTriangleCount = 0, mPolygonCount = 0;
    bool mOverflow = false;
    uint32_t mSkippedInstances = 0;
};
