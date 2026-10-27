#include "VoxelInspector.h"
#include <algorithm>
#include <cmath>

void VoxelInspector::reset()
{
    mpClip = nullptr;
    mpRender = nullptr;
    mSelection = {};
    mDirty = true;
    mDragging = false;
    mFrame = 0;
}

void VoxelInspector::select(const VoxelSelection& selection)
{
    mSelection = selection;
    const float3 v = selection.viewDirection;
    mYaw = std::atan2(v.x, v.z) * 180.f / 3.14159265f;
    mPitch = std::asin(std::clamp(v.y, -1.f, 1.f)) * 180.f / 3.14159265f;
    mPitch = std::clamp(mPitch, -85.f, 85.f);
    mDistance = 2.8f;
    mDragging = false;
    mLightDirection = normalize(v + float3(0, 0.5f, 0));
    mDirty = true;
    mFrame = 0;
}

void VoxelInspector::clip(RenderContext* context, const ref<Scene>& scene, float3 gridMin, float3 voxelSize)
{
    if (!mpClip)
    {
        ProgramDesc desc;
        desc.addShaderLibrary("RenderPasses/Voxelization/ClipInspector.cs.slang").csEntry("main");
        desc.setShaderModel(ShaderModel::SM6_5);
        mpClip = ComputePass::create(mpDevice, desc, scene->getSceneDefines(), true);
    }
    if (!mpTriangles)
    {
        mpTriangles = mpDevice->createStructuredBuffer(80, kCapacity,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mpStats = mpDevice->createStructuredBuffer(sizeof(uint32_t), 4,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mpStatsReadback = mpDevice->createBuffer(4 * sizeof(uint32_t), ResourceBindFlags::None, MemoryType::ReadBack);
    }
    context->clearUAV(mpStats->getUAV().get(), uint4(0));
    auto var = mpClip->getRootVar();
    scene->bindShaderData(var["gScene"]);
    var["clippedTriangles"] = mpTriangles;
    var["clipStats"] = mpStats;
    var["CB"]["nodeMinW"] = gridMin + float3(mSelection.nodeMinCell) * voxelSize;
    var["CB"]["nodeExtentW"] = voxelSize * float(mSelection.nodeSize);
    var["CB"]["capacity"] = kCapacity;
    mSkippedInstances = 0;
    for (uint32_t i = 0; i < scene->getGeometryInstanceCount(); ++i)
    {
        const auto& instance = scene->getGeometryInstance(i);
        if (instance.getType() != Scene::GeometryType::TriangleMesh)
        {
            ++mSkippedInstances;
            continue;
        }
        const uint32_t count = scene->getMesh(MeshID::fromSlang(instance.geometryID)).getTriangleCount();
        var["CB"]["instanceIndex"] = i;
        var["CB"]["triangleCount"] = count;
        // Respect the D3D dispatch X dimension for very large meshes.
        constexpr uint32_t batch = 128 * 65535;
        for (uint32_t offset = 0; offset < count; offset += std::min(batch, count - offset))
        {
            var["CB"]["triangleOffset"] = offset;
            mpClip->execute(context, uint3(std::min(batch, count - offset), 1, 1));
        }
    }
    // This path only runs in Debug, after a selection or geometry change.
    context->copyResource(mpStatsReadback.get(), mpStats.get());
    context->submit(true);
    const auto* stats = static_cast<const uint32_t*>(mpStatsReadback->map());
    mTriangleCount = stats[0];
    mOverflow = stats[1] != 0;
    mPolygonCount = stats[2];
    mpStatsReadback->unmap();
    mDirty = false;
    mFrame = 0;
}

void VoxelInspector::execute(RenderContext* context, const ref<Scene>& scene, const ref<Texture>& output,
    float3 gridMin, float3 voxelSize, const Resources& resources)
{
    if (!mSelection.valid) return;
    const auto updates = scene->getUpdates();
    if (is_set(updates, IScene::UpdateFlags::GeometryMoved) || is_set(updates, IScene::UpdateFlags::SceneGraphChanged) ||
        is_set(updates, IScene::UpdateFlags::GeometryChanged) || is_set(updates, IScene::UpdateFlags::MeshesChanged))
        mDirty = true;
    if (is_set(updates, IScene::UpdateFlags::SceneDefinesChanged) ||
        is_set(updates, IScene::UpdateFlags::TypeConformancesChanged) || is_set(updates, IScene::UpdateFlags::ShaderCodeChanged))
    {
        mpClip = nullptr;
        mpRender = nullptr;
        mDirty = true;
    }
    if (mDirty) clip(context, scene, gridMin, voxelSize);
    if (!mpRender || mVisible != resources.visible)
    {
        mVisible = resources.visible;
        ProgramDesc desc;
        desc.addShaderLibrary("RenderPasses/Voxelization/InspectVoxel.ps.slang").psEntry("main");
        desc.setShaderModel(ShaderModel::SM6_5);
        desc.addTypeConformances(scene->getTypeConformances());
        auto defines = scene->getSceneDefines();
        defines.add("USE_VISIBLE_MATERIAL", mVisible ? "1" : "0");
        defines.add("CHECK_COVERAGE", "1");
        defines.add("CHECK_VISIBILITY", "1");
        mpRender = FullScreenPass::create(mpDevice, desc, defines);
    }
    if (!mpFbo) mpFbo = Fbo::create(mpDevice);
    const float yaw = mYaw * 3.14159265f / 180.f, pitch = mPitch * 3.14159265f / 180.f;
    const float3 outward(std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw));
    const float3 forward = -outward;
    const float3 right = normalize(cross(forward, float3(0, 1, 0)));
    auto var = mpRender->getRootVar();
    scene->bindShaderData(var["gScene"]);
    var["inspectorMaterial"] = resources.material;
    var["inspectorPrimitive"] = resources.primitive;
    if (mVisible) var["inspectorMasks"] = resources.masks;
    var["clippedTriangles"] = mpTriangles;
    var["clipStats"] = mpStats;
    auto cb = var["CB"];
    cb["localIndex"] = resources.localIndex;
    cb["capacity"] = kCapacity;
    cb["viewMode"] = mViewMode;
    cb["frameIndex"] = mFrame++;
    cb["resolution"] = uint2(output->getWidth(), output->getHeight());
    cb["cameraPosition"] = float3(0.5f) + outward * mDistance;
    cb["cameraRight"] = right;
    cb["cameraUp"] = cross(right, forward);
    cb["cameraForward"] = forward;
    cb["lightDirection"] = dot(mLightDirection, mLightDirection) > 1e-8f ? normalize(mLightDirection) : outward;
    cb["lightIntensity"] = mLightIntensity;
    cb["checkCoverage"] = mCoverage;
    cb["checkVisibility"] = mVisibility;
    cb["coverageBlend"] = mCoverageBlend;
    cb["masksAvailable"] = resources.masksAvailable;
    cb["internalVisibilityMode"] = resources.visibilityMode;
    cb["textureLod"] = mTextureLod;
    mpFbo->attachColorTarget(output, 0);
    mpRender->execute(context, mpFbo);
}

bool VoxelInspector::renderUI(Gui::Widgets& widget)
{
    widget.text("Left: baked voxel | Right: GPU-clipped triangles");
    widget.text("Drag to orbit; mouse wheel to zoom. Directional light, isolated node only.");
    widget.text("Polygons: " + std::to_string(mPolygonCount) + " | Triangles: " + std::to_string(mTriangleCount));
    if (mOverflow) widget.text("Reference exceeds 4096 triangles (magenta). Select a finer node.");
    if (mSkippedInstances) widget.text("Non-triangle geometry is excluded from the reference.");
    widget.text("Opaque Standard materials; shared Frostbite/GGX. Unsupported materials are magenta.");
    Gui::DropdownList modes = {{0, "Shaded"}, {1, "Normals"}, {2, "Diffuse color"}, {3, "Wireframe / silhouette"}};
    bool changed = widget.dropdown("Inspector view", modes, mViewMode);
    changed |= widget.slider("Orbit yaw", mYaw, -180.f, 180.f);
    changed |= widget.slider("Orbit pitch", mPitch, -85.f, 85.f);
    changed |= widget.slider("Orbit distance", mDistance, 1.3f, 10.f);
    changed |= widget.var("Light direction", mLightDirection, -1.f, 1.f, 0.01f);
    changed |= widget.slider("Light intensity", mLightIntensity, 0.f, 20.f);
    changed |= widget.checkbox("Inspector coverage", mCoverage);
    changed |= widget.slider("Inspector coverage blend", mCoverageBlend, 0.f, 1.f);
    changed |= widget.checkbox("Inspector internal visibility / local shadows", mVisibility);
    changed |= widget.slider("Reference texture LOD", mTextureLod, 0.f, 12.f);
    if (widget.button("Reclip selected node")) { mDirty = true; changed = true; }
    if (changed) mFrame = 0;
    return changed;
}

bool VoxelInspector::onMouseEvent(const MouseEvent& event)
{
    if (event.type == MouseEvent::Type::ButtonDown && event.button == Input::MouseButton::Left)
    { mDragging = true; mLastMouse = event.pos; return true; }
    if (event.type == MouseEvent::Type::ButtonUp && event.button == Input::MouseButton::Left)
    { mDragging = false; return true; }
    if (event.type == MouseEvent::Type::Move && mDragging)
    {
        const float2 delta = event.pos - mLastMouse;
        mLastMouse = event.pos;
        mYaw = std::remainder(mYaw - delta.x * 360.f, 360.f);
        mPitch = std::clamp(mPitch + delta.y * 180.f, -85.f, 85.f);
        mFrame = 0;
        return true;
    }
    if (event.type == MouseEvent::Type::Wheel)
    { mDistance = std::clamp(mDistance * std::exp(-event.wheelDelta.y * 0.1f), 1.3f, 10.f); mFrame = 0; return true; }
    return false;
}
