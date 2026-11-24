#include "MeshClipmap.h"
#include "MeshDistanceField.h"
#include "Utils/NumericRange.h"
#include <slang-gfx.h>
#include <algorithm>
#include <cmath>
#include <execution>

using namespace Falcor;

namespace
{
int wrap(int index, int resolution)
{
    const int remainder = index % resolution;
    return remainder < 0 ? remainder + resolution : remainder;
}

void validateSize(ref<Device> device, uint32_t resolution, uint32_t levelCount)
{
    FALCOR_CHECK(resolution >= 9 && resolution <= 129 && ((resolution - 1) & (resolution - 2)) == 0,
                 "LatticeGPIS: clipmap resolution must be 9, 17, 33, 65 or 129 nodes.");
    FALCOR_CHECK(levelCount >= 1 && levelCount <= 12, "LatticeGPIS: clipmap level count must be in [1,12].");
    FALCOR_CHECK(resolution * levelCount <= device->getGfxDevice()->getDeviceInfo().limits.maxTextureDimension3D,
                 "LatticeGPIS: the clipmap atlas exceeds this device's maximum 3D texture dimension.");
}
}

MeshClipmap::MeshClipmap(ref<Device> device, const std::filesystem::path& path, uint32_t resolution, uint32_t levelCount, float padding)
    : MeshClipmap(device, std::make_shared<MeshDistanceField>(path), resolution, levelCount, padding)
{}

MeshClipmap::MeshClipmap(ref<Device> device, std::shared_ptr<const MeshDistanceField> mesh, uint32_t resolution,
                         uint32_t levelCount, float padding)
    : mpMesh(std::move(mesh)), mResolution(int(resolution))
{
    FALCOR_CHECK(mpMesh, "LatticeGPIS: a mesh is required for the SDF clipmap.");
    validateSize(device, resolution, levelCount);
    FALCOR_CHECK(std::isfinite(padding) && padding >= 0.f, "LatticeGPIS: clipmap padding must be finite and nonnegative.");

    const AABB meshBounds = getMeshBounds();
    const float3 extent = meshBounds.extent();
    const double side = double(std::max({extent.x, extent.y, extent.z})) + 2.0 * padding;
    const int cells = mResolution - 1;
    // Each child stays one parent cell inside its parent. Three coarse cells of padding
    // leave all mesh faces reachable by the finest grid, even through all nested insets.
    const float coarseSpacing = float(side / double(cells - 6));
    FALCOR_CHECK(coarseSpacing > 0.f && std::isfinite(coarseSpacing), "LatticeGPIS: mesh clipmap extent is not representable.");
    for (int axis = 0; axis < 3; ++axis)
    {
        const double center = (double(meshBounds.maxPoint[axis]) + meshBounds.minPoint[axis]) * 0.5;
        mBounds.minPoint[axis] = float(center - double(cells) * coarseSpacing * 0.5);
        mBounds.maxPoint[axis] = mBounds.minPoint[axis] + cells * coarseSpacing;
    }
    initialize(device, levelCount, coarseSpacing);
}

MeshClipmap::MeshClipmap(ref<Device> device, uint32_t resolution, uint32_t levelCount, float coarseSpacing, RowSampler sampler)
    : mSampleRow(std::move(sampler)), mResolution(int(resolution))
{
    validateSize(device, resolution, levelCount);
    FALCOR_CHECK(mSampleRow && coarseSpacing > 0.f && std::isfinite(coarseSpacing), "Invalid scalar clipmap layout.");
    mBounds = AABB(float3(0.f), float3((resolution - 1) * coarseSpacing));
    initialize(device, levelCount, coarseSpacing);
}

void MeshClipmap::initialize(ref<Device> device, uint32_t levelCount, float coarseSpacing)
{
    mLevels.resize(levelCount);
    for (uint32_t i = 0; i < levelCount; ++i)
        mLevels[i].spacing = std::ldexp(coarseSpacing, int(i) - int(levelCount - 1));
    const float3 maxPerAxis = max(abs(mBounds.minPoint), abs(mBounds.maxPoint));
    const float maxCoord = std::max({maxPerAxis.x, maxPerAxis.y, maxPerAxis.z});
    FALCOR_CHECK(std::isfinite(maxCoord) && std::nextafter(maxCoord, INFINITY) - maxCoord <= getFinestSpacing(),
                 "LatticeGPIS: the finest spacing is below world-coordinate precision. Recenter the model or reduce its clipmap levels.");
    mpAtlas = device->createTexture3D(mResolution, mResolution, mResolution * levelCount, ResourceFormat::R32Float, 1, nullptr,
                                      ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    mpUploadPass = ComputePass::create(device, "RenderPasses/LatticeGPIS/UploadMeshClipmap.cs.slang");
}

AABB MeshClipmap::getMeshBounds() const
{
    return mpMesh->getBounds();
}

void MeshClipmap::update(RenderContext* context, float3 cameraPosition)
{
    FALCOR_CHECK(std::isfinite(cameraPosition.x) && std::isfinite(cameraPosition.y) && std::isfinite(cameraPosition.z),
                 "LatticeGPIS: clipmap camera position must be finite.");
    mUpdatedVoxels = 0;
    mDistances.clear();
    mRows.clear();
    const int coarsest = int(mLevels.size()) - 1;
    updateLevel(coarsest, int3(0));
    mFirstLevel = uint32_t(coarsest);
    for (int i = coarsest - 1; i >= 0; --i)
    {
        const Level& parent = mLevels[i + 1];
        const int childWidthInParent = (mResolution - 1) / 2;
        int3 origin;
        bool nearWindow = true;
        for (int axis = 0; axis < 3; ++axis)
        {
            const double camera = (double(cameraPosition[axis]) - mBounds.minPoint[axis]) / parent.spacing;
            const int lower = parent.origin[axis] + 1;
            const int upper = parent.origin[axis] + childWidthInParent - 1;
            const int start = int(std::clamp(std::round(camera - childWidthInParent * 0.5), double(lower), double(upper)));
            origin[axis] = 2 * start;
            nearWindow &= camera >= start - 1.0 && camera <= start + childWidthInParent + 1.0;
        }
        // The root remains visible from any distance. A far camera is not pinned to
        // the nearest face of the mesh, which would waste work on distant detail.
        if (!nearWindow) break;
        updateLevel(i, origin);
        mFirstLevel = uint32_t(i);
    }
    if (!mRows.empty())
    {
        upload(context);
        // Commit origins only after every sampled level has been uploaded. A failed update
        // leaves dirty levels invalid so the next attempt completely refreshes them.
        for (size_t i = mFirstLevel; i < mLevels.size(); ++i) mLevels[i].valid = true;
    }
}

void MeshClipmap::updateLevel(int index, int3 origin)
{
    Level& level = mLevels[index];
    const int3 delta = origin - level.origin;
    if (level.valid && all(delta == int3(0))) return;

    const bool refill = !level.valid || any(abs(delta) >= int3(mResolution));
    level.valid = false;
    level.origin = origin;
    if (refill)
    {
        level.ringOffset = int3(0);
        sampleRegion(index, int3(0), int3(mResolution));
    }
    else
    {
        for (int axis = 0; axis < 3; ++axis)
            level.ringOffset[axis] = wrap(level.ringOffset[axis] + delta[axis], mResolution);

        // Remove each axis' newly exposed slab from the remaining box so edges
        // and corners shared by two or three slabs are sampled exactly once.
        int3 lower(0), upper(mResolution);
        for (int axis = 0; axis < 3; ++axis)
        {
            if (delta[axis] > 0)
            {
                int3 slabStart = lower;
                slabStart[axis] = mResolution - delta[axis];
                sampleRegion(index, slabStart, upper);
                upper[axis] = slabStart[axis];
            }
            if (delta[axis] < 0)
            {
                int3 slabEnd = upper;
                slabEnd[axis] = -delta[axis];
                sampleRegion(index, lower, slabEnd);
                lower[axis] = slabEnd[axis];
            }
        }
    }
}

void MeshClipmap::sampleRegion(int index, int3 lower, int3 upper)
{
    const Level& level = mLevels[index];
    const int3 size = upper - lower;
    const size_t base = mDistances.size();
    const size_t rowBase = mRows.size();
    mDistances.resize(base + size_t(size.x) * size.y * size.z);
    mRows.resize(rowBase + size_t(size.y) * size.z);
    NumericRange<int> rows(0, size.y * size.z);
    std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int row)
    {
        const int3 local = lower + int3(0, row % size.y, row / size.y);
        const int3 node = level.origin + local;
        const float3 start(mBounds.minPoint.x, mBounds.minPoint.y + node.y * level.spacing, mBounds.minPoint.z + node.z * level.spacing);
        const uint32_t offset = uint32_t(base + size_t(row) * size.x);
        if (mSampleRow) mSampleRow(index, node, size.x, mDistances.data() + offset);
        else mpMesh->sampleRow(start, level.spacing, size.x, mDistances.data() + offset, node.x);
        int3 dst = (local + level.ringOffset) % int3(mResolution);
        dst.z += index * mResolution;
        mRows[rowBase + row] = uint4(dst.x + mResolution * (dst.y + mResolution * dst.z), offset, size.x, 0);
    });
}

void MeshClipmap::upload(RenderContext* context)
{
    auto stage = [&](ref<Buffer>& buffer, const auto& data)
    {
        const uint32_t stride = sizeof(data[0]);
        if (!buffer || buffer->getElementCount() < data.size())
            buffer = context->getDevice()->createStructuredBuffer(stride, uint32_t(data.size()), ResourceBindFlags::ShaderResource);
        context->updateBuffer(buffer.get(), data.data(), 0, data.size() * stride);
    };
    stage(mpDistances, mDistances);
    stage(mpRows, mRows);
    auto vars = mpUploadPass->getRootVar();
    vars["gAtlas"] = mpAtlas;
    vars["gDistances"] = mpDistances;
    vars["gRows"] = mpRows;
    vars["gResolution"] = uint32_t(mResolution);
    vars["gRowCount"] = uint32_t(mRows.size());
    const uint32_t groupsX = std::min(256u, uint32_t(mRows.size()));
    vars["gGroupsX"] = groupsX;
    mpUploadPass->execute(context, groupsX * 64, (uint32_t(mRows.size()) + groupsX - 1) / groupsX);
    mUpdatedVoxels = mDistances.size();
}

void MeshClipmap::bind(const ShaderVar& var) const
{
    var["clipValues"] = mpAtlas;
    var["boundsMin"] = mBounds.minPoint;
    var["boundsMax"] = mBounds.maxPoint;
    var["clipResolution"] = mResolution;
    var["clipCount"] = int(mLevels.size());
    var["clipFirstLevel"] = int(mFirstLevel);
    const auto levels = var["clipLevels"];
    const auto offsets = var["clipOffsets"];
    for (size_t i = mFirstLevel; i < mLevels.size(); ++i)
    {
        const Level& level = mLevels[i];
        levels[i] = float4(mBounds.minPoint + float3(level.origin) * level.spacing, level.spacing);
        offsets[i] = int4(level.ringOffset, 0);
    }
}
