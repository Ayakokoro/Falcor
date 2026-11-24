#include "SdfField.h"
#include "Core/Pass/ComputePass.h"
#include <slang-gfx.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4146 4244 4267 4275 4456 4996)
#endif
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

using namespace Falcor;

namespace
{
float3 toFloat3(const openvdb::Vec3d& v)
{
    return float3(float(v.x()), float(v.y()), float(v.z()));
}
}

SdfField::SdfField(ref<Device> device, const std::filesystem::path& path, const std::string& gridName)
{
    if (path.empty())
    {
        // Original process box; these nodes coincide with every other node in the 0.005 reference bake.
        constexpr uint32_t size = 301;
        constexpr float spacing = 0.01f;
        mWorldOrigin = float3(-1.5f, 1.001f - 1.5f, -1.5f);
        mBounds = AABB(mWorldOrigin, mWorldOrigin + float3((size - 1) * spacing));
        mIndexMax = int3(size - 1);
        for (int i = 0; i < 3; ++i) mWorldToIndex[i][i] = 1.f / spacing;
        mpValues = device->createTexture3D(size, size, size, ResourceFormat::R32Float, 1, nullptr,
                                           ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        auto bake = ComputePass::create(device, "RenderPasses/LatticeGPIS/BakeShaderball.cs.slang");
        auto vars = bake->getRootVar();
        vars["gValues"] = mpValues;
        vars["gOrigin"] = mWorldOrigin;
        vars["gSpacing"] = spacing;
        vars["gSize"] = size;
        bake->execute(device->getRenderContext(), uint3(size));
        return;
    }

    openvdb::initialize();
    openvdb::io::File file(path.string());
    file.open();
    FALCOR_CHECK(file.hasGrid(gridName), "LatticeGPIS: no grid '{}' in '{}'.", gridName, path);
    auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(gridName));
    file.close();
    FALCOR_CHECK(grid && !grid->empty(), "LatticeGPIS: '{}' in '{}' must be a nonempty float grid.", gridName, path);
    const auto& transform = grid->transform();
    FALCOR_CHECK(transform.isLinear(), "LatticeGPIS: grid '{}' must have an affine index-to-world transform.", gridName);

    const auto box = grid->evalActiveVoxelBoundingBox();
    const auto dim = box.dim();
    const int maxSize = int(device->getGfxDevice()->getDeviceInfo().limits.maxTextureDimension3D);
    for (int axis = 0; axis < 3; ++axis)
    {
        FALCOR_CHECK(dim[axis] >= 2 && dim[axis] <= maxSize,
                     "LatticeGPIS: the dense extent of grid '{}' is {} x {} x {}, supported axis range is [2, {}].",
                     gridName, dim.x(), dim.y(), dim.z(), maxSize);
        openvdb::Vec3d direction(0.0);
        direction[axis] = 1.0;
        const auto column = transform.baseMap()->applyInverseJacobian(direction);
        for (int row = 0; row < 3; ++row) mWorldToIndex[row][axis] = float(column[row]);
    }
    mIndexMax = int3(dim.x() - 1, dim.y() - 1, dim.z() - 1);
    mWorldOrigin = toFloat3(transform.indexToWorld(box.min().asVec3d()));
    for (int i = 0; i < 8; ++i)
    {
        const openvdb::Vec3d corner((i & 1) ? box.max().x() : box.min().x(),
                                   (i & 2) ? box.max().y() : box.min().y(),
                                   (i & 4) ? box.max().z() : box.min().z());
        mBounds.include(toFloat3(transform.indexToWorld(corner)));
    }

    // Copy inactive tiles and signed interior/background values as well as active band voxels.
    openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> dense(box);
    openvdb::tools::copyToDense(*grid, dense);
    mpValues = device->createTexture3D(dim.x(), dim.y(), dim.z(), ResourceFormat::R32Float, 1, dense.data());
}

void SdfField::bind(const ShaderVar& var) const
{
    var["values"] = mpValues;
    var["boundsMin"] = mBounds.minPoint;
    var["boundsMax"] = mBounds.maxPoint;
    var["worldOrigin"] = mWorldOrigin;
    var["worldToIndex"] = mWorldToIndex;
    var["indexMax"] = mIndexMax;
}
