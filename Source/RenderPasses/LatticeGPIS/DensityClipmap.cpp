#include "DensityClipmap.h"
#include "MeshClipmap.h"
#include <slang-gfx.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4146 4244 4267 4275 4456 4996)
#endif
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/MultiResGrid.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

using namespace Falcor;

namespace
{
constexpr float kVacuumMean = 32.f;
constexpr int kBakedBrickWidth = 8;
constexpr double kSqrtTwoPi = 2.5066282746310005;

openvdb::FloatGrid::Ptr readFloatGrid(const std::filesystem::path& path, const std::string& name)
{
    openvdb::initialize();
    openvdb::io::File file(path.string());
    file.open();
    FALCOR_CHECK(file.hasGrid(name), "No grid '{}' in '{}'.", name, path);
    auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(name));
    file.close();
    FALCOR_CHECK(grid && grid->transform().isLinear(), "Grid '{}' must be float with an affine index-to-world transform.", name);
    return grid;
}

void validateBake(float ell, float sigma, float extinctionScale, float worldScale)
{
    FALCOR_CHECK(ell > 0.f && std::isfinite(ell) && sigma > 0.f && std::isfinite(sigma) &&
                 extinctionScale >= 0.f && std::isfinite(extinctionScale) && worldScale > 0.f && std::isfinite(worldScale),
                 "GPIS baking requires positive finite correlation length, amplitude and world scale, and nonnegative finite extinction scale.");
}

float validateDensity(openvdb::FloatGrid& grid)
{
    FALCOR_CHECK(grid.getGridClass() != openvdb::GRID_LEVEL_SET && grid.background() == 0.f,
                 "Density '{}' must be a zero-background fog volume.", grid.getName());
    float maximum = 0.f;
    // VDB activation is topology metadata. Honor nonzero inactive tiles and voxels as well.
    for (auto i = grid.tree().beginValueAll(); i; ++i)
    {
        const float value = *i;
        FALCOR_CHECK(value >= 0.f && std::isfinite(value), "Density '{}' contains a negative or non-finite sample.", grid.getName());
        maximum = std::max(maximum, value);
        if ((value > 0.f) != i.isValueOn()) i.setActiveState(value > 0.f);
    }
    return maximum;
}

// Invert phi(u)/Phi(u) = target. The lower-tail expansion avoids 0/0 for
// optically thick voxels. For a positive product of three float32 parameters
// the upper bound +32 is strictly above the root, even with subnormals.
float normalizedMean(double target)
{
    FALCOR_CHECK(target > 0.0 && target <= std::numeric_limits<float>::max(), "Unrepresentable target extinction for the GPIS bake.");
    double lo = -std::max(1.0, target), hi = target > 1.0 ? lo + 1.0 : kVacuumMean;
    for (int i = 0; i < 36; ++i)
    {
        double mid = .5 * (lo + hi);
        double ratio;
        if (mid < -16.0)
        {
            const double inv = -1.0 / mid, s = inv * inv;
            ratio = -mid + inv * (1.0 + s * (-2.0 + s * (10.0 - 74.0 * s)));
        }
        else
            ratio = std::exp(-.5 * mid * mid) / (.5 * kSqrtTwoPi * std::erfc(-mid / std::sqrt(2.0)));
        if (ratio > target) lo = mid;
        else hi = mid;
    }
    return float(.5 * (lo + hi));
}

float3 toFloat3(const openvdb::Vec3d& p)
{
    return float3(float(p.x()), float(p.y()), float(p.z()));
}

AABB transformBox(const openvdb::math::Transform& transform, openvdb::Vec3d lo, openvdb::Vec3d hi)
{
    AABB box;
    for (int i = 0; i < 8; ++i)
        box.include(toFloat3(transform.indexToWorld(openvdb::Vec3d(
            (i & 1) ? hi.x() : lo.x(), (i & 2) ? hi.y() : lo.y(), (i & 4) ? hi.z() : lo.z()))));
    return box;
}

ref<Texture> makeBakedBricks(const ref<Device>& device, const float* nodes, uint3 dim)
{
    const uint3 brickDim = (dim - 1u + uint32_t(kBakedBrickWidth - 1)) / uint32_t(kBakedBrickWidth);
    std::vector<uint8_t> mask(size_t(brickDim.x) * brickDim.y * brickDim.z);
    size_t index = 0;
    for (uint32_t z = 0; z < brickDim.z; ++z)
        for (uint32_t y = 0; y < brickDim.y; ++y)
            for (uint32_t x = 0; x < brickDim.x; ++x, ++index)
            {
                const uint3 lo = uint3(x, y, z) * uint32_t(kBakedBrickWidth);
                // B trilinear cells touch B+1 nodes. A sample on a shared face,
                // edge or corner must mark each incident brick as occupied.
                const uint3 hi = min(lo + uint32_t(kBakedBrickWidth), dim - 1u);
                for (uint32_t iz = lo.z; iz <= hi.z && mask[index] == 0; ++iz)
                    for (uint32_t iy = lo.y; iy <= hi.y && mask[index] == 0; ++iy)
                    {
                        const float* row = nodes + (size_t(iz) * dim.y + iy) * dim.x;
                        mask[index] = std::any_of(row + lo.x, row + hi.x + 1, [](float u) { return u != kVacuumMean; });
                    }
            }
    return device->createTexture3D(brickDim.x, brickDim.y, brickDim.z, ResourceFormat::R8Uint, 1, mask.data(),
                                   ResourceBindFlags::ShaderResource);
}

ref<Texture> uploadNativeDensity(const ref<Device>& device, const openvdb::FloatGrid& grid,
                                 const openvdb::CoordBBox& valuesBox, ref<Texture>* bakedBricks)
{
    const uint32_t maxDim = device->getGfxDevice()->getDeviceInfo().limits.maxTextureDimension3D;
    uint3 dim;
    for (int k = 0; k < 3; ++k)
    {
        const int64_t count = int64_t(valuesBox.max()[k]) - valuesBox.min()[k] + 3;
        FALCOR_CHECK(count <= maxDim, "Native volume grid '{}' needs {} nodes on axis {}, device limit {}.",
                     grid.getName(), count, k, maxDim);
        FALCOR_CHECK(valuesBox.min()[k] > INT32_MIN && valuesBox.max()[k] < INT32_MAX, "Density halo exceeds VDB coordinates.");
        dim[k] = uint32_t(count);
    }
    const uint64_t bytes = uint64_t(dim.x) * dim.y * dim.z * sizeof(float);
    // Allocate on the device before staging CPU samples. Submit the complete
    // subresource together; the gfx Vulkan upload path requires a full 3D image.
    ref<Texture> texture;
    try
    {
        texture = device->createTexture3D(dim.x, dim.y, dim.z, ResourceFormat::R32Float, 1, nullptr, ResourceBindFlags::ShaderResource);
        const openvdb::CoordBBox padded(valuesBox.min().offsetBy(-1), valuesBox.max().offsetBy(1));
        openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> dense(padded);
        openvdb::tools::copyToDense(grid, dense);
        device->getRenderContext()->updateTextureData(texture.get(), dense.data());
        if (bakedBricks) *bakedBricks = makeBakedBricks(device, dense.data(), dim);
    }
    catch (const std::exception& e)
    {
        FALCOR_THROW("Cannot load native volume '{}' ({} x {} x {}, {:.1f} MiB each for CPU staging and GPU values). "
                     "Use a lower bake resolution or a density clipmap for the density source. {}",
                     grid.getName(), dim.x, dim.y, dim.z, double(bytes) / (1024. * 1024.), e.what());
    }
    return texture;
}
}

DensityClipmap::DensityClipmap(ref<Device> device, const std::filesystem::path& path, const std::string& gridName,
                               uint32_t resolution, uint32_t levelCount, bool useClipmap)
    : DensityClipmap(std::move(device), path, gridName, resolution, levelCount, useClipmap, false)
{}

DensityClipmap::DensityClipmap(ref<Device> device, const std::filesystem::path& path)
    : DensityClipmap(std::move(device), path, "gpisMean", 0, 0, false, true)
{}

DensityClipmap::DensityClipmap(ref<Device> device, const std::filesystem::path& path, const std::string& gridName,
                               uint32_t resolution, uint32_t levelCount, bool useClipmap, bool baked)
{
    FALCOR_CHECK(!useClipmap || (resolution >= 9 && resolution <= 129 && ((resolution - 1) & (resolution - 2)) == 0 &&
                 levelCount >= 1 && levelCount <= 12), "Invalid density clipmap resolution or level count.");
    auto grid = readFloatGrid(path, gridName);
    const auto& transform = grid->transform();
    if (baked)
    {
        FALCOR_CHECK(grid->metaValue<int32_t>("gpis.version") == 1 && grid->background() == kVacuumMean,
                     "The volume '{}' has an unsupported GPIS bake format.", path);
        mBakedCorrelationLength = grid->metaValue<float>("gpis.correlationLength");
        mBakedAmplitude = grid->metaValue<float>("gpis.amplitude");
        mBakedExtinctionScale = grid->metaValue<float>("gpis.extinctionScale");
        mBakedWorldScale = grid->metaValue<float>("gpis.worldScale");
        mMaxDensity = grid->metaValue<float>("gpis.maxSourceDensity");
        validateBake(mBakedCorrelationLength, mBakedAmplitude, mBakedExtinctionScale, mBakedWorldScale);
        FALCOR_CHECK(mMaxDensity >= 0.f && std::isfinite(mMaxDensity), "Invalid maximum density in GPIS bake '{}'.", path);
        for (auto i = grid->tree().cbeginValueAll(); i; ++i)
            FALCOR_CHECK(i.isValueOn() ? std::isfinite(*i) && *i < kVacuumMean : *i == kVacuumMean,
                         "Invalid GPIS parameter or vacuum value in '{}'.", path);
    }
    else mMaxDensity = validateDensity(*grid);
    const auto box = grid->empty() ? openvdb::CoordBBox(openvdb::Coord(0), openvdb::Coord(0)) : grid->evalActiveVoxelBoundingBox();
    auto lo = box.min().asVec3d() - openvdb::Vec3d(1.0);
    auto hi = box.max().asVec3d() + openvdb::Vec3d(1.0);
    mBounds = transformBox(transform, lo, hi);
    const auto voxelSize = transform.voxelSize();
    mNativeSpacing = float(std::min({voxelSize.x(), voxelSize.y(), voxelSize.z()}));
    FALCOR_CHECK(std::isfinite(mNativeSpacing) && mNativeSpacing > 0.f, "Invalid density voxel size.");
    for (int axis = 0; axis < 3; ++axis)
    {
        openvdb::Vec3d basis(0.0);
        basis[axis] = 1.0;
        const auto column = transform.baseMap()->applyInverseJacobian(basis);
        for (int row = 0; row < 3; ++row) mWorldToIndex[row][axis] = float(column[row]);
    }

    if (!useClipmap)
    {
        mpNativeValues = uploadNativeDensity(device, *grid, box, baked ? &mpBakedBricks : nullptr);
        mWorldOrigin = toFloat3(transform.indexToWorld(lo));
        mIndexMin = float3(0.f);
        mIndexMax = toFloat3(hi - lo);
        mTraceBounds = mBounds;
        return;
    }

    const int cells = int(resolution) - 1;
    auto extent = hi - lo;
    int rootMip = 0;
    while (double(cells - 6) * std::ldexp(1.0, rootMip) < std::max({extent.x(), extent.y(), extent.z()}))
    {
        ++rootMip;
        FALCOR_CHECK(rootMip < 30, "Density grid extent exceeds the representable VDB index range.");
    }
    const int stride = 1 << rootMip;
    levelCount = std::min(levelCount, uint32_t(rootMip + 1));
    const int firstMip = rootMip + 1 - int(levelCount);
    // Align every cache level with the source integer-node lattice. A near camera
    // sees the original VDB interpolant whenever at least rootMip+1 levels fit.
    openvdb::Coord root;
    for (int a = 0; a < 3; ++a)
    {
        const double index = std::floor(((lo[a] + hi[a]) * 0.5 / stride) - cells * 0.5) * stride;
        FALCOR_CHECK(index >= std::numeric_limits<int>::min() && index + double(cells) * stride <= std::numeric_limits<int>::max(),
                     "Density clipmap root is outside the VDB index range.");
        root[a] = int(index);
    }
    mWorldOrigin = toFloat3(transform.indexToWorld(root.asVec3d()));
    // Full-weight restriction widens a mip's support by up to its node stride;
    // trilinear reconstruction can extend it by one additional stride.
    lo -= openvdb::Vec3d(2.0 * (stride - 1));
    hi += openvdb::Vec3d(2.0 * (stride - 1));
    mTraceBounds = transformBox(transform, lo, hi);
    mIndexMin = toFloat3(lo - root.asVec3d());
    mIndexMax = toFloat3(hi - root.asVec3d());

    std::vector<openvdb::FloatTree::ConstPtr> trees;
    if (rootMip == 0) trees.push_back(grid->treePtr());
    else
    {
        // Full-weight restriction filters every level before decimating. A small
        // isolated plume therefore contributes to distant samples even between nodes.
        openvdb::tools::MultiResGrid<openvdb::FloatTree> pyramid(rootMip + 1, grid);
        for (int mip = firstMip; mip <= rootMip; ++mip) trees.push_back(pyramid.constTreePtr(mip));
    }
    auto sampleRow = [trees = std::move(trees), root, firstMip](int level, int3 node, int count, float* dst)
    {
        const int step = 1 << (firstMip + level);
        const openvdb::Coord first(root.x() / step + node.x, root.y() / step + node.y, root.z() / step + node.z);
        // Accessors are local to each parallel row; the trees are immutable.
        openvdb::tree::ValueAccessor<const openvdb::FloatTree, false> accessor(*trees[level]);
        for (int x = 0; x < count; ++x) dst[x] = accessor.getValue(first.offsetBy(x, 0, 0));
    };
    mpCache = std::make_unique<MeshClipmap>(device, resolution, levelCount, float(stride), std::move(sampleRow));
}

DensityClipmap::~DensityClipmap() = default;

void DensityClipmap::bakeGPIS(const std::filesystem::path& source, const std::string& gridName, const std::filesystem::path& output,
                              float ell, float sigma, float extinctionScale, float worldScale, uint32_t samplesPerVoxel)
{
    validateBake(ell, sigma, extinctionScale, worldScale);
    FALCOR_CHECK(samplesPerVoxel == 1 || samplesPerVoxel == 2, "GPIS baking supports 1 or 2 samples per source voxel axis.");
    FALCOR_CHECK(!output.empty() && (!std::filesystem::exists(output) || !std::filesystem::equivalent(source, output)),
                 "The GPIS bake must use a separate output file.");
    auto grid = readFloatGrid(source, gridName);
    const float maxDensity = validateDensity(*grid);
    if (samplesPerVoxel > 1)
    {
        // Resample the source interpolant first; the inverse mean map is nonlinear.
        // Scale index space exactly, including on grids with rotated or translated transforms.
        auto refined = openvdb::FloatGrid::create(0.f);
        auto placement = grid->transform().copy();
        placement->preScale(1.0 / samplesPerVoxel);
        refined->setTransform(placement);
        if (!grid->empty())
        {
            const openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
            for (int axis = 0; axis < 3; ++axis)
            {
                // GridTransformer processes entire 8-node leaves and the interpolation halo.
                const int64_t lo = (int64_t(box.min()[axis]) - 9) * samplesPerVoxel;
                const int64_t hi = (int64_t(box.max()[axis]) + 9) * samplesPerVoxel;
                FALCOR_CHECK(lo > INT32_MIN && hi < INT32_MAX, "Refined density support exceeds the VDB index range.");
            }
            openvdb::Mat4R indexScale = openvdb::Mat4R::identity();
            indexScale.preScale(openvdb::Vec3d(double(samplesPerVoxel)));
            openvdb::tools::GridTransformer resample(indexScale);
            resample.transformGrid<openvdb::tools::BoxSampler>(*grid, *refined);
        }
        grid = std::move(refined);
    }
    const double factor = kSqrtTwoPi * double(ell) * double(extinctionScale);
    for (auto i = grid->tree().beginValueAll(); i; ++i)
    {
        const float density = *i;
        const bool occupied = density > 0.f && extinctionScale > 0.f;
        i.setValue(occupied ? normalizedMean(factor * density) : kVacuumMean);
        i.setActiveState(occupied);
    }
    grid->tree().root().setBackground(kVacuumMean, false);
    grid->tree().prune();
    grid->clearMetadata();
    grid->setName("gpisMean");
    grid->setSaveFloatAsHalf(false);
    grid->insertMeta("gpis.version", openvdb::Int32Metadata(1));
    grid->insertMeta("gpis.samplesPerVoxel", openvdb::Int32Metadata(int32_t(samplesPerVoxel)));
    grid->insertMeta("gpis.correlationLength", openvdb::FloatMetadata(ell));
    grid->insertMeta("gpis.amplitude", openvdb::FloatMetadata(sigma));
    grid->insertMeta("gpis.extinctionScale", openvdb::FloatMetadata(extinctionScale));
    grid->insertMeta("gpis.worldScale", openvdb::FloatMetadata(worldScale));
    grid->insertMeta("gpis.maxSourceDensity", openvdb::FloatMetadata(maxDensity));

    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    // Create a unique sibling directory atomically so concurrent bakes cannot
    // share a partial stream. Rename only a completed VDB on the same volume.
    std::filesystem::path staging;
    std::random_device random;
    do
    {
        staging = output;
        staging += fmt::format(".{:08x}.tmp", random());
    } while (!std::filesystem::create_directory(staging));
    const auto temporary = staging / "bake.vdb";
    try
    {
        openvdb::io::File file(temporary.string());
        file.write(openvdb::GridPtrVec{grid});
        file.close();
        std::filesystem::rename(temporary, output);
    }
    catch (...)
    {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        std::filesystem::remove(staging, ignored);
        throw;
    }
    std::error_code ignored;
    std::filesystem::remove(staging, ignored);
}

void DensityClipmap::update(RenderContext* context, float3 cameraPosition)
{
    if (mpCache) mpCache->update(context, transformVector(mWorldToIndex, cameraPosition - mWorldOrigin));
}

void DensityClipmap::bind(const ShaderVar& var) const
{
    if (mpCache) mpCache->bind(var);
    else
    {
        var["clipValues"] = mpNativeValues;
        var["clipCount"] = 0;
        var["clipFirstLevel"] = 0;
        var["clipLevels"][0] = float4(0.f, 0.f, 0.f, 1.f);
        if (mpBakedBricks && var.hasMember("bakedBricks"))
        {
            var["bakedBricks"] = mpBakedBricks;
            var["bakedBrickWidth"] = kBakedBrickWidth;
        }
    }
    var["boundsMin"] = mTraceBounds.minPoint;
    var["boundsMax"] = mTraceBounds.maxPoint;
    var["worldOrigin"] = mWorldOrigin;
    var["worldToIndex"] = mWorldToIndex;
    var["indexMin"] = mIndexMin;
    var["indexMax"] = mIndexMax;
    var["maxDensity"] = mMaxDensity;
}

float DensityClipmap::getFinestSpacing() const { return mNativeSpacing * (mpCache ? mpCache->getFinestSpacing() : 1.f); }
uint32_t DensityClipmap::getActiveLevelCount() const { return mpCache ? mpCache->getActiveLevelCount() : 0u; }
uint64_t DensityClipmap::getUpdatedVoxelCount() const { return mpCache ? mpCache->getUpdatedVoxelCount() : 0u; }
