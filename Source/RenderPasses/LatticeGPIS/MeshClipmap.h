#pragma once
#include "Falcor.h"
#include "Core/Pass/ComputePass.h"
#include <functional>

class MeshDistanceField;

/** Nested, camera-following node grids. Level 0 is finest; the last level covers the entire mesh. */
class MeshClipmap
{
public:
    MeshClipmap(Falcor::ref<Falcor::Device> device, std::shared_ptr<const MeshDistanceField> mesh,
                uint32_t resolution, uint32_t levelCount, float padding = 0.f);
    MeshClipmap(Falcor::ref<Falcor::Device> device, const std::filesystem::path& path,
                uint32_t resolution, uint32_t levelCount, float padding = 0.f);
    // Use the same toroidal cache for a scalar grid with a known, aligned root.
    // Nodes are relative to the root corner, measured in the requested level's units.
    using RowSampler = std::function<void(int level, Falcor::int3 node, int count, float* dst)>;
    MeshClipmap(Falcor::ref<Falcor::Device> device, uint32_t resolution, uint32_t levelCount,
                float coarseSpacing, RowSampler sampler);

    void update(Falcor::RenderContext* context, Falcor::float3 cameraPosition);
    void bind(const Falcor::ShaderVar& var) const;
    Falcor::AABB getBounds() const { return mBounds; }
    Falcor::AABB getMeshBounds() const;
    float getFinestSpacing() const { return mLevels.front().spacing; }
    uint64_t getUpdatedVoxelCount() const { return mUpdatedVoxels; }
    uint32_t getActiveLevelCount() const { return uint32_t(mLevels.size()) - mFirstLevel; }

private:
    struct Level
    {
        Falcor::int3 origin = Falcor::int3(0); ///< Absolute node indices relative to the root's lower corner, in this level's units.
        Falcor::int3 ringOffset = Falcor::int3(0); ///< Physical texel coordinates of the logical lower corner.
        float spacing = 0.f;
        bool valid = false;
    };

    void updateLevel(int levelIndex, Falcor::int3 origin);
    void initialize(Falcor::ref<Falcor::Device> device, uint32_t levelCount, float coarseSpacing);
    void sampleRegion(int levelIndex, Falcor::int3 lower, Falcor::int3 upper);
    void upload(Falcor::RenderContext* context);

    std::shared_ptr<const MeshDistanceField> mpMesh;
    RowSampler mSampleRow;
    Falcor::ref<Falcor::Texture> mpAtlas;
    Falcor::ref<Falcor::ComputePass> mpUploadPass;
    Falcor::ref<Falcor::Buffer> mpDistances, mpRows;
    std::vector<float> mDistances;
    std::vector<Falcor::uint4> mRows; ///< Destination linear texel, source offset, row length, padding.
    std::vector<Level> mLevels;
    Falcor::AABB mBounds;
    int mResolution;
    uint32_t mFirstLevel = 0;
    uint64_t mUpdatedVoxels = 0;
};
