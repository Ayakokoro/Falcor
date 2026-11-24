#pragma once
#include "Falcor.h"

class MeshClipmap;

/** Native or camera-following, filtered samples of a zero-background OpenVDB density grid.
    Public coordinates are in the source grid's world space. */
class DensityClipmap
{
public:
    DensityClipmap(Falcor::ref<Falcor::Device> device, const std::filesystem::path& path,
                   const std::string& gridName, uint32_t resolution, uint32_t levelCount, bool useClipmap = true);
    /// Load a self-contained GPIS bake at native resolution. Its samples are mu/sigma.
    DensityClipmap(Falcor::ref<Falcor::Device> device, const std::filesystem::path& bakedGPIS);
    static void bakeGPIS(const std::filesystem::path& source, const std::string& gridName, const std::filesystem::path& output,
                         float ell, float sigma, float extinctionScale, float worldScale, uint32_t samplesPerVoxel = 1);
    ~DensityClipmap();

    void update(Falcor::RenderContext* context, Falcor::float3 cameraPosition);
    void bind(const Falcor::ShaderVar& var) const;
    Falcor::AABB getBounds() const { return mBounds; }
    float getMaxDensity() const { return mMaxDensity; }
    float getBakedCorrelationLength() const { return mBakedCorrelationLength; }
    float getBakedAmplitude() const { return mBakedAmplitude; }
    float getBakedExtinctionScale() const { return mBakedExtinctionScale; }
    float getBakedWorldScale() const { return mBakedWorldScale; }
    float getFinestSpacing() const;
    uint32_t getActiveLevelCount() const;
    uint64_t getUpdatedVoxelCount() const;

private:
    DensityClipmap(Falcor::ref<Falcor::Device> device, const std::filesystem::path& path, const std::string& gridName,
                   uint32_t resolution, uint32_t levelCount, bool useClipmap, bool baked);
    std::unique_ptr<MeshClipmap> mpCache;
    Falcor::ref<Falcor::Texture> mpNativeValues;
    Falcor::ref<Falcor::Texture> mpBakedBricks;
    Falcor::AABB mBounds, mTraceBounds;
    Falcor::float4x4 mWorldToIndex;
    Falcor::float3 mWorldOrigin, mIndexMin, mIndexMax;
    float mMaxDensity = 0.f;
    float mNativeSpacing = 0.f;
    float mBakedCorrelationLength = 0.f, mBakedAmplitude = 0.f, mBakedExtinctionScale = 0.f, mBakedWorldScale = 0.f;
};
