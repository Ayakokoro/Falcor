#pragma once
#include "Falcor.h"

/** A scalar SDF sampled at grid nodes. An empty path selects the Tungsten shaderball. */
class SdfField
{
public:
    SdfField(Falcor::ref<Falcor::Device> device, const std::filesystem::path& path, const std::string& gridName);
    void bind(const Falcor::ShaderVar& var) const;
    Falcor::AABB getBounds() const { return mBounds; }

private:
    Falcor::ref<Falcor::Texture> mpValues;
    Falcor::AABB mBounds;
    Falcor::float3 mWorldOrigin;
    Falcor::float4x4 mWorldToIndex;
    Falcor::int3 mIndexMax;
};
