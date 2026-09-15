#pragma once

#include "OctahedralMap.slang"
#include "Polygon.slang"
#include "AABB.slang"
#include "Ellipsoid.slang"
#include "../Voxel/VoxelGrid.slang"

BEGIN_NAMESPACE_FALCOR

#if !defined(HOST_CODE)

// Estimate the projected area of the fitted ellipsoid using the same disk
// ray test as the CPU voxelizer. The Hammersley sequence is shared with the
// octahedral prefilter so the primitive map has the same sampling convention
// on both sides.
inline float estimatePrimitiveProjectionArea(
    Ellipsoid ellipsoid,
    PolygonRange range,
    float3 direction,
    uint sampleFrequency)
{
    sampleFrequency = max(sampleFrequency, 1u);

    float3 cellMin = float3(range.cellInt);
    float3 cellCenter = cellMin + float3(0.5f);
    direction = normalize(direction);
    Basis2 basis = orthonormal_basis(direction);
    uint hitCount = 0;

    for (uint sample = 0; sample < sampleFrequency; ++sample)
    {
        SamplingRay ray = rayToVoxel(
            octahedralHammersley2D(sample, sampleFrequency),
            direction,
            basis,
            cellCenter);

        float3 to = ray.origin + 2.0f * ray.direction;
        float2 inOut = clip(cellMin, cellMin + float3(1.0f), ray.origin, to);
        if (inOut.x < 0.0f)
            continue;

        float3 delta = to - ray.origin;
        float3 fromLocal = ray.origin + inOut.x * delta - cellMin;
        float3 toLocal = ray.origin + inOut.y * delta - cellMin;
        inOut = ellipsoid.clip(fromLocal, toLocal);
        if (inOut.x >= 0.0f)
            ++hitCount;
    }

    return PROJECT_CIRCLE_AREA * float(hitCount) / float(sampleFrequency);
}

#endif

END_NAMESPACE_FALCOR
