#include "../LatticeGPIS/SdfField.slang"
#include "../LatticeGPIS/DensityClipmap.slang"
#include "../LatticeGPIS/GPISCameraRay.slang"
#include "../LatticeGPIS/DensityLighting.slang"

uniform float3 gCameraPosition, gCameraU, gCameraV, gCameraW;
uniform float3 gDensityTranslation;
uniform float gDensityWorldScale, gExtinctionScale;
uniform uint2 gFrameDim;
uniform uint gSeed;
#if WRITE_TRANSMITTANCE
RWTexture2D<float4> gTransmittance;
#endif
#if WRITE_RADIANCE
RWTexture2D<float4> gRadiance;
#endif

// Trilinear interpolation restricted to a line is cubic. Two Gauss-Legendre
// samples integrate each such segment exactly in real arithmetic.
float cellOpticalDepth(SdfCell cell, float3 fraction, float3 cellDir, float width)
{
    float3 segment = width * cellDir;
    float a = cell.value(saturate(fraction + 0.2113248654f * segment));
    float b = cell.value(saturate(fraction + 0.7886751346f * segment));
    return ((0.5f * a + 0.5f * b) * gExtinctionScale) * width;
}

float volumeOpticalDepth<let FindDistance : bool>(float3 origin, float3 direction, float target, out float distance)
{
    distance = -1.f;
    if (gExtinctionScale == 0.f) return 0.f;
    float3 localOrigin = (origin - gDensityTranslation) / gDensityWorldScale;
    float3 localDirection = direction / gDensityWorldScale;
    float2 span = float2(0.f, 1.e30f);
    if (!densityRaySpan(localOrigin, localDirection, span)) return 0.f;

    localOrigin += span.x * localDirection;
    float limit = span.y - span.x;
    float opticalDepth = 0.f;
    float t = 0.f;
    [loop] while (t < limit)
    {
        SdfCell cell;
        float3 fraction, cellDir;
        float end;
        int level;
        densityRayCell(localOrigin, localDirection, t, limit, cell, fraction, cellDir, end, level);
        if (!(end > t)) break;
        float width = end - t;
        float tau = cellOpticalDepth(cell, fraction, cellDir, width);
        if (FindDistance && tau > 0.f && opticalDepth + tau >= target)
        {
            float lo = 0.f, hi = width;
            [unroll] for (int i = 0; i < 16; ++i)
            {
                float mid = .5f * (lo + hi);
                if (cellOpticalDepth(cell, fraction, cellDir, mid) < target - opticalDepth) lo = mid;
                else hi = mid;
            }
            distance = span.x + (t + .5f * (lo + hi));
            return target;
        }
        opticalDepth += tau;
        t = end;
    }
    return opticalDepth;
}

[numthreads(8, 8, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
    uint2 pixel = threadID.xy;
    if (any(pixel >= gFrameDim)) return;
    uint seed;
    float3 direction = normalize(gpisCameraRay(pixel, gFrameDim, gSeed, gCameraU, gCameraV, gCameraW, seed));
    float unused;
    float value = exp(-volumeOpticalDepth<false>(gCameraPosition, direction, 0.f, unused));
#if WRITE_TRANSMITTANCE
    gTransmittance[pixel] = float4(value, value, value, 1.f);
#endif
#if WRITE_RADIANCE
    float3 radiance = value * kDensityBackground;
    float collisionMass = 1.f - value;
    if (collisionMass > 0.f)
    {
        // Sample the first real collision conditioned on one occurring in the
        // volume. Its probability cancels the camera extinction integrand.
        float u = (float(latticeHash(seed ^ 0x7f4a7c15u) >> 9) + .5f) / 8388608.f;
        float target = -log(1.f - u * collisionMass);
        float hit;
        volumeOpticalDepth<true>(gCameraPosition, direction, target, hit);
        if (hit >= 0.f)
        {
            float3 point = gCameraPosition + hit * direction;
            float lightT = exp(-volumeOpticalDepth<false>(point, densityDirectionToLight(), 0.f, unused));
            radiance += collisionMass * densitySingleScatter(lightT);
        }
    }
    gRadiance[pixel] = float4(radiance, 1.f);
#endif
}
