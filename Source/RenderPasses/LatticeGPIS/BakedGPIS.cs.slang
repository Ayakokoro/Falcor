#include "SdfField.slang"
#include "Intersection.slang"
#include "CloudGPIS.slang"
#include "DensityClipmap.slang"
#include "DensityLighting.slang"

uniform uint2 gFrameDim;
uniform uint gSeed;
uniform float3 gCameraPosition, gCameraU, gCameraV, gCameraW, gTranslation;
uniform float gWorldScale, gEll;
#if WRITE_RADIANCE
RWTexture2D<float4> gColor;
#endif
#if WRITE_TRANSMITTANCE
RWTexture2D<float4> gTransmission;
#endif

// All eight grid samples are baked mu/sigma. Normalizing F by positive sigma
// preserves its zeros; the lattice process therefore has unit variance here.
bool bakedFlight(float3 origin, float3 direction, uint seed, out float hitDistance)
{
    hitDistance = 0.f;
    float3 sourceOrigin = (origin - gTranslation) / gWorldScale;
    float3 sourceDirection = direction / gWorldScale;
    float2 span = float2(0.f, 1.e30f);
    if (!densityRaySpan(sourceOrigin, sourceDirection, span)) return false;
    sourceOrigin += span.x * sourceDirection;
    float limit = span.y - span.x;
    LatticeKernel kernel = cloudStationaryKernel(gEll, 1.f);
    LatticeFlight flight = makeLatticeFlight(0.f, direction, kernel, seed, false, 0.f);
    NoiseSegment noise = {};
    bool positiveOnLeft = true;
    float t = 0.f;
    [loop] while (t < limit)
    {
        SdfCell cell;
        float3 f, cellDir;
        float cellEnd;
        int level;
        densityRayCell(sourceOrigin, sourceDirection, t, limit, cell, f, cellDir, cellEnd, level);
        if (!(cellEnd > t)) break;
        if (all(cell.z0 == 32.f) && all(cell.z1 == 32.f))
        {
            t = cellEnd;
            positiveOnLeft = true;
            continue;
        }
        if (t == 0.f)
        {
            // Start each transport leg in the void phase, also when its origin
            // lies inside the plume. Adjust the value while preserving knot slope.
            float threshold = -cell.value(f);
            float prior = stationaryKnot(0, seed, 1.f, 0.f, kernel).x;
            if (prior <= threshold)
            {
                float z = densityTruncatedNormal(threshold, seed ^ 0xc8f4e26du);
                flight.correction.x = (z - prior) * kernel.stationaryInvVariance.x;
            }
        }
        float cellStart = t;
        [loop] while (t < cellEnd)
        {
            if (noise.end <= t)
            {
                int knot = int(floor(t / flight.spacing));
                if (float(knot + 1) * flight.spacing <= t) ++knot;
                else if (float(knot) * flight.spacing > t) --knot;
                float start = float(knot) * flight.spacing;
                if (start != noise.end) noise.inRamp = true; // force a new left jet after a vacuum gap
                noise = nextNoiseSegment(flight, start, knot, noise);
                if (noise.end <= t) return false;
            }
            float finish = min(cellEnd, noise.end);
            float width = finish - t, knotWidth = noise.end - noise.start;
            float3 a = saturate(f + (t - cellStart) * cellDir);
            float3 b = saturate(f + (finish - cellStart) * cellDir);
            float2 left = latticePolynomialJet(noise.polynomial, (t-noise.start) / knotWidth);
            float2 right = latticePolynomialJet(noise.polynomial, (finish-noise.start) / knotWidth);
            left = left * float2(1.f, width/knotWidth) + float2(cell.value(a), width*dot(cell.gradient(a), cellDir));
            right = right * float2(1.f, width/knotWidth) + float2(cell.value(b), width*dot(cell.gradient(b), cellDir));
            // The trilinear mean and Hermite noise are both cubics along this
            // part of the ray, so their sum can be tested without ray marching.
            float u;
            if (latticeFirstDowncrossing(latticeHermite(left, right), right.x, positiveOnLeft, u))
            {
                hitDistance = span.x + t + u * width;
                return true;
            }
            t = finish;
        }
    }
    return false;
}

[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint2 pixel = id.xy;
    if (any(pixel >= gFrameDim)) return;
    uint seed;
    float3 dir = normalize(gpisCameraRay(pixel, gFrameDim, gSeed, gCameraU, gCameraV, gCameraW, seed));
    float distance;
    bool hit = bakedFlight(gCameraPosition, dir, latticeHash(seed ^ 0x650ea347u), distance);
#if WRITE_TRANSMITTANCE
    float T = hit ? 0.f : 1.f;
    gTransmission[pixel] = float4(T, T, T, 1.f);
#endif
#if WRITE_RADIANCE
    float3 color = kDensityBackground;
    if (hit)
    {
        float shadowDistance;
        bool shadow = bakedFlight(gCameraPosition + distance*dir, densityDirectionToLight(),
                                  latticeHash(seed ^ 0x534d83adu), shadowDistance);
        color = densitySingleScatter(shadow ? 0.f : 1.f);
    }
    gColor[pixel] = float4(color, 1.f);
#endif
}
