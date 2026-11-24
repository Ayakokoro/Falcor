#include "SdfField.slang"
#include "Intersection.slang"
#include "CloudGPIS.slang"
#include "BakedLatticeNoise.slang"
#define GPIS_BAKED_BRICKS 1
#include "DensityClipmap.slang"
#include "DensityLighting.slang"

uniform uint2 gFrameDim;
uniform uint gSeed;
uniform float3 gCameraPosition, gCameraU, gCameraV, gCameraW, gTranslation;
uniform float gWorldScale, gEll;
#if SHADOW_STAGE
Texture2D<float> gHitDistanceInput;
RWTexture2D<float4> gColor;
#elif WRITE_RADIANCE
RWTexture2D<float> gHitDistanceOutput;
#endif
#if WRITE_TRANSMITTANCE && !SHADOW_STAGE
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
    float3 indexOrigin = mul((float3x3)gDensity.worldToIndex, sourceOrigin - gDensity.worldOrigin);
    float3 cellDir = mul((float3x3)gDensity.worldToIndex, sourceDirection);
    float limit = span.y - span.x;
    float occupiedEnd = 0.f;
    LatticeKernel kernel = cloudStationaryKernel(gEll, 1.f);
    LatticeFlight flight = makeLatticeFlight(0.f, direction, kernel, seed, false, 0.f);
    NoiseSegment noise = {};
    BakedGaussianWindow pairs = {};
    pairs.firstPair = -5; // First valid pair is -3; force a full load on first use.
    bool positiveOnLeft = true;
    float noiseLowerBound = 0.f;
    float t = 0.f;
    [loop] while (t < limit)
    {
        SdfCell cell;
        float3 f;
        float cellEnd;
        bool occupied = densityBakedRayCell(indexOrigin, cellDir, t, limit, occupiedEnd, cell, f, cellEnd);
        if (!(cellEnd > t)) break;
        if (!occupied || (all(cell.z0 == 32.f) && all(cell.z1 == 32.f)))
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
        // A trilinear mean is bounded by its eight corner values. Round its
        // lower bound down before combining it with the cubic noise bound.
        float4 lower = min(cell.z0, cell.z1);
        float4 magnitude = max(abs(cell.z0), abs(cell.z1));
        float meanLowerBound = min(min(lower.x, lower.y), min(lower.z, lower.w)) -
            4e-6f * max(1.f, max(max(magnitude.x, magnitude.y), max(magnitude.z, magnitude.w)));
        [loop] while (t < cellEnd)
        {
            if (noise.end <= t)
            {
                int knot = int(floor(t / flight.spacing));
                if (float(knot + 1) * flight.spacing <= t) ++knot;
                else if (float(knot) * flight.spacing > t) --knot;
                float start = float(knot) * flight.spacing;
                noise = bakedNoiseSegment(flight, start, knot, noise, pairs);
                if (noise.end <= t) return false;
                float4 p = noise.polynomial;
                // A cubic stays inside the convex hull of its Bezier controls.
                float4 controls = float4(p.x, p.x + p.y / 3.f, p.x + (2.f * p.y + p.z) / 3.f, dot(p, float4(1.f)));
                noiseLowerBound = min(min(controls.x, controls.y), min(controls.z, controls.w)) -
                    4e-6f * max(1.f, dot(abs(p), float4(1.f)));
            }
            float finish = min(cellEnd, noise.end);
            if (meanLowerBound + noiseLowerBound > 0.f)
            {
                positiveOnLeft = true;
                t = finish;
                continue;
            }
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
#if SHADOW_STAGE
    float distance = gHitDistanceInput.Load(int3(pixel, 0));
    if (distance < 0.f)
    {
        gColor[pixel] = float4(kDensityBackground, 1.f);
        return;
    }
#endif
    uint seed;
    float3 dir = normalize(gpisCameraRay(pixel, gFrameDim, gSeed, gCameraU, gCameraV, gCameraW, seed));
#if SHADOW_STAGE
    float shadowDistance;
    bool shadow = bakedFlight(gCameraPosition + distance*dir, densityDirectionToLight(),
                              latticeHash(seed ^ 0x534d83adu), shadowDistance);
    gColor[pixel] = float4(densitySingleScatter(shadow ? 0.f : 1.f), 1.f);
#else
    float distance;
    bool hit = bakedFlight(gCameraPosition, dir, latticeHash(seed ^ 0x650ea347u), distance);
#if WRITE_TRANSMITTANCE
    float T = hit ? 0.f : 1.f;
    gTransmission[pixel] = float4(T, T, T, 1.f);
#endif
#if WRITE_RADIANCE
    gHitDistanceOutput[pixel] = hit ? distance : -1.f;
#endif
#endif
}
