#include "SdfField.slang"
#include "Intersection.slang"

#ifndef GPIS_TEXTURED_MESH
#define GPIS_TEXTURED_MESH 0
#endif
#ifndef GPIS_SECOND_FIELD
#define GPIS_SECOND_FIELD 0
#endif
#ifndef GPIS_DENSITY_CLIPMAP
#define GPIS_DENSITY_CLIPMAP 0
#endif
#if GPIS_SECOND_FIELD
#include "CloudGPIS.slang"
#if GPIS_DENSITY_CLIPMAP
#include "DensityGPIS.slang"
#endif
#endif
#if GPIS_TEXTURED_MESH || GPIS_SECOND_FIELD
import Utils.Math.MathHelpers;
#endif
#if GPIS_TEXTURED_MESH
#include "MeshSurfaceUV.slang"

Texture2D<float4> gMeshBaseColor;
SamplerState gMeshSampler;

float3 meshAlbedo(float3 hit, float3 ray, float3 normal, float pixelWidth)
{
    MeshSurfaceUV surface = meshSurfaceUV(hit);
    // Project the ray cone onto the hit tangent plane. Explicit UV derivatives
    // also avoid a mip discontinuity at a wrapping seam between adjacent pixels.
    float3 dx, dy;
    branchlessONB(ray, dx, dy);
    float cosTheta = min(dot(ray, normal), -1.e-4f);
    dx = pixelWidth * (dx - ray * (dot(dx, normal) / cosTheta));
    dy = pixelWidth * (dy - ray * (dot(dy, normal) / cosTheta));
    float2 duvdx = float2(dot(surface.gradU, dx), dot(surface.gradV, dx));
    float2 duvdy = float2(dot(surface.gradU, dy), dot(surface.gradV, dy));
    return gMeshBaseColor.SampleGrad(gMeshSampler, surface.uv, duvdx, duvdy).rgb;
}
#endif

#if GPIS_TEXTURED_MESH || GPIS_SECOND_FIELD
float3 sampleGPISDiffuse(float3 normal, uint key)
{
    uint2 bits = uint2(latticeHash(key), latticeHash(key ^ 0x27d4eb2fu));
    float2 u = (float2(bits >> 9) + 0.5f) / 8388608.f;
    float pdf;
    float3 local = sample_cosine_hemisphere_concentric(u, pdf);
    float3 tangent, bitangent;
    branchlessONB(normal, tangent, bitangent);
    return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}
#endif

RWTexture2D<float4> gOutput;

struct PassParameters
{
    uint2 gFrameDim;
    uint gSeed;
    uint gKernelMode;
    float3 gCameraPosition;
    float3 gCameraU;
    float3 gCameraV;
    float3 gCameraW;
    float gCorrelationLength;
    float gAmplitude;
    float gCorrelationLengthEnd;
    float gAmplitudeEnd;
    float2 gHeightRange;
    float gLatticeSpacingInEll;
    float gKernelRadiusInEll;
    int gStationaryRadius;
    float2 gStationaryInvVariance;
#if GPIS_SECOND_FIELD
    uint gSurfaceEnabled, gCloudEnabled;
    float3 gCloudCenter, gCloudRadii;
    float gCloudCorrelationLength, gCloudAmplitude, gCloudMeanOffset;
    float3 gCloudColor;
#if GPIS_DENSITY_CLIPMAP
    float3 gDensityTranslation;
    float gDensityWorldScale, gDensityExtinctionScale;
#endif
#endif
};
uniform PassParameters CB;

bool clipBox(float3 origin, float3 direction, float3 lower, float3 upper, inout float2 span)
{
    [unroll]
    for (int axis = 0; axis < 3; ++axis)
    {
        if (direction[axis] == 0)
        {
            if (origin[axis] < lower[axis] || origin[axis] > upper[axis]) return false;
            continue;
        }
        float2 t = (float2(lower[axis], upper[axis]) - origin[axis]) / direction[axis];
        span = float2(max(span.x, min(t.x, t.y)), min(span.y, max(t.x, t.y)));
    }
    return span.x < span.y;
}

float3 incidentLight(float3 origin, float3 direction, out float lightDistance)
{
#if GPIS_MESH_CLIPMAP
    // A distant light and sky preserve their angular size at every mesh scale.
    lightDistance = 1.e30f;
    float sky = saturate(0.5f + 0.5f * direction.y);
    float key = pow(saturate(dot(direction, normalize(float3(-0.55f, 0.8f, 0.3f)))), 16.f);
    return lerp(float3(0.09f, 0.08f, 0.07f), float3(0.55f, 0.68f, 0.86f), sky) + key * float3(3.f, 2.8f, 2.5f);
#else
    // The reference shaderball uses a one-sided 5 x 5 quad rotated 160 degrees about X.
    static const float invPi = 0.31830988618f;
    static const float3 lightCenter = float3(0.f, 5.f, 2.01f);
    static const float3 lightNormal = float3(0.f, -0.93969262f, 0.34202014f);
    static const float3 lightZAxis = float3(0.f, -0.34202014f, -0.93969262f);
    float facing = dot(direction, lightNormal);
    if (facing != 0.f)
    {
        float t = dot(lightCenter - origin, lightNormal) / facing;
        float3 offset = origin + t * direction - lightCenter;
        if (t > 0.f && abs(offset.x) <= 2.5f && abs(dot(offset, lightZAxis)) <= 2.5f)
        {
            lightDistance = t;
            return facing < 0.f ? float3(300.f, 300.f, 200.f) * (invPi / 25.f) : float3(0.f);
        }
    }
    lightDistance = 1.e30f;
    return float3(2.5f, 2.5f, 1.5f) * (0.25f * invPi);
#endif
}

[numthreads(8, 8, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
    uint2 pixel = threadID.xy;
    if (any(pixel >= CB.gFrameDim)) return;
    uint seed;
    float3 cameraRay = gpisCameraRay(pixel, CB.gFrameDim, CB.gSeed, CB.gCameraU, CB.gCameraV, CB.gCameraW, seed);
    float3 origin = CB.gCameraPosition;
    float3 direction = normalize(cameraRay);
#if GPIS_TEXTURED_MESH
    float pixelAngle = 2.f * max(length(CB.gCameraU) / CB.gFrameDim.x, length(CB.gCameraV) / CB.gFrameDim.y) / length(cameraRay);
    float pathLength = 0.f;
#endif

    LatticeKernel kernel;
    kernel.nonstationary = CB.gKernelMode;
    kernel.lengthScale = float2(CB.gCorrelationLength, CB.gCorrelationLengthEnd);
    kernel.amplitude = float2(CB.gAmplitude, CB.gAmplitudeEnd);
    kernel.heightRange = CB.gHeightRange;
    kernel.latticeSpacingInEll = CB.gLatticeSpacingInEll;
    kernel.kernelRadiusInEll = CB.gKernelRadiusInEll;
    kernel.stationaryRadius = CB.gStationaryRadius;
    kernel.stationaryInvVariance = CB.gStationaryInvVariance;

    float3 throughput = float3(1.f);
    float3 radiance = float3(0.f);
    float3 gradient = float3(0.f);
#if GPIS_SECOND_FIELD
    bool previousWasCloud = false;
#endif
    [loop]
    for (uint bounce = 0; bounce <= 20; ++bounce)
    {
        float lightDistance;
        float3 light = incidentLight(origin, direction, lightDistance);
        float2 span = float2(0.f, lightDistance);
        bool hit = false;
        float distance = 0.f;
        uint flightSeed = latticeHash(seed + (bounce + 1u) * 0x9e3779b9u);
        float3 birthGradient = gradient;
        bool traceSurface = true;
        bool conditionSurface = bounce != 0;
#if GPIS_SECOND_FIELD
        traceSurface = CB.gSurfaceEnabled != 0u;
        conditionSurface = conditionSurface && !previousWasCloud;
#endif
        if (traceSurface && clipBox(origin, direction, gField.boundsMin, gField.boundsMax, span))
        {
            hit = latticeIntersect(origin, direction, span.x, span.y, flightSeed, kernel, conditionSurface,
                                   birthGradient, distance, gradient);
        }
#if GPIS_SECOND_FIELD
        // The two GPIS flights have independent random coefficients. Only the
        // field that produced the previous hit inherits its value and gradient.
        bool cloudHit = false;
        if (CB.gCloudEnabled != 0u)
        {
            float cloudDistance;
            float3 cloudGradient;
            float limit = hit ? distance : lightDistance;
            uint cloudSeed = latticeHash(flightSeed ^ 0x650ea347u);
#if GPIS_DENSITY_CLIPMAP
            cloudHit = densityGPISIntersect(origin, direction, limit, CB.gDensityTranslation, CB.gDensityWorldScale,
                                            CB.gCloudCorrelationLength, CB.gCloudAmplitude, CB.gDensityExtinctionScale,
                                            cloudSeed, previousWasCloud, birthGradient, cloudDistance, cloudGradient);
#else
            cloudHit = cloudGPISIntersect(origin, direction, limit, CB.gCloudCenter, CB.gCloudRadii,
                                          CB.gCloudCorrelationLength, CB.gCloudAmplitude, CB.gCloudMeanOffset,
                                          cloudSeed, previousWasCloud, birthGradient, cloudDistance, cloudGradient);
#endif
            if (cloudHit && cloudDistance < limit)
            {
                hit = true;
                distance = cloudDistance;
                gradient = cloudGradient;
            }
            else cloudHit = false;
        }
        previousWasCloud = cloudHit;
#endif
        if (!hit)
        {
            radiance = throughput * light;
            break;
        }
        origin += distance * direction;
        float n2 = dot(gradient, gradient);
        if (!(n2 > 0.f)) break;
#if GPIS_TEXTURED_MESH
        pathLength += distance;
#endif
#if GPIS_SECOND_FIELD
        if (cloudHit)
        {
            float3 normal = gradient * rsqrt(n2);
            if (dot(normal, direction) > 0.f) normal = -normal;
            throughput *= CB.gCloudColor;
            direction = sampleGPISDiffuse(normal, seed ^ ((bounce + 1u) * 0xc2b2ae35u));
            continue;
        }
#endif
#if GPIS_TEXTURED_MESH
        float3 normal = gradient * rsqrt(n2);
        if (dot(normal, direction) > 0.f) normal = -normal;
        // Cosine sampling cancels Lambert's cosine / pi term in the path weight.
        throughput *= meshAlbedo(origin, direction, normal, pathLength * pixelAngle);
        direction = sampleGPISDiffuse(normal, seed ^ ((bounce + 1u) * 0x165667b1u));
#else
        direction = normalize(reflect(direction, gradient * rsqrt(n2)));
        throughput *= float3(0.15f, 0.8f, 0.5f);
#endif
    }
    gOutput[pixel] = float4(radiance, 1.f);
}
