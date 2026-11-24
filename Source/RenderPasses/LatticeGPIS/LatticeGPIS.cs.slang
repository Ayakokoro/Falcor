#include "SdfField.slang"

uniform SdfField gField;
#include "Intersection.slang"

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

// Tungsten's shaderball uses a one-sided 5 x 5 quad rotated 160 degrees about X.
// It specifies emitted power, which is converted to radiance for both emitters.
float3 incidentLight(float3 origin, float3 direction, out float lightDistance)
{
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
}

float tentSample(uint bits)
{
    float u = (float(bits >> 9) + 0.5f) / 8388608.f;
    return u < 0.5f ? sqrt(2.f * u) - 1.f : 1.f - sqrt(2.f * (1.f - u));
}

[numthreads(8, 8, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
    uint2 pixel = threadID.xy;
    if (any(pixel >= CB.gFrameDim)) return;
    uint seed = latticeHash((pixel.y * CB.gFrameDim.x + pixel.x) ^ latticeHash(CB.gSeed + 0x85ebca6bu));
    float2 jitter = float2(tentSample(latticeHash(seed)), tentSample(latticeHash(seed + 1u)));
    float2 ndc = (float2(pixel) + 0.5f + jitter) / float2(CB.gFrameDim) * 2.f - 1.f;
    float3 origin = CB.gCameraPosition;
    float3 direction = normalize(CB.gCameraW + ndc.x * CB.gCameraU - ndc.y * CB.gCameraV);

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
    [loop]
    for (uint bounce = 0; bounce <= 20; ++bounce)
    {
        float lightDistance;
        float3 light = incidentLight(origin, direction, lightDistance);
        float2 span = float2(0.f, lightDistance);
        bool hit = false;
        float distance = 0.f;
        if (clipBox(origin, direction, gField.boundsMin, gField.boundsMax, span))
        {
            uint flightSeed = latticeHash(seed + (bounce + 1u) * 0x9e3779b9u);
            float3 birthGradient = gradient;
            hit = latticeIntersect(origin, direction, span.x, span.y, flightSeed, kernel, bounce != 0,
                                   birthGradient, distance, gradient);
        }
        if (!hit)
        {
            radiance = throughput * light;
            break;
        }
        origin += distance * direction;
        float n2 = dot(gradient, gradient);
        if (!(n2 > 0.f)) break;
        direction = normalize(reflect(direction, gradient * rsqrt(n2)));
        throughput *= float3(0.15f, 0.8f, 0.5f);
    }
    gOutput[pixel] = float4(radiance, 1.f);
}
