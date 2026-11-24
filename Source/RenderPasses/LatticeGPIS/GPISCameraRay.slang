#ifndef GPIS_CAMERA_RAY_SLANG
#define GPIS_CAMERA_RAY_SLANG

uint latticeHash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

float gpisTentSample(uint bits)
{
    float u = (float(bits >> 9) + 0.5f) / 8388608.f;
    return u < 0.5f ? sqrt(2.f * u) - 1.f : 1.f - sqrt(2.f * (1.f - u));
}

// Both passes use the same jitter for a given pixel and frame.
float3 gpisCameraRay(uint2 pixel, uint2 dims, uint frameSeed, float3 cameraU, float3 cameraV, float3 cameraW, out uint seed)
{
    seed = latticeHash((pixel.y * dims.x + pixel.x) ^ latticeHash(frameSeed + 0x85ebca6bu));
    float2 jitter = float2(gpisTentSample(latticeHash(seed)), gpisTentSample(latticeHash(seed + 1u)));
    float2 ndc = (float2(pixel) + 0.5f + jitter) / float2(dims) * 2.f - 1.f;
    return cameraW + ndc.x * cameraU - ndc.y * cameraV;
}

#endif
