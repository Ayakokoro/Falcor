#pragma once
#include "Types.h"
#include "Triangle.h"

#define LOBE_COUNT 4

inline uint NormalIndex(const float3& n) {
    float ax = std::abs(n.x), az = std::abs(n.z);
    const float eps = 1e-6f;
    bool useX;
    if (ax > az + eps)       useX = true;
    else if (az > ax + eps)  useX = false;
    else                     useX = (n.x >= n.z);
    if (useX) return (n.x >= 0.0f) ? 0u : 1u;
    else      return (n.z >= 0.0f) ? 2u : 3u;
}

struct ABSDFInput {
    float3 baseColor;
    float4 specular;    // .g = roughness, .b = metallic
    float3 normal;
    float area;
};

struct ABSDFLobe {
    float weight = 0;
    float3 normal = float3(0);
    float rough = 0;
    float3 diffuse = float3(0);
    float3 specular = float3(0);

    void init() { weight = 0; normal = float3(0); rough = 0; diffuse = float3(0); specular = float3(0); }

    void accumulate(const ABSDFInput& input) {
        float IoR = 1.5f;
        float f = (IoR - 1.0f) / (IoR + 1.0f);
        float F0 = f * f;
        diffuse  += input.area * lerp(input.baseColor, float3(0.0f), input.specular.b);
        specular += input.area * lerp(float3(F0), input.baseColor, input.specular.b);
        rough    += input.area * input.specular.g;
        normal   += input.area * input.normal;
        weight   += input.area;
    }

    void normalizeSelf(float totalArea) {
        if (weight > 0) {
            normal = safeNormalize(normal);
            diffuse /= weight;
            specular /= weight;
            rough /= weight;
            weight /= totalArea;
        }
    }

    bool isValid() const { return weight > 0 && dot(normal, normal) > 0; }
};

struct ABSDF {
    ABSDFLobe lobes[LOBE_COUNT];
    float area = 0;

    void init() {
        for (int i = 0; i < LOBE_COUNT; i++) lobes[i].init();
        area = 0;
    }

    void accumulate(const ABSDFInput& input) {
        // Keep the stored lobe normal consistent with the GPU implementation.
        // The GPU canonicalizes the input normal before both lobe selection and
        // accumulation; pass the canonicalized normal to the lobe as well.
        ABSDFInput canonicalInput = input;
        if (canonicalInput.normal.y < 0)
            canonicalInput.normal = -canonicalInput.normal;

        lobes[NormalIndex(canonicalInput.normal)].accumulate(canonicalInput);
        area += input.area;
    }

    void normalizeSelf() {
        if (area == 0) return;
        for (int i = 0; i < LOBE_COUNT; i++)
            lobes[i].normalizeSelf(area);
    }
};
