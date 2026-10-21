#pragma once
#include "Types.h"
#include "Triangle.h"

inline constexpr uint VISIBLE_LOBE_COUNT = 8;

inline uint VisibleNormalIndex(const float3& n) {
    float ax = std::abs(n.x), az = std::abs(n.z);
    const float eps = 1e-6f;
    bool useX;
    if (ax > az + eps)       useX = true;
    else if (az > ax + eps)  useX = false;
    else                     useX = (n.x >= n.z);
    if (useX) return (n.x >= 0.0f) ? 0u : 1u;
    else      return (n.z >= 0.0f) ? 2u : 3u;
}

// The renderer stores both hemispheres explicitly.  The lower hemisphere is
// folded onto the upper one for the lobe lookup and then offset by four.
inline uint VisibleNormalIndex8(const float3& n) {
    uint hemi = n.y < 0.0f ? 4u : 0u;
    float3 up = hemi != 0u ? -n : n;
    return hemi + VisibleNormalIndex(up);
}

struct VisibleABSDFInput {
    float3 baseColor;
    float4 specular;    // .g = roughness, .b = metallic
    float3 normal;
    float area;         // geometric surface area
    float projArea;     // visible projected area used for material fitting
};

struct VisibleABSDFLobe {
    float weight = 0;
    float3 normal = float3(0);
    float rough = 0;
    float3 diffuse = float3(0);
    float3 specular = float3(0);

    void init() { weight = 0; normal = float3(0); rough = 0; diffuse = float3(0); specular = float3(0); }

    void accumulate(const VisibleABSDFInput& input) {
        float IoR = 1.5f;
        float f = (IoR - 1.0f) / (IoR + 1.0f);
        float F0 = f * f;
        float w = input.projArea;
        diffuse  += w * lerp(input.baseColor, float3(0.0f), input.specular.b);
        specular += w * lerp(float3(F0), input.baseColor, input.specular.b);
        rough    += w * input.specular.g;
        normal   += w * input.normal;
        weight   += w;
    }

    void normalizeSelf() {
        if (weight > 0) {
            normal = safeNormalize(normal);
            diffuse /= weight;
            specular /= weight;
            rough /= weight;
        }
    }

    bool isValid() const { return weight > 0 && dot(normal, normal) > 0; }
};

struct VisibleABSDF {
    VisibleABSDFLobe lobes[VISIBLE_LOBE_COUNT];
    float area = 0;

    void init() {
        for (int i = 0; i < VISIBLE_LOBE_COUNT; i++) lobes[i].init();
        area = 0;
    }

    void accumulate(const VisibleABSDFInput& input) {
        lobes[VisibleNormalIndex8(input.normal)].accumulate(input);
    }

    void normalizeSelf() {
        if (area == 0) return;
        for (int i = 0; i < VISIBLE_LOBE_COUNT; i++)
            lobes[i].normalizeSelf();
        normalizeWeightsOnly();
    }

    void normalizeWeightsOnly() {
        float totalWeight = 0.0f;
        for (uint i = 0; i < VISIBLE_LOBE_COUNT; ++i)
            totalWeight += lobes[i].weight;
        if (totalWeight <= 0.0f) return;

        // Only keeps stored amplitudes in a convenient range. Runtime
        // normalizes the view-facing projected areas to one.
        float scale = 2.0f / totalWeight;
        for (uint i = 0; i < VISIBLE_LOBE_COUNT; ++i)
            lobes[i].weight *= scale;
    }

    bool isSolid() const { return area > 0.0f; }
};
