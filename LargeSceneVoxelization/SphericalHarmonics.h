#pragma once
#include "Types.h"

#define SH_COUNT 16
#define EVEN_SH_COUNT 6

constexpr float SQRTPI = 1.7724538509055160273f;

inline float calcEvenSH(uint idx, const float3& p) {
    float x = p.x, y = p.y, z = p.z;
    float x2 = x*x, y2 = y*y, z2 = z*z;
    switch (idx) {
    case 0: return 1.0f / (2.0f * SQRTPI);
    case 1: return x * y * std::sqrt(15.0f) / (2.0f * SQRTPI);
    case 2: return y * z * std::sqrt(15.0f) / (2.0f * SQRTPI);
    case 3: return (3.0f * z2 - 1.0f) * std::sqrt(5.0f) / (4.0f * SQRTPI);
    case 4: return x * z * std::sqrt(15.0f) / (2.0f * SQRTPI);
    case 5: return (x2 - y2) * std::sqrt(15.0f) / (4.0f * SQRTPI);
    default: return 0.0f;
    }
}

// Even SphericalHarmonics (band 0, 2) — 6 coefficients
struct SphericalFunc {
    float coefficients[EVEN_SH_COUNT] = {};

    void init() { for (int i = 0; i < EVEN_SH_COUNT; i++) coefficients[i] = 0; }

    void accumulate(float valueByWeight, const float3& direction) {
        for (int i = 0; i < EVEN_SH_COUNT; i++)
            coefficients[i] += valueByWeight * calcEvenSH(i, direction);
    }

    float eval(const float3& p) const {
        float sum = 0;
        for (int i = 0; i < EVEN_SH_COUNT; i++)
            sum += coefficients[i] * calcEvenSH(i, p);
        return sum;
    }
};
