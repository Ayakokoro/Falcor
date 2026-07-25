#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <bit>
#include <ostream>

// --- Type aliases matching Falcor/slang conventions ---
using uint = uint32_t;

using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;

using int2  = glm::ivec2;
using int3  = glm::ivec3;

using uint2 = glm::uvec2;
using uint3 = glm::uvec3;

using float2x2 = glm::mat2;
using float3x3 = glm::mat3;

// --- Math constants ---
constexpr float M_PI_F = glm::pi<float>();
constexpr float M_1_PI_F = glm::one_over_pi<float>();

// Per-node polygon safety cap (matches GPU kSafePerNodePolygonLimit)
#define SAFE_PER_NODE_POLYGON_LIMIT 128000

// --- GLM-compatible Matrix helpers ---
inline float3 mul(const float3x3& M, const float3& v) { return M * v; }
inline float3x3 transpose(const float3x3& M) { return glm::transpose(M); }
inline float  dot(const float3& a, const float3& b) { return glm::dot(a, b); }
inline float3 cross(const float3& a, const float3& b) { return glm::cross(a, b); }
inline float3 normalize(const float3& v) { return glm::normalize(v); }
inline float  length(const float3& v) { return glm::length(v); }
inline float3 abs(const float3& v) { return glm::abs(v); }
inline float  absf(float x) { return std::fabs(x); }
inline float  determinant(const float2x2& m) { return glm::determinant(m); }
inline float  lerp(float a, float b, float t) { return a + t * (b - a); }
inline float3 lerp(const float3& a, const float3& b, float t) { return a + t * (b - a); }
inline float4 lerp(const float4& a, const float4& b, float t) { return a + t * (b - a); }

inline float3 floor(const float3& v) { return glm::floor(v); }
inline int3   floorToInt3(const float3& v) {
    return int3((int)std::floor(v.x), (int)std::floor(v.y), (int)std::floor(v.z));
}

inline float3x3 make3x3(float a00, float a01, float a02,
                         float a10, float a11, float a12,
                         float a20, float a21, float a22) {
    float3x3 m;
    m[0][0] = a00; m[0][1] = a01; m[0][2] = a02;
    m[1][0] = a10; m[1][1] = a11; m[1][2] = a12;
    m[2][0] = a20; m[2][1] = a21; m[2][2] = a22;
    return m;
}

inline float3x3 mul(float k, const float3x3& a) {
    float3x3 r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            r[i][j] = k * a[i][j];
    return r;
}

inline float3x3 zeros3x3() {
    return float3x3(0.0f);
}

inline float3x3 identity3x3() {
    return float3x3(1.0f);
}

inline float3x3 diag3(const float3& d) {
    return make3x3(d.x, 0, 0, 0, d.y, 0, 0, 0, d.z);
}

inline float3x3 add(const float3x3& a, const float3x3& b) { return a + b; }
inline float3x3 sub(const float3x3& a, const float3x3& b) { return a - b; }

inline float3 safeNormalize(const float3& n) {
    float len = length(n);
    return (len > 0) ? n / len : float3(0.0f);
}

inline float safeSqrt(float x) { return std::sqrt(std::max(0.0f, x)); }

inline float srgbChannelToLinear(float c) {
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}
inline float3 srgbToLinear(const float3& c) {
    return float3(srgbChannelToLinear(c.x), srgbChannelToLinear(c.y), srgbChannelToLinear(c.z));
}
inline float4 srgbToLinear(const float4& c) {
    return float4(srgbChannelToLinear(c.x), srgbChannelToLinear(c.y), srgbChannelToLinear(c.z), c.w);
}

inline uint32_t countbits(uint32_t x) {
    return std::popcount(x);
}

// --- Stream output operators ---
inline std::ostream& operator<<(std::ostream& os, const float2& v) {
    return os << "(" << v.x << ", " << v.y << ")";
}
inline std::ostream& operator<<(std::ostream& os, const float3& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}
inline std::ostream& operator<<(std::ostream& os, const float4& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
}
inline std::ostream& operator<<(std::ostream& os, const int2& v) {
    return os << "(" << v.x << ", " << v.y << ")";
}
inline std::ostream& operator<<(std::ostream& os, const int3& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}
inline std::ostream& operator<<(std::ostream& os, const uint2& v) {
    return os << "(" << v.x << ", " << v.y << ")";
}
inline std::ostream& operator<<(std::ostream& os, const uint3& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}
inline std::ostream& operator<<(std::ostream& os, const float2x2& m) {
    return os << "[" << m[0] << ", " << m[1] << "]";
}
inline std::ostream& operator<<(std::ostream& os, const float3x3& m) {
    return os << "[" << m[0] << ", " << m[1] << ", " << m[2] << "]";
}
