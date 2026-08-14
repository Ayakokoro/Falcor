#pragma once
#include "../HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

#define OCTA_RES 4
#define OCTA_COUNT (OCTA_RES * OCTA_RES)

// 半球(z>=0) → [0,1]² 满正方形；偶函数折叠 z<0
// 仅在 GPU 侧使用：normalize/abs/max 等符号在主机 C++ 侧无全局定义，会导致 C2447
#if !defined(HOST_CODE)
inline float2 octaEncode(float3 d)
{
    d = normalize(d);
    if (d.z < 0)
        d = -d;
    float2 p;
    p.x = d.x / (abs(d.x) + abs(d.y) + d.z); // 八面体投影 → 菱形 |p.x|+|p.y|<=1
    p.y = d.y / (abs(d.x) + abs(d.y) + d.z);
    float2 s = float2(p.x - p.y, p.x + p.y);        // 菱形 → 正方形
    return s * 0.5f + 0.5f;
}

inline float3 octaDecode(float2 uv)
{
    float2 s = uv * 2.0f - 1.0f;
    float2 p = float2(s.x + s.y, s.y - s.x) * 0.5f; // 正方形 → 菱形
    float z = 1.0f - abs(p.x) - abs(p.y);
    return normalize(float3(p.x, p.y, max(z, 0.0f)));
}
#endif

struct OctahedralFunc
{
    float coefficients[OCTA_COUNT]; // 每个 texel 存该方向上的值

    MUTATING
    void init()
    {
        for (int i = 0; i < OCTA_COUNT; i++)
            coefficients[i] = 0;
    }

#if !defined(HOST_CODE)
    // 双线性插值求值（clamp 边界），镜像 SphericalFunc::calc
    float calc(float3 direction)
    {
        float2 uv = octaEncode(direction);
        float2 g = uv * OCTA_RES - 0.5f;
        int2 i0 = int2(floor(g));
        float2 f = g - float2(i0);
        int2 i1 = min(i0 + 1, OCTA_RES - 1);
        i0 = max(i0, 0);

        float v00 = coefficients[i0.y * OCTA_RES + i0.x];
        float v10 = coefficients[i0.y * OCTA_RES + i1.x];
        float v01 = coefficients[i1.y * OCTA_RES + i0.x];
        float v11 = coefficients[i1.y * OCTA_RES + i1.x];
        return lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
    }
#endif
};

END_NAMESPACE_FALCOR
