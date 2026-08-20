#pragma once
#include "../HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

#define PI 3.14159265359

struct Basis2
{
    float3 u;
    float3 w;
};
inline Basis2 orthonormal_basis(float3 v)
{
    Basis2 ret;
    float3 axis = (abs(v.z) < 0.999) ? float3(0, 0, 1) : float3(1, 0, 0);
    ret.u = normalize(cross(axis, v));
    ret.w = normalize(cross(v, ret.u));
    return ret;
}

#if !defined(HOST_CODE)

#define PROJ_RES 64
#define PROJ_PLANE_SIZE 2
#define PROJ_HALF_SIZE (PROJ_PLANE_SIZE * 0.5f) // 1.0
#define PROJ_PIXEL_SIZE (PROJ_PLANE_SIZE / float(PROJ_RES))
#define PROJ_WORDS ((PROJ_RES * PROJ_RES + 31u) / 32u)

float edgeFunc(float2 a, float2 b, float2 p)
{
    // (p - a) x (b - a) in 2D
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

bool pointInTriangle(float2 p, float2 a, float2 b, float2 c)
{
    float e0 = edgeFunc(a, b, p);
    float e1 = edgeFunc(b, c, p);
    float e2 = edgeFunc(c, a, p);

    const float eps = 1e-6f;
    bool hasNeg = (e0 < -eps) || (e1 < -eps) || (e2 < -eps);
    bool hasPos = (e0 > eps) || (e1 > eps) || (e2 > eps);
    return !(hasNeg && hasPos); // all same sign (or on edges)
}

// ----- Bitmask helpers -----
void maskSet(inout uint mask[PROJ_WORDS], uint linearIndex)
{
    uint word = linearIndex >> 5; // / 32
    uint bit = linearIndex & 31u; // % 32
    mask[word] |= (1u << bit);
}

uint maskCountBits(uint mask[PROJ_WORDS])
{
    uint sum = 0;
    [unroll]
    for (uint i = 0; i < PROJ_WORDS; ++i)
        sum += countbits(mask[i]);
    return sum;
}

// ----- Rasterize one triangle into mask -----
int clampInt(int x, int lo, int hi)
{
    return max(lo, min(hi, x));
}

void rasterizeTriToMask(float2 a, float2 b, float2 c, inout uint mask[PROJ_WORDS])
{
    float2 mn = min(a, min(b, c));
    float2 mx = max(a, max(b, c));

    // Convert bbox in [-1,1] to pixel indices [0, PROJ_RES-1]
    int ix0 = int(floor((mn.x + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE));
    int iy0 = int(floor((mn.y + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE));
    int ix1 = int(floor((mx.x + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE));
    int iy1 = int(floor((mx.y + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE));

    int Rm1 = int(PROJ_RES) - 1;
    ix0 = clampInt(ix0, 0, Rm1);
    iy0 = clampInt(iy0, 0, Rm1);
    ix1 = clampInt(ix1, 0, Rm1);
    iy1 = clampInt(iy1, 0, Rm1);

    for (int iy = iy0; iy <= iy1; ++iy)
    {
        float y = -PROJ_HALF_SIZE + (float(iy) + 0.5f) * PROJ_PIXEL_SIZE;
        for (int ix = ix0; ix <= ix1; ++ix)
        {
            float x = -PROJ_HALF_SIZE + (float(ix) + 0.5f) * PROJ_PIXEL_SIZE;

            if (pointInTriangle(float2(x, y), a, b, c))
            {
                uint idx = uint(iy) * PROJ_RES + uint(ix);
                maskSet(mask, idx);
            }
        }
    }
}

// 逐像素深度光栅化：在给定顶点深度 d0/d1/d2 下，把三角形覆盖到的每个像素，
// 用重心坐标插值出深度 d，若 d < depthArr 则写入（并记录该三角形 id）。
// 仅考虑前表面（插值深度单调），后表面被更小的深度自然覆盖。
void rasterizeTriDepth(
    float2 a, float2 b, float2 c,
    float d0, float d1, float d2,
    uint triId,
    inout float depthArr[PROJ_RES * PROJ_RES],
    inout uint idArr[PROJ_RES * PROJ_RES])
{
    float2 mn = min(a, min(b, c));
    float2 mx = max(a, max(b, c));

    int ix0 = int(floor((mn.x + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE));
    int iy0 = int(floor((mn.y + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE));
    int ix1 = int(floor((mx.x + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE));
    int iy1 = int(floor((mx.y + PROJ_HALF_SIZE) / PROJ_PIXEL_SIZE));

    int Rm1 = int(PROJ_RES) - 1;
    ix0 = clampInt(ix0, 0, Rm1);
    iy0 = clampInt(iy0, 0, Rm1);
    ix1 = clampInt(ix1, 0, Rm1);
    iy1 = clampInt(iy1, 0, Rm1);

    // 重心坐标分母：edgeFunc(a,b,c) = -2×三角形有符号面积（退化为边视时跳过）
    float denom = edgeFunc(a, b, c);
    const float areaEps = 1e-8f;
    if (abs(denom) < areaEps)
        return; // 投影后退化为线段/点，无面积贡献
    float invDenom = 1.0f / denom;

    for (int iy = iy0; iy <= iy1; ++iy)
    {
        float y = -PROJ_HALF_SIZE + (float(iy) + 0.5f) * PROJ_PIXEL_SIZE;
        for (int ix = ix0; ix <= ix1; ++ix)
        {
            float x = -PROJ_HALF_SIZE + (float(ix) + 0.5f) * PROJ_PIXEL_SIZE;
            float2 p = float2(x, y);

            float e0 = edgeFunc(b, c, p);
            float e1 = edgeFunc(c, a, p);
            float e2 = edgeFunc(a, b, p);

            const float eps = 1e-6f;
            bool hasNeg = (e0 < -eps) || (e1 < -eps) || (e2 < -eps);
            bool hasPos = (e0 > eps) || (e1 > eps) || (e2 > eps);
            if (hasNeg && hasPos)
                continue; // 在三角形外

            // 重心坐标（相对顶点 a/b/c 的权重）
            float w0 = e0 * invDenom;
            float w1 = e1 * invDenom;
            float w2 = e2 * invDenom;
            float d = w0 * d0 + w1 * d1 + w2 * d2;

            uint idx = uint(iy) * PROJ_RES + uint(ix);
            if (d < depthArr[idx])
            {
                depthArr[idx] = d;
                idArr[idx] = triId;
            }
        }
    }
}
#endif

END_NAMESPACE_FALCOR
