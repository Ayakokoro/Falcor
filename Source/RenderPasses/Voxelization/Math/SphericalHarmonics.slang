#pragma once
#include "../HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

#define SQRTPI 1.7724538509055160273f
#define SH_COUNT 16
#define EVEN_SH_COUNT 6

inline float calcSH(uint idx, float3 p)
{
    const float x = p.x;
    const float y = p.y;
    const float z = p.z;

    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;

    switch (idx)
    {
    // l = 0
    case 0:
        return 1.f / (2.f * SQRTPI); // m = 0

    // l = 1
    case 1:
        return y * sqrt(3.f) / (2.f * SQRTPI); // m = -1
    case 2:
        return z * sqrt(3.f) / (2.f * SQRTPI); // m =  0
    case 3:
        return x * sqrt(3.f) / (2.f * SQRTPI); // m =  1

    // l = 2
    case 4:
        return x * y * sqrt(15.f) / (2.f * SQRTPI); // m = -2
    case 5:
        return y * z * sqrt(15.f) / (2.f * SQRTPI); // m = -1
    case 6:
        return (3.f * z2 - 1.f) * sqrt(5.f) / (4.f * SQRTPI); // m =  0
    case 7:
        return x * z * sqrt(15.f) / (2.f * SQRTPI); // m =  1
    case 8:
        return (x2 - y2) * sqrt(15.f) / (4.f * SQRTPI); // m =  2

    // l = 3
    case 9:
        return y * (3.f * x2 - y2) * sqrt(70.f) / (8.f * SQRTPI); // m = -3
    case 10:
        return x * y * z * sqrt(105.f) / (2.f * SQRTPI); // m = -2
    case 11:
        return y * (5.f * z2 - 1.f) * sqrt(42.f) / (8.f * SQRTPI); // m = -1
    case 12:
        return z * (5.f * z2 - 3.f) * sqrt(7.f) / (4.f * SQRTPI); // m =  0
    case 13:
        return x * (5.f * z2 - 1.f) * sqrt(42.f) / (8.f * SQRTPI); // m =  1
    case 14:
        return z * (x2 - y2) * sqrt(105.f) / (4.f * SQRTPI); // m =  2
    case 15:
        return x * (x2 - 3.f * y2) * sqrt(70.f) / (8.f * SQRTPI); // m =  3

    // l = 4
    case 16:
        return x * y * (x2 - y2) * 3.f * sqrt(35.f) / (4.f * SQRTPI); // m = -4
    case 17:
        return y * z * (3.f * x2 - y2) * 3.f * sqrt(70.f) / (8.f * SQRTPI); // m = -3
    case 18:
        return x * y * (7.f * z2 - 1.f) * 3.f * sqrt(5.f) / (4.f * SQRTPI); // m = -2
    case 19:
        return y * z * (7.f * z2 - 3.f) * 3.f * sqrt(10.f) / (8.f * SQRTPI); // m = -1
    case 20:
        return (35.f * z2 * z2 - 30.f * z2 + 3.f) * 3.f / (16.f * SQRTPI); // m =  0
    case 21:
        return x * z * (7.f * z2 - 3.f) * 3.f * sqrt(10.f) / (8.f * SQRTPI); // m =  1
    case 22:
        return (x2 - y2) * (7.f * z2 - 1.f) * 3.f * sqrt(5.f) / (8.f * SQRTPI); // m =  2
    case 23:
        return x * z * (x2 - 3.f * y2) * 3.f * sqrt(70.f) / (8.f * SQRTPI); // m =  3
    case 24:
        return (x2 * (x2 - 3.f * y2) - y2 * (3.f * x2 - y2)) * 3.f * sqrt(35.f) / (16.f * SQRTPI); // m =  4
    }
    return 0.f;
}

struct SphericalHarmonics
{
    float coefficients[SH_COUNT];

    MUTATING
    void init()
    {
        for (int i = 0; i < SH_COUNT; i++)
        {
            coefficients[i] = 0;
        }
    }

    MUTATING
    // 所有采样方向的权重总和应当为4PI
    void accumulate(float valueByWeight, float3 direction)
    {
        // 默认direction已经归一化
        for (int i = 0; i < SH_COUNT; i++)
        {
            coefficients[i] += valueByWeight * calcSH(i, direction);
        }
    }

    MUTATING
    // 不带系数统计，最后再统一乘以系数
    void accumulate(float3 direction)
    {
        // 默认direction已经归一化
        for (int i = 0; i < SH_COUNT; i++)
        {
            coefficients[i] += calcSH(i, direction);
        }
    }

    MUTATING
    void mul(float value)
    {
        for (int i = 0; i < SH_COUNT; i++)
        {
            coefficients[i] *= value;
        }
    }

    float calc(float3 p)
    {
        float sum = 0;
        for (int i = 0; i < SH_COUNT; i++)
        {
            sum += coefficients[i] * calcSH(i, p);
        }
        return sum;
    }
};

inline float calcEvenSH(uint idx, float3 p)
{
    const float x = p.x;
    const float y = p.y;
    const float z = p.z;

    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;

    switch (idx)
    {
    // l = 0
    case 0:
        return 1.f / (2.f * SQRTPI); // m = 0

    // l = 2
    case 1:
        return x * y * sqrt(15.f) / (2.f * SQRTPI); // m = -2
    case 2:
        return y * z * sqrt(15.f) / (2.f * SQRTPI); // m = -1
    case 3:
        return (3.f * z2 - 1.f) * sqrt(5.f) / (4.f * SQRTPI); // m =  0
    case 4:
        return x * z * sqrt(15.f) / (2.f * SQRTPI); // m =  1
    case 5:
        return (x2 - y2) * sqrt(15.f) / (4.f * SQRTPI); // m =  2

    // l = 4
    case 6:
        return x * y * (x2 - y2) * 3.f * sqrt(35.f) / (4.f * SQRTPI); // m = -4
    case 7:
        return y * z * (3.f * x2 - y2) * 3.f * sqrt(70.f) / (8.f * SQRTPI); // m = -3
    case 8:
        return x * y * (7.f * z2 - 1.f) * 3.f * sqrt(5.f) / (4.f * SQRTPI); // m = -2
    case 9:
        return y * z * (7.f * z2 - 3.f) * 3.f * sqrt(10.f) / (8.f * SQRTPI); // m = -1
    case 10:
        return (35.f * z2 * z2 - 30.f * z2 + 3.f) * 3.f / (16.f * SQRTPI); // m =  0
    case 11:
        return x * z * (7.f * z2 - 3.f) * 3.f * sqrt(10.f) / (8.f * SQRTPI); // m =  1
    case 12:
        return (x2 - y2) * (7.f * z2 - 1.f) * 3.f * sqrt(5.f) / (8.f * SQRTPI); // m =  2
    case 13:
        return x * z * (x2 - 3.f * y2) * 3.f * sqrt(70.f) / (8.f * SQRTPI); // m =  3
    case 14:
        return (x2 * (x2 - 3.f * y2) - y2 * (3.f * x2 - y2)) * 3.f * sqrt(35.f) / (16.f * SQRTPI); // m =  4
    }

    return 0.f;
}

struct EvenSphericalHarmonics
{
    float coefficients[EVEN_SH_COUNT];

    MUTATING
    void init()
    {
        for (int i = 0; i < EVEN_SH_COUNT; i++)
        {
            coefficients[i] = 0;
        }
    }

    MUTATING
    // 所有采样方向的权重总和应当为4PI
    void accumulate(float valueByWeight, float3 direction)
    {
        // 默认direction已经归一化
        for (int i = 0; i < EVEN_SH_COUNT; i++)
        {
            coefficients[i] += valueByWeight * calcEvenSH(i, direction);
        }
    }

    MUTATING
    // 不带系数统计，最后再统一乘以系数
    void accumulate(float3 direction)
    {
        // 默认direction已经归一化
        for (int i = 0; i < EVEN_SH_COUNT; i++)
        {
            coefficients[i] += calcEvenSH(i, direction);
        }
    }

    MUTATING
    void mul(float value)
    {
        for (int i = 0; i < EVEN_SH_COUNT; i++)
        {
            coefficients[i] *= value;
        }
    }

    float calc(float3 p)
    {
        float sum = 0;
        for (int i = 0; i < EVEN_SH_COUNT; i++)
        {
            sum += coefficients[i] * calcEvenSH(i, p);
        }
        return sum;
    }
};

END_NAMESPACE_FALCOR
