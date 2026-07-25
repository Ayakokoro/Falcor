#pragma once
#include "Types.h"

#define LEBEDEV_DIRECTION_COUNT 26

constexpr float InvSqrt2 = 0.7071067811865475244f;
constexpr float Lebedev_AxisWeight   = 1.0f / 21.0f;
constexpr float Lebedev_EdgeWeight   = 4.0f / 105.0f;
constexpr float Lebedev_CornerWeight = 9.0f / 280.0f;

struct LebedevSample {
    float3 direction;
    float weight;
};

static const LebedevSample LebedevSamples[LEBEDEV_DIRECTION_COUNT] = {
    // 6 axis points
    {{ 1, 0, 0}, Lebedev_AxisWeight}, {{-1, 0, 0}, Lebedev_AxisWeight},
    {{ 0, 1, 0}, Lebedev_AxisWeight}, {{ 0,-1, 0}, Lebedev_AxisWeight},
    {{ 0, 0, 1}, Lebedev_AxisWeight}, {{ 0, 0,-1}, Lebedev_AxisWeight},
    // 12 edge-midpoints
    {{ InvSqrt2, InvSqrt2, 0}, Lebedev_EdgeWeight}, {{-InvSqrt2,-InvSqrt2, 0}, Lebedev_EdgeWeight},
    {{ InvSqrt2,-InvSqrt2, 0}, Lebedev_EdgeWeight}, {{-InvSqrt2, InvSqrt2, 0}, Lebedev_EdgeWeight},
    {{ InvSqrt2, 0, InvSqrt2}, Lebedev_EdgeWeight}, {{-InvSqrt2, 0,-InvSqrt2}, Lebedev_EdgeWeight},
    {{ InvSqrt2, 0,-InvSqrt2}, Lebedev_EdgeWeight}, {{-InvSqrt2, 0, InvSqrt2}, Lebedev_EdgeWeight},
    {{ 0, InvSqrt2, InvSqrt2}, Lebedev_EdgeWeight}, {{ 0,-InvSqrt2,-InvSqrt2}, Lebedev_EdgeWeight},
    {{ 0, InvSqrt2,-InvSqrt2}, Lebedev_EdgeWeight}, {{ 0,-InvSqrt2, InvSqrt2}, Lebedev_EdgeWeight},
    // 8 corners: (±1/√3, ±1/√3, ±1/√3)
    {{ 0.5773502691896258f, 0.5773502691896258f, 0.5773502691896258f}, Lebedev_CornerWeight},
    {{-0.5773502691896258f,-0.5773502691896258f,-0.5773502691896258f}, Lebedev_CornerWeight},
    {{ 0.5773502691896258f,-0.5773502691896258f, 0.5773502691896258f}, Lebedev_CornerWeight},
    {{-0.5773502691896258f, 0.5773502691896258f,-0.5773502691896258f}, Lebedev_CornerWeight},
    {{ 0.5773502691896258f, 0.5773502691896258f,-0.5773502691896258f}, Lebedev_CornerWeight},
    {{-0.5773502691896258f,-0.5773502691896258f, 0.5773502691896258f}, Lebedev_CornerWeight},
    {{ 0.5773502691896258f,-0.5773502691896258f,-0.5773502691896258f}, Lebedev_CornerWeight},
    {{-0.5773502691896258f, 0.5773502691896258f, 0.5773502691896258f}, Lebedev_CornerWeight},
};
