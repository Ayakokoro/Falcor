#include "Voxel/ABSDF.slang"

Texture2D<float4> splittingMap;   // R=GT, G=Approx, B=|GT-Approx|, A=|formula4 - lobeBSDF|
SamplerState pointSampler;

cbuffer CB
{
    uint visualizationMode;   // 0=GT, 1=Approx, 2=AbsError, 3=BSDF Lobe Error
    float valueMin;
    float valueMax;
    float4 clearColor;
}

struct PsOut
{
    float4 color : SV_Target0;
};

// 将 [0, 1] 范围的标量映射为 Inferno 颜色
float3 MapToInferno(float t) {
    t = saturate(t);

    // Inferno 色阶的 6 个关键控制点 (黑 -> 紫 -> 紫红 -> 橙红 -> 亮黄 -> 浅黄)
    float3 colors[6] = {
        float3(0.000, 0.000, 0.016), // 0.0
        float3(0.220, 0.035, 0.408), // 0.2
        float3(0.573, 0.137, 0.278), // 0.4
        float3(0.867, 0.322, 0.106), // 0.6
        float3(0.984, 0.682, 0.137), // 0.8
        float3(0.988, 0.996, 0.643)  // 1.0
    };

    float scaled_t = t * 5.0;

    int idx = min((int)scaled_t, 4);

    float frac_t = frac(scaled_t);

    return lerp(colors[idx], colors[idx + 1], frac_t);
}

PsOut main(float2 texCoord : TEXCOORD, float4 pos : SV_POSITION)
{
    PsOut psOut = { clearColor };

    float4 data = splittingMap.SampleLevel(pointSampler, texCoord, 0);

    float value = 0;
    if (visualizationMode == 0)
        value = data.r;   // GT
    else if (visualizationMode == 1)
        value = data.g;   // Approx
    else if (visualizationMode == 2)
        value = data.b;   // absError
    else
        value = data.a;   // BSDF Lobe Error

    float k = saturate((value - valueMin) / max(valueMax - valueMin, 1e-8f));

    psOut.color = float4(MapToInferno(k), 1.0f);
    return psOut;
}
