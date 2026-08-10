#include "Voxel/ABSDF.slang"

Texture2D<float4> splittingMap;   // R=GT, G=Approx, B=absError
SamplerState pointSampler;

cbuffer CB
{
    uint visualizationMode;   // 0=GT, 1=Approx, 2=AbsError
    float valueMin;
    float valueMax;
    float4 clearColor;
}

struct PsOut
{
    float4 color : SV_Target0;
};

PsOut main(float2 texCoord : TEXCOORD, float4 pos : SV_POSITION)
{
    PsOut psOut = { clearColor };

    float4 data = splittingMap.SampleLevel(pointSampler, texCoord, 0);

    float value = 0;
    if (visualizationMode == 0)
        value = data.r;   // GT
    else if (visualizationMode == 1)
        value = data.g;   // Approx
    else
        value = data.b;   // absError

    float k = saturate((value - valueMin) / max(valueMax - valueMin, 1e-8f));
    float3 c;
    // 黑(0) → 红(0.5) → 黄(1)
    c.r = saturate(k * 2.0f);
    c.g = saturate((k - 0.5f) * 2.0f);
    c.b = 0.0f;

    psOut.color = float4(c, 1.0f);
    return psOut;
}
