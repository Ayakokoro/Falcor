#include "Voxel/ABSDF.slang"
#include "Voxel/VoxelGrid.slang"
#include "VoxelizationShared.slang"

StructuredBuffer<VoxelData> gBuffer;
Texture2D<float> exactMap;
SamplerState pointSampler;

cbuffer CB
{
    uint targetGBufferIndex;
    uint2 mapResolution;
    uint functionType;       // 0=primitiveProjArea, 1=polygonsProjArea, 2=totalProjArea
    uint visualizationMode;  // 0=SH Approx, 1=Exact, 2=Error (abs diff)
    float valueMin;          // for color range normalization
    float valueMax;
    float4 clearColor;
}

float3 funcValueToColor(float value)
{
    float k = saturate((value - valueMin) / max(valueMax - valueMin, 1e-8f));
    // Blue(0) → Cyan → Green → Yellow → Red(1)
    float3 c;
    c.r = smoothstep(0.4, 1.0, k) - smoothstep(1.0, 5.0, k);
    c.g = smoothstep(0.0, 0.4, k) - smoothstep(0.6, 1.0, k);
    c.b = smoothstep(0.4, 0.0, k);
    return c;
}

float evalSH(VoxelData data, float3 direction)
{
    if (functionType == 0)
        return data.primitiveProjAreaFunc.calc(direction);
    else if (functionType == 1)
        return data.polygonsProjAreaFunc.calc(direction);
    else
        return data.totalProjAreaFunc.calc(direction);
}

// Convert hemisphere direction to equirectangular texture coordinates
// direction is in the local hemisphere frame with z up (z > 0)
float sampleExactMap(float3 localDir)
{
    float theta = acos(saturate(localDir.z));
    float phi = atan2(localDir.y, localDir.x);
    if (phi < 0) phi += 2.0f * PI;

    float u = phi / (2.0f * PI);
    float v = theta / (PI * 0.5f);

    return exactMap.SampleLevel(pointSampler, float2(u, v), 0);
}

struct PsOut
{
    float4 color : SV_Target0;
};

PsOut main(float2 texCoord : TEXCOORD, float4 pos : SV_POSITION)
{
    PsOut psOut = { clearColor };

    if (targetGBufferIndex == 0xFFFFFFFF)
        return psOut;

    // Map texcoord to disc [-1, 1] centered in the output texture
    float2 localCoord = texCoord * 2.0f - 1.0f;

    // Clip to unit disc (renders a sphere via orthographic hemisphere projection)
    if (length(localCoord) > 1.0f)
        return psOut;

    // Point on hemisphere surface: z = sqrt(1 - x² - y²), z up
    float z = sqrt(1.0f - localCoord.x * localCoord.x - localCoord.y * localCoord.y);
    float3 direction = normalize(float3(localCoord.x, localCoord.y, z));

    VoxelData data = gBuffer[targetGBufferIndex];

    if (!data.IsSolid())
        return psOut;

    float shVal = evalSH(data, direction);
    float exactVal = sampleExactMap(direction);

    float value = 0;

    if (visualizationMode == 0)
    {
        // SH Approximation (smooth)
        value = shVal;
    }
    else if (visualizationMode == 1)
    {
        // Exact ground truth (from precomputed map)
        value = exactVal;
    }
    else
    {
        // Error: absolute difference between SH and exact
        value = abs(shVal - exactVal);
    }

    psOut.color = float4(funcValueToColor(value), 1.0f);
    return psOut;
}
