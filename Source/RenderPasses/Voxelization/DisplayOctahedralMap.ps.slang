#include "Voxel/ABSDF.slang"
#include "Voxel/VoxelGrid.slang"
#include "VoxelizationShared.slang"

StructuredBuffer<VoxelData> gBuffer;
Texture2D<float> exactMap;
SamplerState pointSampler;

cbuffer CB
{
    uint targetGBufferIndex;
    uint2 mapResolution;       // square octahedral UV map
    uint functionType;         // 0=primitiveProjArea, 1=polygonsProjArea, 2=totalProjArea
    uint visualizationMode;    // 0=octahedral approximation, 1=exact, 2=absolute error
    float valueMin;             // for color range normalization
    float valueMax;
    float4 clearColor;
}

float3 funcValueToColor(float value)
{
    float k = saturate((value - valueMin) / max(valueMax - valueMin, 1e-8f));
    // Blue -> Cyan -> Green -> Yellow -> Red.
    float3 c;
    c.r = smoothstep(0.4, 1.0, k) - smoothstep(1.0, 5.0, k);
    c.g = smoothstep(0.0, 0.4, k) - smoothstep(0.6, 1.0, k);
    c.b = smoothstep(0.4, 0.0, k);
    return c;
}

float evalProjectionArea(VoxelData data, float3 direction)
{
    if (functionType == 0)
        return data.primitiveProjAreaFunc.calc(direction);
    else if (functionType == 1)
        return data.polygonsProjAreaFunc.calc(direction);
    else
        return data.totalProjAreaFunc.calc(direction);
}

// Convert a direction in the upper hemisphere to octahedral UV coordinates.
float sampleExactMap(float3 localDir)
{
    return exactMap.SampleLevel(pointSampler, octaEncode(localDir), 0);
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

    // Map texcoord to a disc [-1, 1] centered in the output texture.
    float2 localCoord = texCoord * 2.0f - 1.0f;

    // Render the upper hemisphere as an orthographic disc.
    if (length(localCoord) > 1.0f)
        return psOut;

    float z = sqrt(1.0f - localCoord.x * localCoord.x - localCoord.y * localCoord.y);
    float3 direction = normalize(float3(localCoord.x, localCoord.y, z));

    VoxelData data = gBuffer[targetGBufferIndex];

    if (!data.IsSolid())
        return psOut;

    float approxVal = evalProjectionArea(data, direction);
    float exactVal = sampleExactMap(direction);
    float value = 0.0f;

    if (visualizationMode == 0)
    {
        value = approxVal;
    }
    else if (visualizationMode == 1)
    {
        value = exactVal;
    }
    else
    {
        value = abs(approxVal - exactVal);
    }

    psOut.color = float4(funcValueToColor(value), 1.0f);
    return psOut;
}
