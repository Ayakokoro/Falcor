#include "Voxel/ABSDF.slang"
#include "Voxel/VoxelGrid.slang"
#include "VoxelizationShared.slang"
#include "Math/OctahedralMap.slang"

StructuredBuffer<VoxelData> gBuffer;
Texture2D<float> exactMap;   // 高分辨率 equirectangular 精确值
Texture2D<float> octaMap;    // 8x8 半八面体纹理（每个 texel 存该方向精确值）
SamplerState pointSampler;   // exactMap 采样（point）
SamplerState linearSampler;  // octaMap 采样（bilinear + clamp）

cbuffer CB
{
    uint targetGBufferIndex;
    uint2 mapResolution;
    uint functionType;       // 0=primitiveProjArea, 1=polygonsProjArea, 2=totalProjArea
    uint visualizationMode;  // 0=SH, 1=Exact, 2=Octahedral, 3=|SH-Exact|, 4=|Octa-Exact|
    float valueMin;
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
float sampleExactMap(float3 localDir)
{
    float theta = acos(saturate(localDir.z));
    float phi = atan2(localDir.y, localDir.x);
    if (phi < 0) phi += 2.0f * PI;

    float u = phi / (2.0f * PI);
    float v = theta / (PI * 0.5f);

    return exactMap.SampleLevel(pointSampler, float2(u, v), 0);
}

// 从 8x8 八面体纹理采样（双线性插值），与 buffer 存储方案的采样一致
float sampleOctaMap(float3 localDir)
{
    float2 uv = octaEncode(localDir);
    return octaMap.SampleLevel(linearSampler, uv, 0);
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
    float octaVal = sampleOctaMap(direction);

    float value = 0;

    if (visualizationMode == 0)
        value = shVal;
    else if (visualizationMode == 1)
        value = exactVal;
    else if (visualizationMode == 2)
        value = octaVal;
    else if (visualizationMode == 3)
        value = abs(shVal - exactVal);
    else
        value = abs(octaVal - exactVal);

    psOut.color = float4(funcValueToColor(value), 1.0f);
    return psOut;
}
