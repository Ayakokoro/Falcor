#include "Math/VoxelizationUtility.slang"
#include "Shading.slang"
#include "Voxel/VoxelGrid.slang"
#include "VoxelizationShared.slang"

StructuredBuffer<TEBSDF> gBuffer0;
StructuredBuffer<TEBSDF> gBuffer1;
StructuredBuffer<TEBSDF> gBuffer2;
StructuredBuffer<float4> selectedVoxel; // 前三个分量表示相机指向体素的方向

cbuffer CB
{
    float4 clearColor;
    uint gbSplit0;
    uint gbSplit1;
}

TEBSDF loadGBuffer(uint globalIdx, uint split0, uint split1)
{
    if (globalIdx < split0) return gBuffer0[globalIdx];
    else if (globalIdx < split1) return gBuffer1[globalIdx - split0];
    else return gBuffer2[globalIdx - split1];
}

struct PsOut
{
    float4 color : SV_Target0;
};

float3 calcColor(float2 localCoord, int offset, float3 v, uint gbSplit0, uint gbSplit1)
{
    SurfaceBRDF BRDF = loadGBuffer(offset, gbSplit0, gbSplit1).surface;
    float3 t, b;
    buildTBN(t, b, v);

    float z = sqrt(1 - localCoord.x * localCoord.x - localCoord.y * localCoord.y);
    // 中心对称，显示背面没有意义
    float3 direction = float3(localCoord.x, localCoord.y, z);
    direction = localToWorld(direction, t, b, v);
    float pdf = BRDF.evalNormalDistribution(direction);
    float rough = 0.5f; // 参考值
    float refPdf = 1.0 / (Pi * rough * rough);
    return pdfToColor(pdf, refPdf);
}

PsOut main(float2 texCoord: TEXCOORD, float4 pos: SV_POSITION)
{
    PsOut psOut = { clearColor };
    float4 vo = selectedVoxel[0];
    if (vo.w > 0)
    {
        float2 localCoord = (texCoord - float2(0.25f, 1.0f / 18)) / float2(0.5f, 16.0f / 18); // 默认窗口为16:9
        localCoord = localCoord * 2 - 1;                                                      // 映射到[-1,1]
        if (length(localCoord) <= 1)
        {
            psOut.color = float4(calcColor(localCoord, (int)vo.w, vo.xyz, gbSplit0, gbSplit1), 1);
        }
    }
    return psOut;
}
