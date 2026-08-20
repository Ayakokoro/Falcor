#pragma once
#include "../HostDeviceShared.slang"

BEGIN_NAMESPACE_FALCOR

#define LOBE_COUNT 8

// 上半球 4 个主导轴扇区（保留：用于将法线投影到上半球后判定扇区）
// 恰好和坐标轴平行的情况必须特殊处理，以免相隔90°的法线落入同一个lobe
inline uint NormalIndex(float3 n)
{
    float ax = abs(n.x);
    float az = abs(n.z);

    const float eps = 1e-6f;

    bool useX = false;

    if (ax > az + eps)
    {
        useX = true; // X 主导
    }
    else if (az > ax + eps)
    {
        useX = false; // Z 主导
    }
    else
    {
        // ax ~= az 的 tie-break：保证确定性
        if (n.x >= n.z)
            useX = true;
        else
            useX = false;
    }

    if (useX)
    {
        if (n.x >= 0.0f)
            return 0u; // +X
        else
            return 1u; // -X
    }
    else
    {
        if (n.z >= 0.0f)
            return 2u; // +Z
        else
            return 3u; // -Z
    }
}

// 全球面 8 个 lobe：下半球（n.y < 0）映射到高 4 位，扇区判定复用 NormalIndex（把下半球法线投影回上半球）
inline uint NormalIndex8(float3 n)
{
    uint hemi = (n.y < 0.0f) ? 4u : 0u;
    float3 up = (hemi != 0u) ? -n : n;
    return hemi + NormalIndex(up);
}

struct ABSDFInput
{
    float3 baseColor;
    float4 specular;
    float3 normal;
    float area;       // 裸表面积（用于累加 ABSDF.area 判 solid 与 ellipsoid 拟合）
    float projArea;   // 沿该面片所属 lobe 主视角的可见投影面积（用于材质/权重加权）
};

struct ABSDFLobe
{
    float weight;
    float3 normal;
    float rough;
    float3 diffuse;
    float3 specular;

    MUTATING
    void init()
    {
        weight = 0;
        normal = float3(0);
        rough = 0;
        diffuse = float3(0);
        specular = float3(0);
    }

    MUTATING
    void accumulate(ABSDFInput input)
    {
        // 可见投影面积加权（而非裸表面积）——对应 plan 式(7)(8)的视线可见材质期望
        float w = input.projArea;

        // StandardMaterial.slang
        float IoR = 1.5f;
        float f = (IoR - 1.f) / (IoR + 1.f);
        float F0 = f * f;

        diffuse += w * lerp(input.baseColor, float3(0), input.specular.b);
        specular += w * lerp(float3(F0), input.baseColor, input.specular.b);
        rough += w * input.specular.g;
        normal += w * input.normal;
        weight += w;
    }

    MUTATING
    // visibleProjectedAreaSum：该 lobe 所有可见面片沿主视角的可见投影面积和（= VPA_k）
    void normalizeSelf(float visibleProjectedAreaSum)
    {
        if (weight > 0)
        {
            normal = safeNormalize(normal);
            diffuse /= weight;
            specular /= weight;
            rough /= weight;
            // 波瓣权重 = VPA_k / 总可见投影面积，由调用方在 accumulate 前按分母归一
            weight /= visibleProjectedAreaSum;
        }
    }

    bool IsValid() { return weight > 0 && dot(normal, normal) > 0; }
};

struct ABSDF
{
    ABSDFLobe lobes[LOBE_COUNT];
    float area;

    MUTATING
    void init()
    {
        for (int i = 0; i < LOBE_COUNT; i++)
        {
            lobes[i].init();
        }
        area = 0;
    }

    MUTATING
    void accumulate(ABSDFInput input)
    {
        // 双面材质：调用方（AnalyzePolygon.calc）对正面/背面分别调用，
        // 通过 input.normal 的符号（+n / -n）让 NormalIndex8 归入对应的正/反 lobe。
        // 注意：area（总表面积）由调用方独立累加，这里只做 lobe 材质聚合。
        uint index = NormalIndex8(input.normal);
        lobes[index].accumulate(input);
    }

    MUTATING
    void normalizeSelf()
    {
        if (area == 0)
            return;
        // 分母用总可见投影面积（= Σ_k VPA_k），使 Σ_k w_k == 1
        float totalVisibleProjArea = 0;
        for (int i = 0; i < LOBE_COUNT; i++)
            totalVisibleProjArea += lobes[i].weight;

        for (int i = 0; i < LOBE_COUNT; i++)
        {
            lobes[i].normalizeSelf(totalVisibleProjArea);
        }
    }

    MUTATING
    void normalizeWeightsOnly()
    {
        float totalWeight = 0.0f;

        for (uint i = 0; i < LOBE_COUNT; ++i)
            totalWeight += lobes[i].weight;

        if (totalWeight > 0.0f)
        {
            const float targetWeightSum = 2.0f; // 双面 ±n
            float scale = targetWeightSum / totalWeight;

            for (uint i = 0; i < LOBE_COUNT; ++i)
                lobes[i].weight *= scale;
        }
    }
};

enum class ABSDFDrawMode : uint32_t
{
    Default,
    Normal,
    Ellipsoid,
    PrimitiveProjArea,
    PolygonsProjArea,
    TotalProjArea,
    Coverage,
    InternalVisibility,
    TotalArea,
    LODHeatmap,
    LODLevel,
};
FALCOR_ENUM_INFO(
    ABSDFDrawMode,
    {
        { ABSDFDrawMode::Default, "Default" },
        { ABSDFDrawMode::Normal, "Normal" },
        { ABSDFDrawMode::Ellipsoid, "Ellipsoid" },
        { ABSDFDrawMode::PrimitiveProjArea, "PrimitiveProjArea" },
        { ABSDFDrawMode::PolygonsProjArea, "PolygonsProjArea" },
        { ABSDFDrawMode::TotalProjArea, "TotalProjArea" },
        { ABSDFDrawMode::Coverage, "Coverage" },
        { ABSDFDrawMode::InternalVisibility, "InternalVisibility" },
        { ABSDFDrawMode::TotalArea, "TotalArea" },
        { ABSDFDrawMode::LODHeatmap, "LODHeatmap" },
        { ABSDFDrawMode::LODLevel, "LODLevel" },
    }
);
FALCOR_ENUM_REGISTER(ABSDFDrawMode);

#if !defined(HOST_CODE)
inline float3 valueToColor(float value)
{
    if (value <= 1)
        return float3(value);
    return float3(value, 2 - value, 2 - value);
}

inline float3 pdfToColor(float pdf, float refPdf)
{
    float k = pdf / refPdf;
    // 蓝(0)->青->绿->黄->红(1)->黑(5)渐变
    float3 c;
    c.r = smoothstep(0.4, 1.0, k) - smoothstep(1.0, 5.0, k);
    c.g = smoothstep(0.0, 0.4, k) - smoothstep(0.6, 1.0, k);
    c.b = smoothstep(0.4, 0.0, k);
    return c;
}
#endif

END_NAMESPACE_FALCOR
