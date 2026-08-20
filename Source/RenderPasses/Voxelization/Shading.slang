#pragma once
#include "VoxelizationShared.slang"

#if !defined(HOST_CODE)
#include "Math/Random.slang"
import Utils.Math.MathHelpers;

const static float Pi = 3.14159265359;
const static float InvPi = 0.318309886183790671538;
const static float MinRough = 0.01;

inline float pow5(float x)
{
    x = saturate(x);
    float x2 = x * x;
    return x2 * x2 * x;
}
inline float3 calcSchlick(float3 f0, float3 f90, float cosTheta)
{
    return lerp(f0, f90, pow5(1 - cosTheta));
}
inline float calcSchlick(float f0, float f90, float cosTheta)
{
    return lerp(f0, f90, pow5(1 - cosTheta));
}

inline float ggx(float NdotH, float alpha)
{
    float alpha2 = alpha * alpha;
    float temp = NdotH * NdotH * (alpha2 - 1) + 1;
    temp = max(1e-6, temp);
    return alpha2 / Pi / temp / temp;
}
inline float3 fresnel(float3 v, float3 h, float3 spec)
{
    return calcSchlick(spec, float3(1, 1, 1), dot(v, h));
}

float Lambda(float NdotX, float alpha)
{
    float temp = alpha * alpha + (1 - alpha * alpha) * NdotX * NdotX;
    temp = safeSqrt(temp) / max(1e-6, NdotX) - 1;
    return temp / 2;
}

inline float3 sampleGGX_VNDF(float3 v, float alpha, float2 u)
{
    // 1) stretch view
    float3 Vh = normalize(float3(alpha * v.x, alpha * v.y, v.z));

    // 2) build orthonormal basis around Vh
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = (lensq > 0.0f) ? float3(-Vh.y, Vh.x, 0.0f) * rsqrt(lensq) : float3(1, 0, 0);
    float3 T2 = cross(Vh, T1);

    // 3) uniform sample on disk (no trig)
    float2 p = sample_disk_concentric(u);
    float t1 = p.x;
    float t2 = p.y;

    // 4) Heitz “visible normals” warp
    float s = 0.5f * (1.0f + Vh.z);
    t2 = lerp(safeSqrt(max(0.0f, 1.0f - t1 * t1)), t2, s);

    // 5) reproject to hemisphere
    float3 Nh = t1 * T1 + t2 * T2 + safeSqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // 6) unstretch
    return normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0f, Nh.z)));
}

#endif

struct FrostbiteCosBRDF
{
    float3 diffuse; // Lambert反射率
    float rough;

#if !defined(HOST_CODE)
    [mutating]
    void init(float3 diffuse, float rough)
    {
        this.diffuse = diffuse;
        this.rough = rough;
    }

    float3 eval(float3 l, float3 v, float3 h)
    {
        if (l.z <= 0 || v.z <= 0)
            return 0;
        float VdotH = dot(v, h);
        float energyBias = lerp(0.f, 0.5f, rough);
        float energyFactor = lerp(1.f, 1.f / 1.51f, rough);
        float fd90 = energyBias + 2.f * VdotH * VdotH * rough;
        float fd0 = 1.f;
        float wiScatter = calcSchlick(fd0, fd90, l.z);
        float woScatter = calcSchlick(fd0, fd90, v.z);
        return wiScatter * woScatter * energyFactor * diffuse / Pi * l.z;
    }

    void sample<T : ISampleGenerator>(float3 v, out float3 l, out float3 h, inout T sg)
    {
        float2 d = sample_disk_concentric(Next2(sg));
        float z = safeSqrt(1.f - dot(d, d));
        l = float3(d, z);
        h = normalize(l + v);
    }

    float evalPdf(float3 l, float3 v, float3 h)
    {
        if (l.z <= 0 || v.z <= 0)
            return 0;
        return InvPi * l.z;
    }
#endif
};

struct CookTorrenceCosBRDF
{
    float3 specular; // Fresnel反射率
    float alpha;

#if !defined(HOST_CODE)
    [mutating]
    void init(float3 specular, float rough)
    {
        this.specular = specular;
        alpha = rough * rough;
    }

    float3 eval(float3 l, float3 v, float3 h)
    {
        if (l.z <= 0 || v.z <= 0)
            return 0.f;
        float3 specColor = ggx(h.z, alpha) * fresnel(v, h, specular) / (1 + Lambda(l.z, alpha) + Lambda(v.z, alpha)) / 4 / max(1e-6, v.z);
        return specColor;
    }

    void sample<T : ISampleGenerator>(float3 v, out float3 l, out float3 h, inout T sg)
    {
        float2 u = Next2(sg);
        h = sampleGGX_VNDF(v, alpha, u);
        float VdotH = dot(v, h);
        l = reflect(-v, h);
    }

    float evalPdf(float3 l, float3 v, float3 h)
    {
        if (l.z <= 0 || v.z <= 0)
            return 0;
        float VdotH = dot(v, h);
        if (VdotH <= 0)
            return 0;
        float D = ggx(h.z, alpha);
        float G1 = 1.0 / (1.0 + Lambda(v.z, alpha));
        return D * G1 * h.z / (4.0 * max(1e-6, v.z));
    }
#endif
};

// 从此结构体开始接收和返回世界空间方向；8 个单面 lobe 各对应一个确定法线（不做双面折叠）
struct LobeBRDF
{
    FrostbiteCosBRDF frostBite;
    CookTorrenceCosBRDF cookTorrence;

    float3 t, b, n;
    float specularWeight;

#if !defined(HOST_CODE)
    float evalSpecularWeight()
    {
        float3 specular = cookTorrence.specular;
        float specularWeight = (20 * (specular.x + specular.y + specular.z) / 3 + 1) / 21.0;
        return saturate(specularWeight);
    }

    float evalBRDFWeight()
    {
        float glossy = (1 - frostBite.rough) * evalSpecularWeight();
        return lerp(0.2, 0.8, glossy);
    }

    [mutating]
    void init(ABSDFLobe lobe)
    {
        n = lobe.normal;
        buildTBN(t, b, n);
        float rough = max(lobe.rough, MinRough);
        frostBite.init(lobe.diffuse, rough);
        cookTorrence.init(lobe.specular, rough);
        specularWeight = evalSpecularWeight();
    }

    // 单面：背面视线（dot(v,n)<0）时由底层 v.z<=0 判据返回 0
    float3 eval(float3 l, float3 v, float3 h)
    {
        l = worldToLocal(l, t, b, n);
        v = worldToLocal(v, t, b, n);
        h = worldToLocal(h, t, b, n);
        return frostBite.eval(l, v, h) + cookTorrence.eval(l, v, h);
    }

    void sample<T : ISampleGenerator>(float3 v, out float3 l, out float3 h, inout T sg)
    {
        // 前置条件：dot(v,n)>=0（SurfaceBRDF::sample 只选择前向 lobe）
        v = worldToLocal(v, t, b, n);

        float uEnergy = Next(sg);
        if (uEnergy <= specularWeight)
            cookTorrence.sample(v, l, h, sg);
        else
            frostBite.sample(v, l, h, sg);

        l = localToWorld(l, t, b, n);
        h = localToWorld(h, t, b, n);
    }

    float evalPdf(float3 l, float3 v, float3 h)
    {
        l = worldToLocal(l, t, b, n);
        v = worldToLocal(v, t, b, n);
        h = worldToLocal(h, t, b, n);
        return (1 - specularWeight) * frostBite.evalPdf(l, v, h) + specularWeight * cookTorrence.evalPdf(l, v, h);
    }
#endif
};

struct SurfaceBRDF
{
    LobeBRDF lobes[LOBE_COUNT];
    float weights[LOBE_COUNT];
    uint lobeCount;

#if !defined(HOST_CODE)
    [mutating]
    void init(ABSDF ABSDF)
    {
        lobeCount = 0;
        for (uint i = 0; i < LOBE_COUNT; i++)
        {
            ABSDFLobe lobe = ABSDF.lobes[i];
            if (lobe.IsValid())
            {
                lobes[lobeCount].init(lobe);
                weights[lobeCount] = lobe.weight;
                lobeCount++;
            }
        }
        if (lobeCount == 0)
        {
            // 防止没有有效lobes
            ABSDFLobe defaultLobe;
            defaultLobe.init();
            defaultLobe.normal = float3(0, 0, 1);
            lobes[0].init(defaultLobe);
            weights[0] = 1;
            lobeCount = 1;
        }
    }

    // 当前视线 v 下所有前向 lobe（dot(n,v)>0）的权重和
    float frontWeightSum(float3 v)
    {
        float sum = 0;
        for (uint i = 0; i < lobeCount; i++)
        {
            if (dot(lobes[i].n, v) > 0)
                sum += weights[i];
        }
        return sum;
    }

    // 在前向 lobe 中按（重归一化的）权重随机选一个
    uint getRandomFrontLobe(float uLobe, float3 v, float frontSum)
    {
        for (uint i = 0; i < lobeCount; i++)
        {
            if (dot(lobes[i].n, v) > 0)
            {
                uLobe -= weights[i] / frontSum;
                if (uLobe <= 0)
                    return i;
            }
        }
        // fallback：无命中时返回第一个前向 lobe
        for (uint i = 0; i < lobeCount; i++)
            if (dot(lobes[i].n, v) > 0)
                return i;
        return 0;
    }

    float calcBRDFWeight()
    {
        float sum = 0;
        for (uint i = 0; i < lobeCount; i++)
        {
            sum += weights[i] * lobes[i].evalBRDFWeight();
        }
        return sum;
    }

    // 可视化法线分布时，要渲染的点在球面上的方向充当h即可（单面：仅自身半球贡献）
    inline float evalNormalDistribution(float3 h)
    {
        float sum = 0;
        for (int i = 0; i < lobeCount; i++)
        {
            float NdotH = dot(lobes[i].n, h);
            if (NdotH <= 0)
                continue;
            float alpha = lobes[i].cookTorrence.alpha;
            sum += weights[i] * ggx(NdotH, alpha);
        }
        return sum;
    }

    // 物理 BRDF：sum_k w_k f_k，背向 lobe 底层 v.z<=0 自归零（权重不重归一）
    float3 eval(float3 l, float3 v, float3 h)
    {
        float3 sum = 0;
        for (uint i = 0; i < lobeCount; i++)
        {
            sum += weights[i] * lobes[i].eval(l, v, h);
        }
        return sum;
    }

    void sample<T : ISampleGenerator>(float3 v, out float3 l, out float3 h, inout T sg)
    {
        float frontSum = frontWeightSum(v);
        if (frontSum <= 0)
        {
            // 无前向 lobe（背面视线）：返回无效样本，evalPdf=0
            l = -v;
            h = 0;
            return;
        }
        float uLobe = Next(sg);
        uint index = getRandomFrontLobe(uLobe, v, frontSum);
        lobes[index].sample(v, l, h, sg);
    }

    // 采样密度：与 sample 一致，前向 lobe 权重按 frontSum 重归一
    float evalPdf(float3 l, float3 v, float3 h)
    {
        float frontSum = frontWeightSum(v);
        if (frontSum <= 0)
            return 0;
        float pdf = 0;
        for (uint i = 0; i < lobeCount; i++)
        {
            if (dot(lobes[i].n, v) > 0)
                pdf += (weights[i] / frontSum) * lobes[i].evalPdf(l, v, h);
        }
        return pdf;
    }

    // 返回朝向给定方向（n·dir>0）且权重最大的法线；若无前向 lobe，退回最高权重 lobe 并翻转朝向
    float3 getMainNormal(float3 direction)
    {
        float3 best = 0;
        float maxWeight = 0;
        bool found = false;
        for (int i = 0; i < lobeCount; i++)
        {
            if (dot(lobes[i].n, direction) > 0 && weights[i] > maxWeight)
            {
                best = lobes[i].n;
                maxWeight = weights[i];
                found = true;
            }
        }
        if (found)
            return best;

        maxWeight = 0;
        for (int i = 0; i < lobeCount; i++)
        {
            if (weights[i] > maxWeight)
            {
                best = lobes[i].n;
                maxWeight = weights[i];
            }
        }
        if (dot(best, direction) < 0)
            best = -best;
        return best;
    }
#endif
};

#if !defined(HOST_CODE)
struct DeltaBTDF
{
    float3 eval(float3 l, float3 v, float3 h) { return 1; }

    void sample<T : ISampleGenerator>(float3 v, out float3 l, out float3 h, inout T sg)
    {
        l = -v;
        h = 0;
    }

    float evalPdf(float3 l, float3 v, float3 h)
    {
        return 1; // 实质上是无穷大
    }
};
#endif

struct TEBSDF
{
    SurfaceBRDF surface;
#if !defined(HOST_CODE)
    DeltaBTDF delta; // GPU上，没有成员变量的结构体不占空间(人为添加无意义的成员变量也会被优化掉)
#endif
    SphericalFunc primitiveProjAreaFunc;
    SphericalFunc polygonsProjAreaFunc; // 所有多边形的可见投影面积
    SphericalFunc totalProjAreaFunc;    // 所有多边形的总投影面积
    float area;

    float coverage; // 临时变量

#if !defined(HOST_CODE)
    [mutating]
    void init(VoxelData data)
    {
        area = data.ABSDF.area;
        if (area > 0)
        {
            primitiveProjAreaFunc = data.primitiveProjAreaFunc;
            polygonsProjAreaFunc = data.polygonsProjAreaFunc;
            totalProjAreaFunc = data.totalProjAreaFunc;
            surface.init(data.ABSDF);
        }
    }

    float calcCoverage(float3 direction, float coverageBlend)
    {
#if !CHECK_COVERAGE
        return 1;
#endif
        float primitiveProjArea = primitiveProjAreaFunc.calc(direction);
        if (primitiveProjArea <= 0)
            return 1;
        float polygonsProjArea = polygonsProjAreaFunc.calc(direction);
        float rawCoverage = polygonsProjArea / primitiveProjArea;
        float emptyRatio = (primitiveProjArea - polygonsProjArea) / max(primitiveProjArea, 1e-8f);
        float coverage = rawCoverage + coverageBlend * emptyRatio;
        return saturate(coverage);
    }
    // 采样前必须调用此函数
    [mutating]
    void updateCoverage(float3 direction, float threshould, float coverageBlend)
    {
#if !CHECK_COVERAGE
        coverage = 1;
        return;
#endif
        coverage = calcCoverage(direction, coverageBlend);
        if (1 - coverage < threshould)
            coverage = 1;
    }

    // 自遮挡导致入射光减少（单向，退化形式：Λ_o=0 时的 VPA/TPA）
    float calcInternalVisibility(float3 direction)
    {
#if !CHECK_VISIBILITY
        return 1.f;
#endif
        float totalProjArea = totalProjAreaFunc.calc(direction);
        if (totalProjArea <= 0)
            return 1;
        float polygonsProjArea = polygonsProjAreaFunc.calc(direction);
        return saturate(polygonsProjArea / totalProjArea);
    }

    // Λ(ω) = (TPA(ω) - VPA(ω)) / VPA(ω)，Heitz 高度相关 Smith 的阴影因子
    float calcLambda(float3 direction)
    {
        float vpa = polygonsProjAreaFunc.calc(direction);
        if (vpa <= 0)
            return 1e6f; // 完全遮挡 → Λ 极大，双向可见性趋于 0
        float tpa = totalProjAreaFunc.calc(direction);
        return max(0, (tpa - vpa) / vpa);
    }

    // 双向可见性（视线受光率）：V_vis(ωi, ωo) = (1 + Λ(ωo)) / (1 + Λ(ωi) + Λ(ωo))
    float calcVisibility(float3 wi, float3 wo)
    {
#if !CHECK_VISIBILITY
        return 1.f;
#endif
        float Li = calcLambda(wi);
        float Lo = calcLambda(wo);
        return (1.0f + Lo) / (1.0f + Li + Lo);
    }

    void sample<T : ISampleGenerator>(float3 v, out float3 l, out float3 h, inout T sg)
    {
        if (coverage == 1 || Next(sg) <= coverage)
            surface.sample(v, l, h, sg);
        else
            delta.sample(v, l, h, sg);
    }

    float3 eval(float3 l, float3 v, float3 h)
    {
        if (coverage < 1 && all(h == 0))
            return (1 - coverage) * delta.eval(l, v, h);
        return coverage * surface.eval(l, v, h) * calcVisibility(l, v);
    }

    float evalPdf(float3 l, float3 v, float3 h)
    {
        if (coverage < 1 && all(h == 0))
            return (1 - coverage) * delta.evalPdf(l, v, h);
        return coverage * surface.evalPdf(l, v, h);
    }
#endif
};

