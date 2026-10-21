#pragma once
#include "Shading.slang"
#include "VisibilityMask.slang"

static const uint VISIBLE_LOBE_COUNT = 8;

struct VisibleLobeBRDF
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
        // 前置条件：dot(v,n)>=0（VisibleSurfaceBRDF::sample 只选择前向 lobe）
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

struct VisibleSurfaceBRDF
{
    VisibleLobeBRDF lobes[VISIBLE_LOBE_COUNT];
    float weights[VISIBLE_LOBE_COUNT];
    uint lobeCount;

#if !defined(HOST_CODE)

    // Stored weights are visible-area amplitudes, not view-conditional
    // mixture weights. Project them and normalize only over the visible view.
    float projectedWeight(uint i, float3 v)
    {
        return weights[i] * max(0.0f, dot(lobes[i].n, v));
    }

    // Sum of projected areas of all view-facing lobes.
    float frontWeightSum(float3 v)
    {
        float sum = 0;
        for (uint i = 0; i < lobeCount; i++)
            sum += projectedWeight(i, v);
        return sum;
    }

    // 在前向 lobe 中按（重归一化的）权重随机选一个
    uint getRandomFrontLobe(float uLobe, float3 v, float frontSum)
    {
        for (uint i = 0; i < lobeCount; i++)
        {
            float projected = projectedWeight(i, v);
            if (projected > 0)
            {
                uLobe -= projected / frontSum;
                if (uLobe <= 0)
                    return i;
            }
        }
        // fallback：无命中时返回第一个前向 lobe
        for (uint i = 0; i < lobeCount; i++)
            if (projectedWeight(i, v) > 0)
                return i;
        return 0;
    }

    float calcBRDFWeight(float3 v)
    {
        float frontSum = frontWeightSum(v);
        if (frontSum <= 0.0f)
            return 0.0f;
        float sum = 0;
        for (uint i = 0; i < lobeCount; i++)
        {
            sum += (projectedWeight(i, v) / frontSum) * lobes[i].evalBRDFWeight();
        }
        return saturate(sum);
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

    // Conditional material average over the surface visible from v. Do not
    // renormalize by light-facing lobes: unlit visible faces contribute zero.
    float3 eval(float3 l, float3 v, float3 h)
    {
        float frontSum = frontWeightSum(v);
        if (frontSum <= 0.0f)
            return 0.0f;
        float3 sum = 0;
        for (uint i = 0; i < lobeCount; i++)
        {
            sum += (projectedWeight(i, v) / frontSum) * lobes[i].eval(l, v, h);
        }
        return sum;
    }

    // Preserve the same view-conditioned material mixture as eval()/evalPdf().
    // Visibility belongs inside the lobe sum; it must not renormalize weights.
    float3 evalVisible(float3 l, float3 v, float3 h, VisibilityMaskPair pair)
    {
        float frontSum = frontWeightSum(v);
        if (frontSum <= 0.0f)
            return 0.0f;
        float3 sum = 0;
        for (uint i = 0; i < lobeCount; i++)
        {
            float projected = projectedWeight(i, v);
            if (projected <= 0.0f || dot(lobes[i].n, l) <= 0.0f)
                continue;
            VisibilityMaskQuery visibility = queryVisibilityMask(pair, i, lobeCount);
            sum += (projected / frontSum) * lobes[i].eval(l, v, h) * visibility.value;
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
            float projected = projectedWeight(i, v);
            if (projected > 0.0f)
                pdf += (projected / frontSum) * lobes[i].evalPdf(l, v, h);
        }
        return pdf;
    }

    // Choose the dominant lobe that is both visible from v and light-facing.
    // This normal also controls the shadow-ray origin bias.
    float3 getMainNormal(float3 direction, float3 v)
    {
        float3 best = 0;
        float maxWeight = 0;
        for (int i = 0; i < lobeCount; i++)
        {
            float score = projectedWeight(i, v) * max(0.0f, dot(lobes[i].n, direction));
            if (score > maxWeight)
            {
                best = lobes[i].n;
                maxWeight = score;
            }
        }
        if (maxWeight > 0.0f)
            return best;

        // No lobe is lit. Prefer one visible from the current view, then
        // retain the old direction-facing fallback for degenerate data.
        maxWeight = 0;
        for (int i = 0; i < lobeCount; i++)
        {
            float score = projectedWeight(i, v);
            if (score > maxWeight)
            {
                best = lobes[i].n;
                maxWeight = score;
            }
        }
        if (maxWeight == 0.0f)
        {
            for (int i = 0; i < lobeCount; i++)
            {
                if (weights[i] > maxWeight)
                {
                    best = lobes[i].n;
                    maxWeight = weights[i];
                }
            }
        }
        if (dot(best, direction) < 0)
            best = -best;
        return best;
    }
#endif
};

struct VisibleTEBSDF
{
    VisibleSurfaceBRDF surface;
#if !defined(HOST_CODE)
    DeltaBTDF delta; // GPU上，没有成员变量的结构体不占空间(人为添加无意义的成员变量也会被优化掉)
#endif
    ProjectionAreaFunc primitiveProjAreaFunc;
    ProjectionAreaFunc polygonsProjAreaFunc; // 所有多边形的可见投影面积
    ProjectionAreaFunc totalProjAreaFunc;    // 所有多边形的总投影面积
    float area;

    float coverage; // 临时变量

#if !defined(HOST_CODE)
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

    // 自遮挡导致入射光减少
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
        return coverage * surface.eval(l, v, h) * calcInternalVisibility(l);
    }

    float evalPdf(float3 l, float3 v, float3 h)
    {
        if (coverage < 1 && all(h == 0))
            return (1 - coverage) * delta.evalPdf(l, v, h);
        return coverage * surface.evalPdf(l, v, h);
    }
#endif
};

