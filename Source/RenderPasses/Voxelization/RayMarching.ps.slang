#include "Voxel/VoxelGrid.slang"
#include "Math/VoxelizationUtility.slang"
#include "VoxelizationShared.slang"
#include "MixLightSampler.slang"
import Rendering.Materials.StandardMaterial;
import Utils.Sampling.UniformSampleGenerator;

// 如果外部未定义,则取默认值3
#ifndef MAX_BOUNCE
#define MAX_BOUNCE 3
#endif

StructuredBuffer<TEBSDF> gBuffer;
StructuredBuffer<Ellipsoid> pBuffer;

RWStructuredBuffer<float4> selectedVoxel;

ParameterBlock<OccupancyMap> OM;

cbuffer GridData
{
    float3 gridMin;
    float3 voxelSize;
    uint3 voxelCount;
}

cbuffer CB
{
    uint2 pixelCount;
    uint3 blockCount;
    float4x4 invVP;
    float shadowBias;
    uint drawMode;
    uint sampleStrategy;
    uint frameIndex;
    float minPdf;
    float transmittanceThreshold;
    bool renderBackGround;
    float4 clearColor;
    uint2 selectedPixel;
}

struct HitResult
{
    bool hit;
    int3 cellInt;
    float3 cell;
};

struct PsOut
{
    float4 color : SV_Target0;
};

bool outsideGrid(int3 cellInt)
{
    return any(cellInt < int3(0)) || any(cellInt >= (int3)voxelCount);
}

bool isSolidVoxel(int3 cellInt, inout int offset)
{
    return OM.isSolidVoxel(cellInt, offset) && gBuffer[offset].area > 0;
}

// 先后用体素和椭球裁剪线段
HitResult rayHitTE(Ellipsoid ellipsoid, float3 cell, float3 direction, int3 cellInt)
{
    float3 from = cell - cellInt;
    int3 directions = int3(sign(direction));
    float3 border = (directions + 1) / 2;
    float3 t = (border - from) / direction;
    float tMin = 2.0f;
    if (direction.x != 0 && t.x < tMin)
        tMin = t.x;
    if (direction.y != 0 && t.y < tMin)
        tMin = t.y;
    if (direction.z != 0 && t.z < tMin)
        tMin = t.z;
    float3 to = from + tMin * direction;
    float2 inOut = ellipsoid.clip(from, to);
    HitResult result = { inOut.x >= 0, cellInt, cell + inOut.x * (to - from) };
    return result;
}

// 射线命中体素，可能直接穿过，或命中椭球
HitResult rayHitVoxel(float3 cell, float3 direction, int3 cellInt)
{
    int offset;
    if (isSolidVoxel(cellInt, offset))
    {
#if !CHECK_ELLIPSOID
        HitResult hit = { true, cellInt, cellInt + 0.5f };
        return hit;
#else
        Ellipsoid ellipsoid = pBuffer[offset];
        HitResult hit = rayHitTE(ellipsoid, cell, direction, cellInt);
        if (hit.hit)
            return hit;
#endif
    }
    return { false, cellInt, cellInt + 0.5f };
}

HitResult rayMarching(float3 from, float3 direction, float maxDistance, bool ignoreFirst)
{
    direction = normalizeSafe(direction);
    if (all(direction == float3(0)))
    {
        HitResult hit = { false, int3(from), from };
        return hit;
    }

    float3 cell = from;
    int3 cellInt = int3(floor(from));
    float distance = 0;

    float3 reciprocal_direction = 1.0f / direction;
    int3 directions = int3(sign(direction));
    if (ignoreFirst)
        distance += leave(cell, cellInt, directions, direction, reciprocal_direction);

    while (true)
    {
        if (outsideGrid(cellInt))
            break;
        if (maxDistance > 0 && distance > maxDistance)
            break;
#if USE_MIPMAP
        // Mip Level 2: Hyper Block
        if (!OM.isSolidHyperBlock(cellInt))
        {
            distance += leaveHyperBlock(cell, cellInt, directions, direction, reciprocal_direction);
            continue;
        }
#endif
        // Mip Level 1: Block
        if (!OM.isSolidBlock(cellInt))
        {
            distance += leaveBlock(cell, cellInt, directions, direction, reciprocal_direction);
            continue;
        }

        // Mip Level 0: Voxel
        HitResult hit = rayHitVoxel(cell, direction, cellInt);
        if (hit.hit)
            return hit;
        distance += leave(cell, cellInt, directions, direction, reciprocal_direction);
    }
    HitResult hit = { false, cellInt, cellInt + 0.5f };
    return hit;
}

float calcTrasmissionVisibility(HitResult hit, float3 direction)
{
    hit = rayMarching(hit.cell, direction, -1, true);
    return hit.hit ? 0 : 1;
}

float calcLightVisibility(HitResult hit, float3 direction, float3 mainNormal, float distance)
{
    float ignoreDistance = 2; // 忽略光源附近的体素
    if (distance >= 0 && distance <= ignoreDistance)
        return 1;
    float3 from = hit.cell + shadowBias * mainNormal;
    hit = rayMarching(from, direction, distance - ignoreDistance, true);
    return hit.hit ? 0 : 1;
}

float3 coverageTest(HitResult hit, uint offset, float3 direction)
{
    float coverage = 0;
    float times = MAX_BOUNCE;
    while (times > 0)
    {
        TEBSDF bsdf = gBuffer[offset];
        coverage += (1 - coverage) * bsdf.calcCoverage(direction);
        if (coverage >= 1 - transmittanceThreshold)
        {
            coverage = 1;
            break;
        }
        hit = rayMarching(hit.cell, direction, -1, true);
        if (!hit.hit)
            break;
        offset = OM.vBuffer[hit.cellInt];
        times--;
    }
    return valueToColor(coverage);
}

float3 debugShading(HitResult hit, uint offset, float3 v)
{
    TEBSDF bsdf = gBuffer[offset];
    Ellipsoid e = pBuffer[offset];
    float result = 0;
    switch (drawMode)
    {
    case ABSDFDrawMode::Normal:
        return bsdf.surface.getMainNormal(v) * 0.5 + 0.5;
    case ABSDFDrawMode::Ellipsoid:
        float3 localPos = hit.cell - hit.cellInt;
        float3 d = localPos - e.center;
        float3 p = d - v * dot(d, v);
        float3 Bp = mul(e.B, p);
        float3 Bn = mul(e.B, v);
        float denom = max(dot(v, Bn), 1e-8f);
        float val = dot(p, Bp) - (dot(p, Bn) * dot(p, Bn)) / denom;
        float t = 1.0f - safeSqrt(val);
        return pdfToColor(t, 1.0f);
    case ABSDFDrawMode::PolygonsProjArea:
        result = bsdf.polygonsProjAreaFunc.calc(v);
        break;
    case ABSDFDrawMode::TotalProjArea:
        result = bsdf.totalProjAreaFunc.calc(v);
        break;
    case ABSDFDrawMode::PrimitiveProjArea:
        result = bsdf.primitiveProjAreaFunc.calc(v);
        break;
    case ABSDFDrawMode::InternalVisibility:
        result = bsdf.calcInternalVisibility(v);
        break;
    case ABSDFDrawMode::TotalArea:
        result = bsdf.area;
        break;
    default:
        break;
    }
    return valueToColor(result);
}

float3 shading(uint offset, float3 l, float3 v, float3 h, float3 lightColor)
{
    TEBSDF BSDF = gBuffer[offset];
    BSDF.updateCoverage(v, transmittanceThreshold);
    return lightColor * BSDF.eval(l, v, h);
}

// 随机采样方向，不确定光源是否可见
float3 shadingDirect(HitResult hit, uint offset, float3 v, inout UniformSampleGenerator sg)
{
    TEBSDF BSDF = gBuffer[offset];
    float3 posW = gridMin + hit.cell * voxelSize;
    LightSample ls = {};
    bool valid = sampleLight(posW, v, BSDF, sg, ls);

    if (!valid || ls.pdf <= 0 || all(ls.h == 0))
        return 0;

    ls.pdf = max(ls.pdf, minPdf);
    float3 mainNormal = BSDF.surface.getMainNormal(ls.l);
    float visibility = calcLightVisibility(hit, ls.l, mainNormal, ls.distance / voxelSize.x);

    if (visibility > 0)
        return visibility * shading(offset, ls.l, v, ls.h, ls.lightColor) / ls.pdf;
    return 0;
}

float3 shadingGlobal(HitResult hit, float3 v, inout UniformSampleGenerator sg)
{
    // 非递归版本，先计算所有命中点，再倒回来统计
    HitResult[MAX_BOUNCE + 1] hits;
    float3[MAX_BOUNCE + 1] ls; // 第i个采样方向即第i个体素指向第i+1个体素的方向
    float3[MAX_BOUNCE + 1] hs;
    float[MAX_BOUNCE + 1] pdfs; // 第i个pdf即对第i个体素的BSDF采样得到的权重
    uint bounce = MAX_BOUNCE;
    uint shadingPointCount = bounce + 1; // 要着色的点的个数，未中断时为primary hit + bounce
    hits[0] = hit;
    bool leaveGrid = false;
    for (int i = 0; i < bounce; i++)
    {
        uint offset = OM.vBuffer[hits[i].cellInt];
        TEBSDF BSDF = gBuffer[offset];
        float3 currentV = i == 0 ? v : -ls[i - 1];
        BSDF.updateCoverage(currentV, transmittanceThreshold);
        BSDF.sample(currentV, ls[i], hs[i], sg);

        pdfs[i] = BSDF.evalPdf(ls[i], currentV, hs[i]);
        if (pdfs[i] <= 0)
        {
            shadingPointCount = i + 1; // 0,1,...,i;第i+1个点到第i个点的间接光照无法计算，但仍要计算直接光照
            break;
        }
        pdfs[i] = max(pdfs[i], minPdf);
        hits[i + 1] = rayMarching(hits[i].cell, ls[i], -1, true);
        if (!hits[i + 1].hit)
        {
            leaveGrid = true;
            shadingPointCount = i + 1; // 0,1,...,i
            break;
        }
    }

    float3 reflectLights[MAX_BOUNCE + 1];
    for (int i = shadingPointCount - 1; i >= 0; i--)
    {
        reflectLights[i] = 0;
        HitResult currentHit = hits[i];
        uint currentOffset = OM.vBuffer[currentHit.cellInt];
        float3 currentV = i == 0 ? v : -ls[i - 1];

        reflectLights[i] += shadingDirect(currentHit, currentOffset, currentV, sg); // 直接光照
        if (i == shadingPointCount - 1)                                             // 最后一个着色点不计算间接光照
        {
#if USE_ENV_MAP
            if (leaveGrid && all(hs[i] == 0))
            {
                float3 incidentLight = gEnvMapSampler.eval(ls[i]);
                reflectLights[i] += shading(currentOffset, ls[i], currentV, hs[i], incidentLight) / pdfs[i]; // 透射环境光
            }
#endif
        }
        else
        {
            if (pdfs[i] <= 0)
                continue;
            HitResult nextHit = hits[i + 1];
            if (nextHit.hit)
            {
                uint nextOffset = OM.vBuffer[nextHit.cellInt];
                float3 incidentLight = reflectLights[i + 1];
                reflectLights[i] += shading(currentOffset, ls[i], currentV, hs[i], incidentLight) / pdfs[i]; // 间接光照
            }
        }
    }
    return reflectLights[0];
}

float3 screenCoordToCell(float2 screenCoord, float NDCDepth)
{
    float4 NDC = float4(2 * screenCoord.x - 1, 1 - 2 * screenCoord.y, NDCDepth, 1);
    float4 temp = mul(invVP, NDC);
    float4 posWorld = temp / temp.w;
    float3 cell = (posWorld.xyz - gridMin) / voxelSize;
    return cell;
}
float2 randomJitter()
{
    UniformSampleGenerator sg = UniformSampleGenerator(0, frameIndex);
    float2 jitter = Next2(sg) - 0.5f;
    jitter /= pixelCount;
    return jitter;
}
float4 renderBackground(float3 direction)
{
#if USE_ENV_MAP
    if (renderBackGround)
    {
        float3 color = gScene.envMap.eval(direction);
        return float4(color, 1);
    }
#endif
    return clearColor;
}

PsOut main(float2 texCoord: TEXCOORD, float4 pos: SV_POSITION)
{
    PsOut psOut = {};
    uint2 pixel = uint2(texCoord * pixelCount);
    float2 jitter = randomJitter();
    texCoord = saturate(texCoord + jitter);
    float3 from = screenCoordToCell(texCoord, 0);
    float3 to = screenCoordToCell(texCoord, 1);
    float3 delta = to - from;
    float3 direction = normalize(delta); // 网格/世界空间下的方向相同
    psOut.color = renderBackground(direction);

    float2 inOut = clip(float3(0), float3(voxelCount), from, to);
    if (inOut.x < 0)
        return psOut;

    UniformSampleGenerator sg = UniformSampleGenerator(pixel, frameIndex);
    from = from + (inOut.x + 1e-6) * delta;
    HitResult hit = rayMarching(from, direction, -1, false);
    if (all(pixel == selectedPixel))
    {
        if (hit.hit)
        {
            int offset = OM.vBuffer[hit.cellInt];
            selectedVoxel[0] = float4(-direction, (float)offset);
        }
        else
        {
            selectedVoxel[0] = float4(-1);
        }
    }

    if (hit.hit)
    {
        float3 color = 0;
        uint offset = OM.vBuffer[hit.cellInt];
        switch (drawMode)
        {
        case ABSDFDrawMode::Default:
#if USE_EMISSIVE_LIGHTS
            float3 posW = gridMin + hit.cell * voxelSize;
            if (checkHitEmissive(posW, voxelSize.x, color))
            {
                psOut.color = float4(color, 1);
                return psOut; // 命中自发光面片时中止后续计算
            }
#endif
            color = shadingGlobal(hit, -direction, sg);
            break;
        case ABSDFDrawMode::Coverage:
            color = coverageTest(hit, offset, direction);
            break;
        default:
            color = debugShading(hit, offset, -direction);
            break;
        }
        psOut.color = float4(color, 1);
    }
    return psOut;
}
