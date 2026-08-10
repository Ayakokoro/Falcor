#include "Math/Polygon.slang"
#include "Math/Triangle.slang"
#include "Voxel/VoxelGrid.slang"
import Utils.Sampling.UniformSampleGenerator;
import Scene.SceneTypes;
import Scene.Scene;
import Scene.Raster;
import Utils.Math.MathHelpers;
import Rendering.Materials.TexLODHelpers;
import Scene.Material.BasicMaterialData;

// Utility
inline float pow5(float x)
{
    x = saturate(x);
    float x2 = x * x;
    return x2 * x2 * x;
}

inline float luminance(float3 c)
{
    return dot(c, float3(0.299f, 0.587f, 0.114f));
}

// Deterministic RNG
uint hash32(uint x)
{
    x = x * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return x;
}

float rand01(inout uint seed)
{
    seed = hash32(seed);
    return float(seed) / float(0xFFFFFFFFu);
}

// Fibonacci sphere
float3 fibonacciSphere(uint index, uint total)
{
    float y = 1.0f - (2.0f * float(index) + 1.0f) / float(total);
    float radius = sqrt(max(0.0f, 1.0f - y * y));
    float phi = float(index) * 2.399963229728653f;
    return normalize(float3(cos(phi) * radius, y, sin(phi) * radius));
}

// Sample a uniform random point on a convex polygon surface
float3 samplePointOnPolygon(Polygon poly, float2 u)
{
    if (poly.count < 3) return poly.vertices[0];
    float uTri = u.x * float(poly.count - 2);
    uint triIdx = uint(uTri);
    if (triIdx >= poly.count - 2) triIdx = poly.count - 3;
    float frac = saturate(uTri - float(triIdx));
    float3 v0 = poly.vertices[0];
    float3 v1 = poly.vertices[triIdx + 1];
    float3 v2 = poly.vertices[triIdx + 2];
    float sqrtR1 = sqrt(frac);
    float r2 = u.y;
    return v0 * (1.0f - sqrtR1) + v1 * (sqrtR1 * (1.0f - r2)) + v2 * (sqrtR1 * r2);
}

// Ray vs convex polygon intersection
bool rayPolygonIntersect(float3 origin, float3 dir, Polygon poly, float maxDist, out float t)
{
    float denom = dot(poly.normal, dir);
    if (abs(denom) < 1e-7f) return false;
    t = dot(poly.normal, poly.vertices[0] - origin) / denom;
    if (t <= 1e-5f || t >= maxDist) return false;
    float3 hitPoint = origin + t * dir;
    for (uint i = 0; i < poly.count; i++)
    {
        float3 v0 = poly.vertices[i];
        float3 v1 = poly.vertices[(i + 1) % poly.count];
        if (dot(cross(v1 - v0, hitPoint - v0), poly.normal) < -1e-6f)
            return false;
    }
    return true;
}

// Trace ray against ALL polygons; returns 0 if occluded, 1 if visible
float traceVisibility(
    float3 origin, float3 dir, float maxDist,
    StructuredBuffer<Polygon> polygons, uint polyStart, uint polyCount)
{
    for (uint i = 0; i < polyCount; i++)
    {
        float t;
        if (rayPolygonIntersect(origin, dir, polygons[polyStart + i], maxDist, t))
            return 0.0f;
    }
    return 1.0f;
}

// Texture sampling with LOD (same as AnalyzePolygon)
// maybe has problems
float4 sampleTextureArea(TextureHandle handle, float2 uv, float uvArea, float4 uniformValue)
{
    MaterialSystem ms = gScene.materials;
    TextureInfo info = ms.getTextureInfo(handle);
    float pixelArea = uvArea * info.width * info.height;
    float lodLevel = 0.5f * log2(max(pixelArea, 1.0f));
    ITextureSampler lodTex = ExplicitLodTextureSampler(lodLevel);
    return ms.sampleTexture(handle, sampler, uv, uniformValue, lodTex);
}

float4 sampleTexture(TextureHandle handle, float2 uv, float4 uniformValue)
{
    MaterialSystem ms = gScene.materials;
    ITextureSampler lodTex = ExplicitLodTextureSampler(0);
    return ms.sampleTexture(handle, sampler, uv, uniformValue, lodTex);
}

// Per-point BSDF: Frostbite diffuse + Cook-Torrance specular
float3 evalPointBSDF(
    float3 wi, float3 wo, float3 h,
    float3 shadingNormal,
    float3 diffuse, float3 specular, float roughness
)
{
    float3 t, b;
    {
        float3 axis = abs(shadingNormal.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0);
        t = normalize(cross(axis, shadingNormal));
        b = cross(shadingNormal, t);
    }

    bool negative = dot(wo, shadingNormal) < 0;
    float3 lLocal = float3(dot(wi, t), dot(wi, b), dot(wi, shadingNormal));
    float3 vLocal = float3(dot(wo, t), dot(wo, b), dot(wo, shadingNormal));
    float3 hLocal = float3(dot(h, t), dot(h, b), dot(h, shadingNormal));
    if (negative) { lLocal = -lLocal; vLocal = -vLocal; hLocal = -hLocal; }

    if (lLocal.z <= 0 || vLocal.z <= 0) return float3(0);

    float rough = max(roughness, 0.01f);
    float alpha = rough * rough;

    // Frostbite diffuse
    float VdotH = max(0.0f, dot(vLocal, hLocal));
    float energyBias = lerp(0.0f, 0.5f, rough);
    float energyFactor = lerp(1.0f, 1.0f / 1.51f, rough);
    float fd90 = energyBias + 2.0f * VdotH * VdotH * rough;
    float wiScatter = lerp(1.0f, fd90, pow5(1.0f - lLocal.z));
    float woScatter = lerp(1.0f, fd90, pow5(1.0f - vLocal.z));
    float3 diffuseVal = wiScatter * woScatter * energyFactor * diffuse / PI;

    // Cook-Torrance specular
    float NdotH = max(0.0f, hLocal.z);
    float alpha2 = alpha * alpha;
    float temp = max(NdotH * NdotH * (alpha2 - 1.0f) + 1.0f, 1e-6f);
    float D = alpha2 / (PI * temp * temp);

    float3 F = lerp(specular, float3(1, 1, 1), pow5(1 - dot(vLocal, hLocal)));

    float Nz_v = max(vLocal.z, 1e-8f);
    float Nz_l = max(lLocal.z, 1e-8f);
    float LambdaV = (sqrt(alpha2 + (1.0f - alpha2) * Nz_v * Nz_v) / Nz_v - 1.0f) * 0.5f;
    float LambdaL = (sqrt(alpha2 + (1.0f - alpha2) * Nz_l * Nz_l) / Nz_l - 1.0f) * 0.5f;
    float G = 1.0f / max(1.0f + LambdaV + LambdaL, 1e-6f);

    float3 specularVal = F * D * G / max(4.0f * Nz_v * Nz_l, 1e-6f);

    return diffuseVal + specularVal;
}

// Buffers & CB
StructuredBuffer<Polygon> polygonBuffer;
StructuredBuffer<PolygonRange> polygonRangeBuffer;
SamplerState sampler;

RWTexture2D<float4> outputMap;  // R=GT, G=Approx, B=absError

cbuffer GridData
{
    float3 gridMin;
    float3 voxelSize;
    uint3 voxelCount;
};

cbuffer CB
{
    uint targetGBufferIndex;
    uint blockCount;
    uint blockSize;
    uint samplesPerPolygon;
};

// Uniform hemisphere sampling: map pixel (px, py) to a direction on the
// unit hemisphere (z > 0).  PDF = 1/(2π), uniform over solid angle.
float3 sampleUniformHemisphere(uint px, uint py, uint res)
{
    float u = (float(px) + 0.5f) / float(res);
    float v = (float(py) + 0.5f) / float(res);
    float theta = acos(1.0f - u);   // uniform: CDF(θ) = 1 - cos(θ)
    float phi = 2.0f * PI * v;
    float sinT = sin(theta);
    return float3(sinT * cos(phi), sinT * sin(phi), cos(theta));
}

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint totalPixels = blockCount * blockSize;
    if (tid.x >= totalPixels || tid.y >= totalPixels)
        return;

    uint bx = tid.x / blockSize;
    uint by = tid.y / blockSize;
    uint px = tid.x % blockSize;
    uint py = tid.y % blockSize;

    uint blockIdx = by * blockCount + bx;
    uint pixelIdx = py * blockSize + px;
    uint totalWi = blockCount * blockCount;
    uint totalWo = blockSize * blockSize;

    float3 wi = fibonacciSphere(blockIdx, totalWi);

    // ---- ω_o: uniform hemisphere sample (local space, z > 0) ----
    float3 localWo = sampleUniformHemisphere(px, py, blockSize);

    PolygonRange range = polygonRangeBuffer[targetGBufferIndex];
    float nodeScale = range.nodeScale;
    float maxDist = 2.0f;

    // Pre-load per-polygon triangle data (same for all samples on this polygon)
    // We'll iterate polygons first, then samples within each polygon

    // MC accumulators
    float sum_fViVo_cc = 0;
    float sum_ViVo_cc = 0;
    float sum_Vo_co = 0;
    float sum_f_cc = 0;
    float sum_co = 0;
    float sum_Vo_ci_co = 0;
    float totalWeight = 0;

    MaterialSystem ms = gScene.materials;
    float IoR = 1.5f;
    float f0Term = (IoR - 1.0f) / (IoR + 1.0f);
    float F0 = f0Term * f0Term;

    for (uint s = 0; s < range.count; s++)
    {
        uint polyIdx = range.localHead + s;
        Polygon poly = polygonBuffer[polyIdx];

        float polyArea = poly.calcArea();
        float sampleWeight = polyArea / float(samplesPerPolygon);

        // ---- Look up source triangle once per polygon ----
        TriangleRef triRef = poly.triRef;
        uint3 vtxIndices = gScene.getIndices(triRef.meshID, triRef.triangleID);
        StaticVertexData v[3] = {
            gScene.getVertex(vtxIndices.x),
            gScene.getVertex(vtxIndices.y),
            gScene.getVertex(vtxIndices.z)
        };

        Triangle tri = {};
        for (int i = 0; i < 3; i++)
        {
            tri.vertices[i] = (v[i].position - gridMin) / voxelSize;
            tri.uvs[i] = v[i].texCrd;
            tri.normals[i] = v[i].normal;
        }
        tri.buildTBN();

        BasicMaterialData mat = ms.getBasicMaterialData(triRef.materialID);

        uint baseSeed = hash32(polyIdx * 2654435761u + blockIdx * 3266489917u + pixelIdx);

        for (uint k = 0; k < samplesPerPolygon; k++)
        {
            uint seed = hash32(baseSeed + k * 2246822519u);
            float2 u = float2(rand01(seed), rand01(seed));

            // localPos is in Node-local space (0..1), then scaled to leaf voxel space
            float3 localPos = samplePointOnPolygon(poly, u);

            // Interpolate normal at sample point
            float3 leafCoord = localPos * nodeScale;
            float2 uv = tri.lerpUV(leafCoord);
            float3 sampleNormal = tri.lerpNormal(leafCoord);

            // ---- Flip normal to face ω_i, then transform ω_o to world space ----
            float3 hemAxis = dot(sampleNormal, wi) >= 0.0f ? sampleNormal : -sampleNormal;
            float3 t, b;
            buildTBN(t, b, hemAxis);
            float3 wo = normalize(localToWorld(localWo, t, b, hemAxis));
            float3 h = normalize(wi + wo);

            float dot_sample_i = dot(sampleNormal, wi);
            float dot_sample_o = dot(sampleNormal, wo);

            if (dot_sample_i * dot_sample_o <= 0.0f)
                continue;

            // effectiveNormal is in world space
            float3 effectiveNormal = dot_sample_o < 0.0f ? -sampleNormal : sampleNormal;

            float cosSample_i = abs(dot_sample_i);
            float cosSample_o = abs(dot_sample_o);

            // Sample material textures at interpolated UV
            float3 baseColor = sampleTexture(mat.texBaseColor, uv, mat.baseColor).rgb;
            float4 spec = sampleTexture(mat.texSpecular, uv, mat.specular);

            float3 lobeDiffuse = lerp(baseColor, float3(0), spec.b);
            float3 lobeSpecular = lerp(float3(F0), baseColor, spec.b);
            float lobeRough = max(spec.g, 0.01f);

            // Visibility
            float3 rayOrigin = localPos + effectiveNormal * 1e-5f;
            float Vi = traceVisibility(rayOrigin, wi, maxDist, polygonBuffer, range.localHead, range.count);
            float Vo = traceVisibility(rayOrigin, wo, maxDist, polygonBuffer, range.localHead, range.count);

            // Per-point BSDF
            float3 fVal = abs(evalPointBSDF(wi, wo, h, effectiveNormal, lobeDiffuse, lobeSpecular, lobeRough));
            float fLum = luminance(fVal);

            float w = sampleWeight;
            sum_fViVo_cc += fLum * Vi * Vo * cosSample_i * cosSample_o * w;
            sum_ViVo_cc += Vi * Vo * cosSample_i * cosSample_o * w;
            sum_Vo_co += Vo * cosSample_o * w;
            sum_f_cc += fLum * cosSample_i * cosSample_o * w;
            sum_co += cosSample_o * w;
            sum_Vo_ci_co += Vo * cosSample_i * cosSample_o * w;
            totalWeight += w;
        }
    }

    float gt = 0, approx = 0, error = 0;

    if (totalWeight > 0 && sum_ViVo_cc > 0 && sum_Vo_co > 0)
    {
        gt = sum_fViVo_cc / sum_Vo_co;
    }

    if (totalWeight > 0 && sum_co > 0 && sum_Vo_ci_co > 0 && sum_ViVo_cc > 0)
    {
        float formula4 = sum_f_cc / sum_co;
        float formula5 = sum_ViVo_cc / sum_Vo_ci_co;
        approx = formula4 * formula5;
    }

    error = abs(gt - approx);
    outputMap[tid.xy] = float4(gt, approx, error, 0);
}
