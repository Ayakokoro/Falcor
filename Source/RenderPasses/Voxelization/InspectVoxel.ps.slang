#include "Math/VoxelizationUtility.slang"
#include "RayMarchingMaterial.slang"
#include "VisibleShading.slang"
#include "VoxelInspectorShared.slang"
import Scene.Scene;
import Scene.Material.MaterialFactory;
import Rendering.Materials.StandardMaterial;
import Rendering.Materials.TexLODTypes;
import Utils.Sampling.UniformSampleGenerator;

StructuredBuffer<RayMarchingBSDF> inspectorMaterial;
StructuredBuffer<Ellipsoid> inspectorPrimitive;
StructuredBuffer<uint> inspectorMasks;
StructuredBuffer<InspectorTriangle> clippedTriangles;
StructuredBuffer<uint> clipStats;
cbuffer CB
{
    uint localIndex;
    uint capacity;
    uint viewMode; // 0=shaded, 1=normals, 2=diffuse, 3=wireframe
    uint frameIndex;
    uint2 resolution;
    float3 cameraPosition;
    float3 cameraRight;
    float3 cameraUp;
    float3 cameraForward;
    float3 lightDirection;
    float lightIntensity;
    bool checkCoverage;
    bool checkVisibility;
    bool masksAvailable;
    uint internalVisibilityMode;
    float coverageBlend;
    float textureLod;
};

bool intersectTriangle(float3 origin, float3 direction, InspectorTriangle tri, out float t, out float3 bary)
{
    t = 0; bary = 0;
    float3 e1 = tri.p1 - tri.p0, e2 = tri.p2 - tri.p0;
    float3 p = cross(direction, e2);
    float det = dot(e1, p);
    if (abs(det) < 1e-10f) return false;
    float3 q = origin - tri.p0;
    float u = dot(q, p) / det;
    float3 r = cross(q, e1);
    float v = dot(direction, r) / det;
    t = dot(e2, r) / det;
    bary = float3(1.f - u - v, u, v);
    return min(bary.x, min(bary.y, bary.z)) >= -1e-6f && t > 1e-5f;
}

bool traceTriangles(float3 origin, float3 direction, out uint index, out float distance, out float3 bary)
{
    index = 0xffffffffu; distance = 1e30f; bary = 0;
    for (uint i = 0; i < min(clipStats[0], capacity); ++i)
    {
        float t; float3 b;
        if (intersectTriangle(origin, direction, clippedTriangles[i], t, b) && t < distance)
        { index = i; distance = t; bary = b; }
    }
    return index != 0xffffffffu;
}

float3 shadeReference(float3 origin, float3 direction)
{
    uint index; float distance; float3 bary;
    if (!traceTriangles(origin, direction, index, distance, bary)) return 0;
    InspectorTriangle tri = clippedTriangles[index];
    float3 sourceBary = tri.b0 * bary.x + tri.b1 * bary.y + tri.b2 * bary.z;
    GeometryInstanceID instanceID = { tri.instanceID };
    VertexData vertex = gScene.getVertexData(instanceID, tri.triangleID, sourceBary);
    uint materialID = gScene.getMaterialID(instanceID);
    ShadingData sd = gScene.materials.prepareShadingData(vertex, materialID, -direction);
    if (sd.mtl.getMaterialType() != MaterialType::Standard) return float3(1, 0, 1);
    StandardMaterial material;
    material.header = sd.mtl;
    material.data = gScene.materials.getBasicMaterialData(materialID);
    // Use Falcor's texture decoding and normal mapping, then the same
    // Frostbite/GGX lobes as the voxel model to isolate aggregation errors.
    StandardMaterialInstance source = material.setupMaterialInstance(
        gScene.materials, sd, ExplicitLodTextureSampler(textureLod), 0);
    float3 n = source.sf.N;
    if (dot(n, -direction) < 0) n = -n;
    if (viewMode == 1) return n * 0.5f + 0.5f;
    if (viewMode == 2) return source.data.diffuse;
    if (viewMode == 3) return min(bary.x, min(bary.y, bary.z)) < 0.015f ? float3(1) : float3(0.04f);

    float3 position = origin + distance * direction;
    if (checkVisibility)
    {
        uint shadowIndex; float shadowDistance; float3 shadowBary;
        if (traceTriangles(position + lightDirection * 1e-4f, lightDirection, shadowIndex, shadowDistance, shadowBary)) return 0;
    }
    VisibleLobeBRDF lobe = {};
    lobe.n = n;
    buildTBN(n, lobe.t, lobe.b);
    lobe.frostBite.diffuse = source.data.diffuse;
    lobe.frostBite.rough = max(0.01f, source.data.roughness);
    lobe.cookTorrence.specular = source.data.specular;
    lobe.cookTorrence.alpha = lobe.frostBite.rough * lobe.frostBite.rough;
    float3 h = safeNormalize(lightDirection - direction);
    return lightIntensity * lobe.eval(lightDirection, -direction, h);
}

float3 shadeVoxel(float3 origin, float3 direction)
{
    // Limit the fitted ellipsoid to the selected node, just as traversal does.
    float2 bounds = clip(float3(0), float3(1), origin, origin + direction * 100.f);
    if (bounds.x < 0) return 0;
    float3 start = origin + direction * (100.f * bounds.x);
    float3 end = origin + direction * (100.f * bounds.y);
    Ellipsoid primitive = inspectorPrimitive[localIndex];
    float2 interval = primitive.clip(start, end);
    if (interval.x < 0) return 0;
    float3 position = lerp(start, end, interval.x);
    RayMarchingBSDF bsdf = inspectorMaterial[localIndex];
    float3 v = -direction, l = lightDirection, h = safeNormalize(l + v);
    float coverage = checkCoverage ? bsdf.calcCoverage(v, coverageBlend) : 1.f;
    if (viewMode == 3)
    {
        float3 n = safeNormalize(mul(primitive.B, position - primitive.center));
        return coverage * (abs(dot(n, v)) < 0.15f ? float3(1) : float3(0.04f));
    }
    if (viewMode == 1 || viewMode == 2)
    {
        float total = 0; float3 sum = 0;
        for (uint i = 0; i < bsdf.surface.lobeCount; ++i)
        {
#if USE_VISIBLE_MATERIAL
            float w = bsdf.surface.projectedWeight(i, v);
#else
            float w = bsdf.surface.weights[i];
#endif
            float3 n = bsdf.surface.lobes[i].n;
            if (dot(n, v) < 0) n = -n;
            sum += w * (viewMode == 1 ? n * 0.5f + 0.5f : bsdf.surface.lobes[i].frostBite.diffuse);
            total += w;
        }
        return coverage * sum / max(total, 1e-8f);
    }
    float3 result = bsdf.surface.eval(l, v, h);
    if (checkVisibility)
    {
#if USE_VISIBLE_MATERIAL
        if (internalVisibilityMode == 0) result *= bsdf.calcInternalVisibility(l);
        else if (internalVisibilityMode == 1 && masksAvailable)
        {
            uint li = visibilityMaskDirectionIndex(l), vi = visibilityMaskDirectionIndex(v);
            if (li < kVisibilityMaskDirectionCount && vi < kVisibilityMaskDirectionCount)
            {
                uint first = localIndex * kVisibilityMaskWordsPerRecord;
                VisibilityMaskPair pair = { inspectorMasks[first + li], inspectorMasks[first + vi],
                    inspectorMasks[first + kVisibilityMaskDirectionCount] };
                result = bsdf.surface.evalVisible(l, v, h, pair);
            }
        }
#else
        result *= bsdf.calcInternalVisibility(l);
#endif
    }
    return coverage * lightIntensity * result;
}

float4 main(float2 texCoord : TEXCOORD, float4 pos : SV_POSITION) : SV_Target0
{
    if (abs(texCoord.x - 0.5f) < 1.f / float(resolution.x)) return float4(0.25f, 0.25f, 0.25f, 1);
    // Never present a truncated reference as a valid comparison.
    if (clipStats[1] != 0) return float4(1, 0, 1, 1);
    bool reference = texCoord.x >= 0.5f;
    float2 uv = float2(frac(texCoord.x * 2.f), texCoord.y);
    UniformSampleGenerator sg = UniformSampleGenerator(uint2(uv * float2(resolution.x / 2, resolution.y)), frameIndex);
    uv += (Next2(sg) - 0.5f) / float2(resolution.x / 2, resolution.y);
    float2 screen = float2(uv.x * 2.f - 1.f, 1.f - uv.y * 2.f);
    screen.x *= float(resolution.x) * 0.5f / float(resolution.y);
    float3 ray = normalize(cameraForward + 0.41421356f * (screen.x * cameraRight + screen.y * cameraUp));
    float3 color = reference ? shadeReference(cameraPosition, ray) : shadeVoxel(cameraPosition, ray);
    return float4(color, 1);
}
