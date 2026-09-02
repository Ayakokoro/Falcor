#include "VoxelizationCore.h"

#include "Estimate.h"
#include "PolygonGenerator.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>

namespace VoxelizationCore
{
namespace
{

constexpr uint32_t kIndexHeaderSize = sizeof(PolygonSerializer::NodesIdxHeader);
constexpr uint32_t kIndexEntrySize = sizeof(PolygonSerializer::NodeIndex);

float3 localToVoxel(const float3& localPos, const glm::mat4& worldM,
                    const float3& gridMin, const float3& invVoxelSize)
{
    glm::vec4 worldPos = worldM * glm::vec4(localPos, 1.0f);
    return float3(
        (worldPos.x - gridMin.x) * invVoxelSize.x,
        (worldPos.y - gridMin.y) * invVoxelSize.y,
        (worldPos.z - gridMin.z) * invVoxelSize.z);
}

float3 transformNormal(const float3& localNormal, const glm::mat4& worldM)
{
    glm::mat3 rotation(worldM);
    glm::vec3 worldNormal = rotation * glm::vec3(localNormal.x, localNormal.y, localNormal.z);
    return safeNormalize(float3(worldNormal.x, worldNormal.y, worldNormal.z));
}

bool readPolygonChecked(std::istream& stream, Polygon& polygon, uint64_t fileSize)
{
    const auto start = stream.tellg();
    if (start < 0)
        return false;

    uint32_t vertexCount = 0;
    stream.read(reinterpret_cast<char*>(&polygon.triRef.meshID), sizeof(uint32_t));
    stream.read(reinterpret_cast<char*>(&polygon.triRef.triangleID), sizeof(uint32_t));
    stream.read(reinterpret_cast<char*>(&polygon.triRef.materialID), sizeof(uint32_t));
    stream.read(reinterpret_cast<char*>(&polygon.triRef.instanceIdx), sizeof(uint32_t));
    stream.read(reinterpret_cast<char*>(&vertexCount), sizeof(uint32_t));
    if (!stream || vertexCount < 3 || vertexCount > MAX_VERTEX_COUNT)
        return false;

    stream.read(reinterpret_cast<char*>(&polygon.normal), sizeof(float3));
    if (!stream)
        return false;

    const auto verticesStart = stream.tellg();
    if (verticesStart < 0 || static_cast<uint64_t>(verticesStart) > fileSize)
        return false;

    const uint64_t vertexBytes = uint64_t(vertexCount) * sizeof(float3);
    if (vertexBytes > fileSize - static_cast<uint64_t>(verticesStart))
        return false;

    polygon.count = vertexCount;
    stream.read(reinterpret_cast<char*>(polygon.vertices),
                static_cast<std::streamsize>(vertexBytes));
    return stream.good();
}

template<typename Function>
void parallelRows(uint32_t rowCount, uint32_t requestedThreads, Function&& function)
{
    if (rowCount == 0)
        return;

    uint32_t threadCount = requestedThreads;
    if (threadCount == 0)
        threadCount = std::max(1u, std::thread::hardware_concurrency());
    threadCount = std::min(threadCount, rowCount);

    if (threadCount <= 1)
    {
        function(0, rowCount);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(threadCount);
    const uint32_t rowsPerThread = (rowCount + threadCount - 1) / threadCount;
    for (uint32_t threadId = 0; threadId < threadCount; ++threadId)
    {
        const uint32_t begin = threadId * rowsPerThread;
        const uint32_t end = std::min(rowCount, begin + rowsPerThread);
        if (begin >= end)
            break;
        workers.emplace_back([begin, end, &function]() { function(begin, end); });
    }
    for (auto& worker : workers)
        worker.join();
}

float4 sampleTextureLod0(const std::vector<Texture2D>& textures, uint32_t materialId,
                         const float2& uv, const float4& fallback)
{
    if (materialId >= textures.size() || textures[materialId].width <= 0 ||
        textures[materialId].levels <= 0)
        return fallback;
    return textures[materialId].sample(0, uv);
}

struct PreparedPolygon
{
    const Polygon* polygon = nullptr;
    Triangle sourceTriangle;
    uint32_t materialId = 0;
};

bool preparePolygon(const Polygon& polygon, const AnalysisContext& context,
                    PreparedPolygon& prepared, std::string& error)
{
    const TriangleRef& ref = polygon.triRef;
    if (ref.meshID >= context.scene.meshes.size())
    {
        error = "polygon references an invalid meshID " + std::to_string(ref.meshID);
        return false;
    }
    if (ref.instanceIdx >= context.scene.instances.size())
    {
        error = "polygon references an invalid instanceIdx " + std::to_string(ref.instanceIdx);
        return false;
    }

    const MeshGeometry& mesh = context.scene.meshes[ref.meshID];
    if (ref.triangleID >= mesh.triangles.size())
    {
        error = "polygon references an invalid triangleID " + std::to_string(ref.triangleID);
        return false;
    }

    const uint32_t materialId = ref.materialID;
    if (materialId >= context.scene.materials.size())
    {
        error = "polygon references an invalid materialID " + std::to_string(materialId);
        return false;
    }

    const uint3 indices = mesh.triangles[ref.triangleID];
    const size_t vertexCount = mesh.positions.size();
    if (indices.x >= vertexCount || indices.y >= vertexCount || indices.z >= vertexCount ||
        indices.x >= mesh.normals.size() || indices.y >= mesh.normals.size() || indices.z >= mesh.normals.size() ||
        indices.x >= mesh.texCoords.size() || indices.y >= mesh.texCoords.size() || indices.z >= mesh.texCoords.size())
    {
        error = "triangle references an invalid vertex attribute index";
        return false;
    }

    const glm::mat4& worldM = context.scene.instances[ref.instanceIdx].transform;
    const float3 invVoxelSize(
        1.0f / context.grid.voxelSize.x,
        1.0f / context.grid.voxelSize.y,
        1.0f / context.grid.voxelSize.z);

    Triangle triangle;
    triangle.init();
    triangle.vertices[0] = localToVoxel(mesh.positions[indices.x], worldM,
                                        context.grid.gridMin, invVoxelSize);
    triangle.vertices[1] = localToVoxel(mesh.positions[indices.y], worldM,
                                        context.grid.gridMin, invVoxelSize);
    triangle.vertices[2] = localToVoxel(mesh.positions[indices.z], worldM,
                                        context.grid.gridMin, invVoxelSize);
    triangle.uvs[0] = mesh.texCoords[indices.x];
    triangle.uvs[1] = mesh.texCoords[indices.y];
    triangle.uvs[2] = mesh.texCoords[indices.z];
    triangle.normals[0] = transformNormal(mesh.normals[indices.x], worldM);
    triangle.normals[1] = transformNormal(mesh.normals[indices.y], worldM);
    triangle.normals[2] = transformNormal(mesh.normals[indices.z], worldM);
    triangle.buildTBN();

    prepared.polygon = &polygon;
    prepared.sourceTriangle = triangle;
    prepared.materialId = materialId;
    return true;
}

struct PolygonUvInfo
{
    float2 center = float2(0);
    float area = 0;
};

PolygonUvInfo polygonUvInfo(const Polygon& polygon, const Triangle& triangle,
                            float nodeScale)
{
    PolygonUvInfo result;
    if (polygon.count == 0)
        return result;

    float2 polygonUvs[MAX_VERTEX_COUNT];
    for (uint32_t i = 0; i < polygon.count; ++i)
    {
        const float3 leafVertex = polygon.vertices[i] * nodeScale;
        polygonUvs[i] = triangle.lerpUV(leafVertex);
        result.center += polygonUvs[i];
    }
    result.center /= static_cast<float>(polygon.count);

    for (uint32_t i = 0; i < polygon.count; ++i)
    {
        const float2& a = polygonUvs[i];
        const float2& b = polygonUvs[(i + 1) % polygon.count];
        result.area += a.x * b.y - a.y * b.x;
    }
    result.area = 0.5f * std::abs(result.area);
    return result;
}

ABSDFInput makeAggregateInput(const PreparedPolygon& prepared,
                              const AnalysisContext& context,
                              float nodeScale)
{
    const Polygon& polygon = *prepared.polygon;
    const Triangle& triangle = prepared.sourceTriangle;
    const uint32_t materialId = prepared.materialId;
    const MaterialData& material = context.scene.materials[materialId];

    float dummyArea = 0;
    const float3 centroid = polygon.calcCentroid(dummyArea);
    const float3 leafCentroid = centroid * nodeScale;
    const PolygonUvInfo uvInfo = polygonUvInfo(polygon, triangle, nodeScale);

    float3 barycentric = triangle.barycentricCoordinates(leafCentroid);
    float3 interpolatedNormal = safeNormalize(
        triangle.normals[0] * barycentric.x +
        triangle.normals[1] * barycentric.y +
        triangle.normals[2] * barycentric.z);

    float4 baseColorValue(material.baseColor, 1.0f);
    float roughness = material.specular.g;
    float metallic = material.specular.b;

    if (material.isSpecGloss)
    {
        float4 specGloss = material.specular;
        if (materialId < context.textures.specular.size() &&
            context.textures.specular[materialId].width > 0)
        {
            specGloss = sampleTextureArea(
                context.textures.specular[materialId], uvInfo.center, uvInfo.area,
                float4(material.specular.x, 1.0f, 0.0f, specGloss.w));
        }
        const float specularLuminance =
            specGloss.x * 0.2126f + specGloss.y * 0.7152f + specGloss.z * 0.0722f;
        roughness = 1.0f - specGloss.w;
        metallic = std::min(specularLuminance * 2.0f, 1.0f);
    }
    else
    {
        if (materialId < context.textures.specular.size() &&
            context.textures.specular[materialId].width > 0)
        {
            const float4 sampled = sampleTextureArea(
                context.textures.specular[materialId], uvInfo.center, uvInfo.area,
                float4(0.0f, roughness, metallic, 1.0f));
            roughness = sampled.y;
            if (materialId >= context.textures.metallic.size() ||
                context.textures.metallic[materialId].width == 0)
                metallic = sampled.z;
        }

        if (materialId < context.textures.metallic.size() &&
            context.textures.metallic[materialId].width > 0)
        {
            const float4 sampled = sampleTextureArea(
                context.textures.metallic[materialId], uvInfo.center, uvInfo.area,
                float4(metallic, 0.0f, 0.0f, 1.0f));
            metallic = sampled.x;
        }
    }

    if (materialId < context.textures.baseColor.size() &&
        context.textures.baseColor[materialId].width > 0)
    {
        baseColorValue = sampleTextureArea(
            context.textures.baseColor[materialId], uvInfo.center, uvInfo.area,
            float4(material.baseColor, 1.0f));
    }

    const float4 specularValue(material.specular.x, roughness, metallic, 1.0f);
    float3 shadingNormal = interpolatedNormal;

    if (materialId < context.textures.normalMap.size() &&
        context.textures.normalMap[materialId].width > 0)
    {
        const float4 normalMap = sampleTextureArea(
            context.textures.normalMap[materialId], uvInfo.center, uvInfo.area,
            float4(0.5f, 0.5f, 1.0f, 1.0f));
        const float3 tangentNormal = safeNormalize(float3(
            normalMap.x * 2.0f - 1.0f,
            normalMap.y * 2.0f - 1.0f,
            normalMap.z * 2.0f - 1.0f));
        const float3 tangent(
            triangle.TBN[0].x, triangle.TBN[1].x, triangle.TBN[2].x);
        const float3 bitangent(
            triangle.TBN[0].y, triangle.TBN[1].y, triangle.TBN[2].y);
        const float3 normal(
            triangle.TBN[0].z, triangle.TBN[1].z, triangle.TBN[2].z);
        shadingNormal = safeNormalize(
            tangent * tangentNormal.x +
            bitangent * tangentNormal.y +
            normal * tangentNormal.z);
    }

    return ABSDFInput{
        float3(baseColorValue),
        specularValue,
        shadingNormal,
        polygon.calcArea()};
}

struct PointMaterial
{
    float3 baseColor = float3(1);
    float4 specular = float4(0.04f, 1, 0, 1);
};

PointMaterial samplePointMaterial(const PreparedPolygon& prepared,
                                  const AnalysisContext& context,
                                  const float2& uv)
{
    const MaterialData& material = context.scene.materials[prepared.materialId];
    PointMaterial result;
    result.baseColor = material.baseColor;
    result.specular = material.specular;

    const float4 base = sampleTextureLod0(
        context.textures.baseColor, prepared.materialId, uv,
        float4(material.baseColor, 1.0f));
    const float4 spec = sampleTextureLod0(
        context.textures.specular, prepared.materialId, uv, material.specular);
    result.baseColor = float3(base);
    result.specular = spec;

    // The GPU debug shader samples the BasicMaterialData specular handle at
    // LOD 0.  The CPU loader additionally supports a separate metallic map;
    // applying it here preserves the CPU voxelizer's material convention when
    // such a map exists.
    if (!material.isSpecGloss && prepared.materialId < context.textures.metallic.size() &&
        context.textures.metallic[prepared.materialId].width > 0)
    {
        const float4 metallic = sampleTextureLod0(
            context.textures.metallic, prepared.materialId, uv,
            float4(result.specular.b, 0, 0, 1));
        result.specular.b = metallic.x;
    }
    else if (material.isSpecGloss)
    {
        const float specularLuminance =
            result.specular.x * 0.2126f + result.specular.y * 0.7152f + result.specular.z * 0.0722f;
        result.specular = float4(
            material.specular.x,
            1.0f - result.specular.w,
            std::min(specularLuminance * 2.0f, 1.0f),
            1.0f);
    }
    return result;
}

uint32_t hash32(uint32_t value)
{
    value = value * 747796405u + 2891336453u;
    value = ((value >> ((value >> 28u) + 4u)) ^ value) * 277803737u;
    value = (value >> 22u) ^ value;
    return value;
}

float rand01(uint32_t& seed)
{
    seed = hash32(seed);
    return static_cast<float>(seed) / static_cast<float>(0xFFFFFFFFu);
}

float3 samplePointOnPolygon(const Polygon& polygon, const float2& u)
{
    if (polygon.count < 3)
        return polygon.vertices[0];

    const float uTri = u.x * static_cast<float>(polygon.count - 2);
    uint32_t triangleIndex = static_cast<uint32_t>(uTri);
    if (triangleIndex >= polygon.count - 2)
        triangleIndex = polygon.count - 3;
    const float fraction = std::clamp(
        uTri - static_cast<float>(triangleIndex), 0.0f, 1.0f);
    const float3 v0 = polygon.vertices[0];
    const float3 v1 = polygon.vertices[triangleIndex + 1];
    const float3 v2 = polygon.vertices[triangleIndex + 2];
    const float sqrtR1 = std::sqrt(fraction);
    const float r2 = u.y;
    return v0 * (1.0f - sqrtR1) +
           v1 * (sqrtR1 * (1.0f - r2)) +
           v2 * (sqrtR1 * r2);
}

bool rayPolygonIntersect(const float3& origin, const float3& direction,
                         const Polygon& polygon, float maxDistance, float& t)
{
    const float denominator = dot(polygon.normal, direction);
    if (std::abs(denominator) < 1e-7f)
        return false;

    t = dot(polygon.normal, polygon.vertices[0] - origin) / denominator;
    if (t <= 1e-5f || t >= maxDistance)
        return false;

    const float3 hitPoint = origin + t * direction;
    for (uint32_t i = 0; i < polygon.count; ++i)
    {
        const float3& v0 = polygon.vertices[i];
        const float3& v1 = polygon.vertices[(i + 1) % polygon.count];
        if (dot(cross(v1 - v0, hitPoint - v0), polygon.normal) < -1e-6f)
            return false;
    }
    return true;
}

float traceVisibility(const float3& origin, const float3& direction,
                      float maxDistance, const std::vector<PreparedPolygon>& polygons)
{
    for (const PreparedPolygon& prepared : polygons)
    {
        float t = 0;
        if (rayPolygonIntersect(origin, direction, *prepared.polygon, maxDistance, t))
            return 0.0f;
    }
    return 1.0f;
}

float3 buildTangent(const float3& normal)
{
    const float3 axis = std::abs(normal.z) < 0.999f
        ? float3(0, 0, 1)
        : float3(1, 0, 0);
    return safeNormalize(cross(axis, normal));
}

float pow5(float value)
{
    value = std::clamp(value, 0.0f, 1.0f);
    const float value2 = value * value;
    return value2 * value2 * value;
}

float ggx(float nDotH, float alpha)
{
    const float alpha2 = alpha * alpha;
    const float temp = std::max(nDotH * nDotH * (alpha2 - 1.0f) + 1.0f, 1e-6f);
    return alpha2 / M_PI_F / temp / temp;
}

float lambdaTerm(float nDotX, float alpha)
{
    const float alpha2 = alpha * alpha;
    const float temp = std::sqrt(std::max(alpha2 + (1.0f - alpha2) * nDotX * nDotX, 0.0f));
    return (temp / std::max(1e-6f, nDotX) - 1.0f) * 0.5f;
}

float3 evalPointBSDF(const float3& wi, const float3& wo, const float3& h,
                     const float3& shadingNormal, const float3& diffuse,
                     const float3& specular, float roughness)
{
    const float3 tangent = buildTangent(shadingNormal);
    const float3 bitangent = cross(shadingNormal, tangent);

    bool negative = dot(wo, shadingNormal) < 0.0f;
    float3 lightLocal(
        dot(wi, tangent), dot(wi, bitangent), dot(wi, shadingNormal));
    float3 viewLocal(
        dot(wo, tangent), dot(wo, bitangent), dot(wo, shadingNormal));
    float3 halfLocal(
        dot(h, tangent), dot(h, bitangent), dot(h, shadingNormal));
    if (negative)
    {
        lightLocal = -lightLocal;
        viewLocal = -viewLocal;
        halfLocal = -halfLocal;
    }

    if (lightLocal.z <= 0.0f || viewLocal.z <= 0.0f)
        return float3(0);

    const float rough = std::max(roughness, 0.01f);
    const float alpha = rough * rough;

    const float vDotH = std::max(0.0f, dot(viewLocal, halfLocal));
    const float energyBias = lerp(0.0f, 0.5f, rough);
    const float energyFactor = lerp(1.0f, 1.0f / 1.51f, rough);
    const float fd90 = energyBias + 2.0f * vDotH * vDotH * rough;
    const float wiScatter = lerp(1.0f, fd90, pow5(1.0f - lightLocal.z));
    const float woScatter = lerp(1.0f, fd90, pow5(1.0f - viewLocal.z));
    const float3 diffuseValue =
        wiScatter * woScatter * energyFactor * diffuse / M_PI_F;

    const float nDotH = std::max(0.0f, halfLocal.z);
    const float alpha2 = alpha * alpha;
    const float temp = std::max(
        nDotH * nDotH * (alpha2 - 1.0f) + 1.0f, 1e-6f);
    const float distribution = alpha2 / (M_PI_F * temp * temp);
    const float3 fresnel = lerp(
        specular, float3(1), pow5(1.0f - dot(viewLocal, halfLocal)));

    const float nDotV = std::max(viewLocal.z, 1e-8f);
    const float nDotL = std::max(lightLocal.z, 1e-8f);
    const float lambdaV = lambdaTerm(nDotV, alpha);
    const float lambdaL = lambdaTerm(nDotL, alpha);
    const float geometry = 1.0f / std::max(1.0f + lambdaV + lambdaL, 1e-6f);
    const float3 specularValue =
        fresnel * distribution * geometry /
        std::max(4.0f * nDotV * nDotL, 1e-6f);

    return diffuseValue + specularValue;
}

struct SurfaceLobe
{
    float weight = 0;
    float3 normal = float3(0);
    float3 tangent = float3(0);
    float3 bitangent = float3(0);
    float3 diffuse = float3(0);
    float3 specular = float3(0);
    float roughness = 0;
    float alpha = 0;
};

struct SurfaceBRDF
{
    std::array<SurfaceLobe, LOBE_COUNT> lobes{};
    uint32_t count = 0;

    void init(const ABSDF& absdf)
    {
        count = 0;
        for (uint32_t i = 0; i < LOBE_COUNT; ++i)
        {
            const ABSDFLobe& source = absdf.lobes[i];
            if (!source.isValid())
                continue;

            SurfaceLobe& target = lobes[count++];
            target.weight = source.weight;
            target.normal = safeNormalize(source.normal);
            target.tangent = buildTangent(target.normal);
            target.bitangent = cross(target.normal, target.tangent);
            target.diffuse = source.diffuse;
            target.specular = source.specular;
            target.roughness = std::max(source.rough, 0.01f);
            target.alpha = target.roughness * target.roughness;
        }

        if (count == 0)
        {
            SurfaceLobe& target = lobes[0];
            target.weight = 1.0f;
            target.normal = float3(0, 0, 1);
            target.tangent = buildTangent(target.normal);
            target.bitangent = cross(target.normal, target.tangent);
            target.roughness = 0.01f;
            target.alpha = target.roughness * target.roughness;
            count = 1;
        }
    }

    float3 evalLobe(const SurfaceLobe& lobe, const float3& wi,
                   const float3& wo, const float3& h) const
    {
        bool negative = dot(wo, lobe.normal) < 0.0f;
        float3 lightLocal(
            dot(wi, lobe.tangent), dot(wi, lobe.bitangent), dot(wi, lobe.normal));
        float3 viewLocal(
            dot(wo, lobe.tangent), dot(wo, lobe.bitangent), dot(wo, lobe.normal));
        float3 halfLocal(
            dot(h, lobe.tangent), dot(h, lobe.bitangent), dot(h, lobe.normal));
        if (negative)
        {
            lightLocal = -lightLocal;
            viewLocal = -viewLocal;
            halfLocal = -halfLocal;
        }

        if (lightLocal.z <= 0.0f || viewLocal.z <= 0.0f)
            return float3(0);

        const float vDotH = dot(viewLocal, halfLocal);
        const float energyBias = lerp(0.0f, 0.5f, lobe.roughness);
        const float energyFactor = lerp(1.0f, 1.0f / 1.51f, lobe.roughness);
        const float fd90 = energyBias + 2.0f * vDotH * vDotH * lobe.roughness;
        const float wiScatter = lerp(1.0f, fd90, pow5(1.0f - lightLocal.z));
        const float woScatter = lerp(1.0f, fd90, pow5(1.0f - viewLocal.z));
        const float3 diffuseValue =
            wiScatter * woScatter * energyFactor * lobe.diffuse / M_PI_F * lightLocal.z;

        const float nDotH = std::max(0.0f, halfLocal.z);
        const float distribution = ggx(nDotH, lobe.alpha);
        const float3 fresnel = lerp(
            lobe.specular, float3(1), pow5(1.0f - dot(viewLocal, halfLocal)));
        const float geometry = 1.0f /
            std::max(1.0f + lambdaTerm(lightLocal.z, lobe.alpha) +
                     lambdaTerm(viewLocal.z, lobe.alpha), 1e-6f);
        const float3 specularValue = fresnel * distribution * geometry /
            std::max(4.0f * viewLocal.z, 1e-6f);
        return diffuseValue + specularValue;
    }

    float3 eval(const float3& wi, const float3& wo, const float3& h) const
    {
        float3 value(0);
        for (uint32_t i = 0; i < count; ++i)
            value += lobes[i].weight * evalLobe(lobes[i], wi, wo, h);
        return value;
    }

    float evalNormalDistribution(const float3& h) const
    {
        float result = 0;
        for (uint32_t i = 0; i < count; ++i)
        {
            float nDotH = dot(lobes[i].normal, h);
            if (nDotH < 0)
                nDotH = -nDotH;
            result += lobes[i].weight * ggx(nDotH, lobes[i].alpha);
        }
        return result;
    }
};

float3 octahedralMapping(uint32_t x, uint32_t y, uint32_t resolution)
{
    const float2 uv(
        (static_cast<float>(x) + 0.5f) / static_cast<float>(resolution),
        (static_cast<float>(y) + 0.5f) / static_cast<float>(resolution));
    const float2 p = uv * 2.0f - 1.0f;
    float3 direction(p.x, p.y, 1.0f - std::abs(p.x) - std::abs(p.y));
    if (direction.z < 0.0f)
    {
        const float oldX = direction.x;
        direction.x = (1.0f - std::abs(direction.y)) *
            (oldX > 0 ? 1.0f : oldX < 0 ? -1.0f : 0.0f);
        direction.y = (1.0f - std::abs(oldX)) *
            (direction.y > 0 ? 1.0f : direction.y < 0 ? -1.0f : 0.0f);
    }
    return safeNormalize(direction);
}

// Maps a square image pixel to the upper hemisphere using a unit-disk
// parameterization.  The disk radius is mapped linearly to the polar angle:
// the disk center is +Z and the unit-circle boundary is the XY equator.
bool upperHemisphereDiskDirection(uint32_t x, uint32_t y,
                                  uint32_t resolution, float3& direction)
{
    const float2 uv = (
        float2(static_cast<float>(x) + 0.5f,
               static_cast<float>(y) + 0.5f) /
        float2(static_cast<float>(resolution))) * 2.0f - 1.0f;
    const float radiusSquared = dot(uv, uv);
    if (radiusSquared > 1.0f)
    {
        direction = float3(0);
        return false;
    }

    const float radius = std::sqrt(radiusSquared);
    const float theta = radius * M_PI_F * 0.5f;
    const float phi = std::atan2(uv.y, uv.x);
    direction = float3(
        std::sin(theta) * std::cos(phi),
        std::sin(theta) * std::sin(phi),
        std::cos(theta));
    return true;
}

float primitiveProjectionArea(const Ellipsoid& ellipsoid, const PolygonRange& range,
                              const float3& direction, uint32_t sampleFrequency)
{
    const float3 cellMin = float3(range.cellInt);
    const float3 cellCenter = cellMin + float3(0.5f);
    const Basis2 basis = orthonormal_basis(safeNormalize(direction));
    uint32_t hitCount = 0;
    sampleFrequency = std::max(1u, sampleFrequency);

    for (uint32_t sample = 0; sample < sampleFrequency; ++sample)
    {
        const SamplingRay ray = rayToVoxel(
            Hammersley2D(sample, sampleFrequency), direction, basis, cellCenter);
        const float3 to = ray.origin + 2.0f * ray.direction;
        float2 inOut = clipAABB(cellMin, cellMin + float3(1), ray.origin, to);
        if (inOut.x < 0)
            continue;

        const float3 delta = to - ray.origin;
        const float3 fromLocal = ray.origin + inOut.x * delta - cellMin;
        const float3 toLocal = ray.origin + inOut.y * delta - cellMin;
        inOut = ellipsoid.clip(fromLocal, toLocal);
        if (inOut.x >= 0)
            ++hitCount;
    }
    return PROJECT_CIRCLE_AREA * static_cast<float>(hitCount) /
        static_cast<float>(sampleFrequency);
}

const SphericalFunc& selectSphericalFunction(const VoxelData& voxelData,
                                             SphericalFunctionType type)
{
    switch (type)
    {
    case SphericalFunctionType::PrimitiveProjArea:
        return voxelData.primitiveProjAreaFunc;
    case SphericalFunctionType::PolygonsProjArea:
        return voxelData.polygonsProjAreaFunc;
    case SphericalFunctionType::TotalProjArea:
    default:
        return voxelData.totalProjAreaFunc;
    }
}

void computeSphericalMap(FunctionMap& output, SphericalFunctionType type,
                         const VoxelData& voxelData, const PolygonRange& range,
                         const std::vector<Polygon>& polygons,
                         uint32_t resolution, uint32_t sampleFrequency)
{
    resolution = std::max(2u, resolution);
    output.width = resolution;
    output.height = resolution;
    const size_t count = size_t(output.width) * output.height;
    output.exact.assign(count, 0);
    output.approximation.assign(count, 0);
    output.error.assign(count, 0);

    const SphericalFunc& approximation = selectSphericalFunction(voxelData, type);
    for (uint32_t y = 0; y < output.height; ++y)
    {
        for (uint32_t x = 0; x < output.width; ++x)
        {
            float3 direction;
            if (!upperHemisphereDiskDirection(x, y, resolution, direction))
                continue; // output arrays were initialized to zero

            float exact = 0;
            if (type == SphericalFunctionType::PrimitiveProjArea)
                exact = primitiveProjectionArea(
                    voxelData.ellipsoid, range, direction, sampleFrequency);
            else if (type == SphericalFunctionType::PolygonsProjArea)
                exact = range.calcVisibleProjAreaRaster(polygons, direction);
            else
                exact = range.calcTotalProjArea(polygons, direction);

            const float approx = approximation.eval(direction);
            const size_t index = size_t(y) * output.width + x;
            output.exact[index] = exact;
            output.approximation[index] = approx;
            output.error[index] = std::abs(approx - exact);
        }
    }
}

void computeSplittingMap(Float4Map& output, const VoxelData& voxelData,
                         const std::vector<PreparedPolygon>& polygons,
                         const AnalysisContext& context,
                         float nodeScale, const AnalysisOptions& options)
{
    if (options.splittingBlockCount == 0 || options.splittingBlockSize == 0)
        return;

    const uint64_t totalResolution64 =
        uint64_t(options.splittingBlockCount) * options.splittingBlockSize;
    if (totalResolution64 > std::numeric_limits<uint32_t>::max())
        return;

    output.width = static_cast<uint32_t>(totalResolution64);
    output.height = output.width;
    output.values.assign(size_t(output.width) * output.height, float4(0));

    const SurfaceBRDF surface = [&]() {
        SurfaceBRDF value;
        value.init(voxelData.ABSDF);
        return value;
    }();

    const uint32_t blockCount = options.splittingBlockCount;
    const uint32_t blockSize = options.splittingBlockSize;
    const uint32_t samplesPerPolygon = std::max(1u, options.samplesPerPolygon);

    auto processRows = [&](uint32_t rowBegin, uint32_t rowEnd) {
        for (uint32_t y = rowBegin; y < rowEnd; ++y)
        {
            for (uint32_t x = 0; x < output.width; ++x)
            {
                const uint32_t blockX = x / blockSize;
                const uint32_t blockY = y / blockSize;
                const uint32_t pixelX = x % blockSize;
                const uint32_t pixelY = y % blockSize;
                const uint32_t blockIndex = blockY * blockCount + blockX;
                const uint32_t pixelIndex = pixelY * blockSize + pixelX;

                const float3 wi = octahedralMapping(blockX, blockY, blockCount);
                const float3 wo = octahedralMapping(pixelX, pixelY, blockSize);
                const float3 halfVector = safeNormalize(wi + wo);

                float sumFViVoCc = 0;
                float sumViVoCc = 0;
                float sumVoCo = 0;
                float sumFCc = 0;
                float sumCo = 0;
                float sumVoCiCo = 0;
                float totalWeight = 0;

                for (uint32_t polygonIndex = 0; polygonIndex < polygons.size(); ++polygonIndex)
                {
                    const PreparedPolygon& prepared = polygons[polygonIndex];
                    const Polygon& polygon = *prepared.polygon;
                    const float sampleWeight = polygon.calcArea() /
                        static_cast<float>(samplesPerPolygon);
                    const uint32_t baseSeed = hash32(
                        polygonIndex * 2654435761u +
                        blockIndex * 3266489917u + pixelIndex);

                    for (uint32_t sampleIndex = 0; sampleIndex < samplesPerPolygon; ++sampleIndex)
                    {
                        uint32_t seed = hash32(
                            baseSeed + sampleIndex * 2246822519u);
                        const float2 randomPoint(rand01(seed), rand01(seed));
                        const float3 localPosition = samplePointOnPolygon(polygon, randomPoint);
                        const float3 leafCoordinate = localPosition * nodeScale;
                        const float2 uv = prepared.sourceTriangle.lerpUV(leafCoordinate);
                        const float3 sampleNormal = prepared.sourceTriangle.lerpNormal(leafCoordinate);

                        const float dotSampleI = dot(sampleNormal, wi);
                        const float dotSampleO = dot(sampleNormal, wo);
                        if (dotSampleI * dotSampleO <= 0.0f)
                            continue;

                        const float3 effectiveNormal = dotSampleO < 0.0f
                            ? -sampleNormal : sampleNormal;
                        const float cosSampleI = std::abs(dotSampleI);
                        const float cosSampleO = std::abs(dotSampleO);
                        const PointMaterial material = samplePointMaterial(
                            prepared, context, uv);
                        const float3 rayOrigin = localPosition + effectiveNormal * 1e-6f;
                        const float visibilityI = traceVisibility(
                            rayOrigin, wi, 2.0f, polygons);
                        const float visibilityO = traceVisibility(
                            rayOrigin, wo, 2.0f, polygons);
                        const float3 fValue = glm::abs(evalPointBSDF(
                            wi, wo, halfVector, effectiveNormal,
                            lerp(material.baseColor, float3(0), material.specular.b),
                            lerp(float3(0.04f), material.baseColor, material.specular.b),
                            std::max(material.specular.g, 0.01f)));
                        const float fLuminance =
                            fValue.x * 0.299f + fValue.y * 0.587f + fValue.z * 0.114f;
                        const float weight = sampleWeight;

                        sumFViVoCc += fLuminance * visibilityI * visibilityO *
                            cosSampleI * cosSampleO * weight;
                        sumViVoCc += visibilityI * visibilityO *
                            cosSampleI * cosSampleO * weight;
                        sumVoCo += visibilityO * cosSampleO * weight;
                        sumFCc += fLuminance * cosSampleI * cosSampleO * weight;
                        sumCo += cosSampleO * weight;
                        sumVoCiCo += visibilityO * cosSampleI * cosSampleO * weight;
                        totalWeight += weight;
                    }
                }

                float groundTruth = 0;
                if (totalWeight > 0 && sumViVoCc > 0 && sumVoCo > 0)
                    groundTruth = sumFViVoCc / sumVoCo;

                const float3 lobeValue = surface.eval(wi, wo, halfVector);
                const float lobeBsdf =
                    lobeValue.x * 0.299f + lobeValue.y * 0.587f + lobeValue.z * 0.114f;

                float approximate = 0;
                float formula4 = 0;
                if (totalWeight > 0 && sumCo > 0 && sumVoCiCo > 0 && sumViVoCc > 0)
                {
                    formula4 = sumFCc / sumCo;
                    const float formula5 = sumViVoCc / sumVoCiCo;
                    approximate = formula4 * formula5;
                }

                output.values[size_t(y) * output.width + x] = float4(
                    groundTruth,
                    approximate,
                    std::abs(groundTruth - approximate),
                    std::abs(formula4 - lobeBsdf));
            }
        }
    };

    parallelRows(output.height, options.threadCount, processRows);
}

void computeNdfMap(Float4Map& output, const VoxelData& voxelData,
                   const std::vector<PreparedPolygon>& polygons,
                   const AnalysisContext& context,
                   float nodeScale, const AnalysisOptions& options)
{
    const uint32_t resolution = options.ndfResolution;
    if (resolution == 0)
        return;

    output.width = resolution;
    output.height = resolution;
    output.values.assign(size_t(resolution) * resolution, float4(0));

    const SurfaceBRDF surface = [&]() {
        SurfaceBRDF value;
        value.init(voxelData.ABSDF);
        return value;
    }();
    const uint32_t samplesPerPolygon = std::max(1u, options.samplesPerPolygon);

    auto processRows = [&](uint32_t rowBegin, uint32_t rowEnd) {
        for (uint32_t y = rowBegin; y < rowEnd; ++y)
        {
            for (uint32_t x = 0; x < resolution; ++x)
            {
                float3 halfVector;
                if (!upperHemisphereDiskDirection(x, y, resolution, halfVector))
                    continue;

                float sumNdf = 0;
                float totalArea = 0;
                for (uint32_t polygonIndex = 0; polygonIndex < polygons.size(); ++polygonIndex)
                {
                    const PreparedPolygon& prepared = polygons[polygonIndex];
                    const Polygon& polygon = *prepared.polygon;
                    const float sampleWeight = polygon.calcArea() /
                        static_cast<float>(samplesPerPolygon);
                    const uint32_t baseSeed = hash32(
                        polygonIndex * 2654435761u +
                        x * 3266489917u + y);

                    for (uint32_t sampleIndex = 0; sampleIndex < samplesPerPolygon; ++sampleIndex)
                    {
                        uint32_t seed = hash32(
                            baseSeed + sampleIndex * 2246822519u);
                        const float2 randomPoint(rand01(seed), rand01(seed));
                        const float3 localPosition = samplePointOnPolygon(polygon, randomPoint);
                        const float3 leafCoordinate = localPosition * nodeScale;
                        const float2 sampleUv = prepared.sourceTriangle.lerpUV(leafCoordinate);
                        const float3 sampleNormal = prepared.sourceTriangle.lerpNormal(leafCoordinate);
                        const PointMaterial material = samplePointMaterial(
                            prepared, context, sampleUv);
                        const float roughness = std::max(material.specular.g, 0.01f);
                        const float alpha2 = roughness * roughness;
                        const float nDotH = std::abs(dot(sampleNormal, halfVector));
                        sumNdf += sampleWeight * ggx(nDotH, alpha2);
                        totalArea += sampleWeight;
                    }
                }

                const float groundTruth = totalArea > 0 ? sumNdf / totalArea : 0;
                const float lobeNdf = surface.evalNormalDistribution(halfVector);
                output.values[size_t(y) * resolution + x] = float4(
                    groundTruth,
                    lobeNdf,
                    std::abs(groundTruth - lobeNdf),
                    0);
            }
        }
    };

    parallelRows(output.height, options.threadCount, processRows);
}

} // namespace

uint64_t makeNodeKey(uint32_t level, const int3& cell)
{
    return PolygonGenerator::makeNodeKey(level, cell);
}

uint32_t levelFromNodeKey(uint64_t key)
{
    return PolygonGenerator::levelFromKey(key);
}

int3 cellFromNodeKey(uint64_t key)
{
    return PolygonGenerator::cellFromKey(key);
}

bool TempNodeReader::open(const std::filesystem::path& indexPath,
                          const std::filesystem::path& polygonsPath)
{
    mOpen = false;
    mError.clear();
    mIndexPath = indexPath;
    mPolygonsPath = polygonsPath;
    mHeader = {};

    std::ifstream index(indexPath, std::ios::binary | std::ios::ate);
    if (!index)
    {
        mError = "cannot open index file: " + indexPath.string();
        return false;
    }

    const std::streamoff fileSize = index.tellg();
    if (fileSize < static_cast<std::streamoff>(sizeof(mHeader)))
    {
        mError = "index file is smaller than its header: " + indexPath.string();
        return false;
    }
    index.seekg(0, std::ios::beg);
    if (!PolygonSerializer::readNodesIdxHeader(index, mHeader))
    {
        mError = "invalid node index header: " + indexPath.string();
        return false;
    }

    const uint64_t indexBytes = uint64_t(sizeof(mHeader)) +
        mHeader.leafCount * sizeof(PolygonSerializer::NodeIndex);
    if (indexBytes > static_cast<uint64_t>(fileSize))
    {
        mError = "node index is truncated: " + indexPath.string();
        return false;
    }
    if (!std::filesystem::exists(polygonsPath))
    {
        mError = "cannot find polygon data file: " + polygonsPath.string();
        return false;
    }
    if (mHeader.reserved > mHeader.maxDepth ||
        mHeader.maxDepth > PolygonGenerator::NODE_KEY_MAX_DEPTH)
    {
        mError = "invalid target level/maxDepth in node index";
        return false;
    }

    mOpen = true;
    return true;
}

bool TempNodeReader::findNode(uint64_t nodeKey,
                              PolygonSerializer::NodeIndex& out)
{
    if (!mOpen)
    {
        mError = "node reader is not open";
        return false;
    }

    std::ifstream index(mIndexPath, std::ios::binary);
    if (!index)
    {
        mError = "cannot reopen index file: " + mIndexPath.string();
        return false;
    }

    uint64_t begin = 0;
    uint64_t end = mHeader.leafCount;
    while (begin < end)
    {
        const uint64_t middle = begin + (end - begin) / 2;
        const uint64_t offset = sizeof(mHeader) +
            middle * sizeof(PolygonSerializer::NodeIndex);
        index.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        PolygonSerializer::NodeIndex candidate;
        if (!index.read(reinterpret_cast<char*>(&candidate), sizeof(candidate)))
        {
            mError = "failed reading node index entry";
            return false;
        }

        if (candidate.nodeKey < nodeKey)
            begin = middle + 1;
        else
            end = middle;
    }

    if (begin >= mHeader.leafCount)
    {
        mError = "target node is not present in index";
        return false;
    }

    const uint64_t offset = sizeof(mHeader) +
        begin * sizeof(PolygonSerializer::NodeIndex);
    index.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!index.read(reinterpret_cast<char*>(&out), sizeof(out)) ||
        out.nodeKey != nodeKey)
    {
        mError = "target node is not present in index";
        return false;
    }
    return true;
}

bool TempNodeReader::readNode(uint32_t level, const int3& cell, NodeData& out)
{
    if (level != mHeader.reserved)
    {
        mError = "requested level " + std::to_string(level) +
            " does not match index target level " + std::to_string(mHeader.reserved);
        return false;
    }
    if (level > PolygonGenerator::NODE_KEY_MAX_DEPTH ||
        cell.x < 0 || cell.y < 0 || cell.z < 0 ||
        static_cast<uint32_t>(cell.x) >= (1u << level) ||
        static_cast<uint32_t>(cell.y) >= (1u << level) ||
        static_cast<uint32_t>(cell.z) >= (1u << level))
    {
        mError = "requested cell is outside the selected level";
        return false;
    }
    return readNode(makeNodeKey(level, cell), out);
}

bool TempNodeReader::readNode(uint64_t nodeKey, NodeData& out)
{
    PolygonSerializer::NodeIndex indexEntry;
    if (!findNode(nodeKey, indexEntry))
        return false;

    std::ifstream polygons(mPolygonsPath, std::ios::binary | std::ios::ate);
    if (!polygons)
    {
        mError = "cannot open polygon data file: " + mPolygonsPath.string();
        return false;
    }
    const std::streamoff fileSize = polygons.tellg();
    if (indexEntry.dataOffset > static_cast<uint64_t>(fileSize))
    {
        mError = "polygon data offset is outside polygons.dat";
        return false;
    }
    polygons.seekg(static_cast<std::streamoff>(indexEntry.dataOffset), std::ios::beg);

    out = {};
    out.request.nodeKey = nodeKey;
    out.request.level = levelFromNodeKey(nodeKey);
    out.request.cell = cellFromNodeKey(nodeKey);
    out.storedPolygonCount = indexEntry.polyCount;
    out.polygons.resize(indexEntry.polyCount);

    for (uint32_t i = 0; i < indexEntry.polyCount; ++i)
    {
        out.polygons[i].init();
        if (!readPolygonChecked(polygons, out.polygons[i], static_cast<uint64_t>(fileSize)))
        {
            mError = "invalid or truncated polygon at index " + std::to_string(i);
            out.polygons.clear();
            return false;
        }
    }
    return true;
}

bool readVoxelFileHeader(const std::filesystem::path& binaryPath,
                         VoxelFileHeader& out, std::string& error)
{
    std::ifstream file(binaryPath, std::ios::binary | std::ios::ate);
    if (!file)
    {
        error = "cannot open voxel binary: " + binaryPath.string();
        return false;
    }
    const std::streamoff fileSize = file.tellg();
    const uint64_t minimumSize = sizeof(GridData) + sizeof(uint32_t);
    if (fileSize < static_cast<std::streamoff>(minimumSize))
    {
        error = "voxel binary is too small: " + binaryPath.string();
        return false;
    }

    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(&out.grid), sizeof(GridData));
    file.read(reinterpret_cast<char*>(&out.maxDepth), sizeof(uint32_t));
    if (!file || out.maxDepth > PolygonGenerator::NODE_KEY_MAX_DEPTH)
    {
        error = "invalid voxel binary header: " + binaryPath.string();
        return false;
    }

    const uint64_t countsEnd = minimumSize +
        uint64_t(out.maxDepth + 1) * sizeof(uint32_t);
    if (countsEnd > static_cast<uint64_t>(fileSize))
    {
        error = "voxel binary has a truncated level-count array";
        return false;
    }
    return true;
}

bool loadSceneResources(const std::filesystem::path& scenePath,
                        InstancedScene& scene,
                        std::vector<Texture2D>& baseColorTextures,
                        std::vector<Texture2D>& specularTextures,
                        std::vector<Texture2D>& metallicTextures,
                        std::vector<Texture2D>& normalMapTextures,
                        std::string& error)
{
    SceneLoader loader;
    if (!loader.loadMeshInstances(scenePath.string(), scene))
    {
        error = loader.getError();
        return false;
    }

    baseColorTextures.resize(scene.materials.size());
    specularTextures.resize(scene.materials.size());
    metallicTextures.resize(scene.materials.size());
    normalMapTextures.resize(scene.materials.size());

    for (size_t i = 0; i < scene.materials.size(); ++i)
    {
        const MaterialData& material = scene.materials[i];
        if (!material.texBaseColor.empty() &&
            !baseColorTextures[i].load(material.texBaseColor, true))
        {
            std::cerr << "  [VoxelizationCore] WARNING: cannot load base-color texture "
                      << material.texBaseColor << std::endl;
        }
        if (!material.texSpecular.empty() &&
            !specularTextures[i].load(material.texSpecular))
        {
            std::cerr << "  [VoxelizationCore] WARNING: cannot load specular texture "
                      << material.texSpecular << std::endl;
        }
        if (!material.texMetallic.empty() &&
            !metallicTextures[i].load(material.texMetallic))
        {
            std::cerr << "  [VoxelizationCore] WARNING: cannot load metallic texture "
                      << material.texMetallic << std::endl;
        }
        if (!material.texNormalMap.empty() &&
            !normalMapTextures[i].load(material.texNormalMap))
        {
            std::cerr << "  [VoxelizationCore] WARNING: cannot load normal texture "
                      << material.texNormalMap << std::endl;
        }
    }
    return true;
}

std::vector<float3> defaultValidationDirections()
{
    return {
        safeNormalize(float3(1, 0, 0)),
        safeNormalize(float3(0, 1, 0)),
        safeNormalize(float3(0, 0, 1)),
        safeNormalize(float3(1, 1, 0)),
        safeNormalize(float3(1, 0, 1)),
        safeNormalize(float3(0, 1, 1)),
        safeNormalize(float3(-1, 1, 0.5f)),
        safeNormalize(float3(0.3f, 0.7f, -0.5f)),
        safeNormalize(float3(1, 1, 1)),
        safeNormalize(float3(0.5f, -0.3f, 0.8f))};
}

AnalysisResult analyzeNode(const NodeData& node,
                           const AnalysisContext& context,
                           const AnalysisOptions& options)
{
    AnalysisResult result;
    result.voxelData.init();

    if (node.request.level > context.maxDepth ||
        node.request.level > PolygonGenerator::NODE_KEY_MAX_DEPTH)
    {
        result.success = false;
        result.error = "node level exceeds maxDepth";
        return result;
    }
    if (node.request.cell.x < 0 || node.request.cell.y < 0 || node.request.cell.z < 0 ||
        static_cast<uint32_t>(node.request.cell.x) >= (1u << node.request.level) ||
        static_cast<uint32_t>(node.request.cell.y) >= (1u << node.request.level) ||
        static_cast<uint32_t>(node.request.cell.z) >= (1u << node.request.level))
    {
        result.success = false;
        result.error = "node cell is outside the selected level";
        return result;
    }

    const uint32_t nodeScaleInt = 1u << (context.maxDepth - node.request.level);
    const float nodeScale = static_cast<float>(nodeScaleInt);
    PolygonRange range;
    range.init(node.request.cell);
    range.nodeScale = nodeScale;
    range.count = static_cast<uint32_t>(node.polygons.size());

    std::vector<PreparedPolygon> preparedPolygons;
    preparedPolygons.reserve(node.polygons.size());
    for (const Polygon& polygon : node.polygons)
    {
        if (polygon.count < 3 || polygon.count > MAX_VERTEX_COUNT)
        {
            result.success = false;
            result.error = "polygon has an invalid vertex count";
            return result;
        }

        PreparedPolygon prepared;
        if (!preparePolygon(polygon, context, prepared, result.error))
        {
            result.success = false;
            return result;
        }
        preparedPolygons.push_back(prepared);
    }

    for (const PreparedPolygon& prepared : preparedPolygons)
    {
        const ABSDFInput input = makeAggregateInput(prepared, context, nodeScale);
        result.voxelData.ABSDF.accumulate(input);
    }
    result.voxelData.ABSDF.normalizeSelf();

    if (result.voxelData.isSolid())
    {
        result.voxelData.ellipsoid.fit(node.polygons, range);

        SphericalFunc polygonsFunc = result.voxelData.polygonsProjAreaFunc;
        SphericalFunc primitiveFunc = result.voxelData.primitiveProjAreaFunc;
        SphericalFunc totalFunc = result.voxelData.totalProjAreaFunc;
        Estimate(
            result.voxelData.ellipsoid,
            range,
            polygonsFunc,
            primitiveFunc,
            totalFunc,
            node.polygons,
            std::max(1u, options.sampleFrequency));
        result.voxelData.polygonsProjAreaFunc = polygonsFunc;
        result.voxelData.primitiveProjAreaFunc = primitiveFunc;
        result.voxelData.totalProjAreaFunc = totalFunc;
    }

    if (options.computeProjectionValidation)
    {
        const std::vector<float3> directions = defaultValidationDirections();
        result.projectionValidation.reserve(directions.size());
        for (const float3& direction : directions)
        {
            ProjectionValidation validation;
            validation.direction = direction;
            validation.exactVisibleArea = range.calcVisibleProjAreaRaster(
                node.polygons, direction);
            validation.shVisibleArea = result.voxelData.polygonsProjAreaFunc.eval(direction);
            validation.exactTotalArea = range.calcTotalProjArea(
                node.polygons, direction);
            validation.shTotalArea = result.voxelData.totalProjAreaFunc.eval(direction);
            result.projectionValidation.push_back(validation);
        }
    }

    if (options.computeSphericalMaps)
    {
        computeSphericalMap(
            result.sphericalMaps[static_cast<size_t>(SphericalFunctionType::PrimitiveProjArea)],
            SphericalFunctionType::PrimitiveProjArea,
            result.voxelData, range, node.polygons,
            options.sphericalResolution, options.sampleFrequency);
        computeSphericalMap(
            result.sphericalMaps[static_cast<size_t>(SphericalFunctionType::PolygonsProjArea)],
            SphericalFunctionType::PolygonsProjArea,
            result.voxelData, range, node.polygons,
            options.sphericalResolution, options.sampleFrequency);
        computeSphericalMap(
            result.sphericalMaps[static_cast<size_t>(SphericalFunctionType::TotalProjArea)],
            SphericalFunctionType::TotalProjArea,
            result.voxelData, range, node.polygons,
            options.sphericalResolution, options.sampleFrequency);
    }

    if (options.computeSplittingError)
    {
        computeSplittingMap(
            result.splittingError, result.voxelData, preparedPolygons,
            context, nodeScale, options);
    }

    if (options.computeNdf)
    {
        computeNdfMap(
            result.ndf, result.voxelData, preparedPolygons,
            context, nodeScale, options);
    }

    return result;
}

} // namespace VoxelizationCore
