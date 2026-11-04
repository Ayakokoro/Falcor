#include "VoxelizationCore.h"
#include "VisibilityMask.h"
#include <iostream>
#include <stdexcept>

using namespace VoxelizationCore;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void near(float3 actual, float3 expected, float tolerance, const char* message) {
    require(length(actual - expected) <= tolerance, message);
}

void testAreaSampling() {
    Polygon polygon;
    polygon.count = 4;
    polygon.vertices[0] = float3(0, 0, 0);
    polygon.vertices[1] = float3(1, 0, 0);
    polygon.vertices[2] = float3(1, .2f, 0);
    polygon.vertices[3] = float3(0, 1, 0);
    float3 mean(0);
    constexpr uint32_t count = 65536;
    for (uint32_t i = 0; i < count; ++i) {
        // Stratify both random coordinates without correlating them.
        uint32_t reverse = i;
        reverse = (reverse << 16) | (reverse >> 16);
        reverse = ((reverse & 0x55555555u) << 1) | ((reverse & 0xaaaaaaaau) >> 1);
        reverse = ((reverse & 0x33333333u) << 2) | ((reverse & 0xccccccccu) >> 2);
        reverse = ((reverse & 0x0f0f0f0fu) << 4) | ((reverse & 0xf0f0f0f0u) >> 4);
        reverse = ((reverse & 0x00ff00ffu) << 8) | ((reverse & 0xff00ff00u) >> 8);
        mean += samplePolygonSurface(polygon,
            float2((float(i) + .5f) / count, (double(reverse) + .5) / 4294967296.0));
    }
    mean /= float(count);
    // Exact centroid of fan triangles with areas 0.1 and 0.5.
    near(mean, float3(7.0f / 18, 31.0f / 90, 0), 2e-4f,
         "Reference samples are not uniform in polygon area.");
}

// For roughness 1, GGX D=1/pi and its cosine-weighted specular response
// reduces to F*ci/(2*pi*(ci+co)). This avoids a duplicate general shader.
float3 roughResponse(float3 n, float3 diffuse, float3 wi, float3 wo) {
    const float ci = dot(n, wi), co = dot(n, wo);
    if (ci <= 0 || co <= 0) return float3(0);
    const float vh = dot(wo, safeNormalize(wi + wo));
    const float fd90 = .5f + 2 * vh * vh;
    const float scatter = (1 + (fd90 - 1) * std::pow(1 - ci, 5.0f)) *
                          (1 + (fd90 - 1) * std::pow(1 - co, 5.0f));
    return diffuse * (scatter * ci / (1.51f * M_PI_F)) +
        float3(std::pow(1 - vh, 5.0f) * ci / (2 * M_PI_F * (ci + co)));
}

void testSurfaceModes() {
    VoxelData voxel; voxel.init();
    for (auto& value : voxel.polygonsProjAreaFunc.coefficients) value = .25f;
    for (auto& value : voxel.totalProjAreaFunc.coefficients) value = 1;
    VisibleMaterialFile::Data material;
    const float3 normals[] = {float3(0, 0, 1), float3(.6f, 0, .8f)};
    const float3 colors[] = {float3(1, 0, 0), float3(0, 1, 0)};
    for (uint32_t i = 0; i < 2; ++i) {
        // Noncontiguous source lobes verify compaction matches bit identity.
        auto& lobe = material.lobes[i == 0 ? 1 : 6];
        lobe.weight = 1;
        lobe.rough = 1;
        for (uint32_t axis = 0; axis < 3; ++axis) {
            lobe.normal[axis] = normals[i][axis];
            lobe.diffuse[axis] = colors[i][axis];
        }
    }
    const float3 wi(.8f, 0, .6f), wo(0, 0, 1);
    const float3 first = roughResponse(normals[0], colors[0], wi, wo) / 1.8f;
    const float3 second = .8f * roughResponse(normals[1], colors[1], wi, wo) / 1.8f;
    const auto off = evaluateBakedSurface(voxel, &material, nullptr, wi, wo, InternalVisibilityMode::Off);
    near(off, first + second, 2e-6f, "Baked visible response differs from projected-area mixture.");
    near(evaluateBakedSurface(voxel, &material, nullptr, wi, wo), off, 1e-7f,
         "Missing masks did not use renderer's bright fallback.");
    near(evaluateBakedSurface(voxel, &material, nullptr, wi, wo, InternalVisibilityMode::Legacy),
         .25f * off, 1e-7f, "Legacy one-way ratio differs from renderer.");
    const float3 sideLight = safeNormalize(float3(-.9f, 0, .43589f));
    near(evaluateBakedSurface(voxel, &material, nullptr, sideLight, wo, InternalVisibilityMode::Off),
         roughResponse(normals[0], colors[0], sideLight, wo) / 1.8f, 2e-6f,
         "Light-backfacing lobe was removed from view-area normalization.");

    VisibilityMask::Data mask;
    mask.metadata = material.sourceLobes() | (3u << 8);
    mask.masks[VisibilityMask::directionIndex(wo)] = 0xffffffffu;
    mask.masks[VisibilityMask::directionIndex(wi)] = 0xffff00ffu;
    near(evaluateBakedSurface(voxel, &material, &mask, wi, wo), .5f * first + second,
         2e-6f, "Visibility was not applied inside the compact lobe sum.");
    mask.metadata = 1u | (1u << 8); // Inconsistent count must ignore the payload.
    near(evaluateBakedSurface(voxel, &material, &mask, wi, wo), off, 1e-7f,
         "Mismatched mask lobe count did not fall back.");
}

struct Fixture {
    InstancedScene scene;
    NodeData node;
    GridData grid{};
    std::vector<Texture2D> empty, normalMaps;

    Fixture() {
        grid.gridMin = float3(0);
        grid.voxelSize = float3(1);
        grid.voxelCount = uint3(1);
        scene.materials.resize(1);
        scene.materials[0].baseColor = float3(.7f, .4f, .2f);
        scene.materials[0].specular = float4(0, 1, 0, 1);
        scene.meshes.resize(1);
        scene.instances.push_back({0, glm::mat4(1)});
    }

    void triangle(float3 origin, float3 normal) {
        auto& mesh = scene.meshes[0];
        const uint32_t base = uint32_t(mesh.positions.size());
        const float3 positions[] = {origin, origin + float3(.04f, 0, 0), origin + float3(0, .04f, 0)};
        Polygon polygon;
        polygon.count = 3;
        polygon.normal = float3(0, 0, 1);
        polygon.triRef.triangleID = uint32_t(mesh.triangles.size());
        for (uint32_t i = 0; i < 3; ++i) {
            polygon.vertices[i] = positions[i];
            mesh.positions.push_back(positions[i]);
            mesh.normals.push_back(normal);
            mesh.texCoords.emplace_back(positions[i].x, positions[i].y);
        }
        mesh.triangles.emplace_back(base, base + 1, base + 2);
        node.polygons.push_back(polygon);
    }

    AnalysisResult analyze() const {
        const TextureSet textures{empty, empty, empty, normalMaps};
        const AnalysisContext context{scene, grid, 0, textures};
        AnalysisOptions options;
        options.sampleFrequency = 8;
        options.samplesPerPolygon = 8;
        options.computeSplittingError = true;
        options.splittingBlockCount = 4;
        options.splittingBlockSize = 4;
        options.threadCount = 1;
        options.visibleMaterial = true;
        options.internalVisibilityMode = InternalVisibilityMode::Off;
        auto result = analyzeNode(node, context, options);
        require(result.success, "Synthetic inspector node could not be analyzed.");
        return result;
    }
};

void testReferenceDenominator() {
    const float3 wi = VisibilityMask::direction(11), wo = VisibilityMask::direction(10);
    const float3 litNormal = safeNormalize(wi + wo), unlitNormal = safeNormalize(wo - wi);
    require(dot(unlitNormal, wi) < 0 && dot(unlitNormal, wo) > 0, "Invalid test directions.");
    Fixture fixture;
    fixture.triangle(float3(.1f, .1f, .3f), litNormal);
    const auto lit = fixture.analyze();
    fixture.triangle(float3(.8f, .1f, .7f), unlitNormal);
    const auto both = fixture.analyze();
    // Incident texel 11, outgoing texel 10 within its block.
    const size_t pixel = size_t(2 * 4 + 2) * 16 + (3 * 4 + 2);
    const float expected = lit.splittingError.values[pixel].x * dot(litNormal, wo) /
        (dot(litNormal, wo) + dot(unlitNormal, wo));
    require(expected > 0, "Reference fixture has no lit response.");
    require(std::abs(both.splittingError.values[pixel].x - expected) < 2e-6f,
            "Light-backfacing visible area was removed from the reference denominator.");
}

void testReferenceNormalMap() {
    Fixture mapped;
    mapped.triangle(float3(.1f, .1f, .3f), float3(0, 0, 1));
    Texture2D texture;
    texture.width = texture.height = texture.levels = 1;
    texture.mipLevels = {{.8f, .5f, .9f, 1}}; // Unit normal (.6,0,.8), identity triangle frame.
    mapped.normalMaps.push_back(texture);
    Fixture normals;
    normals.triangle(float3(.1f, .1f, .3f), float3(.6f, 0, .8f));
    const auto a = mapped.analyze(), b = normals.analyze();
    for (size_t i = 0; i < a.splittingError.values.size(); ++i)
        require(std::abs(a.splittingError.values[i].x - b.splittingError.values[i].x) < 2e-6f,
                "Reference ignored or transformed the point normal map differently from the baker.");
}
} // namespace

int main() {
    try {
        testAreaSampling();
        testSurfaceModes();
        testReferenceDenominator();
        testReferenceNormalMap();
        std::cout << "Inspector tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
