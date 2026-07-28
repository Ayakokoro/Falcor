#pragma once
#include "Types.h"
#include "Triangle.h"
#include "GridData.h"
#include "SceneLoader.h"
#include "VoxelData.h"
#include "ABSDF.h"
#include "Ellipsoid.h"
#include "Estimate.h"
#include "TextureSampler.h"
#include "PolygonSerializer.h"
#include "PolygonGenerator.h"
#include "OctreeBuilder.h"
#include "MergePhase.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <iostream>

namespace AnalyzePhase {

// Context carrying all scene data needed for leaf analysis
struct AnalyzeContext {
    const InstancedScene& scene;
    const GridData& grid;
    uint32_t maxDepth;
    uint32_t sampleFrequency;
    const std::vector<Texture2D>& baseColorTextures;
    const std::vector<Texture2D>& specularTextures;
    const std::vector<Texture2D>& normalMapTextures;
};

// ---- Transform helpers (same as ClipPhase) ----

static inline float3 localToVoxel(const float3& localPos, const glm::mat4& worldM,
                                   const float3& gridMin, const float3& invVoxelSize) {
    glm::vec4 wp = worldM * glm::vec4(localPos, 1.0f);
    return float3(
        (wp.x - gridMin.x) * invVoxelSize.x,
        (wp.y - gridMin.y) * invVoxelSize.y,
        (wp.z - gridMin.z) * invVoxelSize.z
    );
}

static inline float3 transformNormal(const float3& localNormal, const glm::mat4& worldM) {
    glm::mat3 rot(worldM);
    glm::vec3 wn = rot * glm::vec3(localNormal.x, localNormal.y, localNormal.z);
    return safeNormalize(float3(wn.x, wn.y, wn.z));
}

// ---- Process a single leaf node ----
// Extracted from SceneVoxelization::process() leaf analysis block (lines 108-217).

inline void analyzeLeaf(
    OctreeBuilder::OctreeResult& octree,
    uint32_t bfsIndex,
    const std::vector<Polygon>& leafPolys,
    const AnalyzeContext& ctx)
{
    auto& bfsItem = octree.bfsOrder[bfsIndex];
    float rangeScale = (float)(1u << (ctx.maxDepth - bfsItem.level));

    // Build a temporary PolygonRange for Ellipsoid::fit and Estimate APIs.
    // localHead = 0 because leafPolys is our local contiguous buffer.
    PolygonRange range;
    range.init(bfsItem.cellInt);
    range.nodeScale = rangeScale;
    range.count = (uint32_t)leafPolys.size();

    VoxelData& vd = octree.gBuffer[bfsIndex];
    vd.init();

    float3 invVoxelSize(1.0f / ctx.grid.voxelSize.x,
                         1.0f / ctx.grid.voxelSize.y,
                         1.0f / ctx.grid.voxelSize.z);

    for (uint32_t pi = 0; pi < leafPolys.size(); pi++) {
        const Polygon& poly = leafPolys[pi];

        uint32_t localTid = poly.triRef.triangleID;
        uint32_t meshID   = poly.triRef.meshID;
        uint32_t instIdx  = poly.triRef.instanceIdx;
        const MeshGeometry& mesh = ctx.scene.meshes[meshID];
        uint32_t matID = mesh.materialID;

        // Reconstruct original triangle in voxel space
        uint3 localIdx = mesh.triangles[localTid];
        const glm::mat4& worldM = ctx.scene.instances[instIdx].transform;

        float3 tv0 = localToVoxel(mesh.positions[localIdx.x], worldM, ctx.grid.gridMin, invVoxelSize);
        float3 tv1 = localToVoxel(mesh.positions[localIdx.y], worldM, ctx.grid.gridMin, invVoxelSize);
        float3 tv2 = localToVoxel(mesh.positions[localIdx.z], worldM, ctx.grid.gridMin, invVoxelSize);

        Triangle origTri;
        origTri.vertices[0] = tv0; origTri.vertices[1] = tv1; origTri.vertices[2] = tv2;
        origTri.uvs[0] = mesh.texCoords[localIdx.x];
        origTri.uvs[1] = mesh.texCoords[localIdx.y];
        origTri.uvs[2] = mesh.texCoords[localIdx.z];
        origTri.buildTBN();

        // Polygon centroid in voxel space
        float dummy;
        float3 centroid = poly.calcCentroid(dummy);
        float3 leafCentroid = centroid * rangeScale;

        // Interpolate UV on polygon vertices
        float2 polyUVs[MAX_VERTEX_COUNT];
        for (uint32_t vi = 0; vi < poly.count; vi++) {
            float3 leafVertex = poly.vertices[vi] * rangeScale;
            polyUVs[vi] = origTri.lerpUV(leafVertex);
        }
        float2 uvCenter(0);
        for (uint32_t vi = 0; vi < poly.count; vi++)
            uvCenter += polyUVs[vi];
        uvCenter /= (float)poly.count;

        float uvArea = 0;
        for (uint32_t vi = 0; vi < poly.count; vi++) {
            const float2& a = polyUVs[vi];
            const float2& b = polyUVs[(vi + 1) % poly.count];
            uvArea += a.x * b.y - a.y * b.x;
        }
        uvArea = 0.5f * std::abs(uvArea);

        // Interpolate normal
        float3 bary = origTri.barycentricCoordinates(leafCentroid);
        float3 n0 = transformNormal(mesh.normals[localIdx.x], worldM);
        float3 n1 = transformNormal(mesh.normals[localIdx.y], worldM);
        float3 n2 = transformNormal(mesh.normals[localIdx.z], worldM);
        float3 interpolatedNormal = safeNormalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);

        // Sample textures
        const MaterialData& mat = ctx.scene.materials[matID];
        float4 baseColorVal = float4(mat.baseColor, 1.0f);
        float4 specVal = mat.specular;

        if (matID < ctx.baseColorTextures.size() &&
            ctx.baseColorTextures[matID].width > 0) {
            baseColorVal = sampleTextureArea(
                ctx.baseColorTextures[matID], uvCenter,
                uvArea, float4(mat.baseColor, 1.0f));
        }
        if (matID < ctx.specularTextures.size() &&
            ctx.specularTextures[matID].width > 0) {
            specVal = sampleTextureArea(
                ctx.specularTextures[matID], uvCenter,
                uvArea, float4(mat.specular.x, mat.specular.y, mat.specular.z, 1.0f));
        }

        float3 shadingNormal = interpolatedNormal;
        if (matID < ctx.normalMapTextures.size() &&
            ctx.normalMapTextures[matID].width > 0) {
            float4 nm = sampleTextureArea(ctx.normalMapTextures[matID], uvCenter, uvArea,
                                          float4(0.5f, 0.5f, 1.0f, 1.0f));
            float3 tn = safeNormalize(float3(nm.x * 2.0f - 1.0f, nm.y * 2.0f - 1.0f, nm.z * 2.0f - 1.0f));
            float3 T = float3(origTri.TBN[0].x, origTri.TBN[1].x, origTri.TBN[2].x);
            float3 B = float3(origTri.TBN[0].y, origTri.TBN[1].y, origTri.TBN[2].y);
            float3 Ns = float3(origTri.TBN[0].z, origTri.TBN[1].z, origTri.TBN[2].z);
            shadingNormal = safeNormalize(T * tn.x + B * tn.y + Ns * tn.z);
        }

        ABSDFInput input = { float3(baseColorVal), specVal, shadingNormal, poly.calcArea() };
        vd.ABSDF.accumulate(input);
    }

    vd.ABSDF.normalizeSelf();

    if (vd.isSolid()) {
        // Ellipsoid::fit accesses polygons via range.localHead offset into the buffer.
        // Since leafPolys is a contiguous local buffer, localHead=0 works correctly.
        vd.ellipsoid.fit(leafPolys, range);

        SphericalFunc polyF  = vd.polygonsProjAreaFunc;
        SphericalFunc primF  = vd.primitiveProjAreaFunc;
        SphericalFunc totalF = vd.totalProjAreaFunc;

        Estimate(vd.ellipsoid, range, polyF, primF, totalF,
                 leafPolys, ctx.sampleFrequency);

        vd.polygonsProjAreaFunc  = polyF;
        vd.primitiveProjAreaFunc = primF;
        vd.totalProjAreaFunc     = totalF;
    }
}

// ---- Main entry point: stream through leaves.idx, analyze each leaf ----

inline void execute(
    const MergePhase::MergeResult& mergeResult,
    const AnalyzeContext& ctx)
{
    // Read leaves.idx
    std::ifstream idxIn(mergeResult.leavesIdxPath, std::ios::binary);
    if (!idxIn) {
        std::cerr << "  [Analyze] ERROR: cannot open " << mergeResult.leavesIdxPath << std::endl;
        return;
    }

    PolygonSerializer::LeavesIdxHeader idxHdr;
    if (!PolygonSerializer::readLeavesIdxHeader(idxIn, idxHdr)) {
        std::cerr << "  [Analyze] ERROR: invalid leaves.idx header" << std::endl;
        return;
    }

    std::vector<PolygonSerializer::LeafIndex> leafIndices;
    PolygonSerializer::readLeafIndices(idxIn, leafIndices, idxHdr.leafCount);
    idxIn.close();

    std::cout << "  [Analyze] " << leafIndices.size() << " leaves to process" << std::endl;

    // Open polygons.dat for random-access reading
    std::ifstream polyIn(mergeResult.polygonsDatPath, std::ios::binary);
    if (!polyIn) {
        std::cerr << "  [Analyze] ERROR: cannot open " << mergeResult.polygonsDatPath << std::endl;
        return;
    }

    auto& octree = const_cast<OctreeBuilder::OctreeResult&>(mergeResult.octree);

    uint64_t processed = 0;
    uint64_t lastReported = 0;
    uint64_t skipped = 0;

    for (uint64_t li = 0; li < leafIndices.size(); li++) {
        const auto& leafIdx = leafIndices[li];

        // Seek to this leaf's polygon block
        polyIn.seekg(leafIdx.dataOffset);
        if (!polyIn) {
            std::cerr << "  [Analyze] ERROR: seek to offset " << leafIdx.dataOffset
                      << " failed" << std::endl;
            skipped++;
            continue;
        }

        // Read polygon block (raw serialized polygons, no count prefix —
        // polyCount comes from the leaf index)
        std::vector<Polygon> leafPolys(leafIdx.polyCount);
        for (uint32_t pi = 0; pi < leafIdx.polyCount; pi++) {
            leafPolys[pi].init();
            PolygonSerializer::readPolygon(polyIn, leafPolys[pi]);
        }

        if (leafPolys.empty()) {
            skipped++;
            processed++;
            continue;
        }

        // Find BFS index for this leaf
        auto bfsIt = octree.leafKeyToBFSIndex.find(leafIdx.nodeKey);
        if (bfsIt == octree.leafKeyToBFSIndex.end()) {
            skipped++;
            processed++;
            continue;
        }

        analyzeLeaf(octree, bfsIt->second, leafPolys, ctx);

        processed++;
        if (processed - lastReported >= 5000) {
            std::cout << "\r  [Analyze] " << processed << "/" << leafIndices.size()
                      << " leaves (" << (processed * 100 / std::max(leafIndices.size(), 1ULL))
                      << "%)" << std::flush;
            lastReported = processed;
        }
    }

    polyIn.close();
    std::cout << "\r  [Analyze] " << processed << "/" << leafIndices.size()
              << " leaves (100%)";
    if (skipped > 0) std::cout << "  skipped=" << skipped;
    std::cout << std::endl;
}

} // namespace AnalyzePhase
