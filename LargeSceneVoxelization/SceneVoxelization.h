#pragma once
#include "Types.h"
#include "GridData.h"
#include "VoxelData.h"
#include "ABSDF.h"
#include "Ellipsoid.h"
#include "SphericalHarmonics.h"
#include "Estimate.h"
#include "TextureSampler.h"
#include "PolygonGenerator.h"
#include "SceneLoader.h"
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>

struct VoxelizationConfig {
    uint baseResolution = 512;     // N = 2^D
    uint sampleFrequency = 1024;   // rays per Lebedev direction
};

// CPU scene voxelization — instanced loading, single hierarchical clip pass,
// leaf-only polygon storage, bottom-up parent aggregation.
// No tile subdivision — one full octree over the scene.
class SceneVoxelization {
public:
    SceneVoxelization(const VoxelizationConfig& cfg) : mConfig(cfg) {}

    // Main entry point: load FBX (instanced), voxelize, write binary output
    bool process(const std::string& fbxPath, const std::string& outputPath) {
        // ---- Phase 0: Load scene (instanced mode: unique meshes + transforms) ----
        std::cout << "Loading (instanced): " << fbxPath << std::endl;
        SceneLoader loader;
        InstancedScene scene;
        if (!loader.loadMeshInstances(fbxPath, scene)) {
            std::cerr << "Failed to load: " << loader.getError() << std::endl;
            return false;
        }
        std::cout << "  Unique meshes: " << scene.meshes.size()
                  << "  Instances: " << scene.instances.size()
                  << "  Materials: " << scene.materials.size() << std::endl;

        // Compute grid from all instances' world-space bounds
        setupGrid(scene);

        uint resolution = std::max({mGrid.voxelCount.x, mGrid.voxelCount.y, mGrid.voxelCount.z});
        mMaxDepth = 0;
        while ((1u << mMaxDepth) < resolution) mMaxDepth++;

        loadTextures(scene.materials);

        std::cout << "Grid: " << mGrid.voxelCount.x << "^3  voxelSize=" << mGrid.voxelSize.x
                  << "  maxDepth=" << mMaxDepth << std::endl;

        // ---- Phase 1: Hierarchical clip (leaf-only polygon storage) ----
        PolygonGenerator gen(mGrid);
        gen.reset();

        float3 invVoxelSize(1.0f / mGrid.voxelSize.x, 1.0f / mGrid.voxelSize.y, 1.0f / mGrid.voxelSize.z);

        for (uint instIdx = 0; instIdx < (uint)scene.instances.size(); instIdx++) {
            const MeshInstance& inst = scene.instances[instIdx];
            const MeshGeometry& mesh = scene.meshes[inst.meshID];
            const glm::mat4& worldM = inst.transform;
            uint matID = mesh.materialID;

            for (uint localTid = 0; localTid < (uint)mesh.triangles.size(); localTid++) {
                uint3 idx = mesh.triangles[localTid];

                Triangle tri;
                tri.vertices[0] = localToVoxel(mesh.positions[idx.x], worldM, mGrid.gridMin, invVoxelSize);
                tri.vertices[1] = localToVoxel(mesh.positions[idx.y], worldM, mGrid.gridMin, invVoxelSize);
                tri.vertices[2] = localToVoxel(mesh.positions[idx.z], worldM, mGrid.gridMin, invVoxelSize);
                tri.uvs[0] = mesh.texCoords[idx.x];
                tri.uvs[1] = mesh.texCoords[idx.y];
                tri.uvs[2] = mesh.texCoords[idx.z];
                tri.normals[0] = transformNormal(mesh.normals[idx.x], worldM);
                tri.normals[1] = transformNormal(mesh.normals[idx.y], worldM);
                tri.normals[2] = transformNormal(mesh.normals[idx.z], worldM);
                tri.buildTBN();

                clipHierarchical(inst.meshID, matID, localTid, instIdx,
                                tri, tri.calcAABBInt(),
                                int3(0, 0, 0), 0, mMaxDepth, gen);
            }
        }

        // ---- mem: after clip (before BFS, mNodePolygonMap has leaf entries) ----
        {
            uint totalPoly = 0, maxPolyPerNode = 0, nonEmptyNodes = 0;
            for (auto& kv : gen.mNodePolygonMap) {
                uint n = (uint)kv.second.size();
                totalPoly += n;
                maxPolyPerNode = std::max(maxPolyPerNode, n);
                if (n > 0) nonEmptyNodes++;
            }
            std::cout << "  [Mem] after-clip: leafNodes=" << gen.mNodePolygonMap.size()
                      << " totalPoly=" << totalPoly
                      << " maxPolyPerNode=" << maxPolyPerNode << std::endl;
        }

        gen.finalizeBFS(mMaxDepth, 0, int3(0, 0, 0));

        // ---- Phase 2: Per-node analysis (reverse BFS: leaves first) ----
        for (int nodeIdx = (int)gen.gBuffer.size() - 1; nodeIdx >= 0; nodeIdx--) {
            uint level = gen.mBFSOrder[nodeIdx].level;

            if (level == mMaxDepth) {
                // ── LEAF: analyze from own polygon data ──
                VoxelData& vd = gen.gBuffer[nodeIdx];
                PolygonRange& range = gen.polygonRangeBuffer[nodeIdx];

                for (uint pi = 0; pi < range.count; pi++) {
                    const Polygon& poly = gen.polygonArrays[nodeIdx][pi];

                    uint localTid   = poly.triRef.triangleID;
                    uint meshID     = poly.triRef.meshID;
                    uint instIdx    = poly.triRef.instanceIdx;
                    const MeshGeometry& mesh = scene.meshes[meshID];
                    uint matID = mesh.materialID;

                    // Reconstruct original triangle in voxel space
                    uint3 localIdx = mesh.triangles[localTid];
                    const glm::mat4& worldM = scene.instances[instIdx].transform;

                    float3 tv0 = localToVoxel(mesh.positions[localIdx.x], worldM, mGrid.gridMin, invVoxelSize);
                    float3 tv1 = localToVoxel(mesh.positions[localIdx.y], worldM, mGrid.gridMin, invVoxelSize);
                    float3 tv2 = localToVoxel(mesh.positions[localIdx.z], worldM, mGrid.gridMin, invVoxelSize);

                    Triangle origTri;
                    origTri.vertices[0] = tv0; origTri.vertices[1] = tv1; origTri.vertices[2] = tv2;
                    origTri.uvs[0] = mesh.texCoords[localIdx.x];
                    origTri.uvs[1] = mesh.texCoords[localIdx.y];
                    origTri.uvs[2] = mesh.texCoords[localIdx.z];
                    origTri.buildTBN();

                    // Polygon centroid
                    float dummy;
                    float3 centroid = poly.calcCentroid(dummy);
                    float3 leafCentroid = centroid * range.nodeScale;

                    // Interpolate UV on polygon vertices
                    float2 polyUVs[MAX_VERTEX_COUNT];
                    for (uint vi = 0; vi < poly.count; vi++) {
                        float3 leafVertex = poly.vertices[vi] * range.nodeScale;
                        polyUVs[vi] = origTri.lerpUV(leafVertex);
                    }
                    float2 uvCenter(0);
                    for (uint vi = 0; vi < poly.count; vi++)
                        uvCenter += polyUVs[vi];
                    uvCenter /= (float)poly.count;

                    float uvArea = 0;
                    for (uint vi = 0; vi < poly.count; vi++) {
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
                    const MaterialData& mat = scene.materials[matID];
                    float4 baseColorVal = float4(mat.baseColor, 1.0f);
                    float4 specVal = mat.specular;

                    if (matID < mBaseColorTextures.size() &&
                        mBaseColorTextures[matID].width > 0) {
                        baseColorVal = sampleTextureArea(
                            mBaseColorTextures[matID], uvCenter,
                            uvArea, float4(mat.baseColor, 1.0f));
                    }
                    if (matID < mSpecularTextures.size() &&
                        mSpecularTextures[matID].width > 0) {
                        specVal = sampleTextureArea(
                            mSpecularTextures[matID], uvCenter,
                            uvArea, float4(mat.specular.x, mat.specular.y, mat.specular.z, 1.0f));
                    }

                    float3 shadingNormal = interpolatedNormal;
                    if (matID < mNormalMapTextures.size() &&
                        mNormalMapTextures[matID].width > 0) {
                        float4 nm = sampleTextureArea(mNormalMapTextures[matID], uvCenter, uvArea,
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
                    vd.ellipsoid.fit(gen.polygonArrays[nodeIdx], range);

                    SphericalFunc polyF = vd.polygonsProjAreaFunc;
                    SphericalFunc primF = vd.primitiveProjAreaFunc;
                    SphericalFunc totalF = vd.totalProjAreaFunc;

                    Estimate(vd.ellipsoid, range, polyF, primF, totalF,
                             gen.polygonArrays[nodeIdx], mConfig.sampleFrequency);

                    vd.polygonsProjAreaFunc = polyF;
                    vd.primitiveProjAreaFunc = primF;
                    vd.totalProjAreaFunc = totalF;
                }

            } else {
                // ── NON-LEAF: aggregate from children's VoxelData ──
                VoxelData& vd = gen.gBuffer[nodeIdx];
                vd.init();

                const OctreeNode& oct = gen.mOctreeNodes[nodeIdx];
                if (!oct.childMask) continue;

                float rawLobeWeight[LOBE_COUNT] = {};
                float3 rawDiffuse[LOBE_COUNT] = {};
                float3 rawSpecular[LOBE_COUNT] = {};
                float rawRough[LOBE_COUNT] = {};
                float3 rawNormal[LOBE_COUNT] = {};

                {
                    uint childBFS = oct.childBase;
                    for (uint ci = 0; ci < 8; ci++) {
                        if (!(oct.childMask & (1u << ci))) continue;
                        VoxelData& childVd = gen.gBuffer[childBFS];
                        float childArea = childVd.ABSDF.area;

                        vd.ABSDF.area += childArea;

                        for (int li = 0; li < LOBE_COUNT; li++) {
                            float lobeArea = childVd.ABSDF.lobes[li].weight * childArea;
                            if (lobeArea <= 0) continue;

                            rawLobeWeight[li] += lobeArea;
                            rawDiffuse[li]   += childVd.ABSDF.lobes[li].diffuse * lobeArea;
                            rawSpecular[li]  += childVd.ABSDF.lobes[li].specular * lobeArea;
                            rawRough[li]     += childVd.ABSDF.lobes[li].rough * lobeArea;
                            rawNormal[li]    += childVd.ABSDF.lobes[li].normal * lobeArea;
                        }

                        // SH: sum children coefficients (placeholder for full leaf iteration)
                        for (int si = 0; si < EVEN_SH_COUNT; si++) {
                            vd.polygonsProjAreaFunc.coefficients[si]  += childVd.polygonsProjAreaFunc.coefficients[si];
                            vd.primitiveProjAreaFunc.coefficients[si] += childVd.primitiveProjAreaFunc.coefficients[si];
                            vd.totalProjAreaFunc.coefficients[si]     += childVd.totalProjAreaFunc.coefficients[si];
                        }

                        childBFS++;
                    }
                }

                for (int li = 0; li < LOBE_COUNT; li++) {
                    if (rawLobeWeight[li] <= 0) continue;
                    vd.ABSDF.lobes[li].diffuse  = rawDiffuse[li] / rawLobeWeight[li];
                    vd.ABSDF.lobes[li].specular = rawSpecular[li] / rawLobeWeight[li];
                    vd.ABSDF.lobes[li].rough    = rawRough[li] / rawLobeWeight[li];
                    vd.ABSDF.lobes[li].normal   = safeNormalize(rawNormal[li]);
                    vd.ABSDF.lobes[li].weight   = rawLobeWeight[li] / vd.ABSDF.area;
                }

                if (!vd.isSolid()) continue;

                // Ellipsoid: approximate from child ellipsoid extreme points
                {
                    std::vector<float3> extPoints;
                    float parentScale = (float)(1u << (mMaxDepth - level));

                    uint childBFS = oct.childBase;
                    for (uint ci = 0; ci < 8; ci++) {
                        if (!(oct.childMask & (1u << ci))) continue;
                        VoxelData& childVd = gen.gBuffer[childBFS];
                        Ellipsoid& childE = childVd.ellipsoid;
                        if (childE.B[0][0] == 0 && childE.B[1][1] == 0 && childE.B[2][2] == 0) {
                            childBFS++; continue;
                        }

                        int3 childCell = gen.mBFSOrder[childBFS].cellInt;
                        float3 childCenter = float3(childCell) + childE.center;

                        float3x3 R;
                        float3 evals;
                        Ellipsoid::eigenSym3_Jacobi(childE.B, R, evals);

                        for (int a = 0; a < 3; a++) {
                            if (evals[a] <= 0) continue;
                            float extent = 1.0f / std::sqrt(evals[a]);
                            float3 dir(R[a].x, R[a].y, R[a].z);

                            float3 p1 = (childCenter + dir * extent - float3(gen.mBFSOrder[nodeIdx].cellInt)) / parentScale;
                            float3 p2 = (childCenter - dir * extent - float3(gen.mBFSOrder[nodeIdx].cellInt)) / parentScale;
                            extPoints.push_back(p1);
                            extPoints.push_back(p2);
                        }

                        childBFS++;
                    }

                    if (extPoints.size() >= 6)
                        vd.ellipsoid.fitFromPoints(extPoints, gen.mBFSOrder[nodeIdx].cellInt);
                }
            }
        }

        // ---- Phase 3: Write output ----
        std::cout << "Writing output: " << outputPath << std::endl;
        writeOutput(outputPath, gen);

        // Free polygon data
        gen.polygonArrays.clear();
        gen.polygonArrays.shrink_to_fit();
        gen.mNodePolygonMap.clear();

        std::cout << "Done." << std::endl;
        return true;
    }

private:
    VoxelizationConfig mConfig;
    GridData mGrid;
    uint mMaxDepth = 0;

    // Loaded textures: materialID -> texture
    std::vector<Texture2D> mBaseColorTextures;
    std::vector<Texture2D> mSpecularTextures;
    std::vector<Texture2D> mNormalMapTextures;

    // ---- Transform helpers ----

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

    // ---- Grid setup ----

    void setupGrid(const InstancedScene& scene) {
        float3 sceneMin(1e30f), sceneMax(-1e30f);

        for (const auto& inst : scene.instances) {
            const auto& mesh = scene.meshes[inst.meshID];
            float3 corners[8] = {
                float3(mesh.localMin.x, mesh.localMin.y, mesh.localMin.z),
                float3(mesh.localMin.x, mesh.localMin.y, mesh.localMax.z),
                float3(mesh.localMin.x, mesh.localMax.y, mesh.localMin.z),
                float3(mesh.localMin.x, mesh.localMax.y, mesh.localMax.z),
                float3(mesh.localMax.x, mesh.localMin.y, mesh.localMin.z),
                float3(mesh.localMax.x, mesh.localMin.y, mesh.localMax.z),
                float3(mesh.localMax.x, mesh.localMax.y, mesh.localMin.z),
                float3(mesh.localMax.x, mesh.localMax.y, mesh.localMax.z),
            };
            for (int c = 0; c < 8; c++) {
                glm::vec4 wp = inst.transform * glm::vec4(corners[c], 1.0f);
                sceneMin = glm::min(sceneMin, float3(wp.x, wp.y, wp.z));
                sceneMax = glm::max(sceneMax, float3(wp.x, wp.y, wp.z));
            }
        }

        float3 diag_scene = sceneMax - sceneMin;
        float3 diag = diag_scene * 1.02f;
        float3 center = (sceneMin + sceneMax) * 0.5f;

        std::cout << "  [Grid] bounds min=(" << sceneMin.x << "," << sceneMin.y << "," << sceneMin.z
                  << ") max=(" << sceneMax.x << "," << sceneMax.y << "," << sceneMax.z << ")" << std::endl;

        uint N = mConfig.baseResolution;
        N--; N |= N >> 1; N |= N >> 2; N |= N >> 4;
        N |= N >> 8; N |= N >> 16; N++;
        mGrid.voxelCount = uint3(N, N, N);

        float maxDim = std::max(diag.z, std::max(diag.x, diag.y));
        float s = maxDim / (float)N;
        mGrid.voxelSize = float3(s);
        mGrid.gridMin = center - 0.5f * s * float3(N);

        std::cout << "  [Grid] N=" << N << " maxDim=" << maxDim << " voxelSize=" << s << std::endl;
    }

    void loadTextures(const std::vector<MaterialData>& materials) {
        mBaseColorTextures.resize(materials.size());
        mSpecularTextures.resize(materials.size());
        mNormalMapTextures.resize(materials.size());
        for (size_t i = 0; i < materials.size(); i++) {
            if (!materials[i].texBaseColor.empty())
                mBaseColorTextures[i].load(materials[i].texBaseColor, true);
            if (!materials[i].texSpecular.empty())
                mSpecularTextures[i].load(materials[i].texSpecular);
            if (!materials[i].texNormalMap.empty())
                mNormalMapTextures[i].load(materials[i].texNormalMap);
        }
    }

    // ---- Hierarchical clip (leaf-only polygon storage) ----
    // Exact BoxClipTriangle at every level for correct AABB test.
    // Polygon is stored ONLY at leaf level (direct push to mNodePolygonMap).

    static void clipHierarchical(
        uint meshID, uint materialID, uint triangleID, uint instanceIdx,
        Triangle& tri, const AABBInt& triAABB, const int3& nodeCell,
        uint level, uint maxDepth,
        PolygonGenerator& gen) {

        uint scale = 1u << (maxDepth - level);
        float3 minPoint = float3(nodeCell) * (float)scale;
        float3 maxPoint = minPoint + float3((float)scale);

        Polygon polygon = VoxelizationUtility::BoxClipTriangle(minPoint, maxPoint, tri);
        if (polygon.count < 3 || polygon.calcArea() <= 1e-8f) return;

        float invScale = 1.0f / (float)scale;
        for (uint vi = 0; vi < polygon.count; vi++)
            polygon.vertices[vi] *= invScale;

        polygon.normal = {tri.TBN[0].z, tri.TBN[1].z, tri.TBN[2].z};
        polygon.triRef.meshID = meshID;
        polygon.triRef.triangleID = triangleID;
        polygon.triRef.materialID = materialID;
        polygon.triRef.instanceIdx = instanceIdx;

        uint64_t nodeKey = PolygonGenerator::makeNodeKey(level, nodeCell);
        gen.mOccupiedNodes.insert(nodeKey);

        if (level == maxDepth) {
            // Leaf: directly push polygon to map
            auto& polys = gen.mNodePolygonMap[nodeKey];
            if (polys.size() < SAFE_PER_NODE_POLYGON_LIMIT)
                polys.push_back(polygon);
        }

        if (level >= maxDepth) return;

        int childScale = (int)(scale >> 1);
        for (uint ci = 0; ci < 8; ci++) {
            int3 childCell = nodeCell * 2 + int3((int)(ci & 1), (int)((ci >> 1) & 1), (int)((ci >> 2) & 1));
            int3 childMin = childCell * childScale;
            int3 childMax = childMin + childScale - 1;
            if (triAABB.xMax < childMin.x || triAABB.xMin > childMax.x ||
                triAABB.yMax < childMin.y || triAABB.yMin > childMax.y ||
                triAABB.zMax < childMin.z || triAABB.zMin > childMax.z)
                continue;
            clipHierarchical(meshID, materialID, triangleID, instanceIdx, tri, triAABB,
                            childCell, level + 1, maxDepth, gen);
        }
    }

    // ---- Output writer ----

    void writeOutput(const std::string& outputPath, const PolygonGenerator& gen) {
        std::ofstream f(outputPath, std::ios::binary);
        if (!f) {
            std::cerr << "Cannot open output: " << outputPath << std::endl;
            return;
        }

        uint totalNodes = (uint)gen.gBuffer.size();

        std::vector<uint32_t> nodeCounts(mMaxDepth + 1, 0);
        for (auto& bfn : gen.mBFSOrder)
            nodeCounts[bfn.level]++;

        uint totalPoly = 0, maxPolyPerNode = 0, nonEmptyNodes = 0;
        for (auto& pa : gen.polygonArrays) {
            uint pc = (uint)pa.size();
            totalPoly += pc;
            maxPolyPerNode = std::max(maxPolyPerNode, pc);
            if (pc > 0) nonEmptyNodes++;
        }
        std::cout << "  [Output] nodes=" << totalNodes << " totalPoly=" << totalPoly
                  << " maxPolyPerNode=" << maxPolyPerNode << std::endl;
        std::cout << "  [Output] per-level:";
        for (uint l = 0; l <= mMaxDepth; l++)
            std::cout << " L" << l << "=" << nodeCounts[l];
        std::cout << std::endl;

        // Consistency check
        {
            uint errors = 0;
            uint offset = 0;
            for (uint l = 0; l <= mMaxDepth; l++) {
                uint count = nodeCounts[l];
                uint nextOff = offset + count;
                for (uint i = offset; i < nextOff; i++) {
                    if (gen.mOctreeNodes[i].dataIndex != i) {
                        if (errors < 5) std::cerr << "  [CONSISTENCY] dataIndex mismatch at node[" << i << "]" << std::endl;
                        errors++;
                    }
                    if (gen.mOctreeNodes[i].childMask) {
                        uint childCount = countbits(gen.mOctreeNodes[i].childMask);
                        uint childStart = gen.mOctreeNodes[i].childBase;
                        uint childLevelOff = 0;
                        for (uint cl = 0; cl <= l; cl++) childLevelOff += nodeCounts[cl];
                        if (childStart < childLevelOff || childStart + childCount > childLevelOff + nodeCounts[l+1]) {
                            if (errors < 5) std::cerr << "  [CONSISTENCY] childBase OOB at node[" << i << "]" << std::endl;
                            errors++;
                        }
                    }
                }
                offset = nextOff;
            }
            if (errors == 0)
                std::cout << "  [CONSISTENCY] All checks passed." << std::endl;
            else
                std::cerr << "  [CONSISTENCY] " << errors << " errors found!" << std::endl;
        }

        GridData outGrid = mGrid;
        outGrid.solidVoxelCount = totalNodes;
        outGrid.maxPolygonCount = maxPolyPerNode;
        outGrid.totalPolygonCount = totalPoly;

        f.write((const char*)&outGrid, sizeof(GridData));
        f.write((const char*)&mMaxDepth, sizeof(uint32_t));
        f.write((const char*)nodeCounts.data(), (mMaxDepth + 1) * sizeof(uint32_t));
        f.write((const char*)gen.mOctreeNodes.data(), totalNodes * sizeof(OctreeNode));
        f.write((const char*)gen.gBuffer.data(), totalNodes * sizeof(VoxelData));

        f.close();
        std::cout << "Wrote " << totalNodes << " nodes." << std::endl;
    }
};
