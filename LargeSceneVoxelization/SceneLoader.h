#pragma once
#include "Types.h"
#include "Triangle.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <cctype>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

struct MeshData {
    uint meshID = 0;
    uint materialID = 0;
    uint vertexCount = 0;
    uint triangleCount = 0;
    uint triangleOffset = 0;  // offset into global triangle index buffer
    std::string name;
};

struct MaterialData {
    uint materialID = 0;
    float3 baseColor = float3(1.0f);
    float4 specular = float4(0.04f, 1.0f, 0.0f, 1.0f); // F0, roughness, metallic, _
    std::string texBaseColor;    // path to base color texture
    std::string texSpecular;     // path to specular/roughness/metallic texture
    std::string texNormalMap;
};

struct LoadedScene {
    // Geometry (world-space, after applying node transforms)
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> texCoords;
    std::vector<uint3>  triangles;       // indices into positions/normals/texCoords
    std::vector<TriangleRef> triRefs;    // per-triangle material/mesh reference

    // Mesh & material metadata
    std::vector<MeshData> meshes;
    std::vector<MaterialData> materials;

    // Scene bounds
    float3 sceneMin = float3(1e30f);
    float3 sceneMax = float3(-1e30f);

    uint totalVertices() const { return (uint)positions.size(); }
    uint totalTriangles() const { return (uint)triangles.size(); }
};

// Load an FBX (or any Assimp-supported) scene with instancing support.
// Each mesh is expanded to world-space using its node transform.
// Instanced meshes (same aiMesh referenced by multiple nodes) reuse the
// same MeshData::meshID but produce separate geometry instances.
class SceneLoader {
public:
    bool load(const std::string& filePath, LoadedScene& scene) {
        Assimp::Importer importer;
        // Match Falcor's AssimpImporter flags exactly
        uint32_t flags = aiProcessPreset_TargetRealtime_MaxQuality
                       | aiProcess_FlipUVs
                       | aiProcess_RemoveComponent;
        flags &= ~(aiProcess_CalcTangentSpace);         // build TBN ourselves
        flags &= ~(aiProcess_FindDegenerates);          // don't convert degenerate tris to lines
        flags &= ~(aiProcess_OptimizeGraph);            // broken with negative-determinant transforms
        flags &= ~(aiProcess_RemoveRedundantMaterials); // merge in SceneBuilder instead
        flags &= ~(aiProcess_SplitLargeMeshes);         // don't split large meshes

        // Remove unsupported vertex components (helps JoinIdenticalVertices)
        int removeFlags = aiComponent_COLORS;
        for (uint32_t uvLayer = 1; uvLayer < AI_MAX_NUMBER_OF_TEXTURECOORDS; uvLayer++)
            removeFlags |= aiComponent_TEXCOORDSn(uvLayer);
        removeFlags |= aiComponent_TANGENTS_AND_BITANGENTS;
        importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, removeFlags);

        const aiScene* ai = importer.ReadFile(filePath, flags);
        if (!ai || !ai->mRootNode) {
            mError = importer.GetErrorString();
            return false;
        }

        mSceneDir = filePath.substr(0, filePath.find_last_of("/\\"));

        // Phase 1: Collect materials
        loadMaterials(ai, scene);

        // Phase 2: Traverse nodes, collect mesh instances with transforms
        std::vector<InstanceInfo> instances;
        traverseNodes(ai->mRootNode, ai, aiMatrix4x4(), instances);

        // Phase 3: Expand each instance to world-space geometry
        for (auto& inst : instances) {
            addMeshInstance(ai, inst, scene);
        }

        // Phase 4: Setup mesh triangle offsets
        uint triOffset = 0;
        for (auto& m : scene.meshes) {
            m.triangleOffset = triOffset;
            triOffset += m.triangleCount;
        }

        // ---- Debug: scene loading stats ----
        std::cout << "  [LoadDebug] Materials: " << scene.materials.size() << std::endl;
        for (size_t i = 0; i < scene.materials.size(); i++) {
            auto& mat = scene.materials[i];
            std::cout << "    mat[" << i << "] baseColor=(" << mat.baseColor.x << ","
                      << mat.baseColor.y << "," << mat.baseColor.z
                      << ") roughness=" << mat.specular.g
                      << " metallic=" << mat.specular.b << std::endl;
        }

        // Count instances per aiMesh
        std::vector<uint> meshInstanceCount(ai->mNumMeshes, 0);
        for (auto& inst : instances)
            meshInstanceCount[inst.meshIndex]++;

        std::cout << "  [LoadDebug] aiMeshes: " << ai->mNumMeshes
                  << "  meshInstances: " << instances.size() << std::endl;
        for (uint mi = 0; mi < ai->mNumMeshes; mi++) {
            aiMesh* aim = ai->mMeshes[mi];
            uint matID = aim->mMaterialIndex;
            std::cout << "    aiMesh[" << mi << "] \"" << aim->mName.C_Str()
                      << "\" verts=" << aim->mNumVertices
                      << " tris=" << aim->mNumFaces
                      << " mat=" << matID
                      << " instances=" << meshInstanceCount[mi] << std::endl;
        }

        return true;
    }

    const std::string& getError() const { return mError; }

private:
    std::string mError;
    std::string mSceneDir;

    struct InstanceInfo {
        uint meshIndex;          // aiMesh index
        aiMatrix4x4 transform;   // world transform
        uint materialID;
    };

    void loadMaterials(const aiScene* ai, LoadedScene& scene) {
        for (uint i = 0; i < ai->mNumMaterials; i++) {
            aiMaterial* aimat = ai->mMaterials[i];
            MaterialData mat;
            mat.materialID = i;

            aiColor4D col;
            if (aimat->Get(AI_MATKEY_COLOR_DIFFUSE, col) == AI_SUCCESS)
                mat.baseColor = srgbToLinear(float3(col.r, col.g, col.b));

            // PBR roughness: prefer explicit roughness factor, fall back to shininess
            float roughness = 1.0f;
            if (aimat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) != AI_SUCCESS)
            {
                float shininess = 0;
                if (aimat->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS)
                    roughness = 1.0f - std::sqrt(shininess / 1000.0f);
            }
            mat.specular.g = roughness;

            // Metallic: FBX uses PBR metallic; check for $raw.roughness and $raw.metalness
            float metallic = 0;
            if (aimat->Get(AI_MATKEY_METALLIC_FACTOR, metallic) == AI_SUCCESS)
                mat.specular.b = metallic;

            // Texture paths: handle both relative and absolute paths from Assimp
            auto resolvePath = [&](const aiString& p) -> std::string {
                std::string s(p.C_Str());
                if (s.empty()) return s;
                // Already absolute (Windows drive letter or Unix root)
                if (s.size() >= 2 && s[1] == ':') return s;
                if (s[0] == '/') return s;
                return mSceneDir + "/" + s;
            };

            // Enumerate all texture slots (Assimp may classify PBR textures under various types)
            for (uint texType = 0; texType <= aiTextureType_UNKNOWN; texType++) {
                uint texCount = aimat->GetTextureCount((aiTextureType)texType);
                for (uint slot = 0; slot < texCount; slot++) {
                    aiString texPath;
                    if (aimat->GetTexture((aiTextureType)texType, slot, &texPath) != AI_SUCCESS)
                        continue;
                    std::string rawPath(texPath.C_Str());

                    if (texType == aiTextureType_DIFFUSE || texType == aiTextureType_BASE_COLOR) {
                        if (mat.texBaseColor.empty())
                            mat.texBaseColor = resolvePath(texPath);
                    } else if (texType == aiTextureType_NORMALS || texType == aiTextureType_HEIGHT || texType == aiTextureType_DISPLACEMENT) {
                        if (mat.texNormalMap.empty())
                            mat.texNormalMap = resolvePath(texPath);
                    } else if (texType == aiTextureType_SPECULAR || texType == aiTextureType_SHININESS
                               || texType == aiTextureType_METALNESS) {
                        if (mat.texSpecular.empty())
                            mat.texSpecular = resolvePath(texPath);
                    } else if (texType == aiTextureType_UNKNOWN) {
                        // Heuristic: match by filename keywords
                        std::string lower = rawPath;
                        for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
                        if (lower.find("normal") != std::string::npos || lower.find("bump") != std::string::npos) {
                            if (mat.texNormalMap.empty())
                                mat.texNormalMap = resolvePath(texPath);
                        } else if (lower.find("rough") != std::string::npos || lower.find("metal") != std::string::npos
                                   || lower.find("spec") != std::string::npos) {
                            if (mat.texSpecular.empty())
                                mat.texSpecular = resolvePath(texPath);
                        }
                    }
                }
            }

            // Try $raw custom properties (Blender FBX exporter uses these for PBR)
            {
                aiString raw;
                if (aimat->Get("$raw.basecolor_texture", 0, 0, raw) == AI_SUCCESS && mat.texBaseColor.empty())
                    mat.texBaseColor = resolvePath(raw);
                if (aimat->Get("$raw.roughness_texture", 0, 0, raw) == AI_SUCCESS && mat.texSpecular.empty())
                    mat.texSpecular = resolvePath(raw);
                if (aimat->Get("$raw.normal_texture", 0, 0, raw) == AI_SUCCESS && mat.texNormalMap.empty())
                    mat.texNormalMap = resolvePath(raw);
                if (aimat->Get("$raw.metallic_texture", 0, 0, raw) == AI_SUCCESS && mat.texSpecular.empty())
                    mat.texSpecular = resolvePath(raw);
            }

            // Print all texture slots for debugging
            std::cout << "    mat[" << i << "] textures:";
            if (!mat.texBaseColor.empty()) std::cout << " baseColor=" << mat.texBaseColor;
            if (!mat.texSpecular.empty()) std::cout << " specular=" << mat.texSpecular;
            if (!mat.texNormalMap.empty()) std::cout << " normal=" << mat.texNormalMap;
            if (mat.texBaseColor.empty() && mat.texSpecular.empty() && mat.texNormalMap.empty())
                std::cout << " (none)";
            std::cout << std::endl;

            scene.materials.push_back(mat);
        }
    }

    void traverseNodes(aiNode* node, const aiScene* ai, const aiMatrix4x4& parentTransform,
                       std::vector<InstanceInfo>& instances) {
        aiMatrix4x4 world = parentTransform * node->mTransformation;

        for (uint i = 0; i < node->mNumMeshes; i++) {
            uint meshIdx = node->mMeshes[i];
            uint matID = ai->mMeshes[meshIdx]->mMaterialIndex;
            instances.push_back({meshIdx, world, matID});
        }

        for (uint i = 0; i < node->mNumChildren; i++)
            traverseNodes(node->mChildren[i], ai, world, instances);
    }

    void addMeshInstance(const aiScene* ai, const InstanceInfo& inst, LoadedScene& scene) {
        aiMesh* aim = ai->mMeshes[inst.meshIndex];
        if (!aim->HasPositions()) return;

        uint baseVertex = (uint)scene.positions.size();

        // Expand vertices to world-space
        for (uint v = 0; v < aim->mNumVertices; v++) {
            aiVector3D pos = inst.transform * aim->mVertices[v];
            scene.positions.push_back(float3(pos.x, pos.y, pos.z));
            scene.sceneMin = glm::min(scene.sceneMin, float3(pos.x, pos.y, pos.z));
            scene.sceneMax = glm::max(scene.sceneMax, float3(pos.x, pos.y, pos.z));

            aiVector3D nrm = aim->HasNormals() ?
                (inst.transform * aiVector3D(aim->mNormals[v].x, aim->mNormals[v].y, aim->mNormals[v].z)).Normalize()
                : aiVector3D(0, 1, 0);
            scene.normals.push_back(float3(nrm.x, nrm.y, nrm.z));

            if (aim->HasTextureCoords(0))
                scene.texCoords.push_back(float2(aim->mTextureCoords[0][v].x, aim->mTextureCoords[0][v].y));
            else
                scene.texCoords.push_back(float2(0));
        }

        // Add triangles
        for (uint f = 0; f < aim->mNumFaces; f++) {
            const aiFace& face = aim->mFaces[f];
            if (face.mNumIndices == 3) {
                scene.triangles.push_back(uint3(
                    baseVertex + face.mIndices[0],
                    baseVertex + face.mIndices[1],
                    baseVertex + face.mIndices[2]
                ));
                TriangleRef ref;
                ref.meshID = scene.meshes.size(); // new mesh instance
                ref.materialID = inst.materialID;
                ref.triangleID = f;
                scene.triRefs.push_back(ref);
            }
        }

        // Register mesh metadata (one per instance — supports instancing)
        MeshData md;
        md.meshID = (uint)scene.meshes.size();
        md.materialID = inst.materialID;
        md.vertexCount = aim->mNumVertices;
        md.triangleCount = aim->mNumFaces;
        md.name = aim->mName.C_Str();
        scene.meshes.push_back(md);
    }
};
