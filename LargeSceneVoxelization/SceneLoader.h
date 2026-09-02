#pragma once
#include "Types.h"
#include "Triangle.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <cctype>
#include <fstream>
#include <filesystem>
#include <iostream>
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
    // Same storage as Falcor's BasicMaterialData. In the default Assimp
    // import path (including FBX), the GPU importer writes Phong
    // SpecularColor to RGB and Shininess to A without converting it to PBR.
    float4 specular = float4(0.0f); // default BasicMaterialData value
    std::string texBaseColor;    // path to base color texture
    std::string texSpecular;     // path to ORM combined / roughness-only / Phong specular
    std::string texMetallic;     // path to separate metallic texture (FBX Blender PBR)
    std::string texNormalMap;
    bool isSpecGloss = false;    // Explicitly selected SpecGloss input; converted to metal-rough for ABSDF
};

enum class MaterialImportMode {
    Default,
    OBJ,
    GLTF2,
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

// ---- Instanced mode: unique meshes in local space ----
struct MeshGeometry {
    uint meshID = 0;
    uint materialID = 0;
    uint aiMeshIndex = 0;  // Assimp mesh index for dedup
    std::string name;
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> texCoords;
    std::vector<uint3>  triangles;
    float3 localMin = float3(1e30f);
    float3 localMax = float3(-1e30f);
    uint vertexCount() const { return (uint)positions.size(); }
    uint triangleCount() const { return (uint)triangles.size(); }
};


// TODO:Add material in MeshInstance
struct MeshInstance {
    uint meshID;
    glm::mat4 transform;  // 4x4 world transform (column-major, like GLM/GPU)
};

struct InstancedScene {
    std::vector<MeshGeometry> meshes;
    std::vector<MeshInstance> instances;
    std::vector<MaterialData> materials;
};

// Load an FBX (or any Assimp-supported) scene with instancing support.
// Each mesh is expanded to world-space using its node transform.
// Instanced meshes (same aiMesh referenced by multiple nodes) reuse the
// same MeshData::meshID but produce separate geometry instances.
class SceneLoader {
public:
    explicit SceneLoader(bool useSpecGlossMaterials = false)
        : mUseSpecGlossMaterials(useSpecGlossMaterials) {}

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

        // Phase 1: Collect materials. Keep the same format-specific branches
        // as Falcor's AssimpImporter.
        loadMaterials(ai, scene.materials, getImportMode(filePath));

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
                      << ") specular=(" << mat.specular.x << ","
                      << mat.specular.y << "," << mat.specular.z << ","
                      << mat.specular.w << ") roughness=" << mat.specular.g
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

    // Load FBX, extract unique meshes in local space + instance transforms
    bool loadMeshInstances(const std::string& filePath, InstancedScene& outScene) {
        Assimp::Importer importer;
        uint32_t flags = aiProcessPreset_TargetRealtime_MaxQuality
                       | aiProcess_FlipUVs
                       | aiProcess_RemoveComponent;
        flags &= ~(aiProcess_CalcTangentSpace);
        flags &= ~(aiProcess_FindDegenerates);
        flags &= ~(aiProcess_OptimizeGraph);
        flags &= ~(aiProcess_RemoveRedundantMaterials);
        flags &= ~(aiProcess_SplitLargeMeshes);

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

        // Phase 1: Load materials
        loadMaterials(ai, outScene.materials, getImportMode(filePath));

        // Phase 2: Collect all instances (aiMeshIndex, transform, materialID)
        struct RawInstance { uint meshIndex; aiMatrix4x4 transform; uint materialID; };
        std::vector<RawInstance> rawInstances;
        auto collectInstances = [&](aiNode* node, const aiMatrix4x4& parentXf, auto& self) -> void {
            aiMatrix4x4 world = parentXf * node->mTransformation;
            for (uint i = 0; i < node->mNumMeshes; i++) {
                uint meshIdx = node->mMeshes[i];
                uint matID = ai->mMeshes[meshIdx]->mMaterialIndex;
                rawInstances.push_back({meshIdx, world, matID});
            }
            for (uint i = 0; i < node->mNumChildren; i++)
                self(node->mChildren[i], world, self);
        };
        collectInstances(ai->mRootNode, aiMatrix4x4(), collectInstances);

        // Phase 3: Deduplicate unique meshes by aiMeshIndex
        std::unordered_map<uint, uint> aiMeshToUnique;  // aiMeshIndex -> uniqueMeshID
        for (auto& ri : rawInstances) {
            if (aiMeshToUnique.find(ri.meshIndex) == aiMeshToUnique.end()) {
                uint uid = (uint)outScene.meshes.size();
                aiMeshToUnique[ri.meshIndex] = uid;
                aiMesh* aim = ai->mMeshes[ri.meshIndex];
                MeshGeometry geom;
                geom.meshID = uid;
                geom.materialID = ri.materialID;
                geom.aiMeshIndex = ri.meshIndex;
                geom.name = aim->mName.C_Str();
                for (uint v = 0; v < aim->mNumVertices; v++) {
                    float3 pos(aim->mVertices[v].x, aim->mVertices[v].y, aim->mVertices[v].z);
                    geom.positions.push_back(pos);
                    geom.localMin = glm::min(geom.localMin, pos);
                    geom.localMax = glm::max(geom.localMax, pos);
                    if (aim->HasNormals())
                        geom.normals.push_back(float3(aim->mNormals[v].x, aim->mNormals[v].y, aim->mNormals[v].z));
                    else
                        geom.normals.push_back(float3(0, 1, 0));
                    if (aim->HasTextureCoords(0))
                        geom.texCoords.push_back(float2(aim->mTextureCoords[0][v].x, aim->mTextureCoords[0][v].y));
                    else
                        geom.texCoords.push_back(float2(0));
                }
                for (uint f = 0; f < aim->mNumFaces; f++) {
                    if (aim->mFaces[f].mNumIndices == 3)
                        geom.triangles.push_back(uint3(aim->mFaces[f].mIndices[0],
                                                        aim->mFaces[f].mIndices[1],
                                                        aim->mFaces[f].mIndices[2]));
                }
                outScene.meshes.push_back(std::move(geom));
            }
        }

        // Phase 4: Build instance list with unique mesh IDs
        for (auto& ri : rawInstances) {
            MeshInstance inst;
            inst.meshID = aiMeshToUnique[ri.meshIndex];
            inst.transform = glm::mat4(
                ri.transform.a1, ri.transform.b1, ri.transform.c1, ri.transform.d1,
                ri.transform.a2, ri.transform.b2, ri.transform.c2, ri.transform.d2,
                ri.transform.a3, ri.transform.b3, ri.transform.c3, ri.transform.d3,
                ri.transform.a4, ri.transform.b4, ri.transform.c4, ri.transform.d4
            );
            outScene.instances.push_back(inst);
        }

        std::cout << "  [InstancedLoad] Unique meshes: " << outScene.meshes.size()
                  << "  Instances: " << outScene.instances.size()
                  << "  Materials: " << outScene.materials.size() << std::endl;
        for (auto& geom : outScene.meshes) {
            std::cout << "    mesh[" << geom.meshID << "] \"" << geom.name
                      << "\" verts=" << geom.positions.size()
                      << " tris=" << geom.triangles.size()
                      << " mat=" << geom.materialID << std::endl;
        }
        return true;
    }

    const std::string& getError() const { return mError; }

private:
    std::string mError;
    std::string mSceneDir;
    bool mUseSpecGlossMaterials = false;

    // Cache of extracted embedded textures (Assimp *N paths → temp file paths)
    std::unordered_map<int, std::string> mEmbeddedTempFiles;

    // Extract an embedded texture from aiScene and write to a temp file.
    // Returns the temp file path, or empty string on failure.
    std::string resolveEmbedded(const aiScene* ai, const std::string& texPath) {
        if (texPath.size() < 2 || texPath[0] != '*')
            return "";

        for (size_t i = 1; i < texPath.size(); i++)
            if (!std::isdigit(static_cast<unsigned char>(texPath[i])))
                return "";

        int index = std::stoi(texPath.substr(1));
        if (index < 0 || index >= (int)ai->mNumTextures)
            return "";

        // Check cache first
        auto it = mEmbeddedTempFiles.find(index);
        if (it != mEmbeddedTempFiles.end())
            return it->second;

        const aiTexture* tex = ai->mTextures[index];
        // mHeight == 0 → compressed (PNG/JPEG), mWidth = byte size
        if (tex->mHeight != 0 || !tex->pcData || tex->mWidth == 0)
            return "";

        const char* fmt = tex->achFormatHint[0] != '\0' ? tex->achFormatHint : "png";
        auto tmp = std::filesystem::temp_directory_path() /
            ("vox_embedded_" + std::to_string(index) + "." + fmt);

        std::ofstream ofs(tmp, std::ios::binary);
        if (!ofs.is_open())
            return "";
        ofs.write(reinterpret_cast<const char*>(tex->pcData), tex->mWidth);
        ofs.close();

        mEmbeddedTempFiles[index] = tmp.string();
        return tmp.string();
    }

    // Get the original filename of an embedded texture (for heuristic matching).
    std::string embeddedOriginalName(const aiScene* ai, const std::string& texPath) const {
        if (texPath.size() < 2 || texPath[0] != '*')
            return "";
        for (size_t i = 1; i < texPath.size(); i++)
            if (!std::isdigit(static_cast<unsigned char>(texPath[i])))
                return "";
        int index = std::stoi(texPath.substr(1));
        if (index < 0 || index >= (int)ai->mNumTextures)
            return "";
        return std::string(ai->mTextures[index]->mFilename.C_Str());
    }

    struct InstanceInfo {
        uint meshIndex;          // aiMesh index
        aiMatrix4x4 transform;   // world transform
        uint materialID;
    };

    static MaterialImportMode getImportMode(const std::string& filePath) {
        std::string extension = std::filesystem::path(filePath).extension().string();
        for (char& c : extension)
            c = (char)std::tolower((unsigned char)c);

        if (extension == ".obj")  return MaterialImportMode::OBJ;
        if (extension == ".gltf" || extension == ".glb") return MaterialImportMode::GLTF2;
        return MaterialImportMode::Default;
    }

    void loadMaterials(const aiScene* ai, std::vector<MaterialData>& materials,
                       MaterialImportMode importMode) {
        for (uint i = 0; i < ai->mNumMaterials; i++) {
            aiMaterial* aimat = ai->mMaterials[i];
            MaterialData mat;
            mat.materialID = i;

            // Match AssimpImporter::createMaterial() ordering and value space.
            // Falcor stores imported material colors directly; texture samples
            // are decoded separately by the texture system when appropriate.
            aiColor3D color;
            if (aimat->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS)
                mat.baseColor = float3(color.r, color.g, color.b);

            float shininess = 0.0f;
            bool hasShininess = (aimat->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS);
            if (hasShininess) {
                // This is the only special conversion performed by the GPU
                // importer, and it is only for OBJ/MTL.
                if (importMode == MaterialImportMode::OBJ) {
                    float roughness = std::clamp(std::sqrt(2.0f / (shininess + 2.0f)), 0.0f, 1.0f);
                    shininess = 1.0f - roughness;
                }
                mat.specular.w = shininess;
            }

            if (aimat->Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS)
                mat.specular = float4(color.r, color.g, color.b, mat.specular.w);

            // GLTF2 is the only path in Falcor's importer that reads explicit
            // metallic/roughness factors and the PBR base-color factor.
            if (importMode == MaterialImportMode::GLTF2) {
                if (aimat->Get(AI_MATKEY_BASE_COLOR, color) == AI_SUCCESS)
                    mat.baseColor = float3(color.r, color.g, color.b);

                float metallic = 0.0f;
                if (aimat->Get(AI_MATKEY_METALLIC_FACTOR, metallic) == AI_SUCCESS)
                    mat.specular.b = metallic;

                float roughness = 0.0f;
                if (aimat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) == AI_SUCCESS)
                    mat.specular.g = roughness;
            }

            if (mUseSpecGlossMaterials) {
                // SpecGloss is opt-in, analogous to Falcor's
                // SceneBuilder::Flags::UseSpecGlossMaterials. Store the source
                // representation correctly: RGB = specular color, A = glossiness.
                aiColor4D specColor;
                float3 specularColor(0.04f);
                if (aimat->Get(AI_MATKEY_COLOR_SPECULAR, specColor) == AI_SUCCESS)
                    specularColor = float3(specColor.r, specColor.g, specColor.b);

                float roughness = 1.0f;
                bool hasPbrRoughness = (aimat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) == AI_SUCCESS);
                if (!hasPbrRoughness && hasShininess)
                    roughness = 1.0f - std::sqrt(shininess / 1000.0f);

                mat.isSpecGloss = true;
                mat.specular = float4(specularColor, 1.0f - roughness);
            }

            // Texture paths: handle both relative and absolute paths from Assimp.
            // Embedded textures (*N) are extracted to temp files.
            auto resolvePath = [&](const aiString& p) -> std::string {
                std::string s(p.C_Str());
                if (s.empty()) return s;
                // Handle embedded texture (*0, *1, ...)
                if (s[0] == '*') {
                    std::string tmp = resolveEmbedded(ai, s);
                    if (!tmp.empty()) return tmp;
                }
                // Already absolute (Windows drive letter or Unix root)
                if (s.size() >= 2 && s[1] == ':') return s;
                if (s[0] == '/') return s;
                return mSceneDir + "/" + s;
            };

            // Enumerate texture slots. For the Default path, keep the exact
            // mappings used by AssimpImporter::kTextureMappings[Default].
            // GLTF/OBJ retain the broader legacy handling below.
            const bool useGpuDefaultTextureMapping = (importMode == MaterialImportMode::Default);
            for (uint texType = 0; texType <= aiTextureType_UNKNOWN; texType++) {
                uint texCount = aimat->GetTextureCount((aiTextureType)texType);
                for (uint slot = 0; slot < texCount; slot++) {
                    aiString texPath;
                    if (aimat->GetTexture((aiTextureType)texType, slot, &texPath) != AI_SUCCESS)
                        continue;
                    std::string rawPath(texPath.C_Str());

                    if (texType == aiTextureType_DIFFUSE ||
                        (!useGpuDefaultTextureMapping && texType == aiTextureType_BASE_COLOR)) {
                        if (mat.texBaseColor.empty())
                            mat.texBaseColor = resolvePath(texPath);
                    } else if (texType == aiTextureType_NORMALS ||
                               (!useGpuDefaultTextureMapping &&
                                (texType == aiTextureType_HEIGHT || texType == aiTextureType_DISPLACEMENT))) {
                        if (mat.texNormalMap.empty())
                            mat.texNormalMap = resolvePath(texPath);
                    } else if (texType == aiTextureType_SPECULAR ||
                               (!useGpuDefaultTextureMapping &&
                                (texType == aiTextureType_SHININESS || texType == aiTextureType_METALNESS))) {
                        if (mat.texSpecular.empty())
                            mat.texSpecular = resolvePath(texPath);
                    } else if (!useGpuDefaultTextureMapping && texType == aiTextureType_UNKNOWN) {
                        // Heuristic: match by filename keywords.
                        // For embedded textures (*N), use the original filename from aiScene.
                        std::string matchName = rawPath;
                        if (rawPath.size() >= 2 && rawPath[0] == '*') {
                            std::string orig = embeddedOriginalName(ai, rawPath);
                            if (!orig.empty()) matchName = orig;
                        }
                        std::string lower = matchName;
                        for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
                        if (lower.find("normal") != std::string::npos || lower.find("bump") != std::string::npos) {
                            if (mat.texNormalMap.empty())
                                mat.texNormalMap = resolvePath(texPath);
                        } else if (lower.find("rough") != std::string::npos) {
                            if (mat.texSpecular.empty())
                                mat.texSpecular = resolvePath(texPath);
                        } else if (lower.find("metal") != std::string::npos) {
                            if (mat.texMetallic.empty())
                                mat.texMetallic = resolvePath(texPath);
                        } else if (lower.find("spec") != std::string::npos) {
                            if (mat.texSpecular.empty())
                                mat.texSpecular = resolvePath(texPath);
                        }
                    }
                }
            }

            // Try $raw custom properties only outside the GPU-compatible
            // Default path. Falcor's AssimpImporter does not inspect these
            // properties for FBX/other Default imports.
            if (!useGpuDefaultTextureMapping) {
                aiString raw;
                if (aimat->Get("$raw.basecolor_texture", 0, 0, raw) == AI_SUCCESS && mat.texBaseColor.empty())
                    mat.texBaseColor = resolvePath(raw);
                if (aimat->Get("$raw.roughness_texture", 0, 0, raw) == AI_SUCCESS && mat.texSpecular.empty())
                    mat.texSpecular = resolvePath(raw);
                if (aimat->Get("$raw.normal_texture", 0, 0, raw) == AI_SUCCESS && mat.texNormalMap.empty())
                    mat.texNormalMap = resolvePath(raw);
                if (aimat->Get("$raw.metallic_texture", 0, 0, raw) == AI_SUCCESS && mat.texMetallic.empty())
                    mat.texMetallic = resolvePath(raw);
            }

            // Print all texture slots for debugging
            std::cout << "    mat[" << i << "] textures:";
            if (!mat.texBaseColor.empty()) std::cout << " baseColor=" << mat.texBaseColor;
            if (!mat.texSpecular.empty()) std::cout << " roughness=" << mat.texSpecular;
            if (!mat.texMetallic.empty()) std::cout << " metallic=" << mat.texMetallic;
            if (!mat.texNormalMap.empty()) std::cout << " normal=" << mat.texNormalMap;
            if (mat.isSpecGloss) std::cout << " [SpecGloss]";
            if (mat.texBaseColor.empty() && mat.texSpecular.empty() && mat.texMetallic.empty() && mat.texNormalMap.empty())
                std::cout << " (none)";
            std::cout << std::endl;

            materials.push_back(mat);
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
