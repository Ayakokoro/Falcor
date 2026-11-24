#include "MeshDistanceField.h"
#include "Core/API/Device.h"
#include "Core/Error.h"
#include "Core/Platform/OS.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/StringFormatters.h"
#include <assimp/DefaultIOSystem.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace Falcor;

namespace
{
using Vec3d = math::vector<double, 3>;

double boxDistanceSquared(const AABB& bounds, float3 p)
{
    double sum = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        double d = double(p[i]) - std::clamp(p[i], bounds.minPoint[i], bounds.maxPoint[i]);
        sum += d * d;
    }
    return sum;
}

double edgeDistanceSquared(Vec3d edge, Vec3d delta)
{
    double t = std::clamp(dot(edge, delta) / dot(edge, edge), 0.0, 1.0);
    Vec3d closest = delta - t * edge;
    return dot(closest, closest);
}

// Half-open ownership in the YZ projection. Two projected triangles sharing an
// edge count exactly one crossing there, or two/zero when the ray only grazes it.
bool ownsEdge(float3 a, float3 b)
{
    return a.z < b.z || (a.z == b.z && a.y > b.y);
}

// Assimp returns map_Kd as written in the MTL, without its source library path.
// Capture the library directory during the ordinary OBJ import (including OBJ.GZ).
class ObjMaterialIO : public Assimp::DefaultIOSystem
{
public:
    explicit ObjMaterialIO(std::filesystem::path directory) : mDirectory(std::move(directory)) {}

    bool Exists(const char* file) const override
    {
        return DefaultIOSystem::Exists((mDirectory / std::filesystem::u8path(file)).u8string().c_str());
    }

    Assimp::IOStream* Open(const char* file, const char* mode = "rb") override
    {
        const auto path = (mDirectory / std::filesystem::u8path(file)).lexically_normal();
        auto* stream = DefaultIOSystem::Open(path.u8string().c_str(), mode);
        if (stream && hasExtension(path, "mtl"))
        {
            if (mLibraryDirectory.empty()) mLibraryDirectory = path.parent_path();
            else if (mLibraryDirectory != path.parent_path()) mMultipleDirectories = true;
        }
        return stream;
    }

    std::filesystem::path texturePath(const std::filesystem::path& texture) const
    {
        if (texture.is_absolute()) return texture.lexically_normal();
        // Assimp does not say which library owns a merged material. An explicit texture override
        // remains available when libraries in different directories could mean different files.
        if (mMultipleDirectories) return {};
        auto fromObj = (mDirectory / texture).lexically_normal();
        if (mLibraryDirectory.empty() || mLibraryDirectory == mDirectory) return fromObj;
        auto fromMtl = (mLibraryDirectory / texture).lexically_normal();
        return std::filesystem::is_regular_file(fromMtl) || !std::filesystem::is_regular_file(fromObj) ? fromMtl : fromObj;
    }

private:
    std::filesystem::path mDirectory, mLibraryDirectory;
    bool mMultipleDirectories = false;
};

std::filesystem::path baseColorMap(const aiScene& scene, const aiMesh& mesh, const std::filesystem::path& directory,
                                   const ObjMaterialIO* objIO)
{
    if (mesh.mMaterialIndex >= scene.mNumMaterials) return {};
    const aiMaterial& material = *scene.mMaterials[mesh.mMaterialIndex];
    const auto type = material.GetTextureCount(aiTextureType_BASE_COLOR) ? aiTextureType_BASE_COLOR : aiTextureType_DIFFUSE;
    aiString texture;
    aiTextureMapping mapping = aiTextureMapping_UV;
    unsigned int uvSet = 0;
    if (material.GetTextureCount(type) != 1 ||
        material.GetTexture(type, 0, &texture, &mapping, &uvSet) != AI_SUCCESS ||
        mapping != aiTextureMapping_UV || uvSet != 0 || !texture.length || scene.GetEmbeddedTexture(texture.C_Str()))
        return {};
    std::string textureName = texture.C_Str();
    std::replace(textureName.begin(), textureName.end(), '\\', '/');
    const auto texturePath = std::filesystem::u8path(textureName);
    return objIO ? objIO->texturePath(texturePath) : (directory / texturePath).lexically_normal();
}
}

double MeshDistanceField::Triangle::distanceSquared(float3 p) const
{
    Vec3d ab = Vec3d(b) - Vec3d(a);
    Vec3d bc = Vec3d(c) - Vec3d(b);
    Vec3d ca = Vec3d(a) - Vec3d(c);
    Vec3d ap = Vec3d(p) - Vec3d(a);
    Vec3d bp = Vec3d(p) - Vec3d(b);
    Vec3d cp = Vec3d(p) - Vec3d(c);
    Vec3d normal = cross(ab, -ca);
    double best = std::numeric_limits<double>::infinity();
    if (dot(cross(ab, ap), normal) < 0.0) best = edgeDistanceSquared(ab, ap);
    if (dot(cross(bc, bp), normal) < 0.0) best = std::min(best, edgeDistanceSquared(bc, bp));
    if (dot(cross(ca, cp), normal) < 0.0) best = std::min(best, edgeDistanceSquared(ca, cp));
    if (std::isfinite(best)) return best;
    double height = dot(ap, normal);
    return height * height / dot(normal, normal);
}

bool MeshDistanceField::Triangle::intersectX(float y, float z, double& x) const
{
    double ay = double(a.y) - y, az = double(a.z) - z;
    double by = double(b.y) - y, bz = double(b.z) - z;
    double cy = double(c.y) - y, cz = double(c.z) - z;
    double wa = by * cz - bz * cy;
    double wb = cy * az - cz * ay;
    double wc = ay * bz - az * by;
    double area = (double(b.y) - a.y) * (double(c.z) - a.z) - (double(b.z) - a.z) * (double(c.y) - a.y);
    if (area > 0.0)
    {
        if (wa < 0.0 || wb < 0.0 || wc < 0.0) return false;
        if ((wa == 0.0 && !ownsEdge(b, c)) || (wb == 0.0 && !ownsEdge(c, a)) || (wc == 0.0 && !ownsEdge(a, b))) return false;
    }
    else if (area < 0.0)
    {
        if (wa > 0.0 || wb > 0.0 || wc > 0.0) return false;
        if ((wa == 0.0 && !ownsEdge(c, b)) || (wb == 0.0 && !ownsEdge(a, c)) || (wc == 0.0 && !ownsEdge(b, a))) return false;
    }
    else return false;
    x = double(a.x) + ((double(b.x) - a.x) * wb + (double(c.x) - a.x) * wc) / area;
    return true;
}

MeshDistanceField::MeshDistanceField(const std::filesystem::path& path, bool retainTexCoords)
{
    // Match Falcor's TriangleMesh coordinate and texture conventions while retaining source material/UV information.
    Assimp::Importer importer;
    const auto flags = aiProcess_Triangulate | aiProcess_PreTransformVertices | aiProcess_FlipUVs;
    const auto absolutePath = std::filesystem::absolute(path).lexically_normal();
    const auto directory = absolutePath.parent_path();
    const bool compressed = hasExtension(path, "gz");
    const bool obj = hasExtension(compressed ? path.stem() : path, "obj");
    ObjMaterialIO* objIO = nullptr;
    if (obj)
    {
        objIO = new ObjMaterialIO(directory);
        importer.SetIOHandler(objIO); // The importer owns the handler.
    }
    const aiScene* scene;
    if (compressed)
    {
        const auto data = decompressFile(path);
        scene = importer.ReadFileFromMemory(data.data(), data.size(), flags, obj ? "obj" : "");
    }
    else scene = importer.ReadFile(absolutePath.u8string(), flags);
    FALCOR_CHECK(scene, "LatticeGPIS: could not read triangle mesh '{}': {}", path, importer.GetErrorString());

    size_t faceCount = 0;
    bool hasUV = true, commonMap = true;
    for (size_t i = 0; i < scene->mNumMeshes; ++i)
    {
        const auto& mesh = *scene->mMeshes[i];
        if (!mesh.mNumFaces) continue;
        const auto map = baseColorMap(*scene, mesh, directory, objIO);
        if (faceCount == 0) mBaseColorTexturePath = map;
        else if (map != mBaseColorTexturePath) commonMap = false;
        faceCount += mesh.mNumFaces;
        hasUV &= mesh.HasTextureCoords(0);
    }
    FALCOR_CHECK(faceCount <= std::numeric_limits<uint32_t>::max(), "LatticeGPIS: too many mesh triangles in '{}'.", path);
    if (!commonMap || !hasUV) mBaseColorTexturePath.clear();
    const bool keepUV = hasUV && (retainTexCoords || !mBaseColorTexturePath.empty());
    if (keepUV) mTexCoords.reserve(faceCount);
    mTriangles.reserve(faceCount);
    for (size_t i = 0; i < scene->mNumMeshes; ++i)
    {
        const auto& mesh = *scene->mMeshes[i];
        for (size_t j = 0; j < mesh.mNumFaces; ++j)
        {
            const aiFace& face = mesh.mFaces[j];
            FALCOR_CHECK(face.mNumIndices == 3, "LatticeGPIS: mesh '{}' contains a non-triangular primitive.", path);
            auto position = [&](int corner) {
                const auto& p = mesh.mVertices[face.mIndices[corner]];
                return float3(p.x, p.y, p.z);
            };
            Triangle tri{position(0), position(1), position(2), uint32_t(mTexCoords.size())};
            Vec3d normal = cross(Vec3d(tri.b) - Vec3d(tri.a), Vec3d(tri.c) - Vec3d(tri.a));
            double areaSquared = dot(normal, normal);
            FALCOR_CHECK(std::isfinite(areaSquared), "LatticeGPIS: mesh '{}' has a non-finite position.", path);
            if (areaSquared == 0.0) continue;
            mTriangles.push_back(tri);
            if (keepUV)
            {
                auto uv = [&](int corner) {
                    const auto& t = mesh.mTextureCoords[0][face.mIndices[corner]];
                    FALCOR_CHECK(std::isfinite(t.x) && std::isfinite(t.y), "LatticeGPIS: mesh '{}' has a non-finite UV.", path);
                    return float2(t.x, t.y);
                };
                mTexCoords.push_back({uv(0), uv(1), uv(2)});
            }
        }
    }
    FALCOR_CHECK(!mTriangles.empty(), "LatticeGPIS: no nondegenerate triangles in '{}'.", path);
    importer.FreeScene();
    buildNode(0, uint32_t(mTriangles.size()));
}

MeshDistanceField::SurfaceUVBuffers MeshDistanceField::createSurfaceUVBuffers(ref<Device> device) const
{
    FALCOR_CHECK(hasTexCoords(), "LatticeGPIS: applying a texture requires UV coordinates on every submesh.");
    // MeshSurfaceUV.slang uses the same layout. Keep the three UVs face-local so a shared
    // position can belong to different UV charts on its neighboring faces.
    struct SurfaceNode { float3 lower; uint32_t first; float3 upper; uint32_t count; };
    struct SurfaceTriangle { float4 a, ab, ac, uvEdges; };
    static_assert(sizeof(SurfaceNode) == 32 && sizeof(SurfaceTriangle) == 64);
    std::vector<SurfaceNode> nodes;
    nodes.reserve(mNodes.size());
    for (const auto& node : mNodes)
        nodes.push_back({node.bounds.minPoint, node.first, node.bounds.maxPoint, node.count});
    std::vector<SurfaceTriangle> faces;
    faces.reserve(mTriangles.size());
    for (const auto& tri : mTriangles)
    {
        const auto& uv = mTexCoords[tri.uvIndex];
        const float3 ab = tri.b - tri.a, ac = tri.c - tri.a;
        const auto extent = max(abs(ab), abs(ac));
        float scale = std::max({extent.x, extent.y, extent.z});
        faces.push_back({float4(tri.a, uv.a.x), float4(ab, uv.a.y), float4(ac, scale),
                         float4(uv.b.x - uv.a.x, uv.b.y - uv.a.y, uv.c.x - uv.a.x, uv.c.y - uv.a.y)});
    }
    return {
        device->createStructuredBuffer(sizeof(SurfaceNode), uint32_t(nodes.size()), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, nodes.data()),
        device->createStructuredBuffer(sizeof(SurfaceTriangle), uint32_t(faces.size()), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, faces.data())
    };
}

void MeshDistanceField::SurfaceUVBuffers::bind(const ShaderVar& root) const
{
    root["gSurfaceUVNodes"] = nodes;
    root["gSurfaceUVTriangles"] = triangles;
}

uint32_t MeshDistanceField::buildNode(uint32_t begin, uint32_t end)
{
    uint32_t index = uint32_t(mNodes.size());
    AABB bounds;
    for (uint32_t i = begin; i < end; ++i)
    {
        const auto& tri = mTriangles[i];
        bounds.include(tri.a).include(tri.b).include(tri.c);
    }
    mNodes.push_back({bounds, begin, end - begin});
    if (end - begin > 6)
    {
        Vec3d extent = Vec3d(bounds.maxPoint) - Vec3d(bounds.minPoint);
        int axis = extent.y > extent.x ? 1 : 0;
        if (extent.z > extent[axis]) axis = 2;
        uint32_t mid = begin + (end - begin) / 2;
        auto center = [axis](const Triangle& t) { return double(t.a[axis]) + t.b[axis] + t.c[axis]; };
        std::nth_element(mTriangles.begin() + begin, mTriangles.begin() + mid, mTriangles.begin() + end,
                         [&](const Triangle& a, const Triangle& b) { return center(a) < center(b); });
        buildNode(begin, mid);
        uint32_t right = buildNode(mid, end);
        mNodes[index].first = right;
        mNodes[index].count = 0;
    }
    return index;
}

void MeshDistanceField::findNearest(uint32_t nodeIndex, float3 p, double& best, uint32_t& nearest) const
{
    const Node& node = mNodes[nodeIndex];
    if (node.count > 0)
    {
        for (uint32_t i = node.first; i < node.first + node.count; ++i)
        {
            if (i == nearest) continue;
            double d = mTriangles[i].distanceSquared(p);
            if (d < best) { best = d; nearest = i; }
        }
        return;
    }
    uint32_t first = nodeIndex + 1, second = node.first;
    double firstDistance = boxDistanceSquared(mNodes[first].bounds, p);
    double secondDistance = boxDistanceSquared(mNodes[second].bounds, p);
    if (secondDistance < firstDistance)
    {
        std::swap(first, second);
        std::swap(firstDistance, secondDistance);
    }
    if (firstDistance < best) findNearest(first, p, best, nearest);
    if (secondDistance < best) findNearest(second, p, best, nearest);
}

void MeshDistanceField::collectCrossings(uint32_t nodeIndex, float y, float z, std::vector<double>& crossings) const
{
    const Node& node = mNodes[nodeIndex];
    if (y < node.bounds.minPoint.y || y > node.bounds.maxPoint.y || z < node.bounds.minPoint.z || z > node.bounds.maxPoint.z) return;
    if (node.count > 0)
    {
        for (uint32_t i = node.first; i < node.first + node.count; ++i)
        {
            double x;
            if (mTriangles[i].intersectX(y, z, x)) crossings.push_back(x);
        }
        return;
    }
    collectCrossings(nodeIndex + 1, y, z, crossings);
    collectCrossings(node.first, y, z, crossings);
}

void MeshDistanceField::sampleRow(float3 rowOrigin, float voxelSize, int count, float* result, int firstXIndex) const
{
    std::vector<double> crossings;
    collectCrossings(0, rowOrigin.y, rowOrigin.z, crossings);
    std::sort(crossings.begin(), crossings.end());
    size_t crossed = 0;
    uint32_t nearest = std::numeric_limits<uint32_t>::max();
    for (int i = 0; i < count; ++i)
    {
        float3 p(rowOrigin.x + float(firstXIndex + i) * voxelSize, rowOrigin.y, rowOrigin.z);
        while (crossed < crossings.size() && crossings[crossed] <= p.x) ++crossed;
        double best = i == 0 ? std::numeric_limits<double>::infinity() : mTriangles[nearest].distanceSquared(p);
        findNearest(0, p, best, nearest);
        float distance = float(std::sqrt(best));
        result[i] = (crossed & 1) ? -distance : distance;
    }
}

float MeshDistanceField::signedDistance(float3 worldPosition) const
{
    float result;
    sampleRow(worldPosition, 0.f, 1, &result);
    return result;
}
