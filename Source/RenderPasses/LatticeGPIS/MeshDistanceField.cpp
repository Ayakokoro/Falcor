#include "MeshDistanceField.h"
#include "Core/Error.h"
#include "Scene/TriangleMesh.h"
#include "Utils/StringFormatters.h"
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

MeshDistanceField::MeshDistanceField(const std::filesystem::path& path)
{
    auto mesh = TriangleMesh::createFromFile(path);
    FALCOR_CHECK(mesh, "LatticeGPIS: could not read triangle mesh '{}'.", path);
    const auto& vertices = mesh->getVertices();
    const auto& indices = mesh->getIndices();
    FALCOR_CHECK(indices.size() / 3 <= std::numeric_limits<uint32_t>::max(), "LatticeGPIS: too many mesh triangles in '{}'.", path);
    mTriangles.reserve(indices.size() / 3);
    for (size_t i = 0; i < indices.size(); i += 3)
    {
        Triangle tri{vertices[indices[i]].position, vertices[indices[i + 1]].position, vertices[indices[i + 2]].position};
        Vec3d normal = cross(Vec3d(tri.b) - Vec3d(tri.a), Vec3d(tri.c) - Vec3d(tri.a));
        double areaSquared = dot(normal, normal);
        FALCOR_CHECK(std::isfinite(areaSquared), "LatticeGPIS: mesh '{}' has a non-finite position.", path);
        if (areaSquared > 0.0) mTriangles.push_back(tri);
    }
    FALCOR_CHECK(!mTriangles.empty(), "LatticeGPIS: no nondegenerate triangles in '{}'.", path);
    mesh.reset();
    buildNode(0, uint32_t(mTriangles.size()));
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
