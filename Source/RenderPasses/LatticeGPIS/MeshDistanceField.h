#pragma once
#include "Utils/Math/AABB.h"
#include <filesystem>
#include <vector>

/** World-space signed distance to a static closed mesh. Concurrent readers share an immutable triangle BVH. */
class MeshDistanceField
{
public:
    explicit MeshDistanceField(const std::filesystem::path& path);
    Falcor::AABB getBounds() const { return mNodes.front().bounds; }
    float signedDistance(Falcor::float3 worldPosition) const;

    /** Evaluate consecutive samples along +X from an indexed grid origin. Splitting a row into slabs preserves each sample position. */
    void sampleRow(Falcor::float3 rowOrigin, float voxelSize, int count, float* result, int firstXIndex = 0) const;

private:
    struct Triangle
    {
        Falcor::float3 a, b, c;
        double distanceSquared(Falcor::float3 p) const;
        bool intersectX(float y, float z, double& x) const;
    };
    struct Node
    {
        Falcor::AABB bounds;
        uint32_t first = 0; // First triangle for a leaf, right child for an interior node. Its left child follows the node.
        uint32_t count = 0;
    };
    uint32_t buildNode(uint32_t begin, uint32_t end);
    void findNearest(uint32_t nodeIndex, Falcor::float3 p, double& best, uint32_t& nearest) const;
    void collectCrossings(uint32_t nodeIndex, float y, float z, std::vector<double>& crossings) const;

    std::vector<Triangle> mTriangles;
    std::vector<Node> mNodes;
};
