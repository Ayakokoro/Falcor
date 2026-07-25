#pragma once
#include "Types.h"
#include <cstddef>

#define MIPMAP_SHRINK_BIT 3
#define MIPMAP_SHRINK 8

struct GridData {
    float3 gridMin;
    float3 voxelSize;
    uint3 voxelCount;
    size_t solidVoxelCount = 0;
    uint maxPolygonCount = 0;
    uint totalPolygonCount = 0;

    size_t totalVoxelCount() const {
        return (size_t)voxelCount.x * voxelCount.y * voxelCount.z;
    }
};

inline int CellToIndex(const int3& cell, const uint3& size) {
    return cell.x + cell.y * (int)size.x + cell.z * (int)size.x * (int)size.y;
}

inline int3 IndexToCell(int index, const uint3& size) {
    int z = index / ((int)size.x * (int)size.y);
    int y = (index % ((int)size.x * (int)size.y)) / (int)size.x;
    int x = index % (int)size.x;
    return int3(x, y, z);
}

// DDA: advance ray from current cell to the next cell boundary
inline float leaveCell(float3& cell, int3& cellInt, const int3& directions,
                       const float3& direction, const float3& rcpDir) {
    float3 border = float3(cellInt) + float3((directions + 1) / 2);
    float3 t = (border - cell) * rcpDir;
    float tMin = 2.0f;
    int3 sel(0);
    if (direction.x != 0 && t.x < tMin) { tMin = t.x; sel = int3(directions.x, 0, 0); }
    if (direction.y != 0 && t.y < tMin) { tMin = t.y; sel = int3(0, directions.y, 0); }
    if (direction.z != 0 && t.z < tMin) { tMin = t.z; sel = int3(0, 0, directions.z); }
    cell += tMin * direction;
    cellInt += sel;
    return tMin;
}
