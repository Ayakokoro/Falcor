#pragma once
#include "Types.h"
#include <climits>

struct AABBInt {
    int xMin, xMax, yMin, yMax, zMin, zMax;
    int deltaX, deltaY, deltaZ, deltaXY;

    void init() {
        xMin = yMin = zMin = INT_MAX;
        xMax = yMax = zMax = INT_MIN;
        deltaX = deltaY = deltaZ = deltaXY = 0;
    }

    void accumulate(const int3& point) {
        xMin = std::min(xMin, point.x);
        yMin = std::min(yMin, point.y);
        zMin = std::min(zMin, point.z);
        xMax = std::max(xMax, point.x);
        yMax = std::max(yMax, point.y);
        zMax = std::max(zMax, point.z);
    }

    void complete() {
        deltaX = xMax - xMin + 1;
        deltaY = yMax - yMin + 1;
        deltaZ = zMax - zMin + 1;
        deltaXY = deltaX * deltaY;
    }

    int count() const { return deltaX * deltaY * deltaZ; }

    int3 indexToCell(int index) const {
        int zLocal = index / deltaXY;
        int temp   = index % deltaXY;
        int yLocal = temp / deltaX;
        int xLocal = temp % deltaX;
        return int3(xMin + xLocal, yMin + yLocal, zMin + zLocal);
    }

    int cellToIndex(const int3& cell) const {
        return (cell.z - zMin) * deltaXY + (cell.y - yMin) * deltaX + (cell.x - xMin);
    }
};
