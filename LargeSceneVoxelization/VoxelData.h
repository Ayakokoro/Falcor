#pragma once
#include "ABSDF.h"
#include "Ellipsoid.h"
#include "SphericalHarmonics.h"

#define PROJECT_CIRCLE_AREA 2.35619449f

struct VoxelData {
    ABSDF ABSDF;
    Ellipsoid ellipsoid;
    SphericalFunc primitiveProjAreaFunc;
    SphericalFunc polygonsProjAreaFunc;
    SphericalFunc totalProjAreaFunc;

    void init() {
        ABSDF.init();
        primitiveProjAreaFunc.init();
        polygonsProjAreaFunc.init();
        totalProjAreaFunc.init();
    }

    bool isSolid() const { return ABSDF.area > 0; }
};

struct OctreeNode {
    uint childBase = 0;   // first child index in next BFS level
    uint childMask = 0;   // bits 0-7: valid children
    uint dataIndex = 0;   // index into gBuffer for this node's data
};
