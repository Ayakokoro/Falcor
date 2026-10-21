#pragma once
#include "Shading.slang"

#ifndef USE_VISIBLE_MATERIAL
#define USE_VISIBLE_MATERIAL 0
#endif

#if USE_VISIBLE_MATERIAL
#include "VisibleShading.slang"
typedef VisibleTEBSDF RayMarchingBSDF;
typedef VisibleSurfaceBRDF RayMarchingSurface;
#else
typedef TEBSDF RayMarchingBSDF;
typedef SurfaceBRDF RayMarchingSurface;
#endif
