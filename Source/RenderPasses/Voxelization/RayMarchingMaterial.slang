#pragma once
#include "Shading.slang"
#include "VisibilityMask.slang"

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

#if !defined(HOST_CODE)
// Shared by scene rendering and the isolated-node Inspector. The surface
// result already includes the incident cosine; coverage is applied by callers.
float3 evalRayMarchingSurface(RayMarchingBSDF bsdf, float3 l, float3 v, float3 h,
    bool checkVisibility, uint visibilityMode, bool masksAvailable, VisibilityMaskPair pair)
{
#if CHECK_VISIBILITY
    if (checkVisibility)
    {
#if USE_VISIBLE_MATERIAL
        if (visibilityMode == 0)
            return bsdf.surface.eval(l, v, h) * bsdf.calcInternalVisibility(l);
        if (visibilityMode == 1 && masksAvailable)
            return bsdf.surface.evalVisible(l, v, h, pair);
#else
        return bsdf.surface.eval(l, v, h) * bsdf.calcInternalVisibility(l);
#endif
    }
#endif
    return bsdf.surface.eval(l, v, h);
}
#endif
