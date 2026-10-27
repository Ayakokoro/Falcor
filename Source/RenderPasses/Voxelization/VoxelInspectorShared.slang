#pragma once
// Node-local positions and barycentrics in the ORIGINAL scene triangle.
// This is a transient debug buffer, independent of the on-disk Polygon ABI.
struct InspectorTriangle
{
    float3 p0;
    float3 p1;
    float3 p2;
    float3 b0;
    float3 b1;
    float3 b2;
    uint instanceID;
    uint triangleID;
};
