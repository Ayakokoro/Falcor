#ifndef GPIS_MESH_CLIPMAP
#define GPIS_MESH_CLIPMAP 0
#endif

// Texels are SDF samples at integer index positions (OpenVDB node convention).
struct SdfCell
{
    // x varies fastest inside each z plane.
    float4 z0;
    float4 z1;

    float value(float3 p)
    {
        float4 xy = lerp(z0, z1, p.z);
        float2 y = lerp(xy.xz, xy.yw, p.x);
        return lerp(y.x, y.y, p.y);
    }

    float3 gradient(float3 p)
    {
        float4 xy = lerp(z0, z1, p.z);
        float2 y = lerp(xy.xz, xy.yw, p.x);
        float4 dz = z1 - z0;
        float2 yz = lerp(dz.xz, dz.yw, p.x);
        return float3(lerp(xy.y - xy.x, xy.w - xy.z, p.y), y.y - y.x, lerp(yz.x, yz.y, p.y));
    }
};

struct SdfField
{
    float3 boundsMin;
    float3 boundsMax;
#if GPIS_MESH_CLIPMAP
    // Each level occupies N consecutive Z slices in the atlas. Its local
    // (0,0,0) maps to the ring offset, so moving a window reuses old nodes.
    Texture3D<float> clipValues;
    int clipCount;
    int clipFirstLevel;
    int clipResolution;
    float4 clipLevels[12]; // lower corner in world space and world-space voxel width
    int4 clipOffsets[12];
#else
    Texture3D<float> values;
    float3 worldOrigin;
    float4x4 worldToIndex;
    int3 indexMax;

    float3 worldToIndexPoint(float3 p) { return mul((float3x3)worldToIndex, p - worldOrigin); }
    float3 worldToIndexDir(float3 d) { return mul((float3x3)worldToIndex, d); }
    float3 worldGradient(float3 indexGradient) { return mul(indexGradient, (float3x3)worldToIndex); }

    SdfCell loadCell(int3 corner)
    {
        int3 i = clamp(corner, int3(0), indexMax - 1);
        SdfCell cell;
        cell.z0 = float4(values[i], values[i + int3(1, 0, 0)], values[i + int3(0, 1, 0)], values[i + int3(1, 1, 0)]);
        cell.z1 = float4(values[i + int3(0, 0, 1)], values[i + int3(1, 0, 1)], values[i + int3(0, 1, 1)], values[i + 1]);
        return cell;
    }
#endif
};

#if GPIS_MESH_CLIPMAP
// Access level metadata through the parameter block, including inside the
// sampling functions. Passing SdfField by value copies both arrays into
// per-invocation storage for each inlined texture read on the SPIR-V target.
ParameterBlock<SdfField> gField;

float clipNode(int level, int3 node)
{
    int3 address = node + gField.clipOffsets[level].xyz;
    address -= int3(address >= gField.clipResolution) * gField.clipResolution;
    address.z += level * gField.clipResolution;
    return gField.clipValues.Load(int4(address, 0));
}

SdfCell rawClipCell(int level, int3 i)
{
    SdfCell cell;
    cell.z0 = float4(clipNode(level, i), clipNode(level, i + int3(1, 0, 0)),
                     clipNode(level, i + int3(0, 1, 0)), clipNode(level, i + int3(1, 1, 0)));
    cell.z1 = float4(clipNode(level, i + int3(0, 0, 1)), clipNode(level, i + int3(1, 0, 1)),
                     clipNode(level, i + int3(0, 1, 1)), clipNode(level, i + 1));
    return cell;
}

float clipTransitionNode(int level, int3 node, int3 parentOffset)
{
    if (all(node > 0) && all(node < gField.clipResolution - 1))
        return clipNode(level, node);

    // Facing the coarse region, the four children of a coarse face must
    // reproduce its bilinear interpolation. Sample the parent at these
    // nodes before interpolating the fine cell; its ray restriction stays cubic.
    // Origins are parent-aligned and the fine box has a one-parent-cell inset.
    int3 i = parentOffset + (node >> 1);
    int3 odd = node & 1;
    float first = clipNode(level + 1, i);
    if (all(odd == 0)) return first;
    int3 axis = odd.x != 0 ? int3(1, 0, 0) : (odd.y != 0 ? int3(0, 1, 0) : int3(0, 0, 1));
    float row = 0.5f * (first + clipNode(level + 1, i + axis));
    if (all(odd == axis)) return row;
    return 0.5f * (row + 0.5f * (clipNode(level + 1, i + odd - axis) + clipNode(level + 1, i + odd)));
}

SdfCell loadClipCell(int level, int3 i)
{
    if (level == gField.clipCount - 1 || (all(i > 0) && all(i < gField.clipResolution - 2)))
        return rawClipCell(level, i);
    int3 parentOffset = int3(round((gField.clipLevels[level].xyz - gField.clipLevels[level + 1].xyz) / gField.clipLevels[level + 1].w));
    SdfCell cell;
    cell.z0 = float4(clipTransitionNode(level, i, parentOffset), clipTransitionNode(level, i + int3(1, 0, 0), parentOffset),
                     clipTransitionNode(level, i + int3(0, 1, 0), parentOffset), clipTransitionNode(level, i + int3(1, 1, 0), parentOffset));
    cell.z1 = float4(clipTransitionNode(level, i + int3(0, 0, 1), parentOffset), clipTransitionNode(level, i + int3(1, 0, 1), parentOffset),
                     clipTransitionNode(level, i + int3(0, 1, 1), parentOffset), clipTransitionNode(level, i + 1, parentOffset));
    return cell;
}
#else
uniform SdfField gField;
#endif
