// Include after SdfField.slang for SdfCell's trilinear interpolation.
struct DensityClipmapData
{
    Texture3D<float> clipValues;
    int clipResolution, clipCount, clipFirstLevel;
    float4 clipLevels[12]; // Origin and stride in native voxel coordinates, relative to the root.
    int4 clipOffsets[12];
    float3 boundsMin, boundsMax;
    float3 worldOrigin;
    float4x4 worldToIndex;
    float3 indexMin, indexMax;
    float maxDensity;
};
ParameterBlock<DensityClipmapData> gDensity;

bool densitySlab(float3 p, float3 d, float3 lo, float3 hi, inout float2 span)
{
    [unroll] for (int k = 0; k < 3; ++k)
    {
        if (d[k] == 0.f)
        {
            if (p[k] < lo[k] || p[k] > hi[k]) return false;
        }
        else
        {
            float2 faces = (float2(lo[k], hi[k]) - p[k]) / d[k];
            span.x = max(span.x, min(faces.x, faces.y));
            span.y = min(span.y, max(faces.x, faces.y));
        }
    }
    return span.x < span.y;
}

bool densityRaySpan(float3 origin, float3 direction, inout float2 span)
{
    float3 p = mul((float3x3)gDensity.worldToIndex, origin - gDensity.worldOrigin);
    float3 d = mul((float3x3)gDensity.worldToIndex, direction);
    return gDensity.maxDensity > 0.f && densitySlab(p, d, gDensity.indexMin, gDensity.indexMax, span);
}

float densityRawNode(int level, int3 node)
{
    int3 address = node + gDensity.clipOffsets[level].xyz;
    address -= int3(address >= gDensity.clipResolution) * gDensity.clipResolution;
    address.z += level * gDensity.clipResolution;
    return gDensity.clipValues.Load(int4(address, 0));
}

float densityNode(int level, int3 node, int3 parentOrigin)
{
    if (gDensity.clipCount == 0) return gDensity.clipValues.Load(int4(node, 0));
    int width = gDensity.clipResolution - 1;
    if (level == gDensity.clipCount - 1 || all(node > 0) && all(node < width))
        return densityRawNode(level, node);

    // Child boundary nodes reproduce the parent's trilinear face. At least one
    // coordinate is even; at most four parent samples contribute to the face.
    int3 i = parentOrigin + (node >> 1), odd = node & 1;
    float value = densityRawNode(level + 1, i);
    if (all(odd == 0)) return value;
    int3 axis = odd.x != 0 ? int3(1, 0, 0) : odd.y != 0 ? int3(0, 1, 0) : int3(0, 0, 1);
    value = .5f * (value + densityRawNode(level + 1, i + axis));
    if (all(axis == odd)) return value;
    return .5f * value + .25f * (densityRawNode(level + 1, i + odd - axis) + densityRawNode(level + 1, i + odd));
}

void densityRayCell(float3 origin, float3 direction, float t, float limit, out SdfCell cell,
                    out float3 fraction, out float3 cellDir, out float end, out int level)
{
    float3 p = mul((float3x3)gDensity.worldToIndex, origin - gDensity.worldOrigin);
    float3 d = mul((float3x3)gDensity.worldToIndex, direction);
    int3 width = gDensity.clipCount == 0 ? int3(gDensity.indexMax) : int3(gDensity.clipResolution - 1);
    level = max(0, gDensity.clipCount - 1);
    end = limit;
    [loop] for (int i = gDensity.clipFirstLevel; i < gDensity.clipCount - 1; ++i)
    {
        float4 layout = gDensity.clipLevels[i];
        float2 span = float2(0.f, limit);
        if (!densitySlab(p, d, layout.xyz, layout.xyz + width * layout.w, span) || span.y <= t) continue;
        if (span.x > t) end = min(end, span.x);
        else
        {
            level = i;
            end = min(end, span.y);
            break;
        }
    }
    float4 layout = gDensity.clipLevels[level];
    cellDir = d / layout.w;
    float3 position = clamp(((p - layout.xyz) + d * t) / layout.w, float3(0.f), float3(width));
    int3 base = min(int3(floor(position)), int3(width-1));
    [unroll] for (int k = 0; k < 3; ++k)
    {
        if (d[k] == 0.f) continue;
        int step = d[k] > 0.f ? 1 : -1;
        int face = base[k] + (step > 0 ? 1 : 0);
        float next = (layout.xyz[k] + face * layout.w - p[k]) / d[k];
        // Resolve the exact half-open cell at a crossing using the same absolute
        // face time used by the preceding call, for either traversal direction.
        if (next <= t && base[k] + step >= 0 && base[k] + step < width[k])
        {
            base[k] += step;
            face += step;
            next = (layout.xyz[k] + face * layout.w - p[k]) / d[k];
        }
        if (next > t) end = min(end, next);
    }
    fraction = saturate(position - float3(base));
    int3 parentOrigin = 0;
    if (level < gDensity.clipCount - 1)
    {
        float4 parent = gDensity.clipLevels[level + 1];
        parentOrigin = int3(round((layout.xyz - parent.xyz) / parent.w));
    }
    cell.z0 = float4(densityNode(level, base, parentOrigin), densityNode(level, base + int3(1,0,0), parentOrigin),
                     densityNode(level, base + int3(0,1,0), parentOrigin), densityNode(level, base + int3(1,1,0), parentOrigin));
    cell.z1 = float4(densityNode(level, base + int3(0,0,1), parentOrigin), densityNode(level, base + int3(1,0,1), parentOrigin),
                     densityNode(level, base + int3(0,1,1), parentOrigin), densityNode(level, base + 1, parentOrigin));
}

float3 densityWorldGradient(float3 gradient, int level)
{
    return mul(gradient / gDensity.clipLevels[level].w, (float3x3)gDensity.worldToIndex);
}
