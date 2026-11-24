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
    Texture3D<float> values;
    float3 boundsMin;
    float3 boundsMax;
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
};
