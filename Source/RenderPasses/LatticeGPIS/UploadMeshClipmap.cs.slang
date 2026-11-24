StructuredBuffer<float> gDistances;
StructuredBuffer<uint4> gRows;
RWTexture3D<float> gAtlas;
uniform uint gResolution;
uniform uint gRowCount;
uniform uint gGroupsX;

// All dirty rows in all levels share this dispatch. Each row wraps only in X.
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint index = group.x + group.y * gGroupsX;
    if (index >= gRowCount) return;
    uint4 row = gRows[index];
    uint x = row.x % gResolution;
    uint yz = row.x / gResolution;
    uint2 coord = uint2(yz % gResolution, yz / gResolution);
    for (uint i = lane; i < row.z; i += 64)
        gAtlas[uint3((x + i) % gResolution, coord)] = gDistances[row.y + i];
}
