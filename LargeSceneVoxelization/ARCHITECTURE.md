# LargeSceneVoxelization architecture

This tool mirrors the data flow of
`Source/RenderPasses/Voxelization/VoxelizationPass.cpp). It has no debug or
visualization stage.

## Pipeline

Both entry points use the same semantic stages:

1. Load instanced meshes, compute the world-space voxel grid, and load textures.
2. Hierarchical clipping. A triangle is clipped from the root to every
   intersected octree level. Each successful clip is stored in that node's
   polygon block. Polygon vertices are normalized by the current node scale but
   remain in global voxel coordinates within the node cell.
3. Build breadth-first octree arrays.
4. Analyze every occupied node directly from its own polygon block.
5. Write the renderer-compatible binary output.

The disk-backed entry point only changes the transport between stages 2 and 4:
clip workers write shards, merge creates an index and a polygon data file, and
analysis streams those files. It does not aggregate parent VoxelData from
children.

## Node keys and BFS layout

A node key is encoded as:

```
(level << 32) | x | (y << 10) | (z << 20)
```

The key format has 10 bits per coordinate, so the standalone tool accepts up
to a 1024^3 grid. BFS arrays use the same ordering as the renderer:

- `BFSNodeInfo`: cell coordinate and level
- `OctreeNode`: `dataIndex`, `childBase`, and `childMask`
- `polygonArrays` / `PolygonRange`: one polygon range per BFS node
- `gBuffer`: one independently analyzed `VoxelData` per BFS node
- `levelNodeCounts`: node counts for levels 0 through maxDepth

The output file contains, in order:

```
GridData
uint32 maxDepth
uint32 levelNodeCounts[maxDepth + 1]
OctreeNode[bfsNodeCount]
VoxelData[bfsNodeCount]
```

The final file does not contain polygon blocks.

## Disk intermediate files

Phase 1 creates one shard per worker:

```
ShardHeader
[nodeKey:u64][dataSize:u32][serialized polygon bytes]*
```

Phase 2 sorts shard references by node key and creates:

- `merge/nodes.idx`: header plus one 24-byte `NodeIndex` per occupied node
- `merge/polygons.dat`: serialized polygon bytes grouped by node key

`NodeIndex.dataOffset` points directly at the first polygon byte for that
node; `polyCount` supplies the number of polygons to read. Shards are removed
after merge. The merge directory is retained for inspection/reuse by the
caller.

## Per-node analysis

The CPU analysis is a port of `AnalyzePolygon.cs.slang`:

1. Reconstruct the original triangle from `TriangleRef`.
2. Reconstruct UV area and sample base color, specular, metallic, and normal
   textures.
3. Assign each polygon to the two 8-lobe normal hemispheres (+n and -n).
4. Rasterize a 64x64 projected depth buffer for each lobe direction. The
   nearest polygon at each pixel contributes projected pixel area to ABSDF
   material accumulation.
5. Normalize lobe weights to the two-sided target sum, fit the ellipsoid, and
   estimate projected-area spherical harmonics.

There is no child-to-parent material, ellipsoid, or SH aggregation.

## Memory and safety limits

The standalone tool caps the number of polygons retained per node at
128000, matching the renderer's effective per-node safety limit. The cap never
removes the node from the octree occupancy set.

## Relevant implementation files

- `SceneVoxelization.h`: in-memory and disk pipeline orchestration/output
- `PolygonGenerator.h`: hierarchical clip and in-memory BFS construction
- `ClipPhase.h`: threaded shard clipping
- `MergePhase.h`: sort/group merge and octree construction
- `AnalyzePhase.h`: streamed and in-memory per-node analysis
- `ABSDF.h`, `Polygon.h`: renderer-compatible analysis primitives
- `PolygonSerializer.h`: shard and nodes.idx formats
