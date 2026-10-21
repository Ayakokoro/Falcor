example use

cmake --build . --config Release

.\build\Release\LargeSceneVoxelization.exe "..\Scene\Old tree\OldTree.fbx" -o "..\resource\OldTree.bin" -r 128 -t 2 -s 256 -j 20

.\build\Release\LargeSceneVoxelization.exe "..\Scene\Old tree\OldTree.fbx" -o "..\resource\OldTree.bin" -r 256 -s 256

.\build\Release\LargeSceneVoxelization.exe "..\Scene\Metropolis\Metropolis.fbx" -o "..\resource\Metropolis_1.bin" -r 256 -s 1024

## CPU visibility bitmask bake

Add `--visibility-bitmask` to either bake pipeline to write `<output>.vbm`
(VBM1: 4x4 direction words, 32 shared surface samples, 68 bytes per node).
This replaces the legacy scalar-correlation bake; the VPW3 main file retains
its layout and its scalar correlation field is zero in this mode.
`RayMarchingPass` automatically loads and validates the companion when reading
the `.bin`. Set `Draw Mode` to `Default`, then choose `Internal Visibility`:
`Legacy one-way (VPA/TPA)` (default), `Bitmask`, or `Off`.
Missing or rejected companions use Q=1 in Bitmask mode and report their status in
the UI.

The renderer supports up to five shared buffer segments, each below 2 GiB.
The node capacity is `5 * floor((2^31 - 1) / largest record stride)`; with the
current 696-byte `TEBSDF`, this is 15,427,325 nodes across all octree levels.
Available GPU memory can impose a lower limit.
The five-segment update passes the Release plugin build and standalone compilation
of the preparation, ray-marching and NDF shaders. Full-scene rendering has not
been validated as part of this update.

See the [implementation document](../docs/development/voxel-visibility-bitmask.md)
for sampling, binary layout, GPU evaluation, diagnostics and validation scope.

## Directional material validation

Run from `LargeSceneVoxelization/`, using the unchanged source model, an existing
VPW3 `.bin`, and the `merge/` cache from that same disk-backed bake:

```powershell
.\build\Release\LargeSceneVoxelization.exe "..\Scene\Old tree\OldTree.fbx" --validate-material "..\resource\OldTree.bin" --tmp-dir ".\tmp" --lod 0 --cell 32 45 67 --view 0 0 1 --report ".\OldTree_material.csv"
```

The cell above is only an example; select an occupied node from your bake.
`--tmp-dir` must be the actual directory used when baking, not necessarily
`./tmp`. Validation does not clip, analyze, rewrite the bake, or delete caches.
It reads only the selected node's polygon/material data and prints a report.
`--report` additionally writes CSV and refuses to overwrite existing files.

- `--lod`: renderer convention, finest leaf = 0. Cache root-level index is
  `maxDepth - lod`.
- `--cell`: integer coordinates **at the selected LOD**. If the RayMarching
  selected-cell display gives a minimum corner in leaf-grid coordinates,
  divide each coordinate by `2^lod` to get these coordinates.
- `--view`: finite, nonzero direction from the node **toward the observer**,
  in world-space axes. It is normalized internally; `0 0 1` views from +Z.
- Reference: nearest-polygon ownership from the same 64x64 depth rasterizer
  used by the bake. Each polygon uses `AnalyzePhase::sampleMaterial()` with
  the bake's polygon-level UV/texture LOD filtering, not per-pixel UV sampling.
  Blank pixels are excluded from the material average.
- Aggregate: the stored `.bin` lobes, weighted by
  `weight * max(dot(normal, view), 0)` and normalized over valid front-facing
  lobes. The baked data is **not** recomputed for this comparison.
- Output: linear-RGB diffuse, mirror F0, mean roughness, absolute errors,
  reference/aggregate direction-dependent weights for all eight lobes, and
  each stored lobe's normal, amplitude and material parameters. No Coverage,
  visibility, external lighting, BRDF evaluation or ray bounces are applied.
  Empty projections or zero front-facing aggregate weight are reported as
  invalid/N/A, not as black materials. Roughness is a parameter mean, not an
  equivalent fitted BRDF roughness.

Depth, node/polygon counts, file sizes, bake tag, polygon bounds and source
references are checked. Legacy caches have no source fingerprint: these checks
cannot prove that a different model/cache with matching counts is the right
one. Preserve a dedicated `--tmp-dir` per bake and keep the source/textures and
material-loader version unchanged. Missing textures fail validation instead
of silently substituting a potentially misleading reference material.

GLB/glTF color factors are read as linear values using Assimp's PBR base-color
key (with a linear diffuse-key fallback). Base-color textures still receive
sRGB decoding; FBX/non-glTF factor handling is unchanged. Re-bake existing
GLB/glTF `.bin` files made before this fix before comparing them to the updated
reference loader. For the Bedroom `quadro_bbb` material, the untextured linear
diffuse should be approximately `(0.800007, 0.493300, 0.169919)`, not the old
double-decoded `(0.603840, 0.207892, 0.024494)`. This fix does not change texture
factor multiplication, roughness/metallic handling, or workflow classification.

