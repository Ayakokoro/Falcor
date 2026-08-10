# Splitting Approximation Error Visualization Plan

**Date:** 2026-08-06

## Background: What is the Splitting Approximation?

From 可视化方案.md Section 1, the VABSDF derivation involves:

**公式3 (Original / GT):**
$$\hat{f}_v(\omega_i, \omega_o) = \underbrace{\frac{\int f \cdot V_i \cdot V_o \cdot \langle n \cdot \omega_i \rangle \cdot \langle n \cdot \omega_o \rangle}{\int V_i \cdot V_o \cdot \langle n \cdot \omega_i \rangle \cdot \langle n \cdot \omega_o \rangle}}_{\text{Factor A (coupled f-V)}} \cdot \underbrace{\frac{\int V_i \cdot V_o \cdot \langle n \cdot \omega_i \rangle \cdot \langle n \cdot \omega_o \rangle}{\int V_o \cdot \langle n \cdot \omega_o \rangle}}_{\text{Factor B (joint visibility)}}$$

**公式4 (Material-only approximation):**
$$\hat{f}_m(\omega_i, \omega_o) \approx \frac{\int f \cdot \langle n \cdot \omega_i \rangle \cdot \langle n \cdot \omega_o \rangle}{\int \langle n \cdot \omega_o \rangle}$$

**公式5 (Visibility-only approximation):**
$$\hat{V}(\omega_i, \omega_o) \approx \frac{\int V_i \cdot V_o \cdot \langle n \cdot \omega_i \rangle \cdot \langle n \cdot \omega_o \rangle}{\int V_o \cdot \langle n \cdot \omega_i \rangle \cdot \langle n \cdot \omega_o \rangle}$$

**The splitting assumption:** Factor A = 公式4 (i.e., material BSDF f is uncorrelated with joint visibility V_i·V_o). The full approximation is:

**VABSDF_approx(ω_i, ω_o) = 公式4 × 公式5**

**VABSDF_gt(ω_i, ω_o) = Factor A × Factor B**

The **error** = |VABSDF_gt - VABSDF_approx| comes from: Factor A ≠ 公式4.

## Visualization Layout

The user wants a 2D grid where the VABSDF is visualized as a function of BOTH directions:

```
┌──────────────────────────────────────────┐
│  ω_i[0]    │  ω_i[1]    │  ... │ ω_i[7]  │  ← Block row 0
│ ω_o grid   │ ω_o grid   │      │ ω_o grid│
│ 32×32      │ 32×32      │      │ 32×32   │
├────────────┼────────────┼──────┼─────────┤
│  ω_i[8]    │  ω_i[9]    │  ... │ ω_i[15] │  ← Block row 1
│ ω_o grid   │ ω_o grid   │      │ ω_o grid│
├────────────┼────────────┼──────┼─────────┤
│    ...     │    ...     │      │   ...   │
├────────────┼────────────┼──────┼─────────┤
│  ω_i[56]   │  ω_i[57]   │  ... │ ω_i[63] │  ← Block row 7
│ ω_o grid   │ ω_o grid   │      │ ω_o grid│
└──────────────────────────────────────────┘
```

- **N_incident = 64** incident directions (arranged in 8×8 blocks)
- **K = 32** pixels per block side → **1024 outgoing directions** per block
- **Total image: 256×256 pixels**
- Each incident direction ω_i sampled uniformly on sphere (Fibonacci or Hammersley)
- Each outgoing direction ω_o within a block sampled uniformly on sphere
- **Pixel (bx, by, px, py)**: block index (bx, by) = ω_i, pixel within block (px, py) = ω_o
- **Color**: VABSDF value at (ω_i, ω_o), or the difference GT vs approx

## Monte Carlo Integration Method

### Step 1: Pre-sample polygon surface points (Compute Shader, one-time)

For the selected voxel:
- Generate M = 512 sample points uniformly on polygon surfaces (area-weighted)
- For each sample point, store in a structured buffer:
  - `position` (float3, voxel-local coordinates)
  - `normal` (float3, world-space, from the source triangle)
  - `diffuse` (float3)
  - `specular` (float3, RGB with F0 already baked)
  - `roughness` (float)
  - Weight proportional to polygon area

### Step 2: Per-direction-pair integration (Compute Shader)

For each of 64×1024 = 65536 direction pairs:

**Input:** ω_i (incident), ω_o (outgoing), half-vector h = normalize(ω_i + ω_o)

**For GT:**
- Accumulate over M sample points:
  - Compute BSDF f(p, ω_i, ω_o, h) using the point's material params (Frostbite + Cook-Torrance)
  - Check visibility V(p, ω_i) via ellipsoid clip test (ray from point to voxel boundary)
  - Check visibility V(p, ω_o) via ellipsoid clip test
  - cos_i = clamp(⟨n_p·ω_i⟩, 0, 1), cos_o = clamp(⟨n_p·ω_o⟩, 0, 1)
- Factor A numerator = Σ f × V_i × V_o × cos_i × cos_o
- Factor A denominator = Σ V_i × V_o × cos_i × cos_o
- Factor B numerator = Σ V_i × V_o × cos_i × cos_o
- Factor B denominator = Σ V_o × cos_o
- GT = (Factor A num / Factor A denom) × (Factor B num / Factor B denom)

**For Approximation:**
- Formula 4 numerator = Σ f × cos_i × cos_o
- Formula 4 denominator = Σ cos_o
- Formula 5 numerator = Σ V_i × V_o × cos_i × cos_o
- Formula 5 denominator = Σ V_o × cos_i × cos_o
- Approximation = (F4 num / F4 denom) × (F5 num / F5 denom)

### Step 3: Display (Full-screen pixel shader)

Output a 3-channel texture: (GT, Approximation, Error)
- Mode 0: Show GT (heatmap coloring, normalization by max value)
- Mode 1: Show Approximation (same heatmap, same normalization)
- Mode 2: Show absolute error |GT - Approx| (same heatmap)
- Mode 3: Show relative error |GT - Approx| / max(GT, ε)

## Implementation Plan

### New Shaders

#### 1. `GenerateSurfaceSamples.cs.slang`
Pre-computes M sample points on polygon surfaces within the selected voxel.
- Input: polygonBuffer, polygonRangeBuffer, target index
- Output: RWStructuredBuffer<SurfaceSample> (pos, normal, diffuse, specular, rough, weight)
- Uses area-weighted sampling: each polygon contributes proportionally to its area
- Dispatch: 1 thread group, each thread handles a subset of polygons

#### 2. `ComputeSplittingError.cs.slang`
Computes GT and approx VABSDF values for all (ω_i, ω_o) pairs.
- Input: SurfaceSample buffer, ellipsoid data, pre-generated direction arrays
- Output: RWTexture2D<float4> (R=GT, G=Approx, B=absError, A=relError)
- CB: blockCount, blockSize, numSamples
- Each thread = one direction pair, iterates over M surface samples
- 65536 threads → dispatch (256, 256, 1)

#### 3. `DisplaySplittingError.ps.slang`
Renders the 2D grid texture.
- Simple full-screen pass that displays the texture
- No sphere projection - flat 2D display
- CB: visualization mode, min/max value range

### Modified Files

#### 4. `VoxelizationPass.h`
Add members:
```cpp
ref<ComputePass> mSurfaceSamplesPass;
ref<ComputePass> mSplittingErrorPass;
ref<FullScreenPass> mDisplaySplittingErrorPass;
ref<Buffer> mSurfaceSamplesBuffer;
ref<Texture> mSplittingErrorMap;
uint32_t mBlockCount = 8;        // N blocks per side (8×8 = 64 ω_i)
uint32_t mBlockSize = 32;        // pixels per block (32×32 = 1024 ω_o)
uint32_t mNumSurfaceSamples = 512;
uint32_t mSplittingVisMode = 0;  // 0=GT, 1=Approx, 2=AbsError, 3=RelError
bool mShowSplittingError = false;
bool mSplittingErrorDirty = false;
```

#### 5. `VoxelizationPass.cpp`
- `execute()`: Add Phase 8 for splitting error visualization
- `renderUI()`: Add controls (block count, block size, num samples, display mode)
- Helper functions: `generateSurfaceSamples()`, `computeSplittingError()`, `displaySplittingError()`

## Performance Considerations

- M=512 samples × 65536 pairs = 33.5M BSDF evaluations + 67M visibility checks
- Each visibility check: ray-ellipsoid intersection (~10 float ops)
- Each BSDF eval: Frostbite + Cook-Torrance = GGX NDF + Fresnel + Smith G (~50 float ops)
- Total: ~5 billion float ops → ~0.5ms on 10 TFLOPS GPU (compute-bound)
- Bandwidth: 512 samples × ~48 bytes = 24KB → fits in L1 cache
- **Result: Feasible, should complete within a few milliseconds**

## Direction Sampling

Use Fibonacci sphere sampling for uniform direction distribution:
```hlsl
float3 fibonacciSphere(uint index, uint total) {
    float y = 1.0 - (2.0 * index + 1.0) / total;
    float radius = sqrt(1.0 - y * y);
    float phi = index * 2.3999632; // golden angle
    return float3(cos(phi) * radius, y, sin(phi) * radius);
}
```

- N_incident = blockCount² directions for ω_i
- N_outgoing = blockSize² directions for ω_o (within each block)

## Verification

1. **Build**: Compile all shaders and C++ code
2. **Generate voxel data**: Use "Generate" button
3. **Select voxel**: Set LOD + cellInt
4. **Click "Compute Splitting Error"** → runs surface sampling + error computation
5. **Toggle display modes**: GT / Approximation / Error to see the splitting error distribution
6. **Compare**: Error image should highlight angle regions where material-visibility correlation is strong
7. **Validate**: For simple test cases (e.g., flat plane), the splitting error should be minimal
