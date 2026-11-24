// Include after SdfField.slang. Distances and SDF values are world units.
#include "LatticeNoise.slang"

float latticeInteriorCut(float u)
{
    return u > 0.f && u < 1.f ? u : 1.f;
}

// Split at the derivative roots and search the resulting monotone intervals in
// order. Sampling only the two voxel faces would miss a thin negative pocket.
bool latticeFirstDowncrossing(float4 p, float endValue, inout bool positiveOnLeft, out float uHit)
{
    uHit = 0.f;
    float4 controls = float4(p.x, p.x + p.y / 3.f, p.x + (2.f * p.y + p.z) / 3.f, endValue);
    float margin = 8e-7f * dot(abs(p), float4(1.f));
    if (all(controls > margin) || all(controls < -margin))
    {
        positiveOnLeft = controls.x > 0.f;
        return false;
    }
    float2 cuts = 1.f;
    float3 derivative = float3(3.f * p.w, 2.f * p.z, p.y);
    float scale = max(abs(derivative.x), max(abs(derivative.y), abs(derivative.z)));
    if (scale > 0.f) derivative /= scale;
    if (derivative.x == 0.f)
    {
        if (derivative.y != 0.f) cuts.x = latticeInteriorCut(-derivative.z / derivative.y);
    }
    else
    {
        float disc = derivative.y * derivative.y - 4.f * derivative.x * derivative.z;
        if (disc > 0.f)
        {
            float q = -0.5f * (derivative.y + (derivative.y < 0.f ? -sqrt(disc) : sqrt(disc)));
            float r0 = latticeInteriorCut(q / derivative.x);
            float r1 = latticeInteriorCut(derivative.z / q);
            cuts = float2(min(r0, r1), max(r0, r1));
        }
        else if (disc == 0.f)
            cuts.x = latticeInteriorCut(-0.5f * derivative.y / derivative.x);
    }
    float lo = 0.f;
    float left = p.x;
    [unroll]
    for (int i = 0; i < 3; ++i)
    {
        float hi = i == 0 ? cuts.x : (i == 1 ? cuts.y : 1.f);
        float right = hi == 1.f ? endValue : latticePolynomialJet(p, hi).x;
        if (right < 0.f && (left > 0.f || (left == 0.f && positiveOnLeft)))
        {
            if (left == 0.f) { uHit = lo; return true; }
            [unroll]
            for (uint step = 0; step < 16; ++step)
            {
                float mid = 0.5f * (lo + hi);
                if (latticePolynomialJet(p, mid).x > 0.f) lo = mid;
                else hi = mid;
            }
            uHit = 0.5f * (lo + hi);
            return true;
        }
        if (right != 0.f) positiveOnLeft = right > 0.f;
        else if (left != 0.f) positiveOnLeft = left > 0.f;
        if (hi == 1.f) break;
        lo = hi;
        left = right;
    }
    return false;
}

struct LatticeFlight
{
    LatticeKernel kernel;
    uint seed;
    bool varying;
    bool conditioned;
    float originY, dirY, spacing;
    float2 rampPlanes;
    float2 target; // residual value and directional derivative at birth
    float2 correction;
    VaryingBirth varyingBirth;
};

struct NoiseSegment
{
    int cell;
    float start, end;
    bool inRamp;
    float2 rightJet;
    float4 polynomial;
};

bool insideLatticeRamp(LatticeFlight flight, float t)
{
    float height = flight.originY + t * flight.dirY;
    return height > flight.kernel.heightRange.x && height < flight.kernel.heightRange.y;
}

NoiseSegment nextNoiseSegment(LatticeFlight flight, float start, int cell, NoiseSegment previous)
{
    NoiseSegment seg;
    seg.cell = cell;
    seg.start = start;
    seg.end = float(cell + 1) * flight.spacing;
    if (flight.varying)
    {
        if (flight.rampPlanes.x > start) seg.end = min(seg.end, flight.rampPlanes.x);
        if (flight.rampPlanes.y > start) seg.end = min(seg.end, flight.rampPlanes.y);
    }
    float width = seg.end - start;
    seg.inRamp = flight.varying && insideLatticeRamp(flight, start + 0.5f * width);
    bool reuse = start > 0.f && seg.inRamp == previous.inRamp;
#if GPIS_MESH_CLIPMAP
    reuse = reuse && previous.end == start;
#endif
    // Keep jets in physical units so a neighboring, differently sized piece
    // can reuse its endpoint, including when a height plane splits a cell.
    float2 left;
    if (flight.varying)
    {
        if (reuse) left = previous.rightJet;
        else
        {
            LatticeProfile p = latticeProfile(flight.kernel, flight.originY + start * flight.dirY, seg.inRamp);
            VaryingLatticeSample s = varyingSample(start, flight.spacing, flight.kernel.kernelRadiusInEll, flight.dirY, p, flight.seed, flight.varyingBirth);
            left = float2(s.value, s.slope);
        }
        LatticeProfile p = latticeProfile(flight.kernel, flight.originY + seg.end * flight.dirY, seg.inRamp);
        VaryingLatticeSample s = varyingSample(seg.end, flight.spacing, flight.kernel.kernelRadiusInEll, flight.dirY, p, flight.seed, flight.varyingBirth);
        seg.rightJet = float2(s.value, s.slope);
    }
    else
    {
        float2 units = float2(1.f, 1.f / flight.spacing);
        left = reuse ? previous.rightJet : stationaryKnot(cell, flight.seed, flight.kernel.amplitude.x, flight.correction, flight.kernel) * units;
        seg.rightJet = stationaryKnot(cell + 1, flight.seed, flight.kernel.amplitude.x, flight.correction, flight.kernel) * units;
    }
    if (start == 0.f && flight.conditioned) left = flight.target;
    seg.polynomial = latticeHermite(left * float2(1.f, width), seg.rightJet * float2(1.f, width));
    return seg;
}

LatticeFlight makeLatticeFlight(float3 origin, float3 direction, LatticeKernel kernel, uint seed,
                                bool conditioned, float2 residual)
{
    LatticeFlight f = {};
    f.originY = origin.y;
    f.dirY = direction.y;
    f.seed = seed;
    f.kernel = kernel;
    f.varying = kernel.nonstationary != 0 &&
        (kernel.lengthScale.x != kernel.lengthScale.y || kernel.amplitude.x != kernel.amplitude.y);
    f.spacing = kernel.latticeSpacingInEll * (f.varying ? min(kernel.lengthScale.x, kernel.lengthScale.y) : kernel.lengthScale.x);
    if (f.varying && direction.y != 0.f)
        f.rampPlanes = (kernel.heightRange - origin.y) / direction.y;
    f.conditioned = conditioned;
    f.target = residual;
    if (conditioned && f.varying)
    {
        float firstEnd = f.spacing;
        if (f.rampPlanes.x > 0.f) firstEnd = min(firstEnd, f.rampPlanes.x);
        if (f.rampPlanes.y > 0.f) firstEnd = min(firstEnd, f.rampPlanes.y);
        LatticeProfile p = latticeProfile(kernel, origin.y, insideLatticeRamp(f, 0.5f * firstEnd));
        f.varyingBirth = varyBirth(p, f.spacing, kernel.kernelRadiusInEll, direction.y, seed, residual);
    }
    else if (conditioned && kernel.amplitude.x > 0.f)
    {
        float2 prior = stationaryKnot(0, seed, 1.f, 0.f, kernel);
        f.correction = (residual * float2(1.f, f.spacing) / kernel.amplitude.x - prior) * kernel.stationaryInvVariance;
    }
    return f;
}

#if GPIS_MESH_CLIPMAP
// Box-Muller draws from U >= 2^-24, hence each source coefficient has
// magnitude < 6. A cubic Hermite segment is bounded by its Bezier controls
// (endpoint value +/- segment width times endpoint derivative / 3).
float priorNoiseBound(LatticeFlight flight)
{
    LatticeKernel k = flight.kernel;
    if (!flight.varying)
    {
        // Each row has at most 2R nonzero weights. Cauchy-Schwarz bounds
        // its L1 norm using the stored L2 norms of value and slope.
        float2 rowNorm = rsqrt(k.stationaryInvVariance);
        return 6.2f * k.amplitude.x * sqrt(2.f * float(k.stationaryRadius)) * (rowNorm.x + rowNorm.y / 3.f);
    }
    float minEll = min(k.lengthScale.x, k.lengthScale.y);
    float radius = k.kernelRadiusInEll;
    float sigma = max(k.amplitude.x, k.amplitude.y);
    float count = 2.f * ceil(radius * max(k.lengthScale.x, k.lengthScale.y) / flight.spacing) + 2.f;
    float2 slope = abs(float2(k.amplitude.y - k.amplitude.x, k.lengthScale.y - k.lengthScale.x) * flight.dirY)
                 / (k.heightRange.y - k.heightRange.x);
    float maxDx = 2.f * radius + 1.5f / min(0.5f, 0.5f * radius);
    float derivative = slope.x + sigma / minEll * (maxDx + slope.y * (0.5f + radius * maxDx));
    // The Gaussian envelope and its tapered dimensionless derivative are
    // bounded by sqrt(h/ell) and maxDx * sqrt(h/ell), respectively.
    return 6.2f * count * sqrt(flight.spacing / minEll) * (sigma + flight.spacing * derivative / 3.f);
}

float cellClearance(SdfCell cell)
{
    // The trilinear interpolant stays within the range of the eight corners.
    float4 lo = min(cell.z0, cell.z1);
    float4 hi = max(cell.z0, cell.z1);
    return max(min(min(lo.x, lo.y), min(lo.z, lo.w)), -max(max(hi.x, hi.y), max(hi.z, hi.w)));
}

NoiseSegment seekNoiseSegment(LatticeFlight flight, float t)
{
    int cell = int(floor(t / flight.spacing));
    if (float(cell + 1) * flight.spacing <= t) ++cell;
    else if (float(cell) * flight.spacing > t) --cell;
    float start = float(cell) * flight.spacing;
    if (flight.varying)
    {
        if (flight.rampPlanes.x <= t) start = max(start, flight.rampPlanes.x);
        if (flight.rampPlanes.y <= t) start = max(start, flight.rampPlanes.y);
    }
    NoiseSegment empty = {};
    empty.end = -1.f;
    return nextNoiseSegment(flight, start, cell, empty);
}

// Pick the finest covering level for the portion immediately following t.
// A ray can enter a finer window later, including on a reflected flight.
int clipRayRegion(float3 origin, float3 direction, float t, float limit, out float regionEnd)
{
    regionEnd = limit;
    [loop]
    for (int level = gField.clipFirstLevel; level < gField.clipCount - 1; ++level)
    {
        float4 layout = gField.clipLevels[level];
        float3 lower = layout.xyz - origin;
        float3 upper = lower + layout.w * float(gField.clipResolution - 1);
        float2 span = float2(0.f, limit);
        [unroll] for (int k = 0; k < 3; ++k)
        {
            if (direction[k] == 0.f)
            {
                if (lower[k] > 0.f || upper[k] < 0.f) span.y = -1.f;
            }
            else
            {
                float2 faces = float2(lower[k], upper[k]) / direction[k];
                span.x = max(span.x, min(faces.x, faces.y));
                span.y = min(span.y, max(faces.x, faces.y));
            }
        }
        if (span.x >= span.y || span.y <= t) continue;
        if (span.x <= t)
        {
            regionEnd = min(regionEnd, span.y);
            return level;
        }
        regionEnd = min(regionEnd, span.x);
    }
    return gField.clipCount - 1;
}
#else
bool clipLatticeField(float3 p, float3 d, inout float tNear, inout float tFar)
{
    [unroll]
    for (int k = 0; k < 3; ++k)
    {
        if (d[k] == 0.f)
        {
            if (p[k] < 0.f || p[k] > gField.indexMax[k]) return false;
        }
        else
        {
            float2 f = (float2(0.f, gField.indexMax[k]) - p[k]) / d[k];
            tNear = max(tNear, min(f.x, f.y));
            tFar = min(tFar, max(f.x, f.y));
        }
    }
    return tNear < tFar;
}
#endif

float3 nextVoxelFaces(int3 cell, int3 step, float3 p, float3 invD)
{
    float3 edge = float3(cell + max(step, 0));
    float3 t = (edge - p) * invD;
    return float3(step.x != 0 ? t.x : 3.4e38f, step.y != 0 ? t.y : 3.4e38f, step.z != 0 ? t.z : 3.4e38f);
}

#if GPIS_MESH_CLIPMAP
struct ClipRayGrid
{
    int level;
    float begin;
    float end;
    float voxelWidth;
    float3 p; // local index position at begin
    float3 indexDir;
    float3 invD;
    int3 step;
    int3 cell;

    float3 nextFaces()
    {
        float3 edges = begin + nextVoxelFaces(cell, step, p, invD);
        // The slab interval determines the exact switch time. The same
        // outer face evaluated via local DDA arithmetic may differ by one ULP.
        [unroll] for (int k = 0; k < 3; ++k)
            if (cell[k] + step[k] < 0 || cell[k] + step[k] >= gField.clipResolution - 1)
                edges[k] = end;
        return edges;
    }
};

ClipRayGrid beginClipRayGrid(float3 origin, float3 direction, float t, float limit)
{
    ClipRayGrid grid;
    grid.level = clipRayRegion(origin, direction, t, limit, grid.end);
    grid.begin = t;
    float4 layout = gField.clipLevels[grid.level];
    grid.voxelWidth = layout.w;
    grid.p = clamp(((origin - layout.xyz) + t * direction) / layout.w, float3(0.f), float3(gField.clipResolution - 1));
    grid.indexDir = direction / layout.w;
    grid.step = int3(sign(direction));
    grid.invD = float3(direction.x != 0.f ? layout.w / direction.x : 0.f,
                       direction.y != 0.f ? layout.w / direction.y : 0.f,
                       direction.z != 0.f ? layout.w / direction.z : 0.f);
    grid.cell = int3(floor(grid.p));
    [unroll] for (int k = 0; k < 3; ++k)
        if (grid.step[k] < 0 && grid.p[k] == float(grid.cell[k])) --grid.cell[k];
    grid.cell = clamp(grid.cell, int3(0), int3(gField.clipResolution - 2));
    return grid;
}
#endif

bool latticeIntersect(float3 origin, float3 direction, float tNear, float tFar, uint seed, LatticeKernel kernel,
                      bool conditioned, float3 birthGradient, out float hitDistance, out float3 hitGradient)
{
    hitDistance = 0.f;
    hitGradient = 0.f;
#if GPIS_MESH_CLIPMAP
    conditioned = conditioned && tNear == 0.f;
    origin += tNear * direction;
    float limit = tFar - tNear;
    ClipRayGrid grid = beginClipRayGrid(origin, direction, 0.f, limit);
    SdfCell sdf = loadClipCell(grid.level, grid.cell);
    float3 indexDir = grid.indexDir;
    float3 frac = saturate(grid.p - float3(grid.cell));
#else
    float3 indexOrigin = gField.worldToIndexPoint(origin);
    float3 indexDir = gField.worldToIndexDir(direction);
    if (!clipLatticeField(indexOrigin, indexDir, tNear, tFar)) return false;
    conditioned = conditioned && tNear == 0.f;
    origin += tNear * direction;
    float limit = tFar - tNear;
    float3 p = clamp(indexOrigin + tNear * indexDir, float3(0.f), float3(gField.indexMax));
    int3 step = int3(sign(indexDir));
    float3 invD = float3(indexDir.x != 0.f ? 1.f / indexDir.x : 0.f, indexDir.y != 0.f ? 1.f / indexDir.y : 0.f,
                         indexDir.z != 0.f ? 1.f / indexDir.z : 0.f);
    int3 cell = int3(floor(p));
    [unroll] for (int k = 0; k < 3; ++k) if (step[k] < 0 && p[k] == float(cell[k])) --cell[k];
    cell = clamp(cell, int3(0), int3(gField.indexMax) - 1);
    SdfCell sdf = gField.loadCell(cell);
    float3 frac = saturate(p - float3(cell));
#endif
    float2 residual = conditioned ? float2(-sdf.value(frac), dot(birthGradient, direction) - dot(sdf.gradient(frac), indexDir)) : float2(0.f);
#if GPIS_MESH_CLIPMAP
    bool hasNoise = kernel.amplitude.x > 0.f || (kernel.nonstationary != 0 && kernel.amplitude.y > 0.f);
    LatticeFlight flight = {};
    NoiseSegment noise = {};
    noise.end = limit;
    float noiseBound = 0.f;
    float priorFrom = 0.f;
    if (hasNoise)
    {
        flight = makeLatticeFlight(origin, direction, kernel, seed, conditioned, residual);
        noise = nextNoiseSegment(flight, 0.f, 0, noise);
        noiseBound = priorNoiseBound(flight);
        if (conditioned)
        {
            // Beyond twice the widest support plus a Hermite interval, neither
            // endpoint knot uses any coefficient conditioned at the bounce.
            float radius = flight.varying ? kernel.kernelRadiusInEll * max(kernel.lengthScale.x, kernel.lengthScale.y)
                                          : float(kernel.stationaryRadius) * flight.spacing;
            priorFrom = 2.f * (radius + flight.spacing);
        }
    }
#else
    LatticeFlight flight = makeLatticeFlight(origin, direction, kernel, seed, conditioned, residual);
    NoiseSegment empty = {};
    NoiseSegment noise = nextNoiseSegment(flight, 0.f, 0, empty);
#endif
    bool startInside = !conditioned && sdf.value(frac) + noise.polynomial.x < 0.f;
    bool positiveOnLeft = !conditioned;
    float t = 0.f;
#if GPIS_MESH_CLIPMAP
    float3 edges = grid.nextFaces();
    float clearance = cellClearance(sdf);
#else
    float3 edges = nextVoxelFaces(cell, step, p, invD);
#endif
    [loop]
    while (t < limit)
    {
        float nearestFace = min(edges.x, min(edges.y, edges.z));
#if GPIS_MESH_CLIPMAP
        float end = min(nearestFace, grid.end);
        if (clearance > noiseBound && t >= priorFrom)
        {
            // The entire cell is on one side even after adding this flight's
            // noise. Re-enter the same realization by knot index at the next
            // candidate cell, without evaluating intervening noise segments.
            positiveOnLeft = startInside ? sdf.z0.x < 0.f : sdf.z0.x > 0.f;
            if (end > t) t = end;
        }
        else
        {
            if (hasNoise && noise.end <= t)
            {
                if (noise.end == t)
                {
                    int nextCell = noise.cell + (noise.end >= float(noise.cell + 1) * flight.spacing ? 1 : 0);
                    noise = nextNoiseSegment(flight, t, nextCell, noise);
                }
                else noise = seekNoiseSegment(flight, t);
                if (noise.end <= t) break;
            }
            end = min(end, noise.end);
        }
#else
        float end = min(limit, min(noise.end, nearestFace));
#endif
        if (end > t)
        {
            float width = end - t;
            float noiseWidth = noise.end - noise.start;
#if GPIS_MESH_CLIPMAP
            float3 a = saturate(grid.p - float3(grid.cell) + (t - grid.begin) * indexDir);
            float3 b = saturate(grid.p - float3(grid.cell) + (end - grid.begin) * indexDir);
#else
            float3 a = saturate(p - float3(cell) + t * indexDir);
            float3 b = saturate(p - float3(cell) + end * indexDir);
#endif
            float2 left = latticePolynomialJet(noise.polynomial, (t - noise.start) / noiseWidth);
            float2 right = latticePolynomialJet(noise.polynomial, (end - noise.start) / noiseWidth);
            left = float2(sdf.value(a) + left.x, width * (dot(sdf.gradient(a), indexDir) + left.y / noiseWidth));
            right = float2(sdf.value(b) + right.x, width * (dot(sdf.gradient(b), indexDir) + right.y / noiseWidth));
            if (conditioned && t == 0.f) left = float2(0.f, width * dot(birthGradient, direction));
            float4 polynomial = latticeHermite(left, right);
            float orientation = startInside ? -1.f : 1.f;
            float u;
            if (latticeFirstDowncrossing(orientation * polynomial, orientation * right.x, positiveOnLeft, u))
            {
                float hit = t + width * u;
#if GPIS_MESH_CLIPMAP
                float3 meanGradient = sdf.gradient(lerp(a, b, u)) / grid.voxelWidth;
                if (!hasNoise)
                {
                    hitDistance = tNear + hit;
                    hitGradient = dot(meanGradient, meanGradient) > 0.f ? meanGradient : -direction;
                    return true;
                }
#else
                float3 meanGradient = gField.worldGradient(sdf.gradient(lerp(a, b, u)));
#endif
                float axial = latticePolynomialJet(polynomial, u).y / width;
                float responseY = 0.f;
                float sd;
                if (flight.varying)
                {
                    LatticeProfile profile = latticeProfile(kernel, origin.y + hit * direction.y, noise.inRamp);
                    VaryingLatticeSample s = varyingSample(hit, flight.spacing, kernel.kernelRadiusInEll, direction.y, profile, seed, flight.varyingBirth);
                    responseY = s.dY;
                    sd = sqrt(s.variance) / profile.ell;
                }
                else sd = stationaryTransverseSD((hit - noise.start) / noiseWidth, kernel.amplitude.x / kernel.lengthScale.x, kernel);
                hitDistance = tNear + hit;
                hitGradient = latticeSurfaceGradient(direction, meanGradient, axial, responseY, sd, seed);
                return true;
            }
            t = end;
        }
        if (t >= limit) break;
#if GPIS_MESH_CLIPMAP
        if (t >= grid.end)
        {
            grid = beginClipRayGrid(origin, direction, t, limit);
            indexDir = grid.indexDir;
            edges = grid.nextFaces();
            sdf = loadClipCell(grid.level, grid.cell);
            clearance = cellClearance(sdf);
        }
        else if (nearestFace <= t)
        {
            [unroll] for (int k = 0; k < 3; ++k) if (edges[k] <= t) grid.cell[k] += grid.step[k];
            edges = grid.nextFaces();
            sdf = loadClipCell(grid.level, grid.cell);
            clearance = cellClearance(sdf);
        }
#else
        if (nearestFace <= t)
        {
            [unroll] for (int k = 0; k < 3; ++k) if (edges[k] <= t) cell[k] += step[k];
            if (any(cell < 0) || any(cell >= int3(gField.indexMax))) break;
            edges = nextVoxelFaces(cell, step, p, invD);
            sdf = gField.loadCell(cell);
        }
        if (noise.end <= t)
        {
            int nextCell = noise.cell + (noise.end >= float(noise.cell + 1) * flight.spacing ? 1 : 0);
            noise = nextNoiseSegment(flight, noise.end, nextCell, noise);
            if (noise.end <= t) break; // The grid resolution exceeds float arclength precision.
        }
#endif
    }
    return false;
}
