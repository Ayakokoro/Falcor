// Include after Intersection.slang and CloudGPIS.slang.
#include "DensityClipmap.slang"

// For unit-variance SE noise n and extinction chi, the GPIS mean u obeys
// phi(u)/Phi(u) = sqrt(2*pi)*ell*chi (Cui et al., 2026, Eq. 57).
// The SIGN of n+u is the sign of M(n)-sqrt(2*pi)*ell*chi, M(n)=phi(n)/Phi(-n).
// This monotone reparameterization evaluates the inversion AFTER interpolating
// the original trilinear density. Zero density is true vacuum.
float densityMills(float n)
{
    float a = abs(n);
    if (a > 8.f)
    {
        if (n < 0.f) return 0.3989422804f * exp(-0.5f * n * n);
        float inv = 1.f / n, s = inv * inv;
        return n + inv * (1.f + s * (-2.f + s * (10.f - 74.f * s)));
    }
    // Scaled complementary error function, keeping the upper tail out of a
    // subtract-nearly-one operation. The common Gaussian exponential cancels.
    float t = 1.f / (1.f + 0.3535533906f * a);
    float p = mad(t, 0.17087277f, -0.82215223f);
    p = mad(t, p,  1.48851587f);
    p = mad(t, p, -1.13520398f);
    p = mad(t, p,  0.27886807f);
    p = mad(t, p, -0.18628806f);
    p = mad(t, p,  0.09678418f);
    p = mad(t, p,  0.37409196f);
    p = mad(t, p,  1.00002368f);
    float scaled = t * exp(mad(t, p, -1.26551223f));
    if (n >= 0.f) return 0.7978845608f / scaled;
    float tail = exp(-0.5f * n * n);
    return 0.7978845608f * tail / (2.f - scaled * tail);
}

float densityMillsSlope(float n)
{
    if (n > 8.f)
    {
        float s = 1.f / (n * n);
        return 1.f - s * (1.f - s * (6.f - 50.f * s));
    }
    float m = densityMills(n);
    return max(m * (m - n), 1.e-37f);
}

// The inverse is only needed to condition the noise at a flight's birth.
float densityNoiseThreshold(float target)
{
    float lo = target > 1.f ? target - 1.f : -16.f, hi = target;
    [unroll] for (int i = 0; i < 22; ++i)
    {
        float mid = lo + .5f * (hi - lo);
        if (densityMills(mid) < target)
            lo = mid;
        else
            hi = mid;
    }
    return lo + .5f * (hi - lo);
}

float2 densityBezierRange(float2 a, float2 b, float width)
{
    float left = a.x + (width / 3.f) * a.y, right = b.x - (width / 3.f) * b.y;
    return float2(min(min(a.x, b.x), min(left, right)), max(max(a.x, b.x), max(left, right)));
}

// Both trilinear density and Hermite noise are cubic on this interval. Their
// nonlinear combination need not be cubic or monotone. Bound each polynomial
// by its Bezier controls, then search possibly crossing intervals front to back.
// Isolation stops at 1/16384 of the shorter voxel/noise segment.
bool densityFirstCrossing(float4 noise, float4 target, bool bornHere, out float hit)
{
    hit = 0.f;
    const uint count = 16384u;
    uint first = 0, size = count;
    [loop] while (first < count)
    {
        float lo = float(first) / float(count), width = float(size) / float(count), hi = lo + width;
        float2 nl = latticePolynomialJet(noise, lo), nr = latticePolynomialJet(noise, hi);
        float2 dl = latticePolynomialJet(target, lo), dr = latticePolynomialJet(target, hi);
        float2 dn = densityBezierRange(dl, dr, width), nn = densityBezierRange(nl, nr, width);
        if (dn.y > 0.f && densityMills(nn.x) <= dn.y && densityMills(nn.y) >= dn.x)
        {
            // The derivative of a cubic is quadratic; these are its three
            // Bezier controls up to the same positive interval-width factor.
            float nd = 3.f * (nr.x - nl.x) - width * (nl.y + nr.y);
            float dd = 3.f * (dr.x - dl.x) - width * (dl.y + dr.y);
            bool decreasing = max(nd, width * max(nl.y, nr.y)) <= 0.f && min(dd, width * min(dl.y, dr.y)) >= 0.f;
            bool increasing = min(nd, width * min(nl.y, nr.y)) >= 0.f && max(dd, width * max(dl.y, dr.y)) <= 0.f;
            if (decreasing || size == 1)
            {
                if (!(bornHere && first == 0u) && densityMills(nl.x) >= dl.x && densityMills(nr.x) < dr.x)
                {
                    [unroll] for (int k = 0; k < 14; ++k)
                    {
                        float mid = .5f * (lo + hi);
                        if (densityMills(latticePolynomialJet(noise, mid).x) >= latticePolynomialJet(target, mid).x)
                            lo = mid;
                        else
                            hi = mid;
                    }
                    hit = .5f * (lo + hi);
                    return true;
                }
            }
            else if (!increasing)
            {
                size >>= 1;
                continue;
            }
        }
        first += size;
        size = first & (~first + 1u);
    }
    return false;
}

bool densityGPISIntersect(float3 origin, float3 direction, float tMax, float3 translation, float worldScale, float ell,
                          float amplitude, float extinctionScale, uint seed, bool conditioned, float3 birthGradient,
                          out float hitDistance, out float3 hitGradient)
{
    hitDistance = 0.f;
    hitGradient = 0.f;
    if (amplitude == 0.f || extinctionScale == 0.f) return false;
    float3 localOrigin = (origin - translation) / worldScale;
    float3 localDirection = direction / worldScale;
    float2 span = float2(0.f, tMax);
    if (!densityRaySpan(localOrigin, localDirection, span)) return false;
    conditioned = conditioned && span.x == 0.f;
    localOrigin += span.x * localDirection;
    float limit = span.y - span.x;
    float densityFactor = 2.506628275f * ell * extinctionScale;
    LatticeKernel kernel = cloudStationaryKernel(ell, 1.f);
    LatticeFlight flight = makeLatticeFlight(0.f, direction, kernel, seed, false, 0.f);
    NoiseSegment noise = {};
    float t = 0.f;
    [loop] while (t < limit)
    {
        SdfCell cell;
        float3 f, cellDir;
        float end;
        int level;
        densityRayCell(localOrigin, localDirection, t, limit, cell, f, cellDir, end, level);
        if (!(end > t)) break;
        if (all(cell.z0 <= 0.f) && all(cell.z1 <= 0.f))
        {
            t = end;
            continue;
        }
        float cellStart = t;
        if (t == 0.f)
        {
            float target = densityFactor * cell.value(f);
            if (target > 0.f)
            {
                if (conditioned)
                {
                    float n = densityNoiseThreshold(target);
                    float dn = densityFactor * dot(cell.gradient(f), cellDir) / densityMillsSlope(n);
                    flight = makeLatticeFlight(0.f, direction, kernel, seed, true, float2(n, dot(birthGradient, direction) + dn));
                }
                else
                {
                    float prior = stationaryKnot(0, seed, 1.f, 0.f, kernel).x;
                    if (densityMills(prior) <= target)
                    {
                        float z = densityTruncatedNormal(densityNoiseThreshold(target), seed ^ 0xc8f4e26du);
                        flight.correction.x = (z - prior) * kernel.stationaryInvVariance.x;
                    }
                }
            }
        }
        [loop] while (t < end)
        {
            if (noise.end <= t)
            {
                int knot = int(floor(t / flight.spacing));
                if (float(knot + 1) * flight.spacing <= t)
                    ++knot;
                else if (float(knot) * flight.spacing > t)
                    --knot;
                float start = float(knot) * flight.spacing;
                // A gap of vacuum cells has no cached left-end jet.
                if (start != noise.end) noise.inRamp = true;
                noise = nextNoiseSegment(flight, start, knot, noise);
                if (noise.end <= t) return false;
            }
            float finish = min(noise.end, end), width = finish - t, knotWidth = noise.end - noise.start;
            float2 nl = latticePolynomialJet(noise.polynomial, (t - noise.start) / knotWidth);
            float2 nr = latticePolynomialJet(noise.polynomial, (finish - noise.start) / knotWidth);
            float4 np = latticeHermite(nl * float2(1.f, width / knotWidth), nr * float2(1.f, width / knotWidth));
            float3 a = saturate(f + (t - cellStart) * cellDir), b = saturate(f + (finish - cellStart) * cellDir);
            float2 dl = float2(cell.value(a), width * dot(cell.gradient(a), cellDir));
            float2 dr = float2(cell.value(b), width * dot(cell.gradient(b), cellDir));
            float u;
            if (densityFirstCrossing(np, densityFactor * latticeHermite(dl, dr), flight.conditioned && t == 0.f, u))
            {
                float s = t + u * width;
                float3 at = lerp(a, b, u);
                float2 n = latticePolynomialJet(np, u);
                float3 meanGradient =
                    -(densityFactor / densityMillsSlope(n.x)) * densityWorldGradient(cell.gradient(at), level) / worldScale;
                float axial = n.y / width + dot(meanGradient, direction);
                float sd = stationaryTransverseSD((s - noise.start) / knotWidth, 1.f / ell, kernel);
                hitGradient = latticeSurfaceGradient(direction, meanGradient, axial, 0.f, sd, seed);
                hitDistance = span.x + s;
                return hitDistance < tMax;
            }
            t = finish;
        }
    }
    return false;
}
