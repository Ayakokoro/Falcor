/*
 * LatticeNoise.slang and Intersection.slang are altered GPU implementations of
 * the lattice noise and crossing algorithms of sparse-conv-gpis-tungsten.
 *
 * Copyright (c) 2014 Benedikt Bitterli <benedikt.bitterli (at) gmail (dot) com>
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising from
 * the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute
 * it freely, subject to the following restrictions:
 *
 *     1. The origin of this software must not be misrepresented; you
 *        must not claim that you wrote the original software. If you
 *        use this software in a product, an acknowledgment in the
 *        product documentation would be appreciated but is not required.
 *
 *     2. Altered source versions must be plainly marked as such, and
 *        must not be misrepresented as being the original software.
 *
 *     3. This notice may not be removed or altered from any source
 *        distribution.
 */
struct LatticeKernel
{
    uint nonstationary;
    float2 lengthScale;
    float2 amplitude;
    float2 heightRange;
    float latticeSpacingInEll;
    float kernelRadiusInEll;
    int stationaryRadius; // Radius rounded up to a whole lattice interval.
    float2 stationaryInvVariance; // Inverse squared norms of the value and slope rows.
};

StructuredBuffer<float2> gStationaryWeights; // Value and cell-coordinate slope at nonnegative offsets.

uint latticeHash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

// Adjacent source indices share a Box-Muller pair. Cell revisits regenerate
// exactly the same coefficients. Both uniforms are strictly inside (0,1).
float2 latticeGaussianPair(int pairIndex, uint seed)
{
    uint key = (uint(pairIndex) * 2u) ^ seed;
    float2 u = (float2(latticeHash(key) >> 9, latticeHash(key ^ 1u) >> 9) + 0.5f) / 8388608.f;
    float s, c;
    sincos(6.28318530718f * u.y, s, c);
    return sqrt(-2.f * log(u.x)) * float2(c, s);
}

float4 latticeHermite(float2 left, float2 right)
{
    float difference = right.x - left.x;
    return float4(left, 3.f * difference - 2.f * left.y - right.y, left.y + right.y - 2.f * difference);
}

float2 latticePolynomialJet(float4 p, float u)
{
    return float2(((p.w * u + p.z) * u + p.y) * u + p.x, (3.f * p.w * u + 2.f * p.z) * u + p.y);
}

bool hasDefaultStencil(LatticeKernel kernel)
{
    return kernel.latticeSpacingInEll == 0.5f && kernel.stationaryRadius == 6;
}

float2 stationaryWeight<let DefaultStencil : bool>(int offset, int radius)
{
    if (DefaultStencil)
    {
        static const float weights[7] = {
            0.631618778f, 0.491905199f, 0.232359563f, 0.066572129f, 0.011568501f, 0.001219311f, 0.f
        };
        float value = weights[min(abs(offset), 6)];
        return float2(value, -0.5f * offset * value);
    }
    float2 w = gStationaryWeights[min(abs(offset), radius)];
    return float2(w.x, offset < 0 ? -w.y : w.y);
}

float2 stationaryPairJet<let DefaultStencil : bool>(int knot, int pair, uint seed, float2 birthCorrection, int radius)
{
    int offset = knot - 2 * pair;
    float2 z = latticeGaussianPair(pair, seed);
    if (pair >= ((1 - radius) >> 1) && pair <= ((radius - 1) >> 1))
        z += float2(dot(stationaryWeight<DefaultStencil>(-2 * pair, radius), birthCorrection),
                    dot(stationaryWeight<DefaultStencil>(-2 * pair - 1, radius), birthCorrection));
    return z.x * stationaryWeight<DefaultStencil>(offset, radius) + z.y * stationaryWeight<DefaultStencil>(offset - 1, radius);
}

// Value and slope in units of one knot interval.
float2 stationaryKnot(int knot, uint seed, float sigma, float2 birthCorrection, LatticeKernel kernel)
{
    if (sigma == 0.f) return 0.f;
    float2 jet = 0.f;
    if (hasDefaultStencil(kernel))
    {
        int firstPair = (knot - 5) >> 1;
        [unroll] for (int k = 0; k < 6; ++k)
            jet += stationaryPairJet<true>(knot, firstPair + k, seed, birthCorrection, 6);
    }
    else
    {
        int radius = kernel.stationaryRadius;
        int firstPair = (knot + 1 - radius) >> 1;
        [loop] for (int k = 0; k < radius; ++k)
            jet += stationaryPairJet<false>(knot, firstPair + k, seed, birthCorrection, radius);
    }
    return sigma * jet;
}

float stationaryWeightVariance<let DefaultStencil : bool>(int index, int radius, float4 hermite)
{
    float w = dot(hermite, float4(stationaryWeight<DefaultStencil>(-index, radius), stationaryWeight<DefaultStencil>(1 - index, radius)));
    return w * w;
}

// Discrete transverse variance follows the same Hermite basis used along the ray.
float stationaryTransverseSD(float u, float sigmaOverEll, LatticeKernel kernel)
{
    float v = 1.f - u;
    float4 h = float4((1.f + 2.f * u) * v * v, u * v * v, (1.f + 2.f * v) * u * u, -v * u * u);
    float variance = 0.f;
    if (hasDefaultStencil(kernel))
    {
        [unroll] for (int j = -5; j <= 6; ++j)
            variance += stationaryWeightVariance<true>(j, 6, h);
    }
    else
    {
        int radius = kernel.stationaryRadius;
        [loop] for (int j = 1 - radius; j <= radius; ++j)
            variance += stationaryWeightVariance<false>(j, radius, h);
    }
    return sigmaOverEll * sqrt(variance);
}

struct LatticeProfile
{
    float sigma;
    float ell;
    float2 gradientY; // d(sigma)/dy, d(ell)/dy
};

LatticeProfile latticeProfile(LatticeKernel kernel, float height, bool inRamp)
{
    float range = kernel.heightRange.y - kernel.heightRange.x;
    float u = saturate((height - kernel.heightRange.x) / range);
    LatticeProfile profile;
    profile.sigma = lerp(kernel.amplitude.x, kernel.amplitude.y, u);
    profile.ell = lerp(kernel.lengthScale.x, kernel.lengthScale.y, u);
    profile.gradientY = inRamp ? float2(kernel.amplitude.y - kernel.amplitude.x,
                                        kernel.lengthScale.y - kernel.lengthScale.x) / range : float2(0.f);
    return profile;
}

// Return the value, the derivative w.r.t. translation along the ray, and the
// partial derivatives w.r.t. local amplitude and width for one coefficient.
float4 varyingWeight(float distance, float spacing, float radiusInEll, LatticeProfile profile)
{
    float x = distance / profile.ell;
    float invTaperWidth = 1.f / min(0.5f, 0.5f * radiusInEll);
    float v = saturate(invTaperWidth * (radiusInEll - abs(x)));
    float window = v * v * (3.f - 2.f * v);
    float gaussian = 0.893243842f * sqrt(spacing / profile.ell) * exp(-x * x);
    float dSigma = gaussian * window;
    float dX = gaussian * (-6.f * invTaperWidth * sign(x) * v * (1.f - v) - 2.f * x * window);
    float scale = profile.sigma / profile.ell;
    return float4(profile.sigma * dSigma, scale * dX, dSigma, scale * (-0.5f * dSigma - x * dX));
}

struct VaryingLatticeSample
{
    float value;
    float slope;   // per world distance along this flight
    float dY;      // gradient contribution from changing the parameters
    float variance;
};

// Condition the finite source vector on its value and its physical axial
// derivative at the start of a reflected flight. Coefficients outside the
// compact birth support keep their prior samples.
struct VaryingBirth
{
    LatticeProfile profile;
    float2 gains;
    bool active;
    bool centerSlope;
};

float2 observationRow(float4 weights, float x, float2 parameterSlope, bool centerSlope)
{
    // Subtract the part of the derivative proportional to the value analytically.
    // This form has exactly zero slope at x=0 even when the two rows are nearly parallel.
    if (centerSlope)
        return float2(weights.x, weights.y * (1.f - x * parameterSlope.y));
    return float2(weights.x, weights.y + dot(weights.zw, parameterSlope));
}

void varyingBirthMoments(LatticeProfile profile, float spacing, float radiusInEll, float2 slope, uint seed, bool centerSlope,
                         out float2 prior, out float3 gram)
{
    int reach = int(ceil(radiusInEll * profile.ell / spacing));
    prior = 0.f;
    gram = 0.f;
    [loop]
    for (int pair = (-reach) >> 1; pair <= (reach >> 1); ++pair)
    {
        float2 z = latticeGaussianPair(pair, seed);
        [unroll]
        for (int lane = 0; lane < 2; ++lane)
        {
            float distance = -float(2 * pair + lane) * spacing;
            float4 weights = varyingWeight(distance, spacing, radiusInEll, profile);
            float2 row = observationRow(weights, distance / profile.ell, slope, centerSlope);
            prior += row * z[lane];
            gram += float3(row.x * row.x, row.x * row.y, row.y * row.y);
        }
    }
}

VaryingBirth varyBirth(LatticeProfile profile, float spacing, float radiusInEll, float directionY, uint seed, float2 target)
{
    VaryingBirth birth = {};
    birth.profile = profile;
    float2 slope = profile.gradientY * directionY;
    float2 prior;
    float3 gram;
    varyingBirthMoments(profile, spacing, radiusInEll, slope, seed, false, prior, gram);
    float2 residual = target - prior;
    float det = gram.x * gram.z - gram.y * gram.y;
    if (det > 1.e-4f * gram.x * gram.z)
        birth.gains = float2(gram.z * residual.x - gram.y * residual.y, gram.x * residual.y - gram.y * residual.x) / det;
    else if (gram.x > 0.f)
    {
        // Narrow support can leave almost all value and slope weight on the central tap.
        // Condition on the value and the centered derivative to avoid cancellation in det.
        birth.centerSlope = true;
        target.y -= (slope.x / profile.sigma - 0.5f * slope.y / profile.ell) * target.x;
        varyingBirthMoments(profile, spacing, radiusInEll, slope, seed, true, prior, gram);
        residual = target - prior;
        float projection = gram.y / gram.x;
        float slopeVariance = gram.z - projection * gram.y;
        birth.gains.y = slopeVariance > 0.f ? (residual.y - projection * residual.x) / slopeVariance : 0.f;
        birth.gains.x = residual.x / gram.x - projection * birth.gains.y;
    }
    else if (gram.z > 0.f)
        birth.gains.y = residual.y / gram.z;
    birth.active = any(birth.gains != 0.f);
    return birth;
}

VaryingLatticeSample varyingSample(float t, float spacing, float radiusInEll, float directionY, LatticeProfile profile, uint seed, VaryingBirth birth)
{
    float grid = t / spacing;
    float radius = radiusInEll * profile.ell / spacing;
    int firstPair = int(floor(grid - radius)) >> 1;
    int lastPair = int(ceil(grid + radius)) >> 1;
    float4 sum = 0.f;
    float variance = 0.f;
    [loop]
    for (int pair = firstPair; pair <= lastPair; ++pair)
    {
        float2 z = latticeGaussianPair(pair, seed);
        [unroll]
        for (int lane = 0; lane < 2; ++lane)
        {
            int index = 2 * pair + lane;
            float d = (grid - float(index)) * spacing;
            float4 w = varyingWeight(d, spacing, radiusInEll, profile);
            float coefficient = z[lane];
            if (birth.active && abs(float(index) * spacing) < radiusInEll * birth.profile.ell)
            {
                float distance = -float(index) * spacing;
                float4 b = varyingWeight(distance, spacing, radiusInEll, birth.profile);
                coefficient += dot(observationRow(b, distance / birth.profile.ell, directionY * birth.profile.gradientY, birth.centerSlope),
                                   birth.gains);
            }
            sum += coefficient * w;
            variance += w.x * w.x;
        }
    }
    VaryingLatticeSample sample;
    sample.value = sum.x;
    sample.dY = dot(sum.zw, profile.gradientY);
    sample.slope = sum.y + directionY * sample.dY;
    sample.variance = variance;
    return sample;
}

float3 latticeSurfaceGradient(float3 axis, float3 meanGradient, float axialSlope, float responseY, float transverseSD, uint seed)
{
    float3 tangent = normalize(cross(axis, abs(axis.z) < 0.9f ? float3(0, 0, 1) : float3(0, 1, 0)));
    float3 bitangent = cross(axis, tangent);
    float2 z = latticeGaussianPair(0, latticeHash(seed ^ 0x7c8a3579u));
    float3 gradient = meanGradient + float3(0.f, responseY, 0.f);
    gradient += (axialSlope - dot(gradient, axis)) * axis + transverseSD * (tangent * z.x + bitangent * z.y);
    return dot(gradient, gradient) > 0.f ? gradient : -axis;
}
