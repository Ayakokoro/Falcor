// The baked cloud always uses the unit-amplitude stationary stencil with radius 6.
// Its six Box-Muller pairs move by one pair every other knot. Keep the pairs in
// registers so that following a ray samples each new impulse only once.
struct BakedGaussianWindow
{
    int firstPair;
    float4 lo, mid, hi;
};

float2 bakedGaussianPair(int pairIndex, uint seed, float2 correction)
{
    float2 z = latticeGaussianPair(pairIndex, seed);
    if (pairIndex >= -3 && pairIndex <= 2)
        z += float2(dot(stationaryWeight<true>(-2 * pairIndex, 6), correction),
                    dot(stationaryWeight<true>(-2 * pairIndex - 1, 6), correction));
    return z;
}

float2 bakedPairJet(float2 z, int offset)
{
    return z.x * stationaryWeight<true>(offset, 6) + z.y * stationaryWeight<true>(offset - 1, 6);
}

float2 bakedWindowJet<let FirstOffset : int>(BakedGaussianWindow pairs)
{
    // Match stationaryKnot's summation order; the two parities have fixed weights.
    float2 jet = bakedPairJet(pairs.lo.xy, FirstOffset);
    jet += bakedPairJet(pairs.lo.zw, FirstOffset - 2);
    jet += bakedPairJet(pairs.mid.xy, FirstOffset - 4);
    jet += bakedPairJet(pairs.mid.zw, FirstOffset - 6);
    jet += bakedPairJet(pairs.hi.xy, FirstOffset - 8);
    jet += bakedPairJet(pairs.hi.zw, FirstOffset - 10);
    return jet;
}

float2 bakedKnot(int knot, uint seed, float2 correction, inout BakedGaussianWindow pairs)
{
    int first = (knot - 5) >> 1;
    if (first != pairs.firstPair)
    {
        if (first == pairs.firstPair + 1)
        {
            pairs.lo = float4(pairs.lo.zw, pairs.mid.xy);
            pairs.mid = float4(pairs.mid.zw, pairs.hi.xy);
            pairs.hi = float4(pairs.hi.zw, bakedGaussianPair(first + 5, seed, correction));
        }
        else
        {
            // First active cell, or re-entry after skipping vacuum.
            pairs.lo = float4(bakedGaussianPair(first, seed, correction), bakedGaussianPair(first + 1, seed, correction));
            pairs.mid = float4(bakedGaussianPair(first + 2, seed, correction), bakedGaussianPair(first + 3, seed, correction));
            pairs.hi = float4(bakedGaussianPair(first + 4, seed, correction), bakedGaussianPair(first + 5, seed, correction));
        }
        pairs.firstPair = first;
    }
    return (knot & 1) != 0 ? bakedWindowJet<5>(pairs) : bakedWindowJet<6>(pairs);
}

NoiseSegment bakedNoiseSegment(LatticeFlight flight, float start, int knot, NoiseSegment previous,
                              inout BakedGaussianWindow pairs)
{
    NoiseSegment seg = {};
    seg.start = start;
    seg.end = float(knot + 1) * flight.spacing;
    float2 units = float2(1.f, 1.f / flight.spacing);
    float2 left = previous.rightJet;
    if (start == 0.f || start != previous.end)
        left = bakedKnot(knot, flight.seed, flight.correction, pairs) * units;
    seg.rightJet = bakedKnot(knot + 1, flight.seed, flight.correction, pairs) * units;
    float width = seg.end - start;
    seg.polynomial = latticeHermite(left * float2(1.f, width), seg.rightJet * float2(1.f, width));
    return seg;
}
