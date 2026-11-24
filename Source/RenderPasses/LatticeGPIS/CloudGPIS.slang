// Include after Intersection.slang. The cloud is a second zero-crossing GPIS.
// Its compact ellipsoid bounds the process; entering this domain does not scatter light.
LatticeKernel cloudStationaryKernel(float ell, float sigma)
{
    LatticeKernel kernel = {};
    kernel.lengthScale = float2(ell);
    kernel.amplitude = float2(sigma);
    kernel.latticeSpacingInEll = 0.5f;
    kernel.kernelRadiusInEll = 3.f;
    kernel.stationaryRadius = 6;
    kernel.stationaryInvVariance = float2(1.000000007f, 4.000002577f);
    return kernel;
}

bool cloudEllipsoidRoots(float3 relativeOrigin, float3 direction, float3 radii, float tMax,
                         out float entry, out float exitDistance)
{
    entry = exitDistance = 0.f;
    float boxNear = 0.f, boxFar = tMax;
    [unroll] for (int k = 0; k < 3; ++k)
    {
        if (direction[k] == 0.f)
        {
            if (abs(relativeOrigin[k]) > radii[k]) return false;
        }
        else
        {
            float2 t = (float2(-radii[k], radii[k]) - relativeOrigin[k]) / direction[k];
            boxNear = max(boxNear, min(t.x, t.y));
            boxFar = min(boxFar, max(t.x, t.y));
        }
    }
    if (boxNear >= boxFar) return false;
    // Shift the quadratic's origin close to the ellipsoid before computing the chord.
    float3 p = (relativeOrigin + boxNear * direction) / radii;
    float3 d = direction / radii;
    float speed = length(d);
    float3 axis = d / speed;
    float along = -dot(p, axis);
    float3 closest = p + along * axis;
    float chord2 = 1.f - dot(closest, closest);
    if (chord2 <= 0.f) return false;
    float halfChord = sqrt(chord2);
    entry = boxNear + (along - halfChord) / speed;
    exitDistance = boxNear + (along + halfChord) / speed;
    return exitDistance > 0.f && entry < tMax;
}

float densityUniform(uint key)
{
    return (float(latticeHash(key) >> 9) + .5f) / 8388608.f;
}

// Draw the initial value from the void side even if the camera or a surface
// bounce starts INSIDE the VDB. Gaussian derivatives at the knot are independent
// of its value. The exponential-tail proposal stays efficient in dense regions.
float densityTruncatedNormal(float threshold, uint seed)
{
    [loop] for (;;)
    {
        seed += 0x9e3779b9u;
        float u = densityUniform(seed), v = densityUniform(seed ^ 0x632be5abu);
        if (threshold <= 0.f)
        {
            float z = sqrt(-2.f * log(u)) * cos(6.28318530718f * v);
            if (z > threshold) return z;
        }
        else
        {
            float alpha = threshold > 10.f ? threshold + 1.f / threshold : .5f * threshold + length(float2(.5f * threshold, 1.f));
            float z = threshold - log(u) / alpha;
            if (-2.f * log(v) >= (z - alpha) * (z - alpha)) return z;
        }
    }
}

bool cloudGPISIntersect(float3 origin, float3 direction, float tMax, float3 center, float3 radii,
                        float ell, float sigma, float meanOffset, uint seed, bool conditioned,
                        float3 birthGradient, out float hitDistance, out float3 hitGradient)
{
    hitDistance = 0.f;
    hitGradient = 0.f;
    float3 p = origin - center;
    float meanScale = 0.5f * min(radii.x, min(radii.y, radii.z));
    float entry, exitDistance;
    if (sigma == 0.f)
    {
        // A positive center mean has no zero. A negative offset can expose a smooth
        // ellipsoid wholly inside the process, useful for checking overlap/occlusion.
        float zeroRadius2 = -meanOffset / meanScale;
        if (zeroRadius2 <= 0.f || zeroRadius2 > 1.f) return false;
        if (!cloudEllipsoidRoots(p, direction, radii * sqrt(zeroRadius2), tMax, entry, exitDistance)) return false;
        if (conditioned && dot(birthGradient, direction) >= 0.f) return false;
        float hit = conditioned || entry < 0.f ? exitDistance : entry;
        if (hit < 0.f || hit >= tMax) return false;
        hitDistance = hit;
        hitGradient = (2.f * meanScale) * ((p + hit * direction) / radii) / radii;
        return true;
    }

    if (!cloudEllipsoidRoots(p, direction, radii, tMax, entry, exitDistance)) return false;
    entry = max(0.f, entry);
    float limit = min(tMax, exitDistance) - entry;
    if (limit <= 0.f) return false;
    conditioned = conditioned && entry == 0.f;
    float3 qOrigin = (p + entry * direction) / radii;
    float3 qDir = direction / radii;
    float meanAtEntry = meanOffset + meanScale * dot(qOrigin, qOrigin);
    float meanSlopeAtEntry = (2.f * meanScale) * dot(qOrigin, qDir);
    float birthSlope = dot(birthGradient, direction);

    // The cloud uses the reference stationary stencil regardless of the first GPIS's kernel or LUT.
    LatticeKernel kernel = cloudStationaryKernel(ell, sigma);
    LatticeFlight flight = makeLatticeFlight(origin + entry * direction, direction, kernel, seed, conditioned,
                                             float2(-meanAtEntry, birthSlope - meanSlopeAtEntry));
    NoiseSegment noise = {};
    bool positiveOnLeft = !conditioned;
    float orientation = 1.f;
    float start = 0.f;
    int cell = 0;
    [loop] while (start < limit)
    {
        noise = nextNoiseSegment(flight, start, cell, noise);
        float width = min(limit, noise.end) - start;
        if (width <= 0.f) break;
        float fraction = width / (noise.end - start);
        float3 q = qOrigin + start * qDir;
        float3 dq = width * qDir;
        // Hermite noise is cubic and the analytic mean is quadratic on this same segment.
        float4 poly = noise.polynomial * float4(1.f, fraction, fraction * fraction, fraction * fraction * fraction);
        poly.x += meanOffset + meanScale * dot(q, q);
        poly.y += (2.f * meanScale) * dot(q, dq);
        poly.z += meanScale * dot(dq, dq);
        if (cell == 0)
        {
            if (conditioned)
            {
                poly.x = 0.f;
                poly.y = width * birthSlope;
            }
            orientation = (conditioned ? birthSlope : poly.x) < 0.f ? -1.f : 1.f;
        }
        float endValue = meanOffset + meanScale * dot(q + dq, q + dq) +
                         (fraction == 1.f ? noise.rightJet.x : latticePolynomialJet(noise.polynomial, fraction).x);
        float u;
        if (latticeFirstDowncrossing(orientation * poly, orientation * endValue, positiveOnLeft, u))
        {
            float3 meanGradient = (2.f * meanScale) * (q + u * dq) / radii;
            float axial = latticePolynomialJet(poly, u).y / width;
            float transverse = stationaryTransverseSD(u * fraction, sigma / ell, kernel);
            hitGradient = latticeSurfaceGradient(direction, meanGradient, axial, 0.f, transverse, seed);
            hitDistance = entry + (start + u * width);
            return hitDistance < tMax;
        }
        start = noise.end;
        ++cell;
    }
    return false;
}
