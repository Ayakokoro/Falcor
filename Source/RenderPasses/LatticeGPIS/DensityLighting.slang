#ifndef DENSITY_LIGHTING_SLANG
#define DENSITY_LIGHTING_SLANG

// Shared radiometric setup for the VDB and baked GPIS comparison. The smoke
// scatters once under collimated illumination with an isotropic phase function.
static const float3 kDensityBackground = float3(.22f, .30f, .42f);
static const float3 kDensityLightIrradiance = float3(6.f, 5.5f, 5.f);
static const float3 kDensityScatteringAlbedo = float3(.84f, .88f, .94f);

float3 densityDirectionToLight()
{
    return normalize(float3(-.60f, .80f, .35f));
}

float3 densitySingleScatter(float lightTransmittance)
{
    static const float invFourPi = 0.07957747154595f;
    return kDensityScatteringAlbedo * kDensityLightIrradiance * (invFourPi * lightTransmittance);
}

#endif
