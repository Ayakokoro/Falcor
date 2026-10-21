add_plugin(Voxelization)

option(VOXELIZATION_ENABLE_PIXEL_INSPECTION "Enable voxel picking, NDF inspection, and synchronous GPU readback" OFF)
target_compile_definitions(Voxelization PRIVATE
    VOXELIZATION_ENABLE_PIXEL_INSPECTION=$<BOOL:${VOXELIZATION_ENABLE_PIXEL_INSPECTION}>
)

target_sources(Voxelization PRIVATE
    HostDeviceShared.slang

    Voxel/ABSDF.slang
    Voxel/VoxelGrid.slang

    Math/AABB.slang
    Math/Projection.slang
    Math/Ellipsoid.slang
    Math/Polygon.slang
    Math/Triangle.slang
    Math/Sampling.slang
    Math/SphericalHarmonics.slang
    Math/LebedevQuadrature.slang
    Math/VoxelizationUtility.h
    Math/VoxelizationUtility.slang
    Math/Random.h
    Math/Random.slang

    VoxelizationBase.h
    VoxelizationBase.cpp

    VoxelizationShared.slang
    VisibilityMask.slang
    Shading.slang
    MixLightSampler.slang

    VoxelizationPass.cpp
    VoxelizationPass.h
    AnalyzePolygon.cs.slang
    PolygonGenerator.h
    LoadMesh.cs.slang

    RayMarchingPass.cpp
    RayMarchingPass.h
    PrepareShadingData.cs.slang
    RayMarching.ps.slang
    DisplayNDF.ps.slang

    ComputeSphericalFuncMap.cs.slang
    ComputeSplittingError.cs.slang
    DisplaySphericalFunc.ps.slang
    DisplaySplittingError.ps.slang
)
target_copy_shaders(Voxelization RenderPasses/Voxelization)
target_source_group(Voxelization "RenderPasses")
