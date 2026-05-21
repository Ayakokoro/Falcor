add_plugin(Voxelization)

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

    ReadVoxelPass.cpp
    ReadVoxelPass.h
)
target_copy_shaders(Voxelization RenderPasses/Voxelization)
target_source_group(Voxelization "RenderPasses")
