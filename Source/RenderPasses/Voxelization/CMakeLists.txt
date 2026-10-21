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
    Math/OctahedralMap.slang
    Math/OctahedralProjection.slang
    Math/VoxelizationUtility.h
    Math/VoxelizationUtility.slang
    Math/Random.h
    Math/Random.slang

    VoxelizationBase.h
    VoxelizationBase.cpp
    VoxelizationMetadata.h

    VoxelizationShared.slang
    Shading.slang
    VisibleShading.slang
    RayMarchingMaterial.slang
    VisibilityMask.slang
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

    ComputeOctahedralMap.cs.slang
    ComputeSplittingError.cs.slang
    DisplayOctahedralMap.ps.slang
    DisplaySplittingError.ps.slang
)
target_copy_shaders(Voxelization RenderPasses/Voxelization)
target_source_group(Voxelization "RenderPasses")
