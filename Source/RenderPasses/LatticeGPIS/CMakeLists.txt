add_library(GPISDensityStorage STATIC
    DensityClipmap.h
    DensityClipmap.cpp
    DensityClipmap.slang
    MeshClipmap.h
    MeshClipmap.cpp
    UploadMeshClipmap.cs.slang
    MeshDistanceField.h
    MeshDistanceField.cpp
    SdfField.slang
    GPISCameraRay.slang
    DensityLighting.slang
)
set_target_properties(GPISDensityStorage PROPERTIES POSITION_INDEPENDENT_CODE ON)
# OpenEXR's interface provides the codec library search paths at final DLL link time.
target_link_libraries(GPISDensityStorage PRIVATE Falcor assimp PUBLIC OpenVDB OpenEXR)
target_copy_shaders(GPISDensityStorage RenderPasses/LatticeGPIS)
target_source_group(GPISDensityStorage "RenderPasses")

add_plugin(LatticeGPISPass)
target_sources(LatticeGPISPass PRIVATE
    LatticeGPISPass.h
    LatticeGPISPass.cpp
    LatticeGPIS.cs.slang
    BakedGPISPass.cpp
    BakedGPISPass.h
    BakedGPIS.cs.slang
    BakedLatticeNoise.slang
    LatticeNoise.slang
    Intersection.slang
    CloudGPIS.slang
    DensityGPIS.slang
    MeshSurfaceUV.slang
    SdfField.h
    SdfField.cpp
    BakeShaderball.cs.slang
)

target_link_libraries(LatticeGPISPass PRIVATE GPISDensityStorage)
target_copy_shaders(LatticeGPISPass RenderPasses/LatticeGPIS)
target_source_group(LatticeGPISPass "RenderPasses")
