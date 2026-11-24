add_plugin(LatticeGPISPass)

target_sources(LatticeGPISPass PRIVATE
    LatticeGPISPass.h
    LatticeGPISPass.cpp
    LatticeGPIS.cs.slang
    LatticeNoise.slang
    Intersection.slang
    MeshClipmap.h
    MeshClipmap.cpp
    UploadMeshClipmap.cs.slang
    MeshDistanceField.h
    MeshDistanceField.cpp
    SdfField.h
    SdfField.cpp
    SdfField.slang
    BakeShaderball.cs.slang
)

# OpenVDB's float-grid codec uses Imath's half conversion table.
target_link_libraries(LatticeGPISPass PRIVATE OpenVDB OpenEXR)
target_copy_shaders(LatticeGPISPass RenderPasses/LatticeGPIS)
target_source_group(LatticeGPISPass "RenderPasses")
