add_plugin(VolumeReferencePass)
target_sources(VolumeReferencePass PRIVATE
    VolumeReferencePass.cpp
    VolumeReferencePass.h
    VolumeReference.cs.slang
)
target_link_libraries(VolumeReferencePass PRIVATE GPISDensityStorage)
target_copy_shaders(VolumeReferencePass RenderPasses/VolumeReferencePass)
target_source_group(VolumeReferencePass "RenderPasses")
