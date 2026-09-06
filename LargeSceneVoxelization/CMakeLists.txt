cmake_minimum_required(VERSION 3.20)
project(LargeSceneVoxelization
    DESCRIPTION "Pure-CPU tiled voxelization for large scenes"
    LANGUAGES CXX
)

# C++20 for std::popcount, std::bit, designated initializers
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Dependencies via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

# --- GLM (header-only math library) ---
FetchContent_Declare(glm
    GIT_REPOSITORY  https://github.com/g-truc/glm.git
    GIT_TAG         1.0.1
    GIT_SHALLOW     TRUE
)
FetchContent_MakeAvailable(glm)

# --- stb_image (single-header image loader) ---
FetchContent_Declare(stb
    GIT_REPOSITORY  https://github.com/nothings/stb.git
    GIT_TAG         master
    GIT_SHALLOW     TRUE
)
FetchContent_MakeAvailable(stb)

# --- fmt (optional: nicer logging; falls back to iostream if unavailable) ---
FetchContent_Declare(fmt
    GIT_REPOSITORY  https://github.com/fmtlib/fmt.git
    GIT_TAG         10.2.1
    GIT_SHALLOW     TRUE
)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(fmt)

# --- Assimp (FBX / model loading) ---
# Prefer system/vcpkg installation; fall back to FetchContent
find_package(assimp QUIET)
if(NOT assimp_FOUND)
    message(STATUS "assimp not found via find_package, building from source...")
    FetchContent_Declare(assimp
        GIT_REPOSITORY  https://github.com/assimp/assimp.git
        GIT_TAG         v5.4.3
        GIT_SHALLOW     TRUE
    )
    set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_FBX_IMPORTER ON CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_OBJ_IMPORTER ON CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_GLTF_IMPORTER ON CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_BLEND_IMPORTER ON CACHE BOOL "" FORCE)
    set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(assimp)
endif()

# ---------------------------------------------------------------------------
# Shared CPU voxelization/inspection core
# ---------------------------------------------------------------------------
# This is intentionally a static library.  Both executables link the same
# single-node reader and analysis implementation without introducing a DLL
# ABI/export boundary on Windows.
add_library(VoxelizationCore STATIC
    VoxelizationCore.cpp
    VoxelizationCore.h
)

target_include_directories(VoxelizationCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${stb_SOURCE_DIR}           # stb_image.h
    ${glm_SOURCE_DIR}           # GLM header-only
)

target_link_libraries(VoxelizationCore PUBLIC
    fmt::fmt
    assimp
)

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(LargeSceneVoxelization
    main.cpp
)

add_executable(VoxelizationInspector
    VoxelizationInspector.cpp
)

target_link_libraries(LargeSceneVoxelization PRIVATE VoxelizationCore)
target_link_libraries(VoxelizationInspector PRIVATE VoxelizationCore)

target_include_directories(LargeSceneVoxelization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${stb_SOURCE_DIR}
    ${glm_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/include  # nlohmann/json.hpp for scene manifests
)

target_include_directories(VoxelizationInspector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ---------------------------------------------------------------------------
# Compiler options
# ---------------------------------------------------------------------------
if(MSVC)
    foreach(target VoxelizationCore LargeSceneVoxelization VoxelizationInspector)
        target_compile_options(${target} PRIVATE
        /W3 /MP /arch:AVX2 /fp:fast
        $<$<CONFIG:Release>:/O2 /GL>
        $<$<CONFIG:Debug>:/Od /Zi>
        )
    endforeach()
    foreach(target LargeSceneVoxelization VoxelizationInspector)
        target_link_options(${target} PRIVATE
        $<$<CONFIG:Release>:/LTCG>
        )
    endforeach()
else()
    foreach(target VoxelizationCore LargeSceneVoxelization VoxelizationInspector)
        target_compile_options(${target} PRIVATE
        -Wall -Wextra -march=native -ffast-math
        $<$<CONFIG:Release>:-O3 -flto>
        $<$<CONFIG:Debug>:-O0 -g>
        )
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS LargeSceneVoxelization VoxelizationInspector RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# Post-build: copy assimp DLL to output directory
# ---------------------------------------------------------------------------
if(assimp_BINARY_DIR AND WIN32)
    foreach(target LargeSceneVoxelization VoxelizationInspector)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:assimp>"
                "$<TARGET_FILE_DIR:${target}>"
            COMMENT "Copying assimp DLL to output directory"
        )
    endforeach()
endif()
