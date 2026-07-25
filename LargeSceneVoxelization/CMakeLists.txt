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
# Executable
# ---------------------------------------------------------------------------
add_executable(LargeSceneVoxelization
    main.cpp
)

target_include_directories(LargeSceneVoxelization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${stb_SOURCE_DIR}           # stb_image.h
    ${glm_SOURCE_DIR}           # GLM header-only
)

target_link_libraries(LargeSceneVoxelization PRIVATE
    fmt::fmt
    assimp
)

# ---------------------------------------------------------------------------
# Compiler options
# ---------------------------------------------------------------------------
if(MSVC)
    target_compile_options(LargeSceneVoxelization PRIVATE
        /W3 /MP /arch:AVX2 /fp:fast
        $<$<CONFIG:Release>:/O2 /GL>
        $<$<CONFIG:Debug>:/Od /Zi>
    )
    target_link_options(LargeSceneVoxelization PRIVATE
        $<$<CONFIG:Release>:/LTCG>
    )
else()
    target_compile_options(LargeSceneVoxelization PRIVATE
        -Wall -Wextra -march=native -ffast-math
        $<$<CONFIG:Release>:-O3 -flto>
        $<$<CONFIG:Debug>:-O0 -g>
    )
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS LargeSceneVoxelization RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# Post-build: copy assimp DLL to output directory
# ---------------------------------------------------------------------------
if(assimp_BINARY_DIR AND WIN32)
    add_custom_command(TARGET LargeSceneVoxelization POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${assimp_BINARY_DIR}/bin/$<CONFIG>/assimp-vc143-mt.dll"
            "$<TARGET_FILE_DIR:LargeSceneVoxelization>"
        COMMENT "Copying assimp DLL to output directory"
    )
endif()
