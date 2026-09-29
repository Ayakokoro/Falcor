cmake_minimum_required(VERSION 3.21)
project(LargeSceneVoxelization
    DESCRIPTION "Pure-CPU tiled voxelization for large scenes"
    LANGUAGES CXX
)

# C++20 for std::popcount, std::bit, designated initializers
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Dependencies
#
# Every dependency is taken from an already installed package (vcpkg, system)
# when one is available, and only falls back to FetchContent otherwise. To use
# vcpkg, point CMake at its toolchain file:
#   -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
# Packages: assimp (pulls in stb, zlib, draco, ...), glm, fmt
# ---------------------------------------------------------------------------
include(FetchContent)

# --- GLM (header-only math library) ---
find_path(GLM_INCLUDE_DIR glm/glm.hpp)
if(NOT GLM_INCLUDE_DIR)
    FetchContent_Declare(glm
        GIT_REPOSITORY  https://github.com/g-truc/glm.git
        GIT_TAG         1.0.1
        GIT_SHALLOW     TRUE
    )
    FetchContent_MakeAvailable(glm)
    set(GLM_INCLUDE_DIR ${glm_SOURCE_DIR})
endif()

# --- stb_image (single-header image loader) ---
find_package(Stb QUIET)                     # vcpkg ships a FindStb.cmake
if(NOT Stb_FOUND)
    FetchContent_Declare(stb
        GIT_REPOSITORY  https://github.com/nothings/stb.git
        GIT_TAG         master
        GIT_SHALLOW     TRUE
    )
    FetchContent_MakeAvailable(stb)
    set(Stb_INCLUDE_DIR ${stb_SOURCE_DIR})
endif()

# --- fmt (optional: nicer logging; falls back to iostream if unavailable) ---
find_package(fmt CONFIG QUIET)
if(NOT fmt_FOUND)
    FetchContent_Declare(fmt
        GIT_REPOSITORY  https://github.com/fmtlib/fmt.git
        GIT_TAG         10.2.1
        GIT_SHALLOW     TRUE
    )
    set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(fmt)
endif()

# --- Assimp (FBX / model loading) ---
# Prefer system/vcpkg installation; fall back to FetchContent
find_package(assimp CONFIG QUIET)
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

# A packaged assimp exports assimp::assimp; the in-tree build provides both the
# plain and the namespaced target, so the namespaced one is always available.
if(TARGET assimp::assimp)
    set(LSV_ASSIMP_TARGET assimp::assimp)
else()
    set(LSV_ASSIMP_TARGET assimp)
endif()
get_target_property(LSV_ASSIMP_TYPE ${LSV_ASSIMP_TARGET} TYPE)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(LargeSceneVoxelization
    main.cpp
)

target_include_directories(LargeSceneVoxelization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GLM_INCLUDE_DIR}          # GLM header-only
    ${Stb_INCLUDE_DIR}          # stb_image.h
)

target_link_libraries(LargeSceneVoxelization PRIVATE
    fmt::fmt
    ${LSV_ASSIMP_TARGET}
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
# Post-build: copy dependency DLLs next to the executable
# ---------------------------------------------------------------------------
if(WIN32 AND LSV_ASSIMP_TYPE STREQUAL "SHARED_LIBRARY")
    if(assimp_FOUND)
        # A packaged assimp is a DLL that itself depends on further DLLs
        # (draco, pugixml, zlib, ...), so copy the whole runtime closure.
        add_custom_command(TARGET LargeSceneVoxelization POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:LargeSceneVoxelization>
                "$<TARGET_FILE_DIR:LargeSceneVoxelization>"
            COMMAND_EXPAND_LISTS
            COMMENT "Copying runtime DLLs to output directory"
        )
    else()
        # Source-built assimp only needs its own DLL next to the executable.
        add_custom_command(TARGET LargeSceneVoxelization POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:${LSV_ASSIMP_TARGET}>"
                "$<TARGET_FILE_DIR:LargeSceneVoxelization>"
            COMMENT "Copying assimp DLL to output directory"
        )
    endif()
endif()
