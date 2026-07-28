#include "SceneVoxelization.h"
#include <iostream>
#include <string>
#include <chrono>

int main(int argc, char* argv[]) {
    std::string inputPath;
    std::string outputPath;
    uint resolution = 512;
    uint sampleFrequency = 1024;

    // CLI: voxelize <input.fbx> [-o output] [-r resolution] [-s samples]
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "-r" && i + 1 < argc) {
            resolution = (uint)std::stoul(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            sampleFrequency = (uint)std::stoul(argv[++i]);
        } else if (arg[0] != '-') {
            inputPath = arg;
        }
    }

    if (inputPath.empty()) {
        std::cout << "Usage: LargeSceneVoxelization <input.fbx> [options]\n"
                  << "Options:\n"
                  << "  -o <path>      Output file (default: <input>_voxelized.bin)\n"
                  << "  -r <N>         Grid resolution (default: 512)\n"
                  << "  -s <N>         Sample frequency per Lebedev direction (default: 1024)\n";
        return 0;
    }

    if (outputPath.empty()) {
        auto dot = inputPath.find_last_of('.');
        outputPath = (dot != std::string::npos ? inputPath.substr(0, dot) : inputPath) + "_voxelized.bin";
    }

    VoxelizationConfig config;
    config.baseResolution = resolution;
    config.sampleFrequency = sampleFrequency;

    auto start = std::chrono::high_resolution_clock::now();

    SceneVoxelization voxelizer(config);
    bool ok = voxelizer.process(inputPath, outputPath);

    auto end = std::chrono::high_resolution_clock::now();

    if (ok) {
        auto seconds = std::chrono::duration<double>(end - start).count();
        std::cout << "Voxelization completed in " << seconds << "s." << std::endl;
    }

    return ok ? 0 : 1;
}
