#include "TiledVoxelization.h"
#include <iostream>
#include <string>
#include <chrono>

int main(int argc, char* argv[]) {
    std::string inputPath;
    std::string outputPath;
    uint resolution = 512;
    uint tileLevel = 3;
    uint sampleFrequency = 1024;
    uint threads = 0;

    // Simple CLI: voxelize <input.fbx> [-o output.bin] [-r resolution] [-t tile_level] [-s samples] [-j threads]
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "-r" && i + 1 < argc) {
            resolution = (uint)std::stoul(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            tileLevel = (uint)std::stoul(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            sampleFrequency = (uint)std::stoul(argv[++i]);
        } else if (arg == "-j" && i + 1 < argc) {
            threads = (uint)std::stoul(argv[++i]);
        } else if (arg[0] != '-') {
            inputPath = arg;
        }
    }

    if (inputPath.empty()) {
        std::cout << "Usage: LargeSceneVoxelization <input.fbx> [options]\n"
                  << "Options:\n"
                  << "  -o <path>    Output binary file (default: <input>_voxelized.bin)\n"
                  << "  -r <N>       Grid resolution (default: 512)\n"
                  << "  -t <L>       Tile level: 2^L * 2^L * 2^L tiles (default: 3)\n"
                  << "  -s <N>       Sample frequency per Lebedev direction (default: 1024)\n"
                  << "  -j <N>       Number of threads (default: auto)\n";
        return 0;
    }

    if (outputPath.empty()) {
        auto dot = inputPath.find_last_of('.');
        outputPath = (dot != std::string::npos ? inputPath.substr(0, dot) : inputPath) + "_voxelized.bin";
    }

    TiledVoxelizationConfig config;
    config.baseResolution = resolution;
    config.tileLevel = tileLevel;
    config.sampleFrequency = sampleFrequency;
    config.maxThreads = threads;

    TiledVoxelization voxelizer(config);

    auto start = std::chrono::high_resolution_clock::now();
    bool ok = voxelizer.process(inputPath, outputPath);
    auto end = std::chrono::high_resolution_clock::now();

    if (ok) {
        auto seconds = std::chrono::duration<double>(end - start).count();
        std::cout << "Voxelization completed in " << seconds << "s." << std::endl;
    }

    return ok ? 0 : 1;
}
