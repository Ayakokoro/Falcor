#include "SceneVoxelization.h"
#include <iostream>
#include <string>
#include <chrono>
#include <thread>

int main(int argc, char* argv[]) {
    std::string inputPath;
    std::string outputPath;
    uint32_t resolution = 512;
    uint32_t sampleFrequency = 1024;
    uint32_t lodLevels = 0;
    LODBuildMode lodMode = LODBuildMode::Approximate;
    uint32_t maxPolygonsPerNode = SAFE_PER_NODE_POLYGON_LIMIT;
    uint32_t numThreads = 0;           // 0 = auto-detect
    std::string tmpDir = "./tmp";
    bool keepTemp = true;              // default: preserve temp files
    bool useInMemory = false;
    bool useSpecGlossMaterials = false; // default matches Falcor FBX import

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "-r" && i + 1 < argc) {
            resolution = (uint32_t)std::stoul(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            sampleFrequency = (uint32_t)std::stoul(argv[++i]);
        } else if ((arg == "-l" || arg == "--lod-levels") && i + 1 < argc) {
            lodLevels = (uint32_t)std::stoul(argv[++i]);
        } else if ((arg == "--lod-mode" || arg == "--lod-method") && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "approximate" || mode == "approx") {
                lodMode = LODBuildMode::Approximate;
            } else if (mode == "brute-force" || mode == "bruteforce" || mode == "exact") {
                lodMode = LODBuildMode::BruteForce;
            } else {
                std::cerr << "Unknown LOD mode: " << mode
                          << " (expected approximate or brute-force)\n";
                return 1;
            }
        } else if (arg == "--max-polygons-per-node" && i + 1 < argc) {
            maxPolygonsPerNode = (uint32_t)std::stoul(argv[++i]);
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            numThreads = (uint32_t)std::stoul(argv[++i]);
        } else if (arg == "--tmp-dir" && i + 1 < argc) {
            tmpDir = argv[++i];
        } else if (arg == "--keep-temp") {
            keepTemp = true;
        } else if (arg == "--clean") {
            keepTemp = false;
        } else if (arg == "--in-memory") {
            useInMemory = true;
        } else if (arg == "--spec-gloss") {
            useSpecGlossMaterials = true;
        } else if (arg[0] != '-') {
            inputPath = arg;
        }
    }

    if (inputPath.empty()) {
        std::cout << "Usage: LargeSceneVoxelization <input.fbx> [options]\n\n"
                  << "Options:\n"
                  << "  -o <path>         Output file (default: <input>_voxelized.bin)\n"
                  << "  -r <N>            Grid resolution (default: 512)\n"
                  << "  -s <N>            Sample frequency per Lebedev direction (default: 1024)\n"
                  << "  -l, --lod-levels <N>  Additional coarse LOD levels (0 = leaves only)\n"
                  << "  --lod-mode <mode>     approximate or brute-force (default: approximate)\n"
                  << "  --max-polygons-per-node <N>  Per-node cap; 0 = unlimited\n"
                  << "  -t, --threads <N> Number of clip threads (default: auto-detect)\n"
                  << "  --tmp-dir <path>  Temp directory for intermediate files (default: ./tmp/)\n"
                  << "  --keep-temp       Keep temp files after completion (default)\n"
                  << "  --clean           Delete temp files after completion\n"
                  << "  --in-memory       Use original in-memory pipeline (default: disk-backed)\n"
                  << "  --spec-gloss      Explicitly use SpecGloss input conversion (default: MetalRough)\n";
        return 0;
    }

    if (outputPath.empty()) {
        auto dot = inputPath.find_last_of('.');
        outputPath = (dot != std::string::npos ? inputPath.substr(0, dot) : inputPath) + "_voxelized.bin";
    }

    if (numThreads == 0)
        numThreads = std::max(1u, std::thread::hardware_concurrency());

    VoxelizationConfig config;
    config.baseResolution = resolution;
    config.sampleFrequency = sampleFrequency;
    config.useSpecGlossMaterials = useSpecGlossMaterials;
    config.lodLevels = lodLevels;
    config.lodMode = lodMode;
    config.maxPolygonsPerNode = maxPolygonsPerNode;

    std::cout << "=== LargeSceneVoxelization ===" << std::endl;
    std::cout << "  Input:       " << inputPath << std::endl;
    std::cout << "  Output:      " << outputPath << std::endl;
    std::cout << "  Resolution:  " << resolution << std::endl;
    std::cout << "  Samples:     " << sampleFrequency << std::endl;
    std::cout << "  LOD levels:  " << lodLevels << std::endl;
    std::cout << "  LOD mode:    "
              << (lodMode == LODBuildMode::Approximate ? "approximate" : "brute-force")
              << std::endl;
    std::cout << "  Poly cap:    " << maxPolygonsPerNode
              << (maxPolygonsPerNode == 0 ? " (unlimited)" : "") << std::endl;
    std::cout << "  Threads:     " << numThreads << std::endl;
    std::cout << "  Mode:        " << (useInMemory ? "in-memory" : "disk-backed") << std::endl;
    std::cout << "  Materials:   " << (useSpecGlossMaterials ? "SpecGloss" : "MetalRough") << std::endl;
    if (!useInMemory) {
        std::cout << "  Temp dir:    " << tmpDir << std::endl;
        std::cout << "  Keep temp:   " << (keepTemp ? "yes" : "no") << std::endl;
    }

    auto start = std::chrono::high_resolution_clock::now();

    SceneVoxelization voxelizer(config);
    voxelizer.setTmpDir(tmpDir);
    voxelizer.setNumThreads(numThreads);
    voxelizer.setKeepTemp(keepTemp);

    bool ok;
    if (useInMemory) {
        ok = voxelizer.process(inputPath, outputPath);
    } else {
        ok = voxelizer.processDisk(inputPath, outputPath);
    }

    auto end = std::chrono::high_resolution_clock::now();

    if (ok) {
        auto seconds = std::chrono::duration<double>(end - start).count();
        std::cout << "\nVoxelization completed in " << seconds << "s." << std::endl;
    }

    return ok ? 0 : 1;
}
