#include "SceneVoxelization.h"
#include "MaterialValidation.h"
#include <iostream>
#include <string>
#include <chrono>
#include <thread>

int main(int argc, char* argv[]) {
    std::string inputPath;
    std::string outputPath;
    uint32_t resolution = 512;
    uint32_t sampleFrequency = 1024;
    uint32_t numThreads = 0;           // 0 = auto-detect
    std::string tmpDir = "./tmp";
    bool keepTemp = true;              // default: preserve temp files
    bool useInMemory = false;
    bool visibilityBitmask = false;
    MaterialValidation::Options validation;
    bool hasCell = false, hasView = false, validationOptionsUsed = false;
    bool showHelp = false;

    try {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
                const std::string value = argv[++i];
                if (value.empty() || value.rfind("--", 0) == 0)
                    throw std::runtime_error("Missing value for " + arg);
                return value;
            };
            auto integer = [&]() -> int64_t {
                const std::string value = next();
                size_t used = 0;
                const int64_t result = std::stoll(value, &used);
                if (used != value.size()) throw std::runtime_error("Invalid integer for " + arg);
                return result;
            };
            auto unsignedInteger = [&]() -> uint32_t {
                const int64_t value = integer();
                if (value < 0 || value > UINT32_MAX) throw std::runtime_error("Out-of-range integer for " + arg);
                return static_cast<uint32_t>(value);
            };
            if (arg == "-o") {
                outputPath = next();
            } else if (arg == "-r") {
                resolution = unsignedInteger();
            } else if (arg == "-s") {
                sampleFrequency = unsignedInteger();
            } else if (arg == "-t" || arg == "--threads") {
                numThreads = unsignedInteger();
            } else if (arg == "--tmp-dir") {
                tmpDir = next();
            } else if (arg == "--validate-material") {
                validation.voxelPath = next();
            } else if (arg == "--lod") {
                validation.lod = unsignedInteger();
                validationOptionsUsed = true;
            } else if (arg == "--cell") {
                for (int c = 0; c < 3; ++c) {
                    const int64_t value = integer();
                    if (value < 0 || value > INT32_MAX) throw std::runtime_error("Invalid coordinate for --cell");
                    validation.cell[c] = static_cast<int>(value);
                }
                hasCell = true;
                validationOptionsUsed = true;
            } else if (arg == "--view") {
                for (int c = 0; c < 3; ++c) {
                    const std::string value = next();
                    size_t used = 0;
                    validation.view[c] = std::stof(value, &used);
                    if (used != value.size()) throw std::runtime_error("Invalid number for --view");
                }
                hasView = true;
                validationOptionsUsed = true;
            } else if (arg == "--report") {
                validation.reportPath = next();
                validationOptionsUsed = true;
            } else if (arg == "--keep-temp") {
                keepTemp = true;
            } else if (arg == "--clean") {
                keepTemp = false;
            } else if (arg == "--in-memory") {
                useInMemory = true;
            } else if (arg == "--visibility-bitmask") {
                visibilityBitmask = true;
            } else if (arg == "--help" || arg == "-h") {
                showHelp = true;
            } else if (!arg.empty() && arg[0] != '-') {
                if (!inputPath.empty()) throw std::runtime_error("Only one source model can be specified.");
                inputPath = arg;
            } else {
                throw std::runtime_error("Unknown option: " + arg);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Argument error: " << error.what() << '\n';
        return 1;
    }

    if (inputPath.empty() || showHelp) {
        std::cout << "Usage: LargeSceneVoxelization <input.fbx> [options]\n\n"
                  << "Options:\n"
                  << "  -o <path>         Output file (default: <input>_voxelized.bin)\n"
                  << "  -r <N>            Grid resolution (default: 512)\n"
                  << "  -s <N>            Sample frequency per Lebedev direction (default: 1024)\n"
                  << "  -t, --threads <N> Number of clip threads (default: auto-detect)\n"
                  << "  --tmp-dir <path>  Temp directory for intermediate files (default: ./tmp/)\n"
                  << "  --keep-temp       Keep temp files after completion (default)\n"
                  << "  --clean           Delete temp files after completion\n"
                  << "  --in-memory       Use original in-memory pipeline (default: disk-backed)\n"
                  << "  --visibility-bitmask  Bake <output>.vbm for the renderer's Bitmask visibility mode\n\n"
                  << "Read-only directional material validation (no voxelization):\n"
                  << "  --validate-material <bin>  Read an existing VPW3 voxel bake\n"
                  << "  --tmp-dir <path>          Matching cache directory containing merge/\n"
                  << "  --lod <N>                 Finest leaf = 0 (default: 0)\n"
                  << "  --cell <x> <y> <z>       Required; coordinates in the selected LOD grid\n"
                  << "  --view <x> <y> <z>       Required; world-axis node-to-observer direction\n"
                  << "  --report <new.csv>        Optional CSV report; refuses existing files\n";
        return showHelp || argc == 1 ? 0 : 1;
    }

    if (!validation.voxelPath.empty()) {
        if (!hasCell || !hasView || useInMemory || visibilityBitmask || !outputPath.empty()) {
            std::cerr << "Validation requires --cell and --view; use --report, not -o, and omit --in-memory/--visibility-bitmask.\n";
            return 1;
        }
        validation.tmpDir = tmpDir;
        return MaterialValidation::execute(inputPath, validation) ? 0 : 1;
    }
    if (validationOptionsUsed) {
        std::cerr << "--lod/--cell/--view/--report require --validate-material.\n";
        return 1;
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
    config.visibilityBitmask = visibilityBitmask;

    std::cout << "=== LargeSceneVoxelization ===" << std::endl;
    std::cout << "  Input:       " << inputPath << std::endl;
    std::cout << "  Output:      " << outputPath << std::endl;
    std::cout << "  Resolution:  " << resolution << std::endl;
    std::cout << "  Samples:     " << sampleFrequency << std::endl;
    std::cout << "  Threads:     " << numThreads << std::endl;
    std::cout << "  Mode:        " << (useInMemory ? "in-memory" : "disk-backed") << std::endl;
    if (!useInMemory) {
        std::cout << "  Temp dir:    " << tmpDir << std::endl;
        std::cout << "  Keep temp:   " << (keepTemp ? "yes" : "no") << std::endl;
    }

    auto start = std::chrono::high_resolution_clock::now();
    if (visibilityBitmask)
        std::cout << "  Visibility:  4x4 / 32-sample companion for the renderer's Bitmask mode" << std::endl;

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
