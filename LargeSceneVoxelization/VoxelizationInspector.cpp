#include "VoxelizationCore.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

struct Arguments
{
    fs::path scenePath;
    fs::path binaryPath;
    fs::path tmpDir = "./tmp";
    fs::path mergeDir;
    fs::path indexPath;
    fs::path polygonsPath;
    fs::path outputDir = "./voxel_debug";

    uint32_t lod = 0;
    int level = -1;
    int3 cell = int3(0);
    bool hasCell = false;
    std::string mode = "all";
    bool dumpPolygons = false;

    VoxelizationCore::AnalysisOptions analysis;
};

void printUsage()
{
    std::cout
        << "Usage: VoxelizationInspector --scene <scene.fbx|glb> --bin <result.bin> "
           "--cell <x> <y> <z> [options]\n\n"
        << "The tool reads one selected node from the CPU merge files and runs "
           "the CPU-side equivalents of the GPU diagnostics.\n\n"
        << "Node selection:\n"
        << "  --lod <N>                 LOD 0 = leaves, 1 = maxDepth-1 (default 0)\n"
        << "  --level <N>               Explicit tree level; overrides --lod\n"
        << "  --cell <x> <y> <z>        Cell coordinate at the selected level\n\n"
        << "Input files:\n"
        << "  --tmp-dir <path>          Temp root (default ./tmp)\n"
        << "  --merge-dir <path>       Direct merge_lod directory\n"
        << "  --index <path>            Explicit nodes.idx/leaves.idx\n"
        << "  --polygons <path>         Explicit polygons.dat\n\n"
        << "Analysis:\n"
        << "  --mode projection|spherical|splitting|ndf|all\n"
        << "  --sample-frequency <N>    Primitive projection samples (default 1024)\n"
        << "  --samples-per-polygon <N> MC samples for splitting/NDF (default 4)\n"
        << "  --spherical-resolution <N> Upper-hemisphere disk map size (N x N)\n"
        << "  --block-count <N>         Incident octahedral grid per side\n"
        << "  --block-size <N>          Outgoing octahedral grid per side\n"
        << "  --ndf-resolution <N>      NDF disk map resolution\n"
        << "  --threads <N>             Map worker threads; 0 = auto\n"
        << "  --dump-polygons           Write every clipped polygon to polygons.txt\n"
        << "  --out <path>              Output directory (default ./voxel_debug)\n"
        << "  --help                    Show this help\n";
}

uint32_t parseUint(const std::string& text, const char* option)
{
    try
    {
        size_t consumed = 0;
        const unsigned long long value = std::stoull(text, &consumed, 10);
        if (consumed != text.size() || value > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("out of range");
        return static_cast<uint32_t>(value);
    }
    catch (...)
    {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + text);
    }
}

int parseInt(const std::string& text, const char* option)
{
    try
    {
        size_t consumed = 0;
        const long long value = std::stoll(text, &consumed, 10);
        if (consumed != text.size() ||
            value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max())
            throw std::runtime_error("out of range");
        return static_cast<int>(value);
    }
    catch (...)
    {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + text);
    }
}

std::string nextValue(int& index, int argc, char* argv[], const char* option)
{
    if (index + 1 >= argc)
        throw std::runtime_error(std::string("missing value for ") + option);
    return argv[++index];
}

Arguments parseArguments(int argc, char* argv[])
{
    Arguments args;
    for (int i = 1; i < argc; ++i)
    {
        const std::string option = argv[i];
        if (option == "--help" || option == "-h")
        {
            printUsage();
            std::exit(0);
        }
        else if (option == "--scene")
            args.scenePath = nextValue(i, argc, argv, "--scene");
        else if (option == "--bin")
            args.binaryPath = nextValue(i, argc, argv, "--bin");
        else if (option == "--tmp-dir")
            args.tmpDir = nextValue(i, argc, argv, "--tmp-dir");
        else if (option == "--merge-dir")
            args.mergeDir = nextValue(i, argc, argv, "--merge-dir");
        else if (option == "--index")
            args.indexPath = nextValue(i, argc, argv, "--index");
        else if (option == "--polygons")
            args.polygonsPath = nextValue(i, argc, argv, "--polygons");
        else if (option == "--out")
            args.outputDir = nextValue(i, argc, argv, "--out");
        else if (option == "--lod")
            args.lod = parseUint(nextValue(i, argc, argv, "--lod"), "--lod");
        else if (option == "--level")
            args.level = parseInt(nextValue(i, argc, argv, "--level"), "--level");
        else if (option == "--cell")
        {
            args.cell.x = parseInt(nextValue(i, argc, argv, "--cell"), "--cell");
            args.cell.y = parseInt(nextValue(i, argc, argv, "--cell"), "--cell");
            args.cell.z = parseInt(nextValue(i, argc, argv, "--cell"), "--cell");
            args.hasCell = true;
        }
        else if (option == "--mode")
            args.mode = nextValue(i, argc, argv, "--mode");
        else if (option == "--sample-frequency")
            args.analysis.sampleFrequency = parseUint(
                nextValue(i, argc, argv, "--sample-frequency"), "--sample-frequency");
        else if (option == "--samples-per-polygon")
            args.analysis.samplesPerPolygon = parseUint(
                nextValue(i, argc, argv, "--samples-per-polygon"), "--samples-per-polygon");
        else if (option == "--spherical-resolution")
            args.analysis.sphericalResolution = parseUint(
                nextValue(i, argc, argv, "--spherical-resolution"), "--spherical-resolution");
        else if (option == "--block-count")
            args.analysis.splittingBlockCount = parseUint(
                nextValue(i, argc, argv, "--block-count"), "--block-count");
        else if (option == "--block-size")
            args.analysis.splittingBlockSize = parseUint(
                nextValue(i, argc, argv, "--block-size"), "--block-size");
        else if (option == "--ndf-resolution")
            args.analysis.ndfResolution = parseUint(
                nextValue(i, argc, argv, "--ndf-resolution"), "--ndf-resolution");
        else if (option == "--threads")
            args.analysis.threadCount = parseUint(
                nextValue(i, argc, argv, "--threads"), "--threads");
        else if (option == "--dump-polygons")
            args.dumpPolygons = true;
        else
            throw std::runtime_error("unknown option: " + option);
    }

    if (args.scenePath.empty() || args.binaryPath.empty() || !args.hasCell)
        throw std::runtime_error("--scene, --bin and --cell are required");
    if (args.mode != "projection" && args.mode != "spherical" &&
        args.mode != "splitting" && args.mode != "ndf" && args.mode != "all")
        throw std::runtime_error("unknown --mode: " + args.mode);
    if (args.level < -1)
        throw std::runtime_error("--level must be non-negative");
    return args;
}

void selectAnalysisModes(Arguments& args)
{
    args.analysis.computeProjectionValidation =
        args.mode == "projection" || args.mode == "all";
    args.analysis.computeSphericalMaps =
        args.mode == "spherical" || args.mode == "all";
    args.analysis.computeSplittingError =
        args.mode == "splitting" || args.mode == "all";
    args.analysis.computeNdf =
        args.mode == "ndf" || args.mode == "all";
}

bool isUnitDiskPixel(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
{
    const float u = (static_cast<float>(x) + 0.5f) /
        static_cast<float>(width) * 2.0f - 1.0f;
    const float v = (static_cast<float>(y) + 0.5f) /
        static_cast<float>(height) * 2.0f - 1.0f;
    return u * u + v * v <= 1.0f;
}

void writeScalarPgm(const fs::path& path, uint32_t width, uint32_t height,
                    const std::vector<float>& values, bool unitDiskMask = false)
{
    if (width == 0 || height == 0 || values.size() != size_t(width) * height)
        return;

    float minimum = std::numeric_limits<float>::max();
    float maximum = std::numeric_limits<float>::lowest();
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
        {
            if (unitDiskMask && !isUnitDiskPixel(x, y, width, height))
                continue;
            const float value = values[size_t(y) * width + x];
            if (!std::isfinite(value))
                continue;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
    if (minimum == std::numeric_limits<float>::max())
        minimum = maximum = 0;
    if (maximum <= minimum)
        maximum = minimum + 1.0f;

    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot write " + path.string());
    out << "P5\n" << width << " " << height << "\n255\n";
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
        {
            const size_t index = size_t(y) * width + x;
            unsigned char byte = 0;
            if (!unitDiskMask || isUnitDiskPixel(x, y, width, height))
            {
                const float value = values[index];
                const float normalized = std::clamp(
                    (value - minimum) / (maximum - minimum), 0.0f, 1.0f);
                byte = static_cast<unsigned char>(std::lround(normalized * 255.0f));
            }
            out.write(reinterpret_cast<const char*>(&byte), 1);
        }
}

void writeFunctionMap(const fs::path& outputDir, const std::string& name,
                      const VoxelizationCore::FunctionMap& map)
{
    if (map.width == 0 || map.height == 0)
        return;

    std::ofstream csv(outputDir / (name + ".csv"));
    if (!csv)
        throw std::runtime_error("cannot write " + (outputDir / (name + ".csv")).string());
    csv << "x,y,exact,approximation,error\n";
    for (uint32_t y = 0; y < map.height; ++y)
        for (uint32_t x = 0; x < map.width; ++x)
        {
            const size_t index = size_t(y) * map.width + x;
            csv << x << ',' << y << ',' << std::setprecision(9)
                << map.exact[index] << ',' << map.approximation[index] << ','
                << map.error[index] << '\n';
        }

    writeScalarPgm(outputDir / (name + "_exact.pgm"), map.width, map.height,
                   map.exact, true);
    writeScalarPgm(outputDir / (name + "_approx.pgm"), map.width, map.height,
                   map.approximation, true);
    writeScalarPgm(outputDir / (name + "_error.pgm"), map.width, map.height,
                   map.error, true);
}

void writeFloat4Map(const fs::path& outputDir, const std::string& name,
                    const VoxelizationCore::Float4Map& map)
{
    if (map.width == 0 || map.height == 0)
        return;

    std::ofstream csv(outputDir / (name + ".csv"));
    if (!csv)
        throw std::runtime_error("cannot write " + (outputDir / (name + ".csv")).string());
    csv << "x,y,r,g,b,a\n";
    std::vector<float> channels[4];
    for (auto& channel : channels)
        channel.reserve(map.values.size());
    for (uint32_t y = 0; y < map.height; ++y)
        for (uint32_t x = 0; x < map.width; ++x)
        {
            const float4 value = map.values[size_t(y) * map.width + x];
            csv << x << ',' << y << ',' << std::setprecision(9)
                << value.x << ',' << value.y << ',' << value.z << ',' << value.w << '\n';
            channels[0].push_back(value.x);
            channels[1].push_back(value.y);
            channels[2].push_back(value.z);
            channels[3].push_back(value.w);
        }

    writeScalarPgm(outputDir / (name + "_gt.pgm"), map.width, map.height, channels[0]);
    writeScalarPgm(outputDir / (name + "_approx.pgm"), map.width, map.height, channels[1]);
    writeScalarPgm(outputDir / (name + "_error.pgm"), map.width, map.height, channels[2]);
    writeScalarPgm(outputDir / (name + "_aux.pgm"), map.width, map.height, channels[3]);
}

void writePolygonDump(const fs::path& outputDir,
                      const VoxelizationCore::NodeData& node)
{
    std::ofstream out(outputDir / "polygons.txt");
    if (!out)
        throw std::runtime_error("cannot write polygons.txt");

    out << "nodeKey=" << node.request.nodeKey
        << " level=" << node.request.level
        << " cell=" << node.request.cell.x << ','
        << node.request.cell.y << ',' << node.request.cell.z << '\n';
    out << "polygonCount=" << node.polygons.size() << '\n';
    out << std::setprecision(9);
    for (size_t i = 0; i < node.polygons.size(); ++i)
    {
        const Polygon& polygon = node.polygons[i];
        out << "polygon[" << i << "] mesh=" << polygon.triRef.meshID
            << " triangle=" << polygon.triRef.triangleID
            << " material=" << polygon.triRef.materialID
            << " instance=" << polygon.triRef.instanceIdx
            << " count=" << polygon.count
            << " area=" << polygon.calcArea()
            << " normal=" << polygon.normal.x << ',' << polygon.normal.y << ','
            << polygon.normal.z << '\n';
        for (uint32_t v = 0; v < polygon.count; ++v)
        {
            const float3& p = polygon.vertices[v];
            out << "  v[" << v << "]=" << p.x << ',' << p.y << ',' << p.z << '\n';
        }
    }
}

void writeSummary(const fs::path& outputDir,
                  const VoxelizationCore::NodeData& node,
                  const VoxelizationCore::AnalysisResult& result,
                  const VoxelizationCore::AnalysisOptions& options)
{
    std::ofstream out(outputDir / "summary.txt");
    if (!out)
        throw std::runtime_error("cannot write summary.txt");

    out << std::setprecision(9);
    out << "nodeKey=" << node.request.nodeKey << '\n';
    out << "level=" << node.request.level << '\n';
    out << "cell=" << node.request.cell.x << ',' << node.request.cell.y << ','
        << node.request.cell.z << '\n';
    out << "polygonCount=" << node.polygons.size() << '\n';
    out << "storedPolygonCount=" << node.storedPolygonCount << '\n';
    out << "area=" << result.voxelData.ABSDF.area << '\n';
    out << "sampleFrequency=" << options.sampleFrequency << '\n';
    out << "samplesPerPolygon=" << options.samplesPerPolygon << '\n';
    for (uint32_t i = 0; i < LOBE_COUNT; ++i)
    {
        const ABSDFLobe& lobe = result.voxelData.ABSDF.lobes[i];
        out << "lobe[" << i << "].weight=" << lobe.weight << '\n';
        out << "lobe[" << i << "].normal=" << lobe.normal.x << ','
            << lobe.normal.y << ',' << lobe.normal.z << '\n';
        out << "lobe[" << i << "].roughness=" << lobe.rough << '\n';
        out << "lobe[" << i << "].diffuse=" << lobe.diffuse.x << ','
            << lobe.diffuse.y << ',' << lobe.diffuse.z << '\n';
        out << "lobe[" << i << "].specular=" << lobe.specular.x << ','
            << lobe.specular.y << ',' << lobe.specular.z << '\n';
    }
}

fs::path chooseIndexPath(const Arguments& args, uint32_t level, uint32_t maxDepth)
{
    if (!args.indexPath.empty())
        return args.indexPath;
    const uint32_t lod = maxDepth - level;
    const fs::path directory = !args.mergeDir.empty()
        ? args.mergeDir
        : (args.tmpDir / (lod == 0 ? "merge" : "merge_lod_" + std::to_string(lod)));
    return directory / (lod == 0 ? "leaves.idx" : "nodes.idx");
}

fs::path choosePolygonPath(const Arguments& args, uint32_t level, uint32_t maxDepth)
{
    if (!args.polygonsPath.empty())
        return args.polygonsPath;
    const uint32_t lod = maxDepth - level;
    const fs::path directory = !args.mergeDir.empty()
        ? args.mergeDir
        : (args.tmpDir / (lod == 0 ? "merge" : "merge_lod_" + std::to_string(lod)));
    return directory / "polygons.dat";
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        Arguments args = parseArguments(argc, argv);
        selectAnalysisModes(args);

        VoxelizationCore::VoxelFileHeader voxelHeader;
        std::string error;
        if (!VoxelizationCore::readVoxelFileHeader(args.binaryPath, voxelHeader, error))
            throw std::runtime_error(error);

        uint32_t level = 0;
        if (args.level >= 0)
            level = static_cast<uint32_t>(args.level);
        else
        {
            if (args.lod > voxelHeader.maxDepth)
                throw std::runtime_error("--lod exceeds maxDepth");
            level = voxelHeader.maxDepth - args.lod;
        }
        if (level > voxelHeader.maxDepth)
            throw std::runtime_error("selected level exceeds maxDepth");

        const fs::path indexPath = chooseIndexPath(args, level, voxelHeader.maxDepth);
        const fs::path polygonsPath = choosePolygonPath(args, level, voxelHeader.maxDepth);
        VoxelizationCore::TempNodeReader reader;
        if (!reader.open(indexPath, polygonsPath))
            throw std::runtime_error(reader.error());
        if (reader.maxDepth() != voxelHeader.maxDepth)
            throw std::runtime_error("node index maxDepth does not match voxel binary");
        if (reader.targetLevel() != level)
            throw std::runtime_error(
                "selected index stores tree level " + std::to_string(reader.targetLevel()) +
                ", requested " + std::to_string(level));

        VoxelizationCore::NodeData node;
        if (!reader.readNode(level, args.cell, node))
            throw std::runtime_error(reader.error());

        InstancedScene scene;
        std::vector<Texture2D> baseColorTextures;
        std::vector<Texture2D> specularTextures;
        std::vector<Texture2D> metallicTextures;
        std::vector<Texture2D> normalMapTextures;
        if (!VoxelizationCore::loadSceneResources(
                args.scenePath, scene, baseColorTextures, specularTextures,
                metallicTextures, normalMapTextures, error))
            throw std::runtime_error("scene loading failed: " + error);

        VoxelizationCore::TextureSet textures{
            baseColorTextures, specularTextures, metallicTextures, normalMapTextures};
        VoxelizationCore::AnalysisContext context{
            scene, voxelHeader.grid, voxelHeader.maxDepth, textures};

        const auto result = VoxelizationCore::analyzeNode(
            node, context, args.analysis);
        if (!result.success)
            throw std::runtime_error("node analysis failed: " + result.error);

        fs::create_directories(args.outputDir);
        writeSummary(args.outputDir, node, result, args.analysis);
        if (args.dumpPolygons)
            writePolygonDump(args.outputDir, node);

        if (args.analysis.computeProjectionValidation)
        {
            std::ofstream csv(args.outputDir / "projection.csv");
            if (!csv)
                throw std::runtime_error("cannot write projection.csv");
            csv << "index,dx,dy,dz,exactVisible,shVisible,exactTotal,shTotal,visibleError,totalError\n";
            csv << std::setprecision(9);
            for (size_t i = 0; i < result.projectionValidation.size(); ++i)
            {
                const auto& value = result.projectionValidation[i];
                csv << i << ',' << value.direction.x << ',' << value.direction.y << ','
                    << value.direction.z << ',' << value.exactVisibleArea << ','
                    << value.shVisibleArea << ',' << value.exactTotalArea << ','
                    << value.shTotalArea << ','
                    << std::abs(value.exactVisibleArea - value.shVisibleArea) << ','
                    << std::abs(value.exactTotalArea - value.shTotalArea) << '\n';
            }
        }

        if (args.analysis.computeSphericalMaps)
        {
            writeFunctionMap(args.outputDir, "spherical_primitive",
                result.sphericalMaps[static_cast<size_t>(VoxelizationCore::SphericalFunctionType::PrimitiveProjArea)]);
            writeFunctionMap(args.outputDir, "spherical_polygons",
                result.sphericalMaps[static_cast<size_t>(VoxelizationCore::SphericalFunctionType::PolygonsProjArea)]);
            writeFunctionMap(args.outputDir, "spherical_total",
                result.sphericalMaps[static_cast<size_t>(VoxelizationCore::SphericalFunctionType::TotalProjArea)]);
        }

        if (args.analysis.computeSplittingError)
            writeFloat4Map(args.outputDir, "splitting_error", result.splittingError);
        if (args.analysis.computeNdf)
            writeFloat4Map(args.outputDir, "ndf", result.ndf);

        std::cout << "VoxelizationInspector completed.\n"
                  << "  node level=" << node.request.level
                  << " cell=" << node.request.cell
                  << " polygons=" << node.polygons.size() << '\n'
                  << "  area=" << result.voxelData.ABSDF.area << '\n'
                  << "  output=" << args.outputDir << '\n';
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "VoxelizationInspector ERROR: " << exception.what() << '\n';
        if (argc <= 1)
            printUsage();
        return 2;
    }
}
