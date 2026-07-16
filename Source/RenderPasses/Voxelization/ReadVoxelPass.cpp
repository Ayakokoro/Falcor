#include "ReadVoxelPass.h"
#include "Shading.slang"
#include "RenderGraph/RenderPassStandardFlags.h"

namespace
{
const std::string kPrepareProgramFile = "RenderPasses/Voxelization/PrepareShadingData.cs.slang";
}; // namespace

ReadVoxelPass::ReadVoxelPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice), gridData(VoxelizationBase::GlobalGridData)
{
    mComplete = true;
    mOptionsChanged = false;
    selectedFile = 0;
    mpDevice = pDevice;
}

RenderPassReflection ReadVoxelPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    reflector.addInput("dummy", "Dummy")
        .bindFlags(ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(0, 0, 1, 1);

    reflector.addOutput("dummy", "Dummy")
        .bindFlags(ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(0, 0, 1, 1);

    return reflector;
}

void ReadVoxelPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mComplete)
        return;

    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    if (!mPreparePass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kPrepareProgramFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        mPreparePass = ComputePass::create(mpDevice, desc, defines, true);
    }

    GridData& gd = VoxelizationBase::GlobalGridData;

    std::ifstream f;
    f.open(filePaths[selectedFile], std::ios::binary | std::ios::ate);
    if (!f.is_open())
        return;

    std::cout << "Reading voxel data from: " << filePaths[selectedFile] << std::endl;
    size_t fileSize = std::filesystem::file_size(filePaths[selectedFile]);
    size_t offset = 0;

    // Read GridData header
    tryRead(f, offset, sizeof(GridData), &gd, fileSize);

    // Read octree header
    uint32_t maxDepth = 0;
    tryRead(f, offset, sizeof(uint32_t), &maxDepth, fileSize);

    std::vector<uint32_t> nodeCounts(maxDepth + 1);
    tryRead(f, offset, (maxDepth + 1) * sizeof(uint32_t), nodeCounts.data(), fileSize);

    uint32_t totalNodes = 0;
    for (uint32_t i = 0; i <= maxDepth; i++)
        totalNodes += nodeCounts[i];

    std::cout << "Octree: maxDepth=" << maxDepth << ", totalNodes=" << totalNodes;

    // Read all octree nodes
    std::vector<OctreeNode> octreeNodes(totalNodes);
    tryRead(f, offset, totalNodes * sizeof(OctreeNode), octreeNodes.data(), fileSize);

    // Read VoxelData
    std::vector<VoxelData> voxelData(gd.solidVoxelCount);
    tryRead(f, offset, gd.solidVoxelCount * sizeof(VoxelData), voxelData.data(), fileSize);

    float maxArea = 0, minArea = FLT_MAX;
    uint zeroAreaCount = 0;
    for (auto& vd : voxelData)
    {
        float a = vd.ABSDF.area;
        maxArea = max(maxArea, a);
        minArea = min(minArea, a);
        if (a <= 0) zeroAreaCount++;
    }
    std::cout << "VoxelData area: min=" << minArea << " max=" << maxArea
            << " zeroCount=" << zeroAreaCount << "/" << voxelData.size() << std::endl;

    f.close();

    std::cout << ", solidVoxels=" << gd.solidVoxelCount << std::endl;
    for (uint32_t i = 0; i <= maxDepth; i++)
        std::cout << "  Level " << i << ": " << nodeCounts[i] << " nodes" << std::endl;

    // Create GPU buffer for octree nodes
    auto pOctreeBuffer = mpDevice->createStructuredBuffer(
        sizeof(OctreeNode), totalNodes, ResourceBindFlags::ShaderResource
    );
    pOctreeBuffer->setBlob(octreeNodes.data(), 0, totalNodes * sizeof(OctreeNode));

    // Create temp VoxelData buffer and upload
    auto pVoxelDataBuffer = mpDevice->createStructuredBuffer(
        sizeof(VoxelData), gd.solidVoxelCount, ResourceBindFlags::ShaderResource
    );
    pVoxelDataBuffer->setBlob(voxelData.data(), 0, gd.solidVoxelCount * sizeof(VoxelData));

    // Create gBuffer/pBuffer
    auto pGBuffer = mpDevice->createStructuredBuffer(
        sizeof(TEBSDF), gd.solidVoxelCount,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    auto pPBuffer = mpDevice->createStructuredBuffer(
        sizeof(Ellipsoid), gd.solidVoxelCount,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    // Prepare pass: split VoxelData into TEBSDF and Ellipsoid
    {
        ShaderVar var = mPreparePass->getRootVar();
        var["voxelDataBuffer"] = pVoxelDataBuffer;
        var[kGBuffer] = pGBuffer;
        var[kPBuffer] = pPBuffer;

        auto cb = var["CB"];
        cb["voxelCount"] = (uint)gd.solidVoxelCount;

        uint dispatchX = (gd.solidVoxelCount + 255) / 256;
        if (dispatchX > 0)
            mPreparePass->execute(pRenderContext, uint3(dispatchX, 1, 1));
    }
    // Store in statics
    VoxelizationBase::GBuffer = pGBuffer;
    VoxelizationBase::PBuffer = pPBuffer;
    VoxelizationBase::OctreeBuffer = pOctreeBuffer;
    VoxelizationBase::OctreeMaxDepth = maxDepth;
    VoxelizationBase::OctreeNodeCounts = std::move(nodeCounts);

    pRenderContext->submit(true);
    mComplete = true;
}

void ReadVoxelPass::compile(RenderContext* pRenderContext, const CompileData& compileData) {}

void ReadVoxelPass::renderUI(Gui::Widgets& widget)
{
    if (VoxelizationBase::FileUpdated)
    {
        filePaths.clear();
        for (const auto& entry : std::filesystem::directory_iterator(VoxelizationBase::ResourceFolder))
        {
            if (std::filesystem::is_regular_file(entry))
            {
                filePaths.push_back(entry.path());
            }
        }
        VoxelizationBase::FileUpdated = false;
    }
    Gui::DropdownList list;
    for (uint i = 0; i < filePaths.size(); i++)
    {
        list.push_back({i, filePaths[i].filename().string()});
    }
    widget.dropdown("File", list, selectedFile);

    if (mpScene && widget.button("Read"))
    {
        // Read GridData header from the selected file
        std::ifstream f;
        f.open(filePaths[selectedFile], std::ios::binary | std::ios::ate);
        if (!f.is_open())
            return;

        size_t fileSize = std::filesystem::file_size(filePaths[selectedFile]);
        size_t offset = 0;
        tryRead(f, offset, sizeof(GridData), &gridData, fileSize);
        f.close();

        requestRecompile();
        mComplete = false;
        mOptionsChanged = true;
    }

    widget.text("Voxel Size: " + ToString(gridData.voxelSize));
    widget.text("Voxel Count: " + ToString((int3)gridData.voxelCount));
    widget.text("Block Count: " + ToString((int3)gridData.blockCount3D()));
    widget.text("Hyper Block Count: " + ToString((int3)gridData.hyperBlockCount3D()));
    widget.text("Grid Min: " + ToString(gridData.gridMin));
    widget.text("Solid Voxel Count: " + std::to_string(gridData.solidVoxelCount));
    widget.text("Solid Rate: " + std::to_string(gridData.solidVoxelCount / (float)gridData.totalVoxelCount()));
    widget.text("Max Polygon Count: " + std::to_string(gridData.maxPolygonCount));
    widget.text("Total Polygon Count: " + std::to_string(gridData.totalPolygonCount));

    if (VoxelizationBase::OctreeMaxDepth > 0)
    {
        widget.text("Octree Max Depth: " + std::to_string(VoxelizationBase::OctreeMaxDepth));
        uint32_t totalNodes = 0;
        for (auto c : VoxelizationBase::OctreeNodeCounts)
            totalNodes += c;
        widget.text("Octree Total Nodes: " + std::to_string(totalNodes));
    }
}

void ReadVoxelPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}

bool ReadVoxelPass::tryRead(std::ifstream& f, size_t& offset, size_t bytes, void* dst, size_t fileSize)
{
    if (offset + bytes > fileSize)
        return false;
    if (dst)
    {
        f.seekg(offset, std::ios::beg);
        f.read(reinterpret_cast<char*>(dst), bytes);
    }
    offset += bytes;
    return true;
}
