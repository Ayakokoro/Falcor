#include "VoxelizationPass.h"
#include <fstream>
#include <algorithm>

namespace
{
const std::string kAnalyzePolygonProgramFile = "RenderPasses/Voxelization/AnalyzePolygon.cs.slang";
const std::string kLoadMeshProgramFile = "RenderPasses/Voxelization/LoadMesh.cs.slang";
}; // namespace

VoxelizationPass::VoxelizationPass(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice), polygonGroup(pDevice, VoxelizationBase::GlobalGridData), gridData(VoxelizationBase::GlobalGridData)
{
    mVoxelizationComplete = true;
    mSamplingComplete = true;
    mCompleteTimes = 0;

    mSampleFrequency = 1024;

    mMaxVoxelResolution = 512;
    VoxelizationBase::UpdateVoxelGrid(nullptr, mMaxVoxelResolution);

    mpDevice = pDevice;

    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_DEFAULT);
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear)
        .setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
    mpSampler = pDevice->createSampler(samplerDesc);
}

RenderPassReflection VoxelizationPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addOutput("dummy", "Dummy")
        .bindFlags(ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(0, 0, 1, 1);
    return reflector;
}

void VoxelizationPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene)
        return;

    if (!mVoxelizationComplete)
    {
        clipPolygon(pRenderContext, renderData);
        mVoxelizationComplete = true;
    }
    else if (!mSamplingComplete)
    {
        if (mCompleteTimes < polygonGroup.size())
        {
            if (mCompleteTimes == 0)
                Tools::Profiler::BeginSample("Analyze Polygon");
            analyzePolygon(pRenderContext, renderData);
            mCompleteTimes++;
        }
        else
        {
            pRenderContext->submit(true);
            Tools::Profiler::EndSample("Analyze Polygon");

            Tools::Profiler::BeginSample("Write File");
            ref<Buffer> cpuGBuffer = copyToCpu(mpDevice, pRenderContext, gBuffer);
            pRenderContext->submit(true);
            void* pGBuffer_CPU = cpuGBuffer->map();
            write(getFileName(), pGBuffer_CPU);
            cpuGBuffer->unmap();

            mSamplingComplete = true;

            Tools::Profiler::EndSample("Write File");
            Tools::Profiler::Print();
            Tools::Profiler::Reset();
        }
    }
}

void VoxelizationPass::compile(RenderContext* pRenderContext, const CompileData& compileData) {}

void VoxelizationPass::renderUI(Gui::Widgets& widget)
{
    static const uint resolutions[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 1280, 1344, 1408, 1472, 1536};
    {
        Gui::DropdownList list;
        for (uint32_t i = 0; i < sizeof(resolutions) / sizeof(uint); i++)
        {
            list.push_back({resolutions[i], std::to_string(resolutions[i])});
        }
        widget.dropdown("Voxel Resolution", list, mMaxVoxelResolution);
    }
    static const uint sampleFrequencies[] = {0, 64, 256, 512, 1024, 2048, 4096};
    {
        Gui::DropdownList list;
        for (uint32_t i = 0; i < sizeof(sampleFrequencies) / sizeof(uint); i++)
        {
            list.push_back({sampleFrequencies[i], std::to_string(sampleFrequencies[i])});
        }
        widget.dropdown("Sample Frequency", list, mSampleFrequency);
    }

    static const uint polygonPerFrames[] = {1000, 4000, 16000, 64000, 128000, 256000, 512000, 1024000};
    {
        Gui::DropdownList list;
        for (uint32_t i = 0; i < sizeof(polygonPerFrames) / sizeof(uint); i++)
        {
            list.push_back({polygonPerFrames[i], std::to_string(polygonPerFrames[i])});
        }
        widget.dropdown("Polygon Per Frame", list, polygonGroup.maxPolygonCount);
    }

    widget.checkbox("Multi-threaded Clip", mUseMultiThread);

    if (mpScene && mVoxelizationComplete && mSamplingComplete && widget.button("Generate"))
    {
        VoxelizationBase::UpdateVoxelGrid(mpScene, mMaxVoxelResolution);
        mVoxelizationComplete = false;
        mSamplingComplete = false;
        mCompleteTimes = 0;
        requestRecompile();
    }
}

void VoxelizationPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mAnalyzePolygonPass = nullptr;
    mLoadMeshPass = nullptr;
    VoxelizationBase::UpdateVoxelGrid(mpScene, mMaxVoxelResolution);
}

void VoxelizationPass::clipPolygon(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mLoadMeshPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kLoadMeshProgramFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());

        mLoadMeshPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    uint meshCount = mpScene->getMeshCount();
    uint vertexCount = 0;
    uint triangleCount = 0;
    for (MeshID meshID{0}; meshID.get() < meshCount; ++meshID)
    {
        MeshDesc meshDesc = mpScene->getMesh(meshID);
        vertexCount += meshDesc.vertexCount;
        triangleCount += meshDesc.getTriangleCount();
    }

    ref<Buffer> positions = mpDevice->createStructuredBuffer(sizeof(float3), vertexCount, ResourceBindFlags::UnorderedAccess);
    ref<Buffer> normals = mpDevice->createStructuredBuffer(sizeof(float3), vertexCount, ResourceBindFlags::UnorderedAccess);
    ref<Buffer> texCoords = mpDevice->createStructuredBuffer(sizeof(float2), vertexCount, ResourceBindFlags::UnorderedAccess);
    ref<Buffer> triangles = mpDevice->createStructuredBuffer(sizeof(uint3), triangleCount, ResourceBindFlags::UnorderedAccess);

    std::vector<MeshHeader> meshList;

    ShaderVar var = mLoadMeshPass->getRootVar();
    mpScene->bindShaderData(var["gScene"]);
    var["positions"] = positions;
    var["normals"] = normals;
    var["texCoords"] = texCoords;
    var["triangles"] = triangles;
    uint triangleOffset = 0;
    for (MeshID meshID{0}; meshID.get() < meshCount; ++meshID)
    {
        MeshDesc meshDesc = mpScene->getMesh(meshID);
        MeshHeader mesh = {meshID.get(), meshDesc.materialID, meshDesc.vertexCount, meshDesc.getTriangleCount(), triangleOffset};
        meshList.push_back(mesh);

        auto meshData = mLoadMeshPass->getRootVar()["MeshData"];
        meshData["vertexCount"] = meshDesc.vertexCount;
        meshData["vbOffset"] = meshDesc.vbOffset;
        meshData["triangleCount"] = meshDesc.getTriangleCount();
        meshData["ibOffset"] = meshDesc.ibOffset;
        meshData["triOffset"] = triangleOffset;
        meshData["use16BitIndices"] = meshDesc.use16BitIndices();
        mLoadMeshPass->execute(pRenderContext, uint3(meshDesc.getTriangleCount(), 1, 1));

        triangleOffset += meshDesc.getTriangleCount();
    }

    ref<Buffer> cpuPositions = copyToCpu(mpDevice, pRenderContext, positions);
    ref<Buffer> cpuNormals = copyToCpu(mpDevice, pRenderContext, normals);
    ref<Buffer> cpuTexCoords = copyToCpu(mpDevice, pRenderContext, texCoords);
    ref<Buffer> cpuTriangles = copyToCpu(mpDevice, pRenderContext, triangles);
    pRenderContext->submit(true);

    float3* pPos = reinterpret_cast<float3*>(cpuPositions->map());
    float3* pNormal = reinterpret_cast<float3*>(cpuNormals->map());
    float2* pUV = reinterpret_cast<float2*>(cpuTexCoords->map());
    uint3* pTri = reinterpret_cast<uint3*>(cpuTriangles->map());
    SceneHeader header = {meshCount, vertexCount, triangleCount};

    Tools::Profiler::BeginSample("Clip");
    polygonGenerator.reset();
    polygonGenerator.clipAll(header, meshList, pPos, pNormal, pUV, pTri, mUseMultiThread ? 0 : 1);
    Tools::Profiler::EndSample("Clip");

    polygonGroup.setBlob(polygonGenerator.polygonArrays, polygonGenerator.polygonRangeBuffer);
    cpuPositions->unmap();
    cpuNormals->unmap();
    cpuTexCoords->unmap();
    cpuTriangles->unmap();

    buildOctree();

    gBuffer = mpDevice->createStructuredBuffer(sizeof(VoxelData), gridData.solidVoxelCount, ResourceBindFlags::UnorderedAccess);
    gBuffer->setBlob(polygonGenerator.gBuffer.data(), 0, gridData.solidVoxelCount * sizeof(VoxelData));

    polygonRangeBuffer = mpDevice->createStructuredBuffer(
        sizeof(PolygonRange), gridData.solidVoxelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    polygonRangeBuffer->setBlob(polygonGenerator.polygonRangeBuffer.data(), 0, gridData.solidVoxelCount * sizeof(PolygonRange));

    pRenderContext->submit(true);
}

void VoxelizationPass::analyzePolygon(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mAnalyzePolygonPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kAnalyzePolygonProgramFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        mAnalyzePolygonPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    ShaderVar var = mAnalyzePolygonPass->getRootVar();
    mpScene->bindShaderData(var["gScene"]);
    var["sampler"] = mpSampler;
    var[kGBuffer] = gBuffer;
    var[kPolygonRangeBuffer] = polygonRangeBuffer;
    var[kPolygonBuffer] = polygonGroup.get(mCompleteTimes);

    uint groupVoxelCount = polygonGroup.getVoxelCount(mCompleteTimes);

    auto cb = var["CB"];
    cb["groupVoxelCount"] = groupVoxelCount;
    cb["sampleFrequency"] = mSampleFrequency;
    cb["gBufferOffset"] = polygonGroup.getVoxelOffset(mCompleteTimes);

    auto cb_grid = var["GridData"];
    cb_grid["gridMin"] = gridData.gridMin;
    cb_grid["voxelSize"] = gridData.voxelSize;
    cb_grid["voxelCount"] = gridData.voxelCount;

    mAnalyzePolygonPass->execute(pRenderContext, uint3(groupVoxelCount, 1, 1));
}

static std::string trim_non_alnum_ends(std::string s)
{
    auto is_alnum = [](unsigned char c) { return std::isalnum(c) != 0; };

    size_t b = 0;
    while (b < s.size() && !is_alnum((unsigned char)s[b]))
        ++b;

    size_t e = s.size();
    while (e > b && !is_alnum((unsigned char)s[e - 1]))
        --e;

    return s.substr(b, e - b);
}

std::string VoxelizationPass::getFileName()
{
    std::ostringstream oss;
    oss << trim_non_alnum_ends(mpScene->getPath().stem().string());
    oss << "_";
    oss << ToString((int3)VoxelizationBase::GlobalGridData.voxelCount);
    oss << "_";
    oss << std::to_string(mSampleFrequency);
    oss << ".bin";
    return oss.str();
}

uint64_t VoxelizationPass::morton3(uint32_t x, uint32_t y, uint32_t z)
{
    uint64_t result = 0;
    for (int i = 0; i < 21; i++)
    {
        result |= ((uint64_t)(x & 1) << (3 * i)) | ((uint64_t)(y & 1) << (3 * i + 1)) |
                  ((uint64_t)(z & 1) << (3 * i + 2));
        x >>= 1;
        y >>= 1;
        z >>= 1;
    }
    return result;
}

void VoxelizationPass::buildOctree()
{
    uint32_t resolution = std::max({gridData.voxelCount.x, gridData.voxelCount.y, gridData.voxelCount.z});
    uint32_t maxDepth = 0;
    while ((1u << maxDepth) < resolution)
        maxDepth++;

    uint32_t solidCount = (uint32_t)gridData.solidVoxelCount;
    if (solidCount == 0)
    {
        mOctreeNodes.clear();
        mOctreeNodeCounts.assign(maxDepth + 1, 0);
        mOctreeMaxDepth = maxDepth;
        return;
    }

    // Step 1: Build (morton, gBufferIndex) pairs
    struct VoxelItem
    {
        uint64_t morton;
        uint32_t gbIndex;
    };
    std::vector<VoxelItem> items;
    items.reserve(solidCount);

    for (uint32_t i = 0; i < solidCount; i++)
    {
        int3 cell = polygonGenerator.polygonRangeBuffer[i].cellInt;
        if(cell.x < 0 || cell.y < 0 || cell.z < 0 || cell.x >= (int)gridData.voxelCount.x || cell.y >= (int)gridData.voxelCount.y || cell.z >= (int)gridData.voxelCount.z)
        {
            std::cerr << "Warning: Solid voxel at index " << i << " has out-of-bounds cell " << ToString(cell) << std::endl;
            continue;
        }
        items.push_back({morton3((uint32_t)cell.x, (uint32_t)cell.y, (uint32_t)cell.z), i});
    }

    int3 minCell = int3(INT_MAX);
    int3 maxCell = int3(INT_MIN);
    for (auto& item : items)
    {
        int3 cell = polygonGenerator.polygonRangeBuffer[item.gbIndex].cellInt;
        minCell = min(minCell, cell);
        maxCell = max(maxCell, cell);
    }
    std::cout << "Solid voxel range: min=" << ToString(minCell)
              << " max=" << ToString(maxCell)
              << " extent=" << ToString(maxCell - minCell + 1) << std::endl;
    std::cout << "Solid voxel count: " << solidCount
              << " / " << gridData.totalVoxelCount()
              << " (" << (100.f * solidCount / gridData.totalVoxelCount()) << "%)" << std::endl;

    // Step 2: Sort by Morton code
    std::sort(items.begin(), items.end(), [](const VoxelItem& a, const VoxelItem& b) { return a.morton < b.morton; });

    for (int idx : {0, 1, 2, (int)items.size()-3, (int)items.size()-2, (int)items.size()-1})
    {
        if (idx < 0 || idx >= (int)items.size()) continue;
        int3 cell = polygonGenerator.polygonRangeBuffer[items[idx].gbIndex].cellInt;
        uint64_t m = morton3(cell.x, cell.y, cell.z);
        std::cout << "  [" << idx << "] cell=" << ToString(cell)
                  << " morton=0x" << std::hex << m << std::dec
                  << " gbIndex=" << items[idx].gbIndex << std::endl;
    }


    // Step 3: Build levels bottom-up
    std::vector<OctreeNode> allNodes;
    allNodes.reserve(solidCount * 2);

    std::vector<uint32_t> levelStarts(maxDepth + 1);

    struct BuildItem
    {
        uint64_t morton;
        uint32_t nodeIndex;
    };
    std::vector<BuildItem> curItems;
    curItems.reserve(solidCount);

    // Leaf level (maxDepth): one node per solid voxel
    levelStarts[maxDepth] = (uint32_t)allNodes.size();
    for (auto& item : items)
        allNodes.push_back({item.gbIndex, 0u});

    for (uint32_t k = 0; k < items.size(); k++)
        curItems.push_back({items[k].morton, levelStarts[maxDepth] + k});

    // Build internal levels bottom-up
    for (int level = (int)maxDepth - 1; level >= 0; level--)
    {
        levelStarts[level] = (uint32_t)allNodes.size();
        std::vector<BuildItem> parentItems;

        size_t i = 0;
        while (i < curItems.size())
        {
            uint64_t prefix = curItems[i].morton >> 3;
            size_t j = i + 1;

            while (j < curItems.size() && (curItems[j].morton >> 3) == prefix)
                j++;

            uint32_t childMask = 0;
            for (size_t k = i; k < j; k++)
            {
                // 当前这3位就是子节点在父节点中的索引
                uint32_t childIdx = curItems[k].morton & 7;
                childMask |= (1u << childIdx);
            }

            uint32_t childBase = curItems[i].nodeIndex;
            allNodes.push_back({childBase, childMask});

            // 将 prefix 作为父节点的 morton 码传递给上一层
            parentItems.push_back({prefix, levelStarts[level] + (uint32_t)parentItems.size()});
            i = j;
        }
        curItems = std::move(parentItems);
    }

    // Step 4: Extract per-level node slices (bottom-up order in allNodes)
    // allNodes = [leaves (L_maxDepth)] [L_(maxDepth-1)] ... [L_0 (root)]
    // levelStarts[l] = start index of level l in allNodes (0 for leaves, larger for root)
    std::vector<std::vector<OctreeNode>> levelNodes(maxDepth + 1);
    for (int l = (int)maxDepth; l >= 0; l--)
    {
        uint32_t start = levelStarts[l];
        uint32_t end = (l == 0) ? (uint32_t)allNodes.size() : levelStarts[l - 1];
        levelNodes[l].assign(allNodes.begin() + start, allNodes.begin() + end);
    }

    // Compute node counts and top-down starting positions
    mOctreeNodeCounts.resize(maxDepth + 1);
    std::vector<uint32_t> tdStart(maxDepth + 1);
    tdStart[0] = 0;
    for (uint32_t l = 0; l <= maxDepth; l++)
    {
        mOctreeNodeCounts[l] = (uint32_t)levelNodes[l].size();
        if (l < maxDepth)
            tdStart[l + 1] = tdStart[l] + mOctreeNodeCounts[l];
    }

    // Step 5: Rearrange to top-down order (L0 first), remapping childBase
    std::vector<OctreeNode> topDown;
    topDown.reserve(allNodes.size());
    for (uint32_t l = 0; l <= maxDepth; l++)
    {
        for (OctreeNode node : levelNodes[l])
        {
            if (node.childMask != 0 && l < maxDepth)
            {
                // Remap childBase from bottom-up index to top-down index.
                // Children are at level l+1; their bottom-up start is levelStarts[l+1].
                // Offset within that level = childBase - levelStarts[l+1].
                // New childBase = tdStart[l+1] + offset.
                node.childBase = tdStart[l + 1] + (node.childBase - levelStarts[l + 1]);
            }
            topDown.push_back(node);
        }
    }

    mOctreeNodes = std::move(topDown);
    mOctreeMaxDepth = maxDepth;
}

void VoxelizationPass::write(std::string fileName, void* pGBuffer)
{
    std::ofstream f;
    std::string s = VoxelizationBase::ResourceFolder + fileName;
    f.open(s, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&gridData), sizeof(GridData));

    f.write(reinterpret_cast<const char*>(&mOctreeMaxDepth), sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(mOctreeNodeCounts.data()), mOctreeNodeCounts.size() * sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(mOctreeNodes.data()), mOctreeNodes.size() * sizeof(OctreeNode));

    f.write(reinterpret_cast<const char*>(pGBuffer), gridData.solidVoxelCount * sizeof(VoxelData));

    f.close();
    VoxelizationBase::FileUpdated = true;
}
