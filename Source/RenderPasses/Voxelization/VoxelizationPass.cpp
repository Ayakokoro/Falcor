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
    if (!mpScene || !mVoxelizationDirty)
        return;

    // ===== Phase 1: Load mesh + hierarchical clip (CPU) =====
    runHierarchicalClip(pRenderContext);

    // ===== Phase 2: Upload buffers to GPU =====
    uploadBuffers(pRenderContext);

    // ===== Phase 3: GPU analyze all nodes =====
    analyzeAllNodes(pRenderContext);

    // ===== Phase 4: Readback + write file =====
    readbackAndWrite(pRenderContext);

    // ===== Phase 5: Debug output =====
    if (mEnableDebug)
        outputDebugInfo();

    Tools::Profiler::Print();
    Tools::Profiler::Reset();
    mVoxelizationDirty = false;
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
    widget.checkbox("Debug Output", mEnableDebug);

    if (mpScene && widget.button("Generate"))
    {
        VoxelizationBase::UpdateVoxelGrid(mpScene, mMaxVoxelResolution);
        mVoxelizationDirty = true;
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

// ========================================================================
// Phase 1: Load mesh data + hierarchical clip
// ========================================================================

void VoxelizationPass::runHierarchicalClip(RenderContext* pRenderContext)
{
    Tools::Profiler::BeginSample("Load Mesh");

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

    Tools::Profiler::EndSample("Load Mesh");

    // Compute maxDepth from resolution
    uint32_t resolution = std::max({gridData.voxelCount.x, gridData.voxelCount.y, gridData.voxelCount.z});
    uint32_t maxDepth = 0;
    while ((1u << maxDepth) < resolution)
        maxDepth++;

    // Hierarchical clip
    Tools::Profiler::BeginSample("Hierarchical Clip");
    polygonGenerator.reset();
    polygonGenerator.clipHierarchicalAll(header, meshList, pPos, pNormal, pUV, pTri, maxDepth,
                                         mUseMultiThread ? 0 : 1);
    Tools::Profiler::EndSample("Hierarchical Clip");

    cpuPositions->unmap();
    cpuNormals->unmap();
    cpuTexCoords->unmap();
    cpuTriangles->unmap();

    // Pack polygon data into GPU buffers (uses polygonGenerator.polygonArrays / polygonRangeBuffer)
    polygonGroup.setBlob(polygonGenerator.polygonArrays, polygonGenerator.polygonRangeBuffer);

    pRenderContext->submit(true);
}

// ========================================================================
// Phase 2: Upload buffers to GPU
// ========================================================================

void VoxelizationPass::uploadBuffers(RenderContext* pRenderContext)
{
    uint totalNodeCount = (uint)polygonGenerator.gBuffer.size();

    // gBuffer: per-node VoxelData (placeholder, filled by GPU)
    gBuffer = mpDevice->createStructuredBuffer(sizeof(VoxelData), totalNodeCount, ResourceBindFlags::UnorderedAccess);
    gBuffer->setBlob(polygonGenerator.gBuffer.data(), 0, totalNodeCount * sizeof(VoxelData));

    // pBuffer: per-node Ellipsoid (filled by GPU alongside gBuffer)
    pBuffer = mpDevice->createStructuredBuffer(sizeof(Ellipsoid), totalNodeCount, ResourceBindFlags::UnorderedAccess);

    // polygonRangeBuffer: per-node range info
    polygonRangeBuffer = mpDevice->createStructuredBuffer(
        sizeof(PolygonRange), totalNodeCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    polygonRangeBuffer->setBlob(polygonGenerator.polygonRangeBuffer.data(), 0, totalNodeCount * sizeof(PolygonRange));

    // octreeBuffer: OctreeNode array (BFS order)
    uint octreeNodeCount = (uint)polygonGenerator.mOctreeNodes.size();
    octreeBuffer = mpDevice->createStructuredBuffer(
        sizeof(OctreeNode), octreeNodeCount, ResourceBindFlags::ShaderResource);
    octreeBuffer->setBlob(polygonGenerator.mOctreeNodes.data(), 0, octreeNodeCount * sizeof(OctreeNode));

    // Store in static storage for file write
    VoxelizationBase::OctreeMaxDepth = polygonGenerator.mOctreeMaxDepth;
    VoxelizationBase::OctreeNodeCounts = polygonGenerator.mOctreeNodeCounts;
    VoxelizationBase::OctreeBuffer = octreeBuffer;
    VoxelizationBase::GBuffer = gBuffer;
    VoxelizationBase::PBuffer = pBuffer;

    if (mEnableDebug)
    {
        std::cout << "[Upload] totalNodeCount=" << totalNodeCount
                  << " octreeNodeCount=" << octreeNodeCount
                  << " maxDepth=" << polygonGenerator.mOctreeMaxDepth << std::endl;
        std::cout << "[Upload] level node counts: ";
        for (uint l = 0; l <= polygonGenerator.mOctreeMaxDepth; l++)
            std::cout << polygonGenerator.mOctreeNodeCounts[l] << " ";
        std::cout << std::endl;
    }

    pRenderContext->submit(true);
}

// ========================================================================
// Phase 3: GPU analyze all nodes (single dispatch, all levels)
// ========================================================================

void VoxelizationPass::analyzeAllNodes(RenderContext* pRenderContext)
{
    Tools::Profiler::BeginSample("Analyze Polygons");

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

    uint totalNodeCount = (uint)polygonGenerator.gBuffer.size();
    uint groupCount = polygonGroup.size();

    // Process polygon batches (same batching logic as before, but nodes span all levels)
    for (uint batch = 0; batch < groupCount; batch++)
    {
        ShaderVar var = mAnalyzePolygonPass->getRootVar();
        mpScene->bindShaderData(var["gScene"]);
        var["sampler"] = mpSampler;
        var[kGBuffer] = gBuffer;
        var[kPBuffer] = pBuffer;
        var[kOctreeBuffer] = octreeBuffer;
        var[kPolygonRangeBuffer] = polygonRangeBuffer;
        var[kPolygonBuffer] = polygonGroup.get(batch);

        uint groupVoxelCount = polygonGroup.getVoxelCount(batch);

        auto cb = var["CB"];
        cb["groupVoxelCount"] = groupVoxelCount;
        cb["sampleFrequency"] = mSampleFrequency;
        cb["gBufferOffset"] = polygonGroup.getVoxelOffset(batch);

        auto cb_grid = var["GridData"];
        cb_grid["gridMin"] = gridData.gridMin;
        cb_grid["voxelSize"] = gridData.voxelSize;
        cb_grid["voxelCount"] = gridData.voxelCount;

        mAnalyzePolygonPass->execute(pRenderContext, uint3(groupVoxelCount, 1, 1));
    }

    pRenderContext->submit(true);
    Tools::Profiler::EndSample("Analyze Polygons");
}

// ========================================================================
// Phase 4: Readback + write file
// ========================================================================

void VoxelizationPass::readbackAndWrite(RenderContext* pRenderContext)
{
    Tools::Profiler::BeginSample("Readback & Write");

    uint totalNodeCount = (uint)polygonGenerator.gBuffer.size();
    ref<Buffer> cpuGBuffer = copyToCpu(mpDevice, pRenderContext, gBuffer);
    pRenderContext->submit(true);
    void* pGBuffer_CPU = cpuGBuffer->map();
    write(getFileName(), pGBuffer_CPU);
    cpuGBuffer->unmap();

    Tools::Profiler::EndSample("Readback & Write");
}

// ========================================================================
// Phase 5: Debug output
// ========================================================================

void VoxelizationPass::outputDebugInfo()
{
    uint totalNodeCount = (uint)polygonGenerator.gBuffer.size();
    uint leafCount = polygonGenerator.mOctreeNodeCounts.back();
    uint internalCount = totalNodeCount - leafCount;

    std::cout << "===== Octree Debug Info =====" << std::endl;
    std::cout << "Max Depth: " << polygonGenerator.mOctreeMaxDepth << std::endl;
    std::cout << "Total nodes: " << totalNodeCount
              << " (internal=" << internalCount << ", leaf=" << leafCount << ")" << std::endl;
    std::cout << "Solid voxel count (leaf): " << gridData.solidVoxelCount << std::endl;
    std::cout << "Grid resolution: " << ToString((int3)gridData.voxelCount) << std::endl;

    std::cout << "Per-level node counts:" << std::endl;
    uint totalPolygons = 0;
    for (uint l = 0; l <= polygonGenerator.mOctreeMaxDepth; l++)
    {
        uint nodeCount = polygonGenerator.mOctreeNodeCounts[l];
        uint polyCount = 0;
        // Count polygons for nodes at this level (nodes are in BFS order)
        uint levelStart = 0;
        for (uint pl = 0; pl < l; pl++)
            levelStart += polygonGenerator.mOctreeNodeCounts[pl];
        for (uint i = levelStart; i < levelStart + nodeCount; i++)
            polyCount += (uint)polygonGenerator.polygonArrays[i].size();
        totalPolygons += polyCount;

        uint voxelWidth = 1u << (polygonGenerator.mOctreeMaxDepth - l);
        std::cout << "  Level " << l << ": " << nodeCount << " nodes, "
                  << polyCount << " polygons (voxel width=" << voxelWidth << ")" << std::endl;
    }
    std::cout << "Total polygons (all levels): " << totalPolygons << std::endl;

    // Validation: BFS ordering check
    std::cout << "Validating BFS ordering..." << std::endl;
    bool valid = true;
    for (size_t i = 0; i < polygonGenerator.mOctreeNodes.size(); i++)
    {
        const OctreeNode& node = polygonGenerator.mOctreeNodes[i];
        if (node.dataIndex != (uint)i)
        {
            std::cerr << "  FAIL: node[" << i << "].dataIndex=" << node.dataIndex << " != " << i << std::endl;
            valid = false;
        }
        if (node.childMask != 0)
        {
            // Check that children are at valid indices
            uint childCount = 0;
            uint temp = node.childMask;
            while (temp) { childCount += temp & 1; temp >>= 1; }
            if (node.childBase + childCount > polygonGenerator.mOctreeNodes.size())
            {
                std::cerr << "  FAIL: node[" << i << "] childBase=" << node.childBase
                          << " + count=" << childCount << " > total=" << polygonGenerator.mOctreeNodes.size() << std::endl;
                valid = false;
            }
        }
    }
    if (valid)
        std::cout << "  BFS ordering: OK" << std::endl;

    std::cout << "=============================" << std::endl;
}

// ========================================================================
// File I/O
// ========================================================================

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
    oss << "_hier";  // mark as hierarchical
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

void VoxelizationPass::write(std::string fileName, void* pGBuffer)
{
    std::ofstream f;
    std::string s = VoxelizationBase::ResourceFolder + fileName;
    f.open(s, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&gridData), sizeof(GridData));

    uint32_t maxDepth = polygonGenerator.mOctreeMaxDepth;
    f.write(reinterpret_cast<const char*>(&maxDepth), sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(polygonGenerator.mOctreeNodeCounts.data()),
            polygonGenerator.mOctreeNodeCounts.size() * sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(polygonGenerator.mOctreeNodes.data()),
            polygonGenerator.mOctreeNodes.size() * sizeof(OctreeNode));

    f.write(reinterpret_cast<const char*>(pGBuffer), polygonGenerator.gBuffer.size() * sizeof(VoxelData));

    f.close();
    VoxelizationBase::FileUpdated = true;
}
