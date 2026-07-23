#include "VoxelizationPass.h"
#include <fstream>
#include <algorithm>
#include <iomanip>

namespace
{
const std::string kAnalyzePolygonProgramFile = "RenderPasses/Voxelization/AnalyzePolygon.cs.slang";
const std::string kLoadMeshProgramFile = "RenderPasses/Voxelization/LoadMesh.cs.slang";
const std::string kValidationProgramFile = "RenderPasses/Voxelization/ValidateProjection.cs.slang";
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
    if (!mpScene)
        return;

    if (mVoxelizationDirty)
    {
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

    // ===== Phase 6: Validate SH fitting (on demand, independent of voxelization) =====
    if (mValidationRequested && gBuffer && polygonGroup.size() > 0)
    {
        std::vector<float3> dirs;
        dirs.push_back(normalize(float3(1, 0, 0)));
        dirs.push_back(normalize(float3(0, 1, 0)));
        dirs.push_back(normalize(float3(0, 0, 1)));
        dirs.push_back(normalize(float3(1, 1, 0)));
        dirs.push_back(normalize(float3(1, 0, 1)));
        dirs.push_back(normalize(float3(0, 1, 1)));
        dirs.push_back(normalize(float3(-1, 1, 0.5f)));
        dirs.push_back(normalize(float3(0.3f, 0.7f, -0.5f)));
        dirs.push_back(normalize(float3(1, 1, 1)));
        dirs.push_back(normalize(float3(0.5f, -0.3f, 0.8f)));
        validateProjection(pRenderContext, mValidationLOD, mValidationCellInt, dirs);
        mValidationRequested = false;
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
    widget.checkbox("Debug Output", mEnableDebug);

    // ---- SH Validation ----
    if (polygonGenerator.mOctreeMaxDepth > 0)
    {
        widget.text("--- SH Validation ---");
        int lodMax = (int)polygonGenerator.mOctreeMaxDepth;
        widget.var("LOD Level", (int&)mValidationLOD, 0, lodMax);

        int maxCoord = (1 << mValidationLOD) - 1;
        if (maxCoord < 0) maxCoord = 0;
        widget.var("CellInt X", (int&)mValidationCellInt.x, 0, maxCoord);
        widget.var("CellInt Y", (int&)mValidationCellInt.y, 0, maxCoord);
        widget.var("CellInt Z", (int&)mValidationCellInt.z, 0, maxCoord);

        if (widget.button("Validate SH"))
        {
            mValidationRequested = true;
            requestRecompile();
        }
    }

    widget.checkbox("LerpNormal", mLerpNormal);

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
    mValidationPass = nullptr;
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
    gBuffer = mpDevice->createStructuredBuffer(sizeof(VoxelData), totalNodeCount,
                                               ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
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

    mAnalyzePolygonPass->addDefine("LERP_NORMAL", mLerpNormal ? "1" : "0");

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
        pRenderContext->submit(true);
    }

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

// ========================================================================
// SH Validation: find octree node + compare exact vs SH projection area
// ========================================================================

uint32_t VoxelizationPass::findNodeByLODAndCell(int3 targetCellInt, uint32_t targetLOD)
{
    const auto& nodes = polygonGenerator.mOctreeNodes;

    std::cout << "Octree Max Depth = " << polygonGenerator.mOctreeMaxDepth << "\n";
    if (nodes.empty() || targetLOD > polygonGenerator.mOctreeMaxDepth)
        return UINT32_MAX;

    uint32_t nodeIndex = 0;
    uint32_t maxDepth = polygonGenerator.mOctreeMaxDepth;

    for (uint32_t level = 0; level <= maxDepth; level++)
    {
        const OctreeNode& node = nodes[nodeIndex];

        if (level == targetLOD)
            return node.dataIndex;

        if (node.childMask == 0)
            return UINT32_MAX;

        uint32_t shift = targetLOD - level - 1;
        uint32_t lx = (targetCellInt.x >> shift) & 1;
        uint32_t ly = (targetCellInt.y >> shift) & 1;
        uint32_t lz = (targetCellInt.z >> shift) & 1;
        uint32_t childSlot = lx + ly * 2 + lz * 4;

        if (node.childMask & (1u << childSlot))
        {
            uint32_t childOffset = node.childMask & ((1u << childSlot) - 1);
            uint32_t childIdx = 0;
            while (childOffset) { childIdx += childOffset & 1; childOffset >>= 1; }
            nodeIndex = node.childBase + childIdx;
        }
        else
        {
            return UINT32_MAX;
        }
    }
    return UINT32_MAX;
}

void VoxelizationPass::validateProjection(RenderContext* pRenderContext, uint32_t lodLevel,
                                          int3 cellInt, const std::vector<float3>& directions)
{
    if (directions.empty())
        return;

    uint32_t targetIdx = findNodeByLODAndCell(cellInt, lodLevel);
    if (targetIdx == UINT32_MAX)
    {
        std::cerr << "[Validate] Node not found at LOD=" << lodLevel
                  << " cellInt=" << ToString(cellInt) << std::endl;
        return;
    }

    // Find which polygon batch contains this node
    int batchIdx = -1;
    for (uint b = 0; b < polygonGroup.size(); b++)
    {
        uint offset = polygonGroup.getVoxelOffset(b);
        uint count = polygonGroup.getVoxelCount(b);
        if (targetIdx >= offset && targetIdx < offset + count)
        {
            batchIdx = (int)b;
            break;
        }
    }
    if (batchIdx < 0)
    {
        std::cerr << "[Validate] Node gbIndex=" << targetIdx << " not found in any batch" << std::endl;
        return;
    }

    Tools::Profiler::BeginSample("Validate Projection");

    // Create or reuse validation pass
    if (!mValidationPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kValidationProgramFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        mValidationPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines(), true);
    }

    uint dirCount = (uint)directions.size();

    // Upload directions buffer
    auto dirBuffer = mpDevice->createStructuredBuffer(
        sizeof(float3), dirCount,
        ResourceBindFlags::ShaderResource
    );
    dirBuffer->setBlob(directions.data(), 0, dirCount * sizeof(float3));

    // Create output buffer
    auto outputBuffer = mpDevice->createStructuredBuffer(
        sizeof(DirectionValidation), dirCount,
        ResourceBindFlags::UnorderedAccess
    );

    ShaderVar var = mValidationPass->getRootVar();
    var["polygonBuffer"] = polygonGroup.get(batchIdx);
    var["polygonRangeBuffer"] = polygonRangeBuffer;
    var["gBuffer"] = gBuffer;
    var["directions"] = dirBuffer;
    var["validationOutput"] = outputBuffer;

    auto cb = var["CB"];
    cb["targetGBufferIndex"] = targetIdx;
    cb["directionCount"] = dirCount;

    uint dispatchX = (dirCount + 63) / 64;
    mValidationPass->execute(pRenderContext, uint3(dispatchX, 1, 1));
    pRenderContext->submit(true);

    // Readback
    ref<Buffer> cpuBuffer = copyToCpu(mpDevice, pRenderContext, outputBuffer);
    pRenderContext->submit(true);
    const DirectionValidation* pResults =
        reinterpret_cast<const DirectionValidation*>(cpuBuffer->map());

    std::cout << "\n===== SH Validation Results =====" << std::endl;
    std::cout << "LOD=" << lodLevel << " cellInt=" << ToString(cellInt)
              << " gbIndex=" << targetIdx << " batch=" << batchIdx << std::endl;
    std::cout << "DirectionCount=" << dirCount << std::endl;
    std::cout << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (uint i = 0; i < dirCount; i++)
    {
        const auto& r = pResults[i];
        float visError = (r.exactVisibleArea > 0)
            ? abs(r.shVisibleArea - r.exactVisibleArea) / r.exactVisibleArea * 100.0f
            : 0.0f;
        float totalError = (r.exactTotalArea > 0)
            ? abs(r.shTotalArea - r.exactTotalArea) / r.exactTotalArea * 100.0f
            : 0.0f;

        std::cout << "Dir[" << i << "] (" << r.direction.x << "," << r.direction.y << "," << r.direction.z << ")"
                  << "  visExact=" << r.exactVisibleArea
                  << " visSH=" << r.shVisibleArea
                  << " err=" << visError << "%"
                  << "  |  totExact=" << r.exactTotalArea
                  << " totSH=" << r.shTotalArea
                  << " err=" << totalError << "%"
                  << std::endl;
    }
    std::cout << "=================================\n" << std::endl;

    cpuBuffer->unmap();

    Tools::Profiler::EndSample("Validate Projection");
}
