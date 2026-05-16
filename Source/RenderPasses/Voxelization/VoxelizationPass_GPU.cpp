#include "VoxelizationPass_GPU.h"

namespace
{
const std::string kLoadMeshProgramFile = "E:/Project/Falcor/Source/RenderPasses/Voxelization/LoadMesh.cs.slang";
const std::string kClipTriangleProgramFile = "E:/Project/Falcor/Source/RenderPasses/Voxelization/ClipTriangle.cs.slang";

}; // namespace

VoxelizationPass_GPU::VoxelizationPass_GPU(ref<Device> pDevice, const Properties& props) : VoxelizationPass(pDevice, props)
{
    mSolidRate = 0.05;
}

void VoxelizationPass_GPU::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    VoxelizationPass::setScene(pRenderContext, pScene);
    mLoadMeshPass = nullptr;
    mClipTrianglePass = nullptr;
}

void VoxelizationPass_GPU::clipPolygon(RenderContext* pRenderContext, const RenderData& renderData)
{
    // === 1. Allocate buffers ===
    double maxCapacity = min(mSolidRate * gridData.totalVoxelCount() * sizeof(VoxelData), 4294967296.0);
    maxSolidVoxelCount = (uint)ceil(maxCapacity / sizeof(VoxelData));

    gBuffer = mpDevice->createStructuredBuffer(sizeof(VoxelData), maxSolidVoxelCount, ResourceBindFlags::UnorderedAccess);
    vBuffer = mpDevice->createStructuredBuffer(sizeof(int), gridData.totalVoxelCount(), ResourceBindFlags::UnorderedAccess);
    polygonCountBuffer = mpDevice->createStructuredBuffer(sizeof(uint), maxSolidVoxelCount, ResourceBindFlags::UnorderedAccess);
    solidVoxelCount = mpDevice->createStructuredBuffer(sizeof(uint), 1, ResourceBindFlags::UnorderedAccess);
    polygonRangeBuffer = mpDevice->createStructuredBuffer(
        sizeof(PolygonRange), maxSolidVoxelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    mPolygonRangeBufferGPU = polygonRangeBuffer;

    pRenderContext->clearUAV(gBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(vBuffer->getUAV().get(), uint4(0xFFFFFFFFu));
    pRenderContext->clearUAV(polygonCountBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(solidVoxelCount->getUAV().get(), uint4(0));

    // Flat buffers for LoadMesh.cs.slang
    uint meshCount = mpScene->getMeshCount();
    uint vertexCountTotal = 0;
    uint triangleCountTotal = 0;
    for (MeshID meshID{0}; meshID.get() < meshCount; ++meshID)
    {
        MeshDesc meshDesc = mpScene->getMesh(meshID);
        vertexCountTotal += meshDesc.vertexCount;
        triangleCountTotal += meshDesc.getTriangleCount();
    }

    positions = mpDevice->createStructuredBuffer(sizeof(float3), vertexCountTotal, ResourceBindFlags::UnorderedAccess);
    normals = mpDevice->createStructuredBuffer(sizeof(float3), vertexCountTotal, ResourceBindFlags::UnorderedAccess);
    texCoords = mpDevice->createStructuredBuffer(sizeof(float2), vertexCountTotal, ResourceBindFlags::UnorderedAccess);
    triangles = mpDevice->createStructuredBuffer(sizeof(uint3), triangleCountTotal, ResourceBindFlags::UnorderedAccess);

    // === 2. LoadMesh.cs.slang: copy scene geometry to flat GPU buffers ===
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

    ShaderVar var = mLoadMeshPass->getRootVar();
    mpScene->bindShaderData(var["gScene"]);
    var["positions"] = positions;
    var["normals"] = normals;
    var["texCoords"] = texCoords;
    var["triangles"] = triangles;

    uint triOffset = 0;
    for (MeshID meshID{0}; meshID.get() < meshCount; ++meshID)
    {
        MeshDesc meshDesc = mpScene->getMesh(meshID);
        uint triangleCount = meshDesc.getTriangleCount();

        auto meshData = mLoadMeshPass->getRootVar()["MeshData"];
        meshData["vertexCount"] = meshDesc.vertexCount;
        meshData["vbOffset"] = meshDesc.vbOffset;
        meshData["triangleCount"] = triangleCount;
        meshData["ibOffset"] = meshDesc.ibOffset;
        meshData["triOffset"] = triOffset;
        meshData["use16BitIndices"] = meshDesc.use16BitIndices();

        mLoadMeshPass->execute(pRenderContext, uint3(triangleCount, 1, 1));
        triOffset += triangleCount;
    }

    // Ensure LoadMesh writes are visible before ClipTriangle reads
    pRenderContext->uavBarrier(positions.get());
    pRenderContext->uavBarrier(normals.get());
    pRenderContext->uavBarrier(texCoords.get());
    pRenderContext->uavBarrier(triangles.get());

    // === 3. ClipTriangle.cs.slang Pass 1: count polygons (atomic) ===
    if (!mClipTrianglePass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kClipTriangleProgramFile).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        mClipTrianglePass = ComputePass::create(mpDevice, desc, defines, true);
    }

    ShaderVar varClip = mClipTrianglePass->getRootVar();
    mpScene->bindShaderData(varClip["gScene"]);
    varClip["positions"] = positions;
    varClip["normals"] = normals;
    varClip["texCoords"] = texCoords;
    varClip["triangles"] = triangles;
    varClip["vBuffer"] = vBuffer;
    varClip["polygonCountBuffer"] = polygonCountBuffer;
    varClip["solidVoxelCount"] = solidVoxelCount;

    auto cb_grid_clip = varClip["GridData"];
    cb_grid_clip["gridMin"] = gridData.gridMin;
    cb_grid_clip["voxelSize"] = gridData.voxelSize;
    cb_grid_clip["voxelCount"] = gridData.voxelCount;

    // Pass 1: count only
    varClip["PassConfig"]["writePass"] = 0u;
    // polygonRangeBuffer is needed by tryGetOffset() even in Pass 1 (for future localHead lookup)
    varClip["polygonRangeBuffer"] = polygonRangeBuffer;

    uint dispatchTriOffset = 0;
    for (MeshID meshID{0}; meshID.get() < meshCount; ++meshID)
    {
        MeshDesc meshDesc = mpScene->getMesh(meshID);
        uint triangleCount = meshDesc.getTriangleCount();

        auto cb_mesh = varClip["MeshData"];
        cb_mesh["vertexCount"] = meshDesc.vertexCount;
        cb_mesh["vbOffset"] = meshDesc.vbOffset;
        cb_mesh["ibOffset"] = meshDesc.ibOffset;
        cb_mesh["triOffset"] = dispatchTriOffset;
        cb_mesh["use16BitIndices"] = meshDesc.use16BitIndices();
        cb_mesh["materialID"] = meshDesc.materialID;
        cb_mesh["meshID"] = meshID.get();

        mClipTrianglePass->execute(pRenderContext, uint3(triangleCount, 1, 1));

        dispatchTriOffset += triangleCount;
        pRenderContext->uavBarrier(vBuffer.get());
        pRenderContext->uavBarrier(solidVoxelCount.get());
        pRenderContext->uavBarrier(polygonCountBuffer.get());
    }
    pRenderContext->submit(true);

    // === 4. CPU: read back counts, pre-allocate polygonBuffer via polygonGroup ===
    Tools::Profiler::BeginSample("Clip");
    ref<Buffer> cpuSolidVoxelCount = copyToCpu(mpDevice, pRenderContext, solidVoxelCount);
    ref<Buffer> cpuPolygonCountBuffer = copyToCpu(mpDevice, pRenderContext, polygonCountBuffer);
    pRenderContext->submit(true);

    uint* pSolidVoxelCount = reinterpret_cast<uint*>(cpuSolidVoxelCount->map());
    uint* pPolygonCount = reinterpret_cast<uint*>(cpuPolygonCountBuffer->map());

    // Debug: print raw GPU readback values (before early-out check)
    uint rawSolidVoxelCount = pSolidVoxelCount[0];
    uint firstPolygonCount = (gridData.totalVoxelCount() > 0) ? pPolygonCount[0] : 0;
    std::cout << "[GPU readback] rawSolidVoxelCount=" << rawSolidVoxelCount
              << " firstPolygonCount[0]=" << firstPolygonCount << std::endl;

    gridData.solidVoxelCount = rawSolidVoxelCount;

    std::vector<uint> polygonCounts;
    polygonCounts.resize(gridData.solidVoxelCount);
    memcpy(polygonCounts.data(), pPolygonCount, sizeof(uint) * gridData.solidVoxelCount);

    cpuSolidVoxelCount->unmap();
    cpuPolygonCountBuffer->unmap();

    // Early-out: no solid voxels found (scene may be empty or outside grid)
    if (gridData.solidVoxelCount == 0)
    {
        // Still fill vBuffer_CPU so downstream can handle empty result
        ref<Buffer> cpuVBuffer = copyToCpu(mpDevice, pRenderContext, vBuffer);
        pRenderContext->submit(true);
        void* pVBuffer = cpuVBuffer->map();
        vBuffer_CPU.resize(gridData.totalVoxelCount());
        memcpy(vBuffer_CPU.data(), pVBuffer, sizeof(int) * gridData.totalVoxelCount());
        cpuVBuffer->unmap();
        pVBuffer_CPU = vBuffer_CPU.data();
        Tools::Profiler::EndSample("Clip");
        return;
    }

    // polygonGroup.reserve() sets localHead/count for each solid voxel
    std::vector<PolygonRange> polygonRanges;
    polygonRanges.resize(gridData.solidVoxelCount);
    polygonGroup.reserve(polygonCounts, polygonRanges);
    polygonRangeBuffer->setBlob(polygonRanges.data(), 0, sizeof(PolygonRange) * gridData.solidVoxelCount);

    // Pre-allocate polygonBuffer GPU buffer
    // Conservative estimate: 64 polygons per voxel on average
    const uint maxPolygonsPerVoxel = 64;
    uint maxTotalPolygons = gridData.solidVoxelCount * maxPolygonsPerVoxel;
    mPolygonBuffer = mpDevice->createStructuredBuffer(
        sizeof(Polygon), maxTotalPolygons, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    polygonGroup.adoptBuffer(mPolygonBuffer, polygonRanges, gridData.solidVoxelCount);

    // Debug: verify polygonGroup state
    std::cout << "[GPU] solidVoxelCount=" << gridData.solidVoxelCount
              << " polygonGroup.size=" << polygonGroup.size()
              << " polygonGroup.voxelCount[0]=" << polygonGroup.getVoxelCount(0) << std::endl;

    // === 5. ClipTriangle.cs.slang Pass 2: write polygons (with correct triRef) ===
    pRenderContext->clearUAV(polygonCountBuffer->getUAV().get(), uint4(0));
    pRenderContext->submit(true); // ensure polygonRangeBuffer upload completes before Pass 2
    varClip["polygonBuffer"] = mPolygonBuffer;
    varClip["polygonRangeBuffer"] = polygonRangeBuffer;

    dispatchTriOffset = 0;
    for (MeshID meshID{0}; meshID.get() < meshCount; ++meshID)
    {
        MeshDesc meshDesc = mpScene->getMesh(meshID);
        uint triangleCount = meshDesc.getTriangleCount();

        auto cb_mesh = varClip["MeshData"];
        cb_mesh["vertexCount"] = meshDesc.vertexCount;
        cb_mesh["vbOffset"] = meshDesc.vbOffset;
        cb_mesh["ibOffset"] = meshDesc.ibOffset;
        cb_mesh["triOffset"] = dispatchTriOffset;
        cb_mesh["use16BitIndices"] = meshDesc.use16BitIndices();
        cb_mesh["materialID"] = meshDesc.materialID;
        cb_mesh["meshID"] = meshID.get();

        mClipTrianglePass->execute(pRenderContext, uint3(triangleCount, 1, 1));

        dispatchTriOffset += triangleCount;
        pRenderContext->uavBarrier(polygonCountBuffer.get());
    }
    pRenderContext->uavBarrier(mPolygonBuffer.get());
    pRenderContext->submit(true);
    Tools::Profiler::EndSample("Clip");

    // === 6. CPU: read back vBuffer, fill cellInt, upload polygonRangeBuffer ===
    ref<Buffer> cpuVBuffer = copyToCpu(mpDevice, pRenderContext, vBuffer);
    pRenderContext->submit(true);

    void* pVBuffer = cpuVBuffer->map();
    vBuffer_CPU.resize(gridData.totalVoxelCount());
    memcpy(vBuffer_CPU.data(), pVBuffer, sizeof(int) * gridData.totalVoxelCount());
    cpuVBuffer->unmap();

    // === 6. CPU: read back polygonRangeBuffer (localHead from reserve()), fill cellInt ===
    ref<Buffer> cpuPolygonRangeBuffer = copyToCpu(mpDevice, pRenderContext, polygonRangeBuffer);
    pRenderContext->submit(true);
    PolygonRange* pPolygonRanges = reinterpret_cast<PolygonRange*>(cpuPolygonRangeBuffer->map());

    for (size_t i = 0; i < vBuffer_CPU.size(); i++)
    {
        int offset = vBuffer_CPU[i];
        if (offset >= 0)
        {
            int3 cellInt = IndexToCell((int)i, gridData.voxelCount);
            pPolygonRanges[offset].cellInt = cellInt;
        }
    }

    cpuPolygonRangeBuffer->unmap();
    polygonRangeBuffer->setBlob(pPolygonRanges, 0, sizeof(PolygonRange) * gridData.solidVoxelCount);
    pVBuffer_CPU = vBuffer_CPU.data();
}

void VoxelizationPass_GPU::analyzePolygon(RenderContext* pRenderContext, const RenderData& renderData)
{
    VoxelizationPass::analyzePolygon(pRenderContext, renderData);
}

void VoxelizationPass_GPU::renderUI(Gui::Widgets& widget)
{
    VoxelizationPass::renderUI(widget);
    widget.var("Solid Rate", mSolidRate, 0.01, 1.0);
}

std::string VoxelizationPass_GPU::getFileName()
{
    return VoxelizationPass::getFileName() + "_GPU";
}
