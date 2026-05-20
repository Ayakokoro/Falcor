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
    // ========================================================================
    // Phase 1: Buffer allocation
    // ========================================================================
    double maxCapacity = min(mSolidRate * gridData.totalVoxelCount() * sizeof(VoxelData), 4294967296.0);
    maxSolidVoxelCount = (uint)ceil(maxCapacity / sizeof(VoxelData));

    gBuffer = mpDevice->createStructuredBuffer(sizeof(VoxelData), maxSolidVoxelCount, ResourceBindFlags::UnorderedAccess);
    vBuffer = mpDevice->createStructuredBuffer(sizeof(int), gridData.totalVoxelCount(), ResourceBindFlags::UnorderedAccess);
    visitedBuffer = mpDevice->createStructuredBuffer(sizeof(uint), gridData.totalVoxelCount(), ResourceBindFlags::UnorderedAccess);
    polygonCountBuffer = mpDevice->createStructuredBuffer(sizeof(uint), maxSolidVoxelCount, ResourceBindFlags::UnorderedAccess);
    polygonRangeBuffer = mpDevice->createStructuredBuffer(
        sizeof(PolygonRange), maxSolidVoxelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    // NOTE: polygonRangeBuffer is the PARENT class member (VoxelizationPass::polygonRangeBuffer).
    // We directly use the parent member so all downstream code sees the same buffer.

    // Flat buffers for LoadMesh.cs.slang — compute totals FIRST (needed by streaming buffers below)
    uint meshCount = mpScene->getMeshCount();
    uint vertexCountTotal = 0;
    uint triangleCountTotal = 0;
    for (MeshID meshID{0}; meshID.get() < meshCount; ++meshID)
    {
        MeshDesc meshDesc = mpScene->getMesh(meshID);
        vertexCountTotal += meshDesc.vertexCount;
        triangleCountTotal += meshDesc.getTriangleCount();
    }

    // Streaming mark buffers for GPU→CPU compaction (replaces full visitedBuffer readback)
    // Conservative upper bound: each triangle marks at most kMaxMarksPerTri cells.
    // Actual usage is far less for thin-shell scenes (typically 0.001x–0.01x of capacity).
    // Overflow is detected and reported; fallback to visitedBuffer path if needed.
    constexpr uint kMaxMarksPerTriangle = 10000;
    uint rawMarkCapacity = min(triangleCountTotal * kMaxMarksPerTriangle, 10000000u); // cap at 10M entries = 40MB
    if (rawMarkCapacity == 0) rawMarkCapacity = 1; // avoid zero-size buffer
    rawMarkBuffer = mpDevice->createStructuredBuffer(sizeof(uint), rawMarkCapacity, ResourceBindFlags::UnorderedAccess);
    rawMarkCounter = mpDevice->createStructuredBuffer(sizeof(uint), 1, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    pRenderContext->clearUAV(gBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(vBuffer->getUAV().get(), uint4((int)0x80000000u)); // -1 as uint = empty voxel (matches downstream semantics)
    pRenderContext->clearUAV(visitedBuffer->getUAV().get(), uint4(0));   // 0=unvisited
    pRenderContext->clearUAV(polygonCountBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(polygonRangeBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(rawMarkBuffer->getUAV().get(), uint4(0));  // empty streaming buffer
    pRenderContext->clearUAV(rawMarkCounter->getUAV().get(), uint4(0)); // counter starts at 0

    positions = mpDevice->createStructuredBuffer(sizeof(float3), vertexCountTotal, ResourceBindFlags::UnorderedAccess);
    normals = mpDevice->createStructuredBuffer(sizeof(float3), vertexCountTotal, ResourceBindFlags::UnorderedAccess);
    texCoords = mpDevice->createStructuredBuffer(sizeof(float2), vertexCountTotal, ResourceBindFlags::UnorderedAccess);
    triangles = mpDevice->createStructuredBuffer(sizeof(uint3), triangleCountTotal, ResourceBindFlags::UnorderedAccess);

    // ========================================================================
    // Phase 2: LoadMesh.cs.slang — copy scene geometry to flat GPU buffers
    // ========================================================================
    Tools::Profiler::BeginSample("Clip"); // Cover ALL phases: LoadMesh + Pass 1 + readback + Pass 2/3

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
    pRenderContext->submit(true);

    // ========================================================================
    // Phase 3: ClipTriangle Pass 1 — DDA traversal + InterlockedOr visited marking
    // ========================================================================
    // Lock-free design: each thread only does InterlockedOr(visitedBuffer[index], 1).
    // No CAS spin-wait, no warp deadlock possible.
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
    varClip["visitedBuffer"] = visitedBuffer;
    varClip["rawMarkBuffer"] = rawMarkBuffer;
    varClip["rawMarkCounter"] = rawMarkCounter;

    auto cb_grid_clip = varClip["GridData"];
    cb_grid_clip["gridMin"] = gridData.gridMin;
    cb_grid_clip["voxelSize"] = gridData.voxelSize;
    cb_grid_clip["voxelCount"] = gridData.voxelCount;

    // Pass 1: mark visited only (writePass == 0)
    varClip["PassConfig"]["writePass"] = 0u;

    uint dispatchTriOffset = 0;
    for (MeshID meshID{0}; meshID.get() < meshCount; ++meshID)
    {
        MeshDesc meshDesc = mpScene->getMesh(meshID);
        uint triangleCount = meshDesc.getTriangleCount();

        auto cb_mesh = varClip["MeshData"];
        cb_mesh["vertexCount"] = meshDesc.vertexCount;
        cb_mesh["vbOffset"] = 0; // flat buffer: vertices start at index 0
        cb_mesh["ibOffset"] = 0; // flat buffer: not used
        cb_mesh["triOffset"] = dispatchTriOffset;
        cb_mesh["use16BitIndices"] = meshDesc.use16BitIndices();
        cb_mesh["materialID"] = meshDesc.materialID;
        cb_mesh["meshID"] = meshID.get();

        // Batched dispatch: safety net against single-triangle TDR from large DDA steps.
        constexpr uint kTrianglesPerBatch = 64;
        uint batchStart = 0;
        while (batchStart < triangleCount)
        {
            uint batchCount = min(kTrianglesPerBatch, triangleCount - batchStart);
            cb_mesh["triangleCount"] = batchCount;
            mClipTrianglePass->execute(pRenderContext, uint3(batchCount, 1, 1));
            pRenderContext->uavBarrier(visitedBuffer.get()); // only visitedBuffer is written in Pass 1
            batchStart += batchCount;
        }
        dispatchTriOffset += triangleCount;
    }
    // Note: no submit() here — copyToCpu in Phase 4 will implicitly synchronize

    // ========================================================================
    // Phase 4: CPU — streaming compaction (read back ONLY raw mark data, KB-scale!)
    // ========================================================================
    // OLD approach: copyToCpu(visitedBuffer) → scan O(gridSize) entries
    //   512³ = 134M × 4B = 537MB PCIe transfer (!!!)
    //
    // NEW approach: copyToCpu(rawMarkCounter) [4 bytes!] + copyToCpu(rawMarkBuffer)
    //   Typical: ~3600 marks × 4B = 14KB for small scenes
    //   Even large scenes: ~10M marks × 4B = 40MB (vs 4GB visited at 1024³)

    // Step 4a: Read atomic counter (4 bytes only)
    ref<Buffer> cpuRawMarkCounter = copyToCpu(mpDevice, pRenderContext, rawMarkCounter);
    pRenderContext->submit(true);
    uint* pCounter = reinterpret_cast<uint*>(cpuRawMarkCounter->map());
    uint totalRawMarks = pCounter[0];
    cpuRawMarkCounter->unmap();

    if (totalRawMarks == 0)
    {
        // No solid voxels found at all
        vBuffer_CPU.resize(gridData.totalVoxelCount(), -1);
        pVBuffer_CPU = vBuffer_CPU.data();
        Tools::Profiler::EndSample("Clip");
        return;
    }

    // Overflow safety check
    if (totalRawMarks > rawMarkCapacity)
    {
        logError("rawMarkBuffer overflow! " + std::to_string(totalRawMarks) + " > " + std::to_string(rawMarkCapacity) +
                 ". Increase kMaxMarksPerTriangle or triangle batch size.");
        // Fallback: could use old visitedBuffer path here, but for now just error out
        Tools::Profiler::EndSample("Clip");
        return;
    }

    // Step 4b: Read compact mark list (totalRawMarks × 4 bytes — typically KB-scale!)
    ref<Buffer> cpuRawMarks = copyToCpu(mpDevice, pRenderContext, rawMarkBuffer);
    pRenderContext->submit(true);
    uint* pRawMarks = reinterpret_cast<uint*>(cpuRawMarks->map());

    // Step 4c: Sort + deduplicate → unique sorted solid voxel linear indices
    // rawMarks[] may contain duplicates (same voxel hit by multiple triangles)
    std::vector<uint> rawMarkList(pRawMarks, pRawMarks + totalRawMarks);
    cpuRawMarks->unmap();

    // Sort ascending
    std::sort(rawMarkList.begin(), rawMarkList.end());

    // Deduplicate → compact unique list
    std::vector<uint> solidVoxelLinearIndices;
    solidVoxelLinearIndices.reserve(totalRawMarks); // upper bound (worst case: all unique)
    for (uint i = 0; i < totalRawMarks; i++)
    {
        if (i == 0 || rawMarkList[i] != rawMarkList[i - 1])
        {
            solidVoxelLinearIndices.push_back(rawMarkList[i]);
        }
    }

    uint validSolidVoxelCount = (uint)solidVoxelLinearIndices.size();
    gridData.solidVoxelCount = validSolidVoxelCount;

    // Build vBuffer: full-grid mapping from cell index → compact offset (or -1 for empty)
    vBuffer_CPU.resize(gridData.totalVoxelCount(), -1);
    for (uint i = 0; i < validSolidVoxelCount; i++)
    {
        vBuffer_CPU[solidVoxelLinearIndices[i]] = (int)i;
    }

    // Upload vBuffer (compact offset mapping) to GPU
    vBuffer->setBlob(vBuffer_CPU.data(), 0, sizeof(int) * gridData.totalVoxelCount());

    // Build initial polygonRangeBuffer (cellInt only; localHead/count set after prefix sum)
    std::vector<PolygonRange> polygonRanges(validSolidVoxelCount);
    for (uint i = 0; i < validSolidVoxelCount; i++)
    {
        polygonRanges[i].cellInt = IndexToCell((int)solidVoxelLinearIndices[i], gridData.voxelCount);
        polygonRanges[i].localHead = 0;   // placeholder, set after prefix sum in Phase 6
        polygonRanges[i].count = 0;       // placeholder, updated after Phase 7
    }
    polygonRangeBuffer->setBlob(polygonRanges.data(), 0, sizeof(PolygonRange) * validSolidVoxelCount);

    // Create solidVoxelOffsetBuffer for per-voxel dispatch in Pass 2/3
    ref<Buffer> solidVoxelOffsetBuffer =
        mpDevice->createStructuredBuffer(sizeof(uint), validSolidVoxelCount, ResourceBindFlags::ShaderResource);
    solidVoxelOffsetBuffer->setBlob(solidVoxelLinearIndices.data(), 0, sizeof(uint) * validSolidVoxelCount);

    // Bind shared resources for Pass 2 and Pass 3
    varClip["polygonRangeBuffer"] = polygonRangeBuffer;
    varClip["solidVoxelOffsetBuffer"] = solidVoxelOffsetBuffer;
    varClip["polygonCountBuffer"] = polygonCountBuffer;

    auto cb_voxel = varClip["VoxelDispatchConfig"];
    cb_voxel["validSolidVoxelCount"] = validSolidVoxelCount;

    // Set MeshData for Pass 2/3: all triangles across all meshes
    auto cb_mesh_pass23 = varClip["MeshData"];
    cb_mesh_pass23["vertexCount"] = vertexCountTotal;
    cb_mesh_pass23["triangleCount"] = triangleCountTotal;
    cb_mesh_pass23["vbOffset"] = 0;
    cb_mesh_pass23["ibOffset"] = 0;
    cb_mesh_pass23["triOffset"] = 0;
    cb_mesh_pass23["use16BitIndices"] = meshCount > 0 ? mpScene->getMesh(MeshID{0}).use16BitIndices() : false;
    cb_mesh_pass23["materialID"] = meshCount > 0 ? mpScene->getMesh(MeshID{0}).materialID : 0;
    cb_mesh_pass23["meshID"] = 0;

    // ========================================================================
    // Phase 5: Pass 2 — GPU per-voxel COUNT (writePass == 1)
    // ========================================================================
    // Each thread owns one solid voxel, iterates all triangles, counts how many
    // clipped polygons its cell produces. Writes exact count to polygonCountBuffer.
    // No polygon data is written yet — this pass is pure counting.
    pRenderContext->clearUAV(polygonCountBuffer->getUAV().get(), uint4(0));
    pRenderContext->submit(true); // ensure all uploads complete before Pass 2

    varClip["PassConfig"]["writePass"] = 1u; // count mode

    mClipTrianglePass->execute(pRenderContext, uint3(validSolidVoxelCount, 1, 1));
    pRenderContext->submit(true);

    // ========================================================================
    // Phase 6: CPU — exclusive prefix sum on counts → exact allocation
    // ========================================================================
    ref<Buffer> cpuRawCounts = copyToCpu(mpDevice, pRenderContext, polygonCountBuffer);
    pRenderContext->submit(true);
    uint* pRawCounts = reinterpret_cast<uint*>(cpuRawCounts->map());

    // Exclusive prefix sum: prefixSums[i] = total polygons in voxels [0, i)
    std::vector<uint> rawCounts(validSolidVoxelCount);
    std::vector<uint> prefixSums(validSolidVoxelCount);
    uint totalPolygons = 0;
    for (uint i = 0; i < validSolidVoxelCount; i++)
    {
        rawCounts[i] = pRawCounts[i];
        prefixSums[i] = totalPolygons;     // exclusive: start offset for voxel i
        totalPolygons += pRawCounts[i];
    }
    cpuRawCounts->unmap();

    // Early-out: no polygons produced (shouldn't happen if visited > 0, but safety)
    if (totalPolygons == 0)
    {
        pVBuffer_CPU = vBuffer_CPU.data();
        Tools::Profiler::EndSample("Clip");
        return;
    }

    // Update polygonRangeBuffer with EXACT localHead from prefix sum
    for (uint i = 0; i < validSolidVoxelCount; i++)
    {
        polygonRanges[i].localHead = prefixSums[i];
        // count stays 0 for now, filled after Pass 3
    }
    polygonRangeBuffer->setBlob(polygonRanges.data(), 0, sizeof(PolygonRange) * validSolidVoxelCount);

    // Allocate EXACT-sized polygonBuffer (zero waste)
    mPolygonBuffer = mpDevice->createStructuredBuffer(
        sizeof(Polygon), totalPolygons, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    // ========================================================================
    // Phase 7: Pass 3 — GPU per-voxel WRITE EXACT (writePass == 2)
    // ========================================================================
    // Each thread writes polygons to [localHead, localHead + count) using
    // pre-computed exact offsets from CPU prefix sum. No atomics, no limits.
    pRenderContext->clearUAV(mPolygonBuffer->getUAV().get(), uint4(0));
    pRenderContext->submit(true); // ensure all uploads complete before Pass 3

    varClip["polygonBuffer"] = mPolygonBuffer;
    varClip["PassConfig"]["writePass"] = 2u; // write-exact mode

    mClipTrianglePass->execute(pRenderContext, uint3(validSolidVoxelCount, 1, 1));
    pRenderContext->submit(true);
    Tools::Profiler::EndSample("Clip");

    // ========================================================================
    // Phase 8: CPU — finalize data structures for downstream (analyzePolygon)
    // ========================================================================
    // polygonCounts already captured from rawCounts[] in Phase 6.
    // Update polygonRanges[].count and re-upload for downstream consumers.
    polygonCounts.resize(validSolidVoxelCount);
    for (uint i = 0; i < validSolidVoxelCount; i++)
    {
        polygonRanges[i].count = rawCounts[i];
        polygonCounts[i] = rawCounts[i];
    }
    polygonRangeBuffer->setBlob(polygonRanges.data(), 0, sizeof(PolygonRange) * validSolidVoxelCount);

    // Adopt the exact-sized polygon buffer into the group structure
    polygonGroup.adoptBuffer(mPolygonBuffer, polygonRanges, validSolidVoxelCount);

    // Set CPU-side pointer for downstream (analyzePolygon, write file)
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

void* VoxelizationPass_GPU::getVBufferCPU() const
{
    return pVBuffer_CPU;
}
