#include "VoxelizationPass.h"
#include <fstream>

namespace
{
const std::string kAnalyzePolygonProgramFile = "E:/Project/Falcor/Source/RenderPasses/Voxelization/AnalyzePolygon.cs.slang";
const std::string kLoadMeshProgramFile = "E:/Project/Falcor/Source/RenderPasses/Voxelization/LoadMesh.cs.slang";
}; // namespace

VoxelizationPass::VoxelizationPass(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice), polygonGroup(pDevice, VoxelizationBase::GlobalGridData), gridData(VoxelizationBase::GlobalGridData)
{
    mVoxelizationComplete = true;
    mSamplingComplete = true;
    mCompleteTimes = 0;

    mSampleFrequency = 1024;
    mVoxelResolution = 512;

    VoxelizationBase::UpdateVoxelGrid(nullptr, mVoxelResolution);

    mpDevice = pDevice;

    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_DEFAULT);
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear)
        .setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
    mpSampler = pDevice->createSampler(samplerDesc);

    pVBuffer_CPU = nullptr;
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
    if (!mpScene || mVoxelizationComplete && mSamplingComplete)
        return;

    if (!mVoxelizationComplete)
    {
        clipPolygon(pRenderContext, renderData);
        mVoxelizationComplete = true;

        blockMap = mpDevice->createStructuredBuffer(sizeof(uint), 4 * gridData.blockTextureCount(), ResourceBindFlags::UnorderedAccess);
        pRenderContext->clearUAV(blockMap->getUAV().get(), uint4(0));

        hyperBlockMap =
            mpDevice->createStructuredBuffer(sizeof(uint), 4 * gridData.hyperBlockTextureCount(), ResourceBindFlags::UnorderedAccess);
        pRenderContext->clearUAV(hyperBlockMap->getUAV().get(), uint4(0));
    }
        else
    {
        if (polygonGroup.size() == 0)
        {
            pRenderContext->submit(true);

            Tools::Profiler::BeginSample("Write File");
            ref<Buffer> cpuGBuffer = copyToCpu(mpDevice, pRenderContext, gBuffer);
            ref<Buffer> cpuBlockMap = copyToCpu(mpDevice, pRenderContext, blockMap);
            ref<Buffer> cpuHyperBlockMap = copyToCpu(mpDevice, pRenderContext, hyperBlockMap);
            pRenderContext->submit(true);
            void* pGBuffer_CPU = cpuGBuffer->map();
            void* pBlockMap_CPU = cpuBlockMap->map();
            void* pHyperBlockMap_CPU = cpuHyperBlockMap->map();
            write(getFileName(), pGBuffer_CPU, getVBufferCPU(), pBlockMap_CPU, pHyperBlockMap_CPU);
            cpuGBuffer->unmap();
            cpuBlockMap->unmap();
            cpuHyperBlockMap->unmap();
            mSamplingComplete = true;
            Tools::Profiler::EndSample("Write File");
            Tools::Profiler::Print();
            Tools::Profiler::Reset();
        }
        else if (mCompleteTimes < polygonGroup.size())
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
            ref<Buffer> cpuBlockMap = copyToCpu(mpDevice, pRenderContext, blockMap);
            ref<Buffer> cpuHyperBlockMap = copyToCpu(mpDevice, pRenderContext, hyperBlockMap);
            pRenderContext->submit(true);
            void* pGBuffer_CPU = cpuGBuffer->map();
            void* pBlockMap_CPU = cpuBlockMap->map();
            void* pHyperBlockMap_CPU = cpuHyperBlockMap->map();
            write(getFileName(), pGBuffer_CPU, getVBufferCPU(), pBlockMap_CPU, pHyperBlockMap_CPU);
            cpuGBuffer->unmap();
            cpuBlockMap->unmap();
            cpuHyperBlockMap->unmap();
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
    static const uint resolutions[] = {64, 128, 256, 512, 960, 1024, 1280, 1344, 1408, 1472, 1536};
    {
        Gui::DropdownList list;
        for (uint32_t i = 0; i < sizeof(resolutions) / sizeof(uint); i++)
        {
            list.push_back({resolutions[i], std::to_string(resolutions[i])});
        }
        widget.dropdown("Voxel Resolution", list, mVoxelResolution);
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
        VoxelizationBase::UpdateVoxelGrid(mpScene, mVoxelResolution);
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
    VoxelizationBase::UpdateVoxelGrid(mpScene, mVoxelResolution);
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

    gBuffer = mpDevice->createStructuredBuffer(sizeof(VoxelData), gridData.solidVoxelCount, ResourceBindFlags::UnorderedAccess);
    gBuffer->setBlob(polygonGenerator.gBuffer.data(), 0, gridData.solidVoxelCount * sizeof(VoxelData));

    polygonRangeBuffer = mpDevice->createStructuredBuffer(
        sizeof(PolygonRange), gridData.solidVoxelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    polygonRangeBuffer->setBlob(polygonGenerator.polygonRangeBuffer.data(), 0, gridData.solidVoxelCount * sizeof(PolygonRange));

    pVBuffer_CPU = polygonGenerator.vBuffer.data();

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
    var[kBlockMap] = blockMap;
    var[kHyperBlockMap] = hyperBlockMap;

    uint groupVoxelCount = polygonGroup.getVoxelCount(mCompleteTimes);

    auto cb = var["CB"];
    cb["groupVoxelCount"] = groupVoxelCount;
    cb["sampleFrequency"] = mSampleFrequency;
    cb["gBufferOffset"] = polygonGroup.getVoxelOffset(mCompleteTimes);
    cb["blockCount"] = gridData.blockCount2D();
    cb["hyperBlockCount"] = gridData.hyperBlockCount2D();

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
    oss << ToString((int3)gridData.voxelCount);
    oss << "_";
    oss << std::to_string(mSampleFrequency);
    oss << ".bin";
    return oss.str();
}

void VoxelizationPass::write(std::string fileName, void* pGBuffer, void* pVBuffer, void* pBlockMap, void* pHyperBlockMap)
{
    std::ofstream f;
    std::string s = VoxelizationBase::ResourceFolder + fileName;
    f.open(s, std::ios::binary);
    f.write(reinterpret_cast<char*>(&gridData), sizeof(GridData));

    f.write(reinterpret_cast<const char*>(pVBuffer), gridData.totalVoxelCount() * sizeof(int));
    f.write(reinterpret_cast<const char*>(pGBuffer), gridData.solidVoxelCount * sizeof(VoxelData));
    f.write(reinterpret_cast<const char*>(pBlockMap), gridData.blockTextureCount() * sizeof(uint4));
    f.write(reinterpret_cast<const char*>(pHyperBlockMap), gridData.hyperBlockTextureCount() * sizeof(uint4));

    f.close();
    VoxelizationBase::FileUpdated = true;
}
