#pragma once
#include "VoxelizationBase.h"
#include "VoxelizationPass.h"

using namespace Falcor;

class VoxelizationPass_GPU : public VoxelizationPass
{
public:
    FALCOR_PLUGIN_CLASS(VoxelizationPass_GPU, "VoxelizationPass_GPU", "Insert pass description here.");

    static ref<VoxelizationPass_GPU> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<VoxelizationPass_GPU>(pDevice, props);
    }

    VoxelizationPass_GPU(ref<Device> pDevice, const Properties& props);
    virtual ~VoxelizationPass_GPU() = default;

    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;

    virtual void clipPolygon(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void analyzePolygon(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual std::string getFileName() override;
    virtual void* getVBufferCPU() const override;

private:
    uint maxSolidVoxelCount;
    ref<ComputePass> mLoadMeshPass;     // LoadMesh.cs.slang (flat buffers)
    ref<ComputePass> mClipTrianglePass; // ClipTriangle.cs.slang (two-pass GPU clipping)

    ref<Buffer> vBuffer;            // int per voxel: cell -> compact solid voxel offset (>=0) or -1 (empty)
    ref<Buffer> visitedBuffer;      // uint per voxel: 0=unvisited, nonzero=visited by Pass 1 (debug/fallback)
    ref<Buffer> rawMarkBuffer;      // uint[rawMarkCapacity]: streaming voxel-index output from Pass 1
    ref<Buffer> rawMarkCounter;     // uint[1]: atomic counter for rawMarkBuffer (read back = only 4 bytes!)
    ref<Buffer> polygonCountBuffer; // uint per solid voxel: actual polygon count (written by Pass 2)
    ref<Buffer> mPolygonBuffer;     // GPU polygon buffer (conservative allocation, Pass 2 writes here)

    // Flat buffers for LoadMesh.cs.slang
    ref<Buffer> positions;
    ref<Buffer> normals;
    ref<Buffer> texCoords;
    ref<Buffer> triangles;

    std::vector<int> vBuffer_CPU; // CPU mirror of vBuffer
    void* pVBuffer_CPU;           // CPU pointer for write file (set after clipPolygon)

    std::vector<uint> polygonCounts; // per-solid-voxel polygon count (needed by analyzePolygon)
    double mSolidRate;
};
