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

private:
    uint maxSolidVoxelCount;
    ref<ComputePass> mLoadMeshPass;         // LoadMesh.cs.slang (flat buffers)
    ref<ComputePass> mClipTrianglePass;       // ClipTriangle.cs.slang (two-pass GPU clipping)

    ref<Buffer> vBuffer;                     // int per voxel: cell -> gBuffer offset
    ref<Buffer> polygonCountBuffer;           // uint per solid voxel: polygon count
    ref<Buffer> solidVoxelCount;             // single-element: total solid voxel count
    ref<Buffer> mPolygonBuffer;               // GPU polygon buffer (ClipTriangle Pass 2 writes here)
    ref<Buffer> mPolygonRangeBufferGPU;       // polygonRangeBuffer GPU 版本

    // Flat buffers for LoadMesh.cs.slang
    ref<Buffer> positions;
    ref<Buffer> normals;
    ref<Buffer> texCoords;
    ref<Buffer> triangles;

    std::vector<uint> vBuffer_CPU;
    double mSolidRate;
};
