import falcor

def render_graph_Pass():
    g = RenderGraph("Voxelization")

    voxel_pass = createPass("VoxelizationPass")

    g.addPass(voxel_pass,"VoxelizationPass")

    g.markOutput("VoxelizationPass.dummy")

    return g

Graph = render_graph_Pass()
try:
    m.addGraph(Graph)
except NameError:
    pass
