import falcor

def render_graph_Pass():
    g = RenderGraph("Voxelization")

    marching_pass = createPass("RayMarchingPass")
    viewport_pass = createPass("RenderToViewportPass")
    accumulate_pass = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single",'maxFrameCount': 1024})
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})

    g.addPass(marching_pass,"RayMarchingPass")
    g.addPass(viewport_pass,"RenderToViewportPass")
    g.addPass(accumulate_pass,"AccumulatePass")
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("RayMarchingPass.color","RenderToViewportPass.input")
    g.addEdge("RenderToViewportPass.output","AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")

    return g

Graph = render_graph_Pass()
try:
    m.addGraph(Graph)
except NameError:
    pass
