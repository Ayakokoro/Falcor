"""Shaderball from sparse-conv-gpis-tungsten/data/example-scenes/lattice-cubic-gpis.

Run in Mogwai at 256 x 256 for the reference framing. Each frame contributes
one sample to the linear HDR average before filmic tone mapping. The default
field is the original knob formula baked at 0.01 world units on the GPU.
For an external OpenVDB float grid, pass sdfFile and gridName to the factory.

For a closed static triangle mesh, open GPIS > Browse and choose an OBJ, PLY
or glTF file. Alternatively set meshFile when creating the graph:
    graph = render_graph_LatticeGPIS(meshFile=r"D:/models/relief.obj",
                                     clipmapResolution=33, clipmapLevels=8)
The coarsest level covers the model. Finer levels follow the camera and halve
the voxel width at each step. The panel reports the finest active voxel width
in model units. Increase the level count to resolve smaller details up close
without increasing the cost of each grid. Supported grids have 9, 17, 33, 65
or 129 nodes per axis, and 1 to 12 levels. Set amplitude to 0 to inspect the
mesh geometry alone; the noise lattice spacing is independent of SDF resolution.

Both kernel modes accept latticeSpacingInEll (default 0.5) and
kernelRadiusInEll (default 3.0). Spacing is measured in units of the
shortest correlation length and support radius in units of the local length.
The stationary support boundary rounds up to the next lattice knot.

The reference nonstationary preset can be selected in the Mogwai console:
    m.getGraph("LatticeGPIS").get_pass("GPIS").set_properties({
        "kernelMode": "linear", "heightRange": float2(0.0, 0.6),
        "correlationLength": 0.1, "correlationLengthEnd": 0.032,
        "amplitude": 0.05, "amplitudeEnd": 0.0016,
    })
"""

from falcor import *


def render_graph_LatticeGPIS(**field_and_kernel):
    graph = RenderGraph("LatticeGPIS")
    field_and_kernel.setdefault("seed", 20261024)
    graph.create_pass("GPIS", "LatticeGPISPass", field_and_kernel)
    graph.create_pass("Accumulate", "AccumulatePass", {"precisionMode": "Single"})
    graph.create_pass("Filmic", "ToneMapper", {
        "operator": "HejiHableAlu",
        "autoExposure": False,
        "useSceneMetadata": False,
        "outputFormat": "RGBA8UnormSrgb",
    })
    graph.add_edge("GPIS.color", "Accumulate.input")
    graph.add_edge("Accumulate.output", "Filmic.src")
    graph.mark_output("Filmic.dst")
    return graph


if "m" in globals():
    m.addGraph(render_graph_LatticeGPIS())
