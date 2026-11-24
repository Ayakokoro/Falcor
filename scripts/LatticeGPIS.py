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

For meshes with UVs, GPIS can use an external base-color texture shared by
the triangles (for OBJ, the material's map_Kd). Override it with the
baseColorTexture property or GPIS > Browse texture. At each hit the pass
projects onto the nearest source triangle and interpolates its corner UVs.
Open Scene/UVSphere/RenderGPIS.py in Mogwai for a textured example.

Open Scene/CloudAndSurface/RenderGPIS.py for a cloud-like GPIS intersecting
the textured sphere. Each ray scatters at the first zero crossing of either
field. Move or resize the cloud with cloudCenter and cloudRadii in world
units. Its stationary kernel uses cloudCorrelationLength and cloudAmplitude.
Increasing cloudMeanOffset above zero makes crossings rarer at fixed amplitude.
Toggle surfaceEnabled and cloudEnabled to inspect either field individually.

Open Scene/SmokeDensity/RenderGPIS.py to use the density grid from Falcor's
media/test_scenes/volumes/smoke.vdb. Set cloudDensityFile and cloudDensityGrid
to load another nonnegative, zero-background OpenVDB float grid. The cloud
has its own camera-following densityClipmapResolution and densityClipmapLevels.
cloudDensityWorldScale and cloudDensityTranslation place the VDB in the scene.
cloudExtinctionScale converts density to target extinction per world unit;
cloudCorrelationLength sets the GPIS microstructure scale in world units.
The conversion approximates volume extinction. Set cloudDensityFile to an
empty string to use the analytic cloud again.

Scene/SmokeComparison/Compare.py compares illuminated smoke on the native
grids with clipmaps disabled. VolumeReferencePass reads the original density.
BakedGPISPass reads a separate gpisMean VDB containing the normalized mean
and stored kernel parameters, then samples the random field's zero crossings.
The script caches the conversion under build/gpis-bakes. Both paths use the
same directional light, isotropic phase, scattering albedo and background.
Their linear transmittance and single-scattering radiance can be captured as EXR.

Both surface kernel modes accept latticeSpacingInEll (default 0.5) and
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
