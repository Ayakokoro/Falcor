"""Compare a conventional density volume and its baked GPIS parameters in Mogwai.

On first load the script bakes the trilinear density at twice the source
resolution along each axis and caches the GPIS parameters in build/gpis-bakes.
Drag the divider to inspect the same pixels with either renderer. Both use
fully resident grids, the same camera and single-scattering lighting. In the console,
set_smoke_view("transmittance") selects camera-ray transmission through the
cloud and set_smoke_view("radiance") restores the illuminated view. The four
full-frame linear averages are available in the output selector and EXR capture.
"""

import hashlib
from pathlib import Path

import falcor
from falcor import RenderGraph, float3


scene_dir = Path(__file__).resolve().parent
root = scene_dir.parents[1]
smoke_file = root / "media" / "test_scenes" / "volumes" / "smoke.vdb"
world_scale = 0.0315
extinction_scale = 4.0
translation = float3(0.83, -0.83, 0.68)
seed = 20261024


def prepare_smoke_bake():
    grid, ell, sigma, samples_per_voxel = "density", 0.004, 1.0, 2
    # The cache key covers both the source bytes and the data that determines u.
    signature = (1, grid, ell, sigma, extinction_scale, world_scale, samples_per_voxel)
    digest = hashlib.sha256(repr(signature).encode("ascii"))
    with smoke_file.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    cache = root / "build" / "gpis-bakes"
    cache.mkdir(parents=True, exist_ok=True)
    baked = cache / ("smoke-gpis-" + digest.hexdigest()[:20] + ".vdb")
    if not baked.is_file():
        falcor.bake_gpis_density(str(smoke_file), grid, str(baked), ell, sigma, extinction_scale, world_scale, samples_per_voxel)
    return baked


def make_smoke_comparison(baked):
    graph = RenderGraph("SmokeComparison")
    graph.create_pass("Volume", "VolumeReferencePass", {
        "cloudDensityFile": str(smoke_file),
        "cloudDensityGrid": "density",
        "cloudDensityWorldScale": world_scale,
        "cloudDensityTranslation": translation,
        "cloudExtinctionScale": extinction_scale,
        "densityUseClipmap": False,
        "seed": seed,
    })
    graph.create_pass("GPIS", "BakedGPISPass", {
        "cloudBakedFile": str(baked),
        "cloudDensityTranslation": translation,
        "seed": seed,
    })
    averages = {
        "VolumeMean": "Volume.radiance",
        "GPISMean": "GPIS.color",
        "VolumeT": "Volume.transmittance",
        "GPIST": "GPIS.transmittance",
    }
    for name, source in averages.items():
        graph.create_pass(name, "AccumulatePass", {"precisionMode": "Single"})
        graph.add_edge(source, name + ".input")
    graph.create_pass("Compare", "SplitScreenPass", {
        "splitLocation": 0.5,
        "showTextLabels": True,
        "leftLabel": "Volume",
        "rightLabel": "Baked GPIS",
    })
    graph.add_edge("VolumeMean.output", "Compare.leftInput")
    graph.add_edge("GPISMean.output", "Compare.rightInput")
    graph.mark_output("Compare.output")
    for name in averages:
        graph.mark_output(name + ".output")
    return graph


def set_smoke_view(view):
    global smoke_view
    outputs = {"radiance": ("VolumeMean", "GPISMean"), "transmittance": ("VolumeT", "GPIST")}
    if view not in outputs:
        raise ValueError("Choose 'radiance' or 'transmittance'.")
    if smoke_view == view:
        return
    graph = m.getGraph("SmokeComparison")
    for source, pin in zip(outputs[smoke_view], ("Compare.leftInput", "Compare.rightInput")):
        graph.remove_edge(source + ".output", pin)
    for source, pin in zip(outputs[view], ("Compare.leftInput", "Compare.rightInput")):
        graph.add_edge(source + ".output", pin)
    smoke_view = view


if "m" in globals():
    smoke_view = "radiance"
    baked_smoke_file = prepare_smoke_bake()
    comparison = make_smoke_comparison(baked_smoke_file)
    m.addGraph(comparison)
    m.setActiveGraph(comparison)
    m.loadScene(str(scene_dir / "SmokeCamera.pyscene"))
