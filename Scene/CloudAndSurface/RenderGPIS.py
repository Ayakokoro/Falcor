"""Open in Mogwai to trace a cloud-like GPIS intersecting a textured surface-like GPIS."""
from pathlib import Path
from runpy import run_path

from falcor import float3


scene_dir = Path(__file__).resolve().parent
sphere_dir = scene_dir.parent / "UVSphere"
create_graph = run_path(str(scene_dir.parents[1] / "scripts" / "LatticeGPIS.py"))["render_graph_LatticeGPIS"]

m.addGraph(create_graph(
    meshFile=str(sphere_dir / "UVSphere.obj"),
    clipmapResolution=65,
    correlationLength=0.12,
    amplitude=0.005,
    cloudEnabled=True,
    cloudCenter=float3(0.98, 0.28, 0.59),
    cloudRadii=float3(1.0, 0.63, 0.70),
    cloudCorrelationLength=0.10,
    cloudAmplitude=0.17,
    cloudMeanOffset=0.07,
    cloudColor=float3(0.38, 0.55, 0.81),
))
m.loadScene(str(sphere_dir / "UVSphere.pyscene"))
camera = m.scene.camera
camera.position = float3(1.30, 0.90, 4.25)
camera.target = float3(0.24, 0.10, 0.10)
