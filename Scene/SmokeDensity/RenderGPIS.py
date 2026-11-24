"""Open in Mogwai to render Falcor's smoke density grid crossing a textured GPIS sphere."""
from pathlib import Path
from runpy import run_path

from falcor import float3


root = Path(__file__).resolve().parents[2]
sphere = root / "Scene" / "UVSphere"
smoke = root / "media" / "test_scenes" / "volumes" / "smoke.vdb"
create_graph = run_path(str(root / "scripts" / "LatticeGPIS.py"))["render_graph_LatticeGPIS"]

m.addGraph(create_graph(
    meshFile=str(sphere / "UVSphere.obj"),
    clipmapResolution=65,
    correlationLength=0.12,
    amplitude=0.005,
    cloudEnabled=True,
    cloudDensityFile=str(smoke),
    cloudDensityGrid="density",
    # One source voxel becomes 0.015 world units. The plume crosses the sphere's near side.
    cloudDensityWorldScale=0.0315,
    cloudDensityTranslation=float3(0.83, -0.83, 0.68),
    densityClipmapResolution=65,
    densityClipmapLevels=3,
    cloudExtinctionScale=4.0,
    cloudCorrelationLength=0.045,
    cloudColor=float3(0.39, 0.46, 0.57),
))
m.loadScene(str(sphere / "UVSphere.pyscene"))
camera = m.scene.camera
camera.position = float3(1.30, 0.65, 5.0)
camera.target = float3(0.28, 0.38, 0.10)
