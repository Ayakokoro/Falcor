"""Open this script in Mogwai to trace a mountain and an intersecting cloud band.

Ridge.obj supplies the mountain's mean surface. CloudBand.vdb stores the
cloud density in the same world coordinates. The GPIS pass converts it to a
mean along each ray and chooses the nearest hit from either field.
The cloud crosses the front slope near y=0.85, below the 2.19-unit summit.
Adjust cloudDensityTranslation to move the band or cloudExtinctionScale to
change its optical thickness. Render surface and Render cloudlike GPIS in
the pass controls show the two components individually.
"""
from pathlib import Path
from runpy import run_path

from falcor import float3


scene_dir = Path(__file__).resolve().parent
make_graph = run_path(str(scene_dir.parents[1] / "scripts" / "LatticeGPIS.py"))["render_graph_LatticeGPIS"]

graph = make_graph(
    meshFile=str(scene_dir / "Ridge.obj"),
    clipmapResolution=65,
    clipmapLevels=6,
    correlationLength=0.055,
    amplitude=0.004,
    cloudEnabled=True,
    cloudDensityFile=str(scene_dir / "CloudBand.vdb"),
    cloudDensityGrid="density",
    densityUseClipmap=False,
    cloudCorrelationLength=0.045,
    cloudExtinctionScale=3.5,
    cloudColor=float3(0.95, 0.97, 0.99),
)
m.addGraph(graph)
m.setActiveGraph(graph)
m.loadScene(str(scene_dir / "MountainCloud.pyscene"))
