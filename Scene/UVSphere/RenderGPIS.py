"""Open this script in Mogwai to render the textured UV sphere through GPIS."""
from pathlib import Path
from runpy import run_path

scene_dir = Path(__file__).resolve().parent
create_graph = run_path(str(scene_dir.parents[1] / "scripts" / "LatticeGPIS.py"))["render_graph_LatticeGPIS"]
m.addGraph(create_graph(meshFile=str(scene_dir / "UVSphere.obj"), clipmapResolution=65,
                        amplitude=0.005, correlationLength=0.12))
m.loadScene(str(scene_dir / "UVSphere.pyscene"))
