#!/usr/bin/env python3
"""Inspect one voxel from a Falcor/LargeSceneVoxelization .bin file.

The final .bin contains the octree and aggregated VoxelData, but not the
per-leaf polygon list.  Pass --merge-dir from a disk-backed CPU run to also
inspect the exact polygons written by MergePhase.

Examples:
  python debug_voxel.py result.bin --cell 249 309 291
  python debug_voxel.py result.bin --cell 249 309 291 \
      --merge-dir LargeSceneVoxelization/tmp/merge \
      --fbx Scene/SimplePlane/simpleplane.fbx \
      --texture C:/Users/Administrator/Desktop/veer-348664429.jpg
"""

from __future__ import annotations

import argparse
import math
import mmap
import os
import struct
import sys
from pathlib import Path
from typing import Any


# The host-side files are written by a 64-bit build.  GridData has 4 bytes of
# padding before size_t so that size_t starts at offset 40.
GRID = struct.Struct("<3f3f3I4xQII")
NODE = struct.Struct("<3I")
LOBE = struct.Struct("<f3ff3f3f")
INDEX_HEADER = struct.Struct("<IIQII")
INDEX_ENTRY = struct.Struct("<QQII")
POLYGON_HEADER = struct.Struct("<5I3f")

GRID_SIZE = GRID.size              # 56
NODE_SIZE = NODE.size              # 12
LOBE_SIZE = LOBE.size              # 44
ABSDF_AREA_OFFSET = 4 * LOBE_SIZE # 176
VOXEL_DATA_MIN_SIZE = ABSDF_AREA_OFFSET + 4

NODE_KEY_COORD_BITS = 19
NODE_KEY_LEVEL_SHIFT = NODE_KEY_COORD_BITS * 3
NODE_KEY_MASK = (1 << NODE_KEY_COORD_BITS) - 1
LEAVES_IDX_MAGIC = 0x49444C56
LEAVES_IDX_VERSION = 2


def fmt_vec(values: Any, digits: int = 7) -> str:
    return "(" + ", ".join(f"{float(v):.{digits}g}" for v in values) + ")"


def parse_meta(binary_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    meta_path = Path(str(binary_path) + ".meta")
    if not meta_path.is_file():
        return result
    for line in meta_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


class VoxelFile:
    def __init__(self, path: Path):
        self.path = path
        # A 515 input is rounded to 1024^3 by the voxelizers.  Its VoxelData
        # section can be hundreds of MB, so map it instead of copying the
        # complete file into Python's heap.
        self._file = path.open("rb")
        self.raw = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self.raw) < GRID_SIZE + 4:
            raise RuntimeError(f"file is too small: {path}")

        values = GRID.unpack_from(self.raw, 0)
        self.grid_min = values[0:3]
        self.voxel_size = values[3:6]
        self.voxel_count = values[6:9]
        self.solid_count = values[9]
        self.max_polygon_count = values[10]
        self.total_polygon_count = values[11]

        self.max_depth = struct.unpack_from("<I", self.raw, GRID_SIZE)[0]
        counts_offset = GRID_SIZE + 4
        counts_size = 4 * (self.max_depth + 1)
        if len(self.raw) < counts_offset + counts_size:
            raise RuntimeError("truncated level-count array")
        self.level_counts = struct.unpack_from(
            f"<{self.max_depth + 1}I", self.raw, counts_offset
        )

        self.nodes_offset = counts_offset + counts_size
        self.nodes_size = int(self.solid_count) * NODE_SIZE
        self.gbuffer_offset = self.nodes_offset + self.nodes_size
        if self.solid_count == 0:
            self.voxel_stride = 0
        else:
            remaining = len(self.raw) - self.gbuffer_offset
            if remaining < 0 or remaining % int(self.solid_count) != 0:
                raise RuntimeError(
                    "cannot infer VoxelData stride; output may be truncated or use a different ABI"
                )
            self.voxel_stride = remaining // int(self.solid_count)

        if self.voxel_stride and self.voxel_stride < VOXEL_DATA_MIN_SIZE:
            raise RuntimeError(
                f"unexpected VoxelData stride {self.voxel_stride} (minimum {VOXEL_DATA_MIN_SIZE})"
            )

    def node(self, index: int) -> tuple[int, int, int]:
        return NODE.unpack_from(self.raw, self.nodes_offset + index * NODE_SIZE)

    def voxel_offset(self, data_index: int) -> int:
        if data_index < 0 or data_index >= int(self.solid_count):
            raise IndexError(f"dataIndex {data_index} outside [0,{self.solid_count})")
        return self.gbuffer_offset + data_index * self.voxel_stride

    def voxel_data(self, data_index: int) -> dict[str, Any]:
        base = self.voxel_offset(data_index)
        lobes = []
        for li in range(4):
            values = LOBE.unpack_from(self.raw, base + li * LOBE_SIZE)
            lobes.append({
                "weight": values[0],
                "normal": values[1:4],
                "rough": values[4],
                "diffuse": values[5:8],
                "specular": values[8:11],
            })
        area = struct.unpack_from("<f", self.raw, base + ABSDF_AREA_OFFSET)[0]
        return {"lobes": lobes, "area": area}

    def find_node(self, cell: tuple[int, int, int], level: int | None = None) -> dict[str, Any] | None:
        target_level = self.max_depth if level is None else level
        if target_level < 0 or target_level > self.max_depth:
            raise ValueError(f"target level must be in [0,{self.max_depth}]")
        limit = 1 << target_level
        if any(c < 0 or c >= limit for c in cell):
            raise ValueError(f"cell {cell} is outside the level-{target_level} range [0,{limit})")

        node_index = 0
        path = []
        for current_level in range(target_level + 1):
            child_base, child_mask, data_index = self.node(node_index)
            entry = {
                "level": current_level,
                "cell": tuple(cell_component >> (target_level - current_level)
                               for cell_component in cell),
                "node_index": node_index,
                "data_index": data_index,
                "child_base": child_base,
                "child_mask": child_mask,
            }
            path.append(entry)
            if current_level == target_level:
                return {"path": path, "node": entry}

            shift = target_level - current_level - 1
            lx = (cell[0] >> shift) & 1
            ly = (cell[1] >> shift) & 1
            lz = (cell[2] >> shift) & 1
            child_slot = lx | (ly << 1) | (lz << 2)
            if (child_mask & (1 << child_slot)) == 0:
                return None

            preceding = child_mask & ((1 << child_slot) - 1)
            child_rank = preceding.bit_count()
            node_index = child_base + child_rank
            if node_index >= int(self.solid_count):
                raise RuntimeError("octree child index is outside the node array")
        return None

    def world_center(self, cell: tuple[int, int, int]) -> tuple[float, float, float]:
        return tuple(
            self.grid_min[i] + (cell[i] + 0.5) * self.voxel_size[i]
            for i in range(3)
        )


def make_node_key(level: int, cell: tuple[int, int, int]) -> int:
    x, y, z = cell
    return (
        (level << NODE_KEY_LEVEL_SHIFT)
        | x
        | (y << NODE_KEY_COORD_BITS)
        | (z << (NODE_KEY_COORD_BITS * 2))
    )


def read_polygons(merge_dir: Path, level: int, cell: tuple[int, int, int]) -> list[dict[str, Any]] | None:
    # MergePhase writes the leaf level to leaves.idx and every independently
    # generated internal level to nodes.idx.  The header's reserved field is
    # the actual target level, so use it to avoid confusing level 0 with the
    # leaf level when the caller asks to inspect an internal node.
    leaf_index = merge_dir / "leaves.idx"
    node_index = merge_dir / "nodes.idx"
    index_name = "leaves.idx" if level == _merge_leaf_level(merge_dir) else "nodes.idx"
    if not (merge_dir / index_name).is_file():
        # Be helpful for old leaf-only output or a directory containing only
        # one of the two index names.
        fallback = node_index if index_name == "leaves.idx" else leaf_index
        if fallback.is_file():
            index_name = fallback.name
    index_path = merge_dir / index_name
    polygons_path = merge_dir / "polygons.dat"
    if not index_path.is_file() or not polygons_path.is_file():
        raise FileNotFoundError(f"expected {index_path} and {polygons_path}")

    with index_path.open("rb") as index_file:
        header_raw = index_file.read(INDEX_HEADER.size)
        if len(header_raw) < INDEX_HEADER.size:
            raise RuntimeError(f"truncated index header: {index_path}")
        magic, version, count, max_depth, reserved = INDEX_HEADER.unpack(header_raw)
    if magic != LEAVES_IDX_MAGIC or version != LEAVES_IDX_VERSION:
        raise RuntimeError(f"unsupported index header in {index_path}: magic/version={magic:x}/{version}")

    if reserved != level:
        raise RuntimeError(
            f"{index_path.name} contains target level {reserved}, but requested level {level}"
        )

    target_key = make_node_key(level, cell)
    found = None
    with index_path.open("rb") as index_file:
        index_file.seek(INDEX_HEADER.size)
        for _ in range(count):
            entry_raw = index_file.read(INDEX_ENTRY.size)
            if len(entry_raw) < INDEX_ENTRY.size:
                raise RuntimeError(f"truncated index entries: {index_path}")
            node_key, data_offset, poly_count, _padding = INDEX_ENTRY.unpack(entry_raw)
            if node_key == target_key:
                found = (data_offset, poly_count)
                break

    if found is None:
        return None

    data_offset, poly_count = found
    polygons = []
    with polygons_path.open("rb") as polygon_file:
        polygon_file.seek(data_offset)
        for pi in range(poly_count):
            offset = polygon_file.tell()
            header_raw = polygon_file.read(POLYGON_HEADER.size)
            if len(header_raw) < POLYGON_HEADER.size:
                raise RuntimeError(f"truncated polygon header at offset {offset}")
            values = POLYGON_HEADER.unpack(header_raw)
            mesh_id, tri_id, material_id, instance_id, vertex_count = values[:5]
            normal = values[5:8]
            if vertex_count < 3 or vertex_count > 6:
                raise RuntimeError(f"invalid polygon vertex count {vertex_count} at offset {offset}")
            vertices_raw = polygon_file.read(vertex_count * 12)
            if len(vertices_raw) < vertex_count * 12:
                raise RuntimeError(f"truncated polygon vertices at offset {offset}")
            flat = struct.unpack(f"<{vertex_count * 3}f", vertices_raw)
            vertices = [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)]
            polygons.append({
                "mesh_id": mesh_id,
                "triangle_id": tri_id,
                "material_id": material_id,
                "instance_id": instance_id,
                "vertex_count": vertex_count,
                "normal": normal,
                "vertices": vertices,
            })
    return polygons


def _merge_leaf_level(merge_dir: Path) -> int:
    """Return the level recorded by leaves.idx, or -1 if it is absent."""
    path = merge_dir / "leaves.idx"
    if path.is_file():
        with path.open("rb") as index_file:
            raw = index_file.read(INDEX_HEADER.size)
        if len(raw) >= INDEX_HEADER.size:
            return INDEX_HEADER.unpack(raw)[3]
    return -1


def _merge_max_depth(merge_dir: Path) -> int:
    # Kept as a small compatibility helper for callers/scripts that imported
    # the old private name.
    for name in ("leaves.idx", "nodes.idx"):
        path = merge_dir / name
        if path.is_file():
            with path.open("rb") as index_file:
                raw = index_file.read(INDEX_HEADER.size)
            if len(raw) >= INDEX_HEADER.size:
                return INDEX_HEADER.unpack(raw)[3]
    return -1


def polygon_area(vertices: list[tuple[float, float, float]]) -> float:
    # Match Polygon::calcArea(), rather than using a triangle fan.  This is
    # important at high voxel indices: calcArea() sums cross products of the
    # absolute coordinates in float32, so a double-precision triangle fan can
    # report a visibly different value after cancellation.  The Release build
    # uses MSVC /fp:fast, which may fuse a*b-c*d; f32/fma32 reproduce that
    # rounding closely enough for this diagnostic.
    def f32(value: float) -> float:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]

    def mul32(a: float, b: float) -> float:
        return f32(f32(a) * f32(b))

    def fma32(a: float, b: float, c: float) -> float:
        return f32(float(a) * float(b) + float(c))

    cross_sum = [f32(0.0), f32(0.0), f32(0.0)]
    for i, a in enumerate(vertices):
        b = vertices[(i + 1) % len(vertices)]
        cross = (
            fma32(a[1], b[2], -mul32(a[2], b[1])),
            fma32(a[2], b[0], -mul32(a[0], b[2])),
            fma32(a[0], b[1], -mul32(a[1], b[0])),
        )
        for axis in range(3):
            cross_sum[axis] = f32(cross_sum[axis] + cross[axis])

    squared = fma32(
        cross_sum[0], cross_sum[0],
        fma32(cross_sum[1], cross_sum[1], mul32(cross_sum[2], cross_sum[2])),
    )
    return f32(f32(math.sqrt(max(0.0, squared))) / 2.0)


def setup_assimp_path() -> None:
    dll_dir = Path(__file__).resolve().parent / "build" / "Release"
    if dll_dir.is_dir():
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(dll_dir))


def load_triangle_data(fbx_path: Path, mesh_id: int, triangle_id: int, instance_id: int):
    """Return the CPU loader's triangle positions/UVs and instance transform."""
    setup_assimp_path()
    try:
        import numpy as np
        import pyassimp
    except ImportError as exc:
        raise RuntimeError("--fbx requires pyassimp and numpy") from exc

    # Match SceneLoader::loadMeshInstances().  In particular, the CPU loader
    # triangulates the FBX and flips V before it stores triangle UVs.
    processing = (
        pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        | pyassimp.postprocess.aiProcess_FlipUVs
        | pyassimp.postprocess.aiProcess_RemoveComponent
    )

    with pyassimp.load(str(fbx_path), processing=processing) as scene:
        mesh_index_by_id = {id(mesh): i for i, mesh in enumerate(scene.meshes)}
        instances = []
        unique_mesh_indices = []
        seen_meshes = set()

        def walk(node, parent):
            local = np.asarray(node.transformation, dtype=np.float64)
            world = parent @ local
            for mesh in node.meshes:
                source_index = mesh_index_by_id[id(mesh)]
                if source_index not in seen_meshes:
                    seen_meshes.add(source_index)
                    unique_mesh_indices.append(source_index)
                instances.append((source_index, world.copy()))
            for child in node.children:
                walk(child, world)

        walk(scene.rootnode, np.eye(4, dtype=np.float64))
        if mesh_id >= len(unique_mesh_indices):
            raise RuntimeError(f"meshID {mesh_id} is not present in FBX")
        if instance_id >= len(instances):
            raise RuntimeError(f"instanceIdx {instance_id} is not present in FBX")

        source_mesh_index = unique_mesh_indices[mesh_id]
        mesh = scene.meshes[source_mesh_index]
        faces = np.asarray(mesh.faces)
        face = np.asarray(faces[triangle_id]).reshape(-1)
        if face.size != 3:
            raise RuntimeError(f"triangle {triangle_id} is not a triangle")
        positions = np.asarray(mesh.vertices)
        texcoords = getattr(mesh, "texturecoords", None)
        if texcoords is None or len(texcoords) == 0 or texcoords[0] is None:
            raise RuntimeError("FBX mesh has no UV channel 0")
        uv_array = np.asarray(texcoords[0])
        return {
            "positions": [tuple(float(x) for x in positions[int(index)][:3]) for index in face],
            "uvs": [tuple(float(x) for x in uv_array[int(index)][:2]) for index in face],
            "transform": world_for_instance(instances, instance_id),
        }


def world_for_instance(instances: list[tuple[int, Any]], instance_id: int):
    if instance_id < 0 or instance_id >= len(instances):
        raise RuntimeError(f"instanceIdx {instance_id} is not present in FBX")
    return instances[instance_id][1]


class CpuTexture:
    def __init__(self, path: Path, srgb: bool = True):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("texture sampling requires Pillow and numpy") from exc

        self.np = np
        image = Image.open(path).convert("RGBA")
        data = np.asarray(image, dtype=np.float32) / 255.0
        if srgb:
            rgb = data[..., :3]
            data[..., :3] = np.where(
                rgb <= 0.04045,
                rgb / 12.92,
                ((rgb + 0.055) / 1.055) ** 2.4,
            )
        self.mips = [data]
        while True:
            previous = self.mips[-1]
            ph, pw = previous.shape[:2]
            if pw == 1 and ph == 1:
                break
            cw, ch = max(1, pw // 2), max(1, ph // 2)
            current = np.zeros((ch, cw, 4), dtype=np.float32)
            for y in range(ch):
                for x in range(cw):
                    sx = min(2 * x, pw - 1)
                    sy = min(2 * y, ph - 1)
                    current[y, x] = (
                        previous[sy, sx]
                        + previous[min(sy + 1, ph - 1), sx]
                        + previous[sy, min(sx + 1, pw - 1)]
                        + previous[min(sy + 1, ph - 1), min(sx + 1, pw - 1)]
                    ) / 4.0
            self.mips.append(current)

    def sample(self, uv: tuple[float, float], uv_area: float) -> tuple[float, float, float, float]:
        height, width = self.mips[0].shape[:2]
        pixel_area = uv_area * width * height
        lod = 0.5 * math.log2(max(pixel_area, 1.0))
        lo = max(0, min(int(math.floor(lod)), len(self.mips) - 1))
        hi = max(0, min(lo + 1, len(self.mips) - 1))
        frac = lod - math.floor(lod)

        def bilinear(level: int):
            image = self.mips[level]
            h, w = image.shape[:2]
            u = uv[0] - math.floor(uv[0])
            v = uv[1] - math.floor(uv[1])
            fx, fy = u * w - 0.5, v * h - 0.5
            x0, y0 = math.floor(fx), math.floor(fy)
            tx, ty = fx - x0, fy - y0
            x0 %= w
            y0 %= h
            x1, y1 = (x0 + 1) % w, (y0 + 1) % h
            value = (
                (1 - tx) * (1 - ty) * image[y0, x0]
                + tx * (1 - ty) * image[y0, x1]
                + (1 - tx) * ty * image[y1, x0]
                + tx * ty * image[y1, x1]
            )
            return value

        value = (1 - frac) * bilinear(lo) + frac * bilinear(hi)
        return tuple(float(x) for x in value)


def polygon_uv_info(uvs: list[tuple[float, float]]) -> tuple[tuple[float, float], float]:
    center = (
        sum(uv[0] for uv in uvs) / len(uvs),
        sum(uv[1] for uv in uvs) / len(uvs),
    )
    area = 0.0
    for i, a in enumerate(uvs):
        b = uvs[(i + 1) % len(uvs)]
        area += a[0] * b[1] - a[1] * b[0]
    return center, 0.5 * abs(area)


def barycentric(point: tuple[float, float, float], triangle: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Match Triangle::barycentricCoordinates()."""
    ab = tuple(triangle[1][i] - triangle[0][i] for i in range(3))
    ac = tuple(triangle[2][i] - triangle[0][i] for i in range(3))
    ap = tuple(point[i] - triangle[0][i] for i in range(3))
    d00 = sum(x * x for x in ab)
    d01 = sum(ab[i] * ac[i] for i in range(3))
    d11 = sum(x * x for x in ac)
    d20 = sum(ap[i] * ab[i] for i in range(3))
    d21 = sum(ap[i] * ac[i] for i in range(3))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-8:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return (1.0 - v - w, v, w)


def polygon_uv_from_triangle(
    polygon_vertices: list[tuple[float, float, float]],
    triangle_positions: list[tuple[float, float, float]],
    triangle_uvs: list[tuple[float, float]],
    range_scale: float,
) -> tuple[list[tuple[float, float]], tuple[float, float], float]:
    """Reproduce AnalyzePhase's clipped-polygon UV interpolation."""
    # Polygon vertices are node-local. AnalyzePhase restores global voxel
    # coordinates with rangeScale before calling Triangle::lerpUV().
    scaled_vertices = [tuple(c * range_scale for c in p) for p in polygon_vertices]
    polygon_uvs = []
    for point in scaled_vertices:
        weights = barycentric(point, triangle_positions)
        polygon_uvs.append(tuple(
            triangle_uvs[0][axis] * weights[0]
            + triangle_uvs[1][axis] * weights[1]
            + triangle_uvs[2][axis] * weights[2]
            for axis in range(2)
        ))
    center, area = polygon_uv_info(polygon_uvs)
    return polygon_uvs, center, area


def cpu_triangle_positions_in_voxel(
    triangle_data: dict[str, Any], vf: VoxelFile
) -> list[tuple[float, float, float]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("--fbx requires numpy") from exc

    transform = triangle_data["transform"]
    result = []
    for local in triangle_data["positions"]:
        world = transform @ np.asarray((*local, 1.0), dtype=np.float64)
        if abs(float(world[3])) > 1e-12:
            world = world / world[3]
        result.append(tuple(
            (float(world[axis]) - vf.grid_min[axis]) / vf.voxel_size[axis]
            for axis in range(3)
        ))
    return result


def print_voxel(vf: VoxelFile, cell: tuple[int, int, int], level: int | None, label: str) -> dict[str, Any] | None:
    print(f"\n[{label}] {vf.path}")
    print(f"  gridMin={fmt_vec(vf.grid_min)} voxelSize={fmt_vec(vf.voxel_size)}")
    print(f"  voxelCount={vf.voxel_count} maxDepth={vf.max_depth} solidNodes={vf.solid_count}")
    print(f"  levelCounts={vf.level_counts} VoxelDataStride={vf.voxel_stride}")
    print(f"  targetCell={cell} worldCenter={fmt_vec(vf.world_center(cell))}")

    found = vf.find_node(cell, level)
    if found is None:
        print("  RESULT: target cell is not occupied / no octree path")
        return None

    print("  octree path:")
    for entry in found["path"]:
        print(
            f"    L{entry['level']} cell={entry['cell']} node={entry['node_index']} "
            f"data={entry['data_index']} mask=0x{entry['child_mask']:02x}"
        )
    node = found["node"]
    data = vf.voxel_data(node["data_index"])
    print(f"  RESULT: node={node['node_index']} dataIndex={node['data_index']} area={data['area']:.9g}")
    active = []
    for li, lobe in enumerate(data["lobes"]):
        if lobe["weight"] > 1e-8:
            active.append(li)
            print(
                f"    lobe[{li}] weight={lobe['weight']:.9g} "
                f"normal={fmt_vec(lobe['normal'])} rough={lobe['rough']:.9g} "
                f"diffuse={fmt_vec(lobe['diffuse'])} specular={fmt_vec(lobe['specular'])}"
            )
    print(f"  activeLobes={active}")
    return {"file": vf, "found": found, "data": data, "active": active}


def print_polygons(polygons: list[dict[str, Any]] | None, target_level: int, fbx: Path | None,
                   texture_path: Path | None, target_data: dict[str, Any] | None,
                   vf: VoxelFile, expected_metallic: float | None) -> None:
    if polygons is None:
        print("\n[polygons] target key is not present in the supplied merge directory")
        return
    print(f"\n[polygons] count={len(polygons)}")
    if len(polygons) == 1:
        print("  CHECK: exactly one serialized polygon")
    else:
        print("  CHECK: more than one polygon; final VoxelData is an accumulation")

    for pi, poly in enumerate(polygons):
        print(
            f"  poly[{pi}] triRef=(mesh={poly['mesh_id']}, tri={poly['triangle_id']}, "
            f"material={poly['material_id']}, instance={poly['instance_id']}) "
            f"vertices={poly['vertex_count']} area={polygon_area(poly['vertices']):.9g}"
        )
        print(f"    normal={fmt_vec(poly['normal'])}")
        for vi, vertex in enumerate(poly["vertices"]):
            print(f"    v[{vi}]={fmt_vec(vertex)}")

    if len(polygons) != 1 or fbx is None or texture_path is None:
        return

    poly = polygons[0]
    try:
        triangle_data = load_triangle_data(
            fbx, poly["mesh_id"], poly["triangle_id"], poly["instance_id"]
        )
        triangle_positions = cpu_triangle_positions_in_voxel(triangle_data, vf)
        uvs, uv_center, uv_area = polygon_uv_from_triangle(
            poly["vertices"], triangle_positions, triangle_data["uvs"],
            1 << (vf.max_depth - target_level),
        )
        texture = CpuTexture(texture_path, srgb=True)
        sampled = texture.sample(uv_center, uv_area)
        print(f"  CPU triangle positions (voxel)={triangle_positions}")
        print(f"  UV triangle={triangle_data['uvs']}")
        print(f"  clipped polygon UVs={uvs}")
        print(f"  UV center={fmt_vec(uv_center)} uvArea={uv_area:.9g}")
        print(f"  CPU-equivalent sampled baseColor (linear)={fmt_vec(sampled)}")

        if target_data is not None:
            active = [l for l in target_data["lobes"] if l["weight"] > 1e-8]
            if len(active) == 1:
                print(f"  final active-lobe diffuse={fmt_vec(active[0]['diffuse'])}")
                if expected_metallic is None:
                    print(
                        "  NOTE: use --metallic to compute the expected ABSDF diffuse/specular; "
                        "raw baseColor equals final diffuse only when metallic=0."
                    )
                else:
                    metallic = max(0.0, min(1.0, expected_metallic))
                    f0 = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2
                    expected_diffuse = tuple(c * (1.0 - metallic) for c in sampled[:3])
                    expected_specular = tuple(
                        f0 * (1.0 - metallic) + c * metallic for c in sampled[:3]
                    )
                    actual_diffuse = active[0]["diffuse"]
                    actual_specular = active[0]["specular"]
                    diffuse_delta = max(
                        abs(actual_diffuse[i] - expected_diffuse[i]) for i in range(3)
                    )
                    specular_delta = max(
                        abs(actual_specular[i] - expected_specular[i]) for i in range(3)
                    )
                    print(f"  expected with metallic={metallic:.9g}: diffuse={fmt_vec(expected_diffuse)}")
                    print(f"  expected with metallic={metallic:.9g}: specular={fmt_vec(expected_specular)}")
                    print(
                        f"  ABSDF delta: diffuseMax={diffuse_delta:.9g} "
                        f"specularMax={specular_delta:.9g} "
                        f"({'PASS' if max(diffuse_delta, specular_delta) < 2e-4 else 'CHECK'})"
                    )
    except Exception as exc:
        print(f"  UV/texture sampling skipped: {exc}")


def compare_results(first: dict[str, Any] | None, second: dict[str, Any] | None) -> None:
    if first is None or second is None:
        return
    a = first["data"]
    b = second["data"]
    print("\n[compare]")
    print(f"  area: {a['area']:.9g} vs {b['area']:.9g}  delta={a['area'] - b['area']:.9g}")
    for li, (la, lb) in enumerate(zip(a["lobes"], b["lobes"])):
        if la["weight"] <= 1e-8 and lb["weight"] <= 1e-8:
            continue
        print(
            f"  lobe[{li}] rough {la['rough']:.9g} vs {lb['rough']:.9g}; "
            f"diffuse {fmt_vec(la['diffuse'])} vs {fmt_vec(lb['diffuse'])}; "
            f"specular {fmt_vec(la['specular'])} vs {fmt_vec(lb['specular'])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="voxelization .bin file")
    parser.add_argument("--cell", nargs=3, type=int, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--level", type=int, default=None,
                        help="tree level to inspect (default: maxDepth / leaf level)")
    parser.add_argument("--merge-dir", type=Path,
                        help="CPU disk pipeline merge directory containing leaves.idx/polygons.dat")
    parser.add_argument("--fbx", type=Path,
                        help="optional FBX used to recover the target triangle UVs")
    parser.add_argument("--texture", type=Path,
                        help="optional BaseColor texture for CPU-equivalent sampling")
    parser.add_argument("--metallic", type=float,
                        help="optional expected metallic value for ABSDF prediction (0 for default MR SimplePlane)")
    parser.add_argument("--compare", type=Path,
                        help="optional second .bin file to compare at the same cell")
    args = parser.parse_args()

    cell = tuple(args.cell)
    try:
        first_file = VoxelFile(args.binary)
        target_level = first_file.max_depth if args.level is None else args.level
        first = print_voxel(first_file, cell, target_level, "first")

        if args.merge_dir is not None:
            polygons = read_polygons(args.merge_dir, target_level, cell)
            print_polygons(
                polygons, target_level, args.fbx, args.texture,
                first["data"] if first else None, first_file, args.metallic
            )

        second = None
        if args.compare is not None:
            second_file = VoxelFile(args.compare)
            second = print_voxel(second_file, cell, target_level, "compare")
        compare_results(first, second)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
