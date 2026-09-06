#!/usr/bin/env python3
"""
extract_dedup.py — 从 GLB 提取去重后的模型，并生成 Falcor voxel scene meta。

用途
====
一个 GLB 的 scene graph 经常是"少量几何 mesh + 大量 node 摆放"。例如
Metropolis_2.glb 只有 1 个 mesh（37 个 primitive），却被 6 个 node 引用，
每个 node 只是 transform 不同。

本脚本做两件事：
1. 按 glTF mesh 去重，把每个唯一 mesh 写成一个独立的 GLB（几何保持原始
   局部坐标，挂一个 identity node）。用 global 模式体素化这些 GLB，每个
   得到一个 .bin。
2. 输出 scene meta（.voxscene.json），记录这些 .bin 如何按原 node 的 world
   transform 摆回场景。

transform 约定
==============
- 提取出的 GLB 用 identity node，几何顶点保持 glb 原始局部坐标。
- 体素化（global 模式）以 mesh 的局部 AABB 中心作为 grid 中心，这个偏移被
  写入 .bin 的 GridData.gridMin。
- 渲染器 RayMarchingPass 内部用 gridMin 把体素 cell 还原到 mesh 局部坐标，
  再乘 scene meta 的 transform 得到世界坐标。

  因此 scene meta 的 transform = "mesh 局部坐标 -> world" 的完整仿射矩阵，
  即原 node 的 world transform，不需要额外补偿"体素 center 为原点"。

- glTF 的 node.matrix 是 column-major；scene meta 要求 row-major
  （VoxelSceneMetadata.h 注释）。本脚本在序列化时做转置。
- glTF 的 node 若用 translation/rotation/scale 而非 matrix，合成顺序为
  M = T * R * S。

用法
====
    python extract_dedup.py <input.glb> --out-dir <dir> [--resolution 512]
"""

import argparse
import json
import shutil
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

GLB_MAGIC = 0x46546C67          # b"glTF"
CHUNK_JSON = 0x4E4F534A         # b"JSON"
CHUNK_BIN = 0x004E4942          # b"BIN\x00"


# --------------------------------------------------------------------------- #
# GLB 读取
# --------------------------------------------------------------------------- #
def load_glb(path):
    """返回 (json_dict, bin_path) 或 (json_dict, None)。bin 数据不读入内存。"""
    with open(path, "rb") as f:
        magic, version, length = struct.unpack("<III", f.read(12))
        if magic != GLB_MAGIC or version != 2:
            raise RuntimeError(f"Not a GLB v2 file: {path}")

        json_dict = None
        bin_path = None
        while f.tell() < length:
            chunk_len, chunk_type = struct.unpack("<II", f.read(8))
            if chunk_type == CHUNK_JSON:
                data = f.read(chunk_len)
                json_dict = json.loads(data.rstrip(b"\x00 \t\r\n").decode("utf-8"))
            elif chunk_type == CHUNK_BIN:
                # 不把 BIN 读进内存，只记录绝对路径偏移，后续 copy 时再 seek。
                bin_path = path
                f.seek(chunk_len, 1)
            else:
                f.seek(chunk_len, 1)

    if json_dict is None:
        raise RuntimeError(f"No JSON chunk in {path}")
    return json_dict, bin_path


def read_bin_chunk(glb_path):
    """读取 glb 的 BIN chunk 全部字节。"""
    with open(glb_path, "rb") as f:
        magic, version, length = struct.unpack("<III", f.read(12))
        while True:
            chunk_len, chunk_type = struct.unpack("<II", f.read(8))
            if chunk_type == CHUNK_BIN:
                return f.read(chunk_len)
            f.seek(chunk_len, 1)


def write_glb(json_dict, bin_path, out_path):
    """写 GLB v2。json_dict 里的 buffer 引用沿用原 BIN 数据。"""
    json_bytes = json.dumps(json_dict, separators=(",", ":")).encode("utf-8")
    # JSON chunk 4 字节对齐
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    json_chunk = json_bytes + b" " * json_pad

    bin_bytes = read_bin_chunk(bin_path) if bin_path is not None else b""
    bin_pad = (4 - (len(bin_bytes) % 4)) % 4
    bin_bytes += b"\x00" * bin_pad

    total_len = 12 + 8 + len(json_chunk) + 8 + len(bin_bytes)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, 2, total_len))
        f.write(struct.pack("<II", len(json_chunk), CHUNK_JSON))
        f.write(json_chunk)
        f.write(struct.pack("<II", len(bin_bytes), CHUNK_BIN))
        f.write(bin_bytes)


# --------------------------------------------------------------------------- #
# transform 数学
# --------------------------------------------------------------------------- #
def quat_to_mat(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),     0],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),     0],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y), 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)


def node_local_matrix(node):
    """glTF node 局部变换 -> 4x4 numpy 矩阵（数学行列，列向量约定）。"""
    if "matrix" in node:
        # column-major 16 floats -> 转置成 numpy 行列矩阵
        m = np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
        return m

    M = np.eye(4, dtype=np.float64)
    # glTF 合成顺序：M = T * R * S
    if "translation" in node:
        M[:3, 3] = node["translation"]
    if "rotation" in node:
        R = quat_to_mat(node["rotation"])
        M = M @ R
    if "scale" in node:
        S = np.diag([*node["scale"], 1.0])
        M = M @ S
    return M


def compute_world_transforms(gltf):
    """返回 {node_index: 4x4 world matrix}。"""
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    default_scene = gltf.get("scene", 0)
    world = {}

    def visit(node_idx, parent_world):
        node = nodes[node_idx]
        local = node_local_matrix(node)
        w = parent_world @ local
        world[node_idx] = w
        for child in node.get("children", []):
            visit(child, w)

    if scenes and default_scene < len(scenes):
        for root in scenes[default_scene].get("nodes", []):
            visit(root, np.eye(4))
    else:
        # 无 scene 定义：遍历所有 node 作为 root
        children_used = set()
        for n in nodes:
            children_used.update(n.get("children", []))
        for i in range(len(nodes)):
            if i not in children_used:
                visit(i, np.eye(4))
    return world


def to_row_major(M):
    """4x4 数学矩阵 -> row-major 16 floats（scene meta transform）。"""
    return [float(M[r, c]) for r in range(4) for c in range(4)]


# --------------------------------------------------------------------------- #
# scene graph 解析
# --------------------------------------------------------------------------- #
def collect_mesh_instances(gltf, world):
    """返回 [(node_index, mesh_index, world_matrix)]，只收集带 mesh 的 node。"""
    nodes = gltf.get("nodes", [])
    out = []
    for ni, node in enumerate(nodes):
        mesh = node.get("mesh")
        if mesh is not None:
            out.append((ni, mesh, world[ni]))
    return out


def sanitize(name):
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(c if c in keep else "_" for c in name) or "unnamed"


def mesh_name(gltf, mesh_index):
    meshes = gltf.get("meshes", [])
    if mesh_index < len(meshes) and meshes[mesh_index].get("name"):
        return sanitize(meshes[mesh_index]["name"])
    return f"mesh_{mesh_index}"


# --------------------------------------------------------------------------- #
# 去重 GLB 提取
# --------------------------------------------------------------------------- #
def extract_mesh_glb(gltf, bin_path, mesh_index, out_path):
    """输出一个只含指定 mesh 的 GLB（identity node）。

    简化策略：保留原 GLB 的所有 mesh/accessor/bufferView/buffer/材质/纹理定义，
    只重写 nodes/scenes，让场景仅包含一个 identity node 指向该 mesh。
    BIN 数据原样复用。对单 mesh 场景零冗余，对多 mesh 场景每个 GLB 会携带
    完整 BIN（冗余但正确）。
    """
    new = dict(gltf)
    new["nodes"] = [{"name": mesh_name(gltf, mesh_index), "mesh": mesh_index}]
    new["scenes"] = [{"nodes": [0]}]
    new["scene"] = 0
    write_glb(new, bin_path, out_path)


# --------------------------------------------------------------------------- #
# scene meta 输出
# --------------------------------------------------------------------------- #
def write_scene_meta(assets, instances, out_path):
    """assets: [(id, voxelFile_relative)]；instances: [(asset_id, row_major_16)]。"""
    doc = {
        "format": "FalcorVoxelScene",
        "version": 2,
        "assets": [
            {"id": aid, "voxelFile": vf} for aid, vf in assets
        ],
        "instances": [
            {
                "id": i,
                "asset": aid,
                "transform": tf,
                "enabled": True,
            }
            for i, (aid, tf) in enumerate(instances)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=4)
        f.write("\n")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="输入 GLB")
    ap.add_argument("--out-dir", required=True, help="输出目录")
    ap.add_argument("--resolution", type=int, default=512, help="体素分辨率（用于打印命令）")
    ap.add_argument("--samples", type=int, default=1024, help="采样频率（用于打印命令）")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gltf, bin_path = load_glb(in_path)
    world = compute_world_transforms(gltf)
    instances = collect_mesh_instances(gltf, world)

    if not instances:
        print("No mesh instances found in scene graph.")
        sys.exit(1)

    # 按 mesh 去重
    mesh_to_nodes = defaultdict(list)
    for ni, mi, w in instances:
        mesh_to_nodes[mi].append((ni, w))

    unique_meshes = sorted(mesh_to_nodes.keys())

    assets = []
    scene_instances = []
    commands = []

    exe = r".\build\Release\LargeSceneVoxelization.exe"

    for mi in unique_meshes:
        name = mesh_name(gltf, mi)
        glb_out = out_dir / f"{name}.glb"
        bin_name = f"{name}.bin"
        bin_out = out_dir / bin_name

        extract_mesh_glb(gltf, bin_path, mi, glb_out)
        assets.append((name, bin_name))

        # 每个引用该 mesh 的 node 是一个 instance，transform = node world transform
        for ni, w in mesh_to_nodes[mi]:
            scene_instances.append((name, to_row_major(w)))

        # 打印体素化命令（global 模式：整份几何烘进一个 bin）
        commands.append(
            f'{exe} "{glb_out}" -o "{bin_out}" '
            f"-r {args.resolution} -s {args.samples} "
            f'--lod-levels 1 --lod-mode approximate --tmp-dir "tmp_{name}" --clean'
        )

    scene_name = in_path.stem or "scene"
    meta_out = out_dir / f"{scene_name}.voxscene.json"
    write_scene_meta(assets, scene_instances, meta_out)

    print(f"\n去重结果：{len(unique_meshes)} 个唯一 mesh，{len(scene_instances)} 个 instance")
    for aid, vf in assets:
        print(f"  asset: {aid}  ->  {vf}")
    print(f"\nscene meta: {meta_out}")
    print("\n批量体素化命令（逐条执行）：")
    for c in commands:
        print(f"\n  {c}")


if __name__ == "__main__":
    main()
