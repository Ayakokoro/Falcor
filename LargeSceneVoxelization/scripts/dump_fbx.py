#!/usr/bin/env python3
"""FBX scene dumper -- prints comprehensive scene summary, node tree, meshes,
materials, textures, and instance analysis using pyassimp."""

import sys
import os
import math
from collections import defaultdict

# ---- Ensure assimp DLL directory is on PATH before importing pyassimp ----
_DLL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "Release")
if os.path.isdir(_DLL_DIR):
    os.environ['PATH'] = _DLL_DIR + os.pathsep + os.environ.get('PATH', '')

import pyassimp
import numpy as np


# ---------------------------------------------------------------------------
# Output encoding -- force UTF-8 on Windows
# ---------------------------------------------------------------------------
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP = "=" * 72
SUB = "-" * 72


def indent(level):
    return "  " * level


def mat4_to_trs(m):
    """Decompose a 4x4 homogeneous matrix into translation, rotation (euler, deg),
    and scale.  Returns (tx, ty, tz), (rx, ry, rz), (sx, sy, sz)."""
    t = (float(m[0][3]), float(m[1][3]), float(m[2][3]))

    # Scale per axis = length of each basis vector
    sx = float(np.linalg.norm(m[0, :3]))
    sy = float(np.linalg.norm(m[1, :3]))
    sz = float(np.linalg.norm(m[2, :3]))

    # Rotation from upper-left 3x3, normalized by scale
    eps = 1e-10
    r00, r01, r02 = m[0, 0] / (sx + eps), m[0, 1] / (sx + eps), m[0, 2] / (sx + eps)
    r10, r11, r12 = m[1, 0] / (sy + eps), m[1, 1] / (sy + eps), m[1, 2] / (sy + eps)
    r20, r21, r22 = m[2, 0] / (sz + eps), m[2, 1] / (sz + eps), m[2, 2] / (sz + eps)

    # Extract euler angles (ZYX order, degrees)
    if r20 < 1.0 - 1e-6:
        if r20 > -1.0 + 1e-6:
            ry = -math.asin(r20)
            rx = math.atan2(r21 / math.cos(ry), r22 / math.cos(ry))
            rz = math.atan2(r10 / math.cos(ry), r00 / math.cos(ry))
        else:
            ry = math.pi / 2
            rx = math.atan2(-r12, r11)
            rz = 0.0
    else:
        ry = -math.pi / 2
        rx = math.atan2(-r12, r11)
        rz = 0.0

    rx_d = math.degrees(rx)
    ry_d = math.degrees(ry)
    rz_d = math.degrees(rz)

    return t, (rx_d, ry_d, rz_d), (sx, sy, sz)


def basename(path):
    return os.path.basename(path) if path else ""


# ---------------------------------------------------------------------------
# Scene statistics collectors
# ---------------------------------------------------------------------------

def collect_mesh_instances(scene):
    """Walk all nodes and return dict: mesh_index -> [(node_path, node)].
    Also returns list of all (node_path, node) tuples.
    A mesh 'instance' is any node that references a mesh."""
    # Build identity -> index mapping (node.meshes contains LP_Mesh objects)
    mesh_id_to_idx = {id(m): i for i, m in enumerate(scene.meshes)}

    mesh_to_nodes = defaultdict(list)
    all_nodes = []

    def walk(node, path_parts):
        node_path = "/".join(path_parts)
        all_nodes.append((node_path, node))
        for m in node.meshes:
            mesh_idx = mesh_id_to_idx.get(id(m))
            if mesh_idx is not None:
                mesh_to_nodes[mesh_idx].append((node_path, node))
        for child in node.children:
            walk(child, path_parts + [child.name or "(unnamed)"])

    walk(scene.rootnode, [scene.rootnode.name or "RootNode"])
    return mesh_to_nodes, all_nodes


def count_nodes(scene):
    """Count total nodes and nodes with mesh references."""

    def walk(node):
        total = 1
        with_mesh = 1 if len(node.meshes) > 0 else 0
        for child in node.children:
            t, m = walk(child)
            total += t
            with_mesh += m
        return total, with_mesh

    return walk(scene.rootnode)


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------

def print_summary(scene, mesh_instances, total_nodes):
    """Print top-level scene summary."""
    n_meshes = len(scene.meshes)
    n_materials = len(scene.materials)
    n_animations = len(scene.animations) if scene.animations else 0
    total_verts = sum(len(m.vertices) for m in scene.meshes)
    total_faces = sum(len(m.faces) for m in scene.meshes)

    total_nodes_count, nodes_with_mesh = total_nodes
    leaf_meshes = [i for i, nodes in mesh_instances.items() if len(nodes) == 1]
    shared_meshes = [i for i, nodes in mesh_instances.items() if len(nodes) > 1]
    total_instances = sum(len(v) for v in mesh_instances.values())

    print(SEP)
    print("  SCENE SUMMARY")
    print(SEP)

    print(f"\n  File info:")
    print(f"    Meshes:        {n_meshes}")
    print(f"    Total verts:   {total_verts:,}")
    print(f"    Total faces:   {total_faces:,}")
    print(f"    Materials:     {n_materials}")
    print(f"    Animations:    {n_animations}")

    print(f"\n  Node hierarchy:")
    print(f"    Total nodes:            {total_nodes_count}")
    print(f"    Nodes with mesh ref:    {nodes_with_mesh}")
    print(f"    Nodes without mesh:     {total_nodes_count - nodes_with_mesh}  (transform / group / camera / light)")

    print(f"\n  Mesh instances:")
    print(f"    Unique meshes:          {n_meshes}")
    print(f"    Mesh references (inst): {total_instances}")
    print(f"    Single-use meshes:      {len(leaf_meshes)}")
    print(f"    Shared meshes:          {len(shared_meshes)}  "
          f"(used by {sum(len(mesh_instances[i]) for i in shared_meshes)} nodes total)")
    if total_instances > 0:
        print(f"    Avg instances / mesh:   {total_instances / n_meshes:.1f}")

    if shared_meshes:
        print(f"\n  Top shared meshes:")
        sorted_shared = sorted(shared_meshes, key=lambda i: len(mesh_instances[i]), reverse=True)
        for mi in sorted_shared[:10]:
            nodes = mesh_instances[mi]
            mesh_name = scene.meshes[mi].name or "(unnamed)"
            print(f"    [{mi}] \"{mesh_name}\"  --  {len(nodes)} instances")
        if len(sorted_shared) > 10:
            print(f"    ... and {len(sorted_shared) - 10} more shared meshes")

    # Unreferenced meshes
    all_mesh_indices = set(range(n_meshes))
    referenced = set(mesh_instances.keys())
    unreferenced = all_mesh_indices - referenced
    if unreferenced:
        print(f"\n  Unreferenced meshes: {len(unreferenced)}")
        for mi in sorted(unreferenced)[:10]:
            print(f"    [{mi}] \"{scene.meshes[mi].name or '(unnamed)'}\"")
        if len(unreferenced) > 10:
            print(f"    ... and {len(unreferenced) - 10} more")


def print_texture_inventory(scene):
    """Collect and categorize all texture paths across materials."""
    textures_by_type = defaultdict(set)  # type -> set of filepath
    all_texture_paths = set()

    type_map = {
        'DiffuseColor':      'BaseColor / Diffuse',
        'NormalMap':         'Normal',
        'ReflectionFactor':  'Metallic',
        'ShininessExponent': 'Roughness',
        'EmissiveColor':     'Emission / Emissive',
        'Opacity':           'Opacity',
        'SpecularColor':     'Specular',
    }

    for mat in scene.materials:
        for key, val in mat.properties.items():
            if key.endswith('|file') and val:
                tex_type = key.split('|')[0]
                display_type = type_map.get(tex_type, tex_type)
                textures_by_type[display_type].add(val)
                all_texture_paths.add(val)

    print(f"\n{SUB}")
    print(f"  TEXTURE INVENTORY")
    print(f"{SUB}")

    if not all_texture_paths:
        print(f"\n  No external textures found (may be embedded or missing).")
        return

    print(f"\n  Unique texture files: {len(all_texture_paths)}")
    print(f"  Texture types found:  {len(textures_by_type)}")

    for tex_type in sorted(textures_by_type.keys()):
        entries = textures_by_type[tex_type]
        unique_files = sorted(set(entries))
        print(f"\n  [{tex_type}]  ({len(unique_files)} unique, {len(entries)} total refs)")
        for f in unique_files[:8]:
            print(f"    {f}")
        if len(unique_files) > 8:
            print(f"    ... and {len(unique_files) - 8} more")

    # Texture directory summary
    dirs = defaultdict(int)
    for p in all_texture_paths:
        d = os.path.dirname(p)
        dirs[d] += 1
    if dirs:
        print(f"\n  Texture source directories:")
        for d, count in sorted(dirs.items(), key=lambda x: -x[1]):
            print(f"    [{count:4d}] {d}")


def print_node_tree(scene):
    """Print scene node hierarchy with transforms and mesh refs."""
    print(f"\n{SUB}")
    print(f"  SCENE NODE TREE")
    print(f"{SUB}\n")

    def walk(node, depth=0):
        name = node.name or "(unnamed)"
        n_children = len(node.children)
        n_meshes = len(node.meshes)
        is_identity = np.allclose(node.transformation, np.eye(4), atol=1e-6)

        # Build branch prefix
        branch = "  " * depth

        # Mesh info
        mesh_info = ""
        if n_meshes > 0:
            parts = []
            for m in node.meshes:
                mn = m.name or f"mesh?"
                nv = len(m.vertices)
                parts.append(f'"{mn}" ({nv:,}v)')
            if len(parts) <= 3:
                mesh_info = "  -> " + ", ".join(parts)
            else:
                mesh_info = "  -> " + ", ".join(parts[:3]) + f", ... ({len(parts)} total)"

        has_trs = False
        if not is_identity:
            t, r, s = mat4_to_trs(node.transformation)
            has_t = not np.allclose([t[0], t[1], t[2]], [0, 0, 0], atol=1e-4)
            has_s = not np.allclose([s[0], s[1], s[2]], [1, 1, 1], atol=1e-4)
            has_r = not np.allclose([r[0], r[1], r[2]], [0, 0, 0], atol=1e-2)
            has_trs = has_t or has_s or has_r

        transform_str = ""
        if has_trs:
            t, r, s = mat4_to_trs(node.transformation)
            trs_parts = []
            if not np.allclose([t[0], t[1], t[2]], [0, 0, 0], atol=1e-4):
                trs_parts.append(f"T=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")
            if not np.allclose([s[0], s[1], s[2]], [1, 1, 1], atol=1e-4):
                trs_parts.append(f"S=({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})")
            if not np.allclose([r[0], r[1], r[2]], [0, 0, 0], atol=1e-2):
                trs_parts.append(f"R=({r[0]:.1f}deg, {r[1]:.1f}deg, {r[2]:.1f}deg)")
            transform_str = "  [" + ", ".join(trs_parts) + "]"

        print(f"{branch}{name}  (children={n_children}){mesh_info}{transform_str}")

        for child in node.children:
            walk(child, depth + 1)

    walk(scene.rootnode)


def print_meshes(scene, mesh_instances):
    """Print per-mesh summary table."""
    print(f"\n{SUB}")
    print(f"  MESH LIST  ({len(scene.meshes)} meshes)")
    print(f"{SUB}\n")

    header = f"  {'Idx':>4s}  {'Name':<45s}  {'Verts':>8s}  {'Faces':>8s}  {'Mat':>3s}  {'Feat':>6s}  {'Inst':>4s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, mesh in enumerate(scene.meshes):
        name = (mesh.name or "(unnamed)")[:45]
        nv = len(mesh.vertices)
        nf = len(mesh.faces)
        has_n = mesh.normals is not None and len(mesh.normals) > 0
        has_uv = mesh.texturecoords is not None and len(mesh.texturecoords) > 0
        features = ("N" if has_n else "") + ("+" if has_n and has_uv else "") + ("UV" if has_uv else "") or "-"
        n_inst = len(mesh_instances.get(i, []))

        print(f"  {i:4d}  {name:<45s}  {nv:>8,d}  {nf:>8,d}  {mesh.materialindex:>3d}  {features:>6s}  {n_inst:>4d}")


def print_materials(scene):
    """Print material list with texture paths."""
    print(f"\n{SUB}")
    print(f"  MATERIALS  ({len(scene.materials)} materials)")
    print(f"{SUB}\n")

    slot_labels = {
        'DiffuseColor':      'BaseColor',
        'NormalMap':         'Normal',
        'ReflectionFactor':  'Metallic',
        'ShininessExponent': 'Roughness',
        'EmissiveColor':     'Emission',
    }

    for i, mat in enumerate(scene.materials):
        props = mat.properties
        name = props.get('name', '(unnamed)')
        print(f"  [{i}] \"{name}\"")

        # Collect texture slots
        tex_slots = defaultdict(list)
        for key, val in props.items():
            if key.endswith('|file') and val:
                slot = key.split('|')[0]
                tex_slots[slot].append(val)

        if 'diffuse' in props:
            print(f"      diffuse color:  {props['diffuse']}")

        for slot, label in slot_labels.items():
            if slot in tex_slots:
                for tp in tex_slots[slot]:
                    print(f"      tex_{label:<12s} {tp}")

        # Other slots
        for slot, paths in sorted(tex_slots.items()):
            if slot not in slot_labels:
                for tp in paths:
                    print(f"      tex_{slot:<14s} {tp}")

        if i < len(scene.materials) - 1:
            print()


def print_instance_transforms(scene, mesh_instances):
    """Print transforms for each node that references a shared mesh."""
    print(f"\n{SUB}")
    print(f"  INSTANCE TRANSFORMS  (only shared meshes with >= 2 instances)")
    print(f"{SUB}\n")

    shared = [(mi, nodes) for mi, nodes in mesh_instances.items() if len(nodes) >= 2]
    shared.sort(key=lambda x: -len(x[1]))

    count = 0
    for mesh_idx, nodes in shared:
        if count >= 30:
            remaining = len(shared) - 30
            if remaining > 0:
                print(f"  ... {remaining} more shared meshes skipped (use --instances for full list)")
            break
        mesh_name = scene.meshes[mesh_idx].name or "(unnamed)"
        print(f"  [{mesh_idx}] \"{mesh_name}\"  ({len(nodes)} instances)")
        for node_path, node in nodes[:5]:
            t, r, s = mat4_to_trs(node.transformation)
            print(f"      {node_path}")
            print(f"        T=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})  "
                  f"R=({r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f})  "
                  f"S=({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})")
        if len(nodes) > 5:
            print(f"      ... and {len(nodes) - 5} more instances")
        print()
        count += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <file.fbx> [options]")
        print()
        print("  Sections (default: --summary + --textures):")
        print("    --summary       Scene overview + instance statistics")
        print("    --tree          Scene node hierarchy with transforms")
        print("    --meshes        Per-mesh detail table")
        print("    --materials     Material list with texture paths")
        print("    --textures      Texture inventory by type")
        print("    --instances     Per-mesh instance transforms (shared meshes only)")
        print("    --all           Everything")
        sys.exit(1)

    path = sys.argv[1]
    flags = sys.argv[2:]

    # Default: summary + textures only (fast overview)
    show_all = '--all' in flags
    any_section_flag = any(f in flags for f in
        ['--summary', '--tree', '--meshes', '--materials', '--textures', '--instances'])

    if not any_section_flag:
        show_summary   = True
        show_textures  = True
        show_tree      = False
        show_meshes    = False
        show_materials = False
        show_instances = False
    else:
        show_summary   = show_all or '--summary'   in flags
        show_tree      = show_all or '--tree'      in flags
        show_meshes    = show_all or '--meshes'    in flags
        show_materials = show_all or '--materials' in flags
        show_textures  = show_all or '--textures'  in flags
        show_instances = show_all or '--instances' in flags

    print(f"File:   {path}")
    print(f"Size:   {os.path.getsize(path):,} bytes")
    print(f"Format: {os.path.splitext(path)[1]}")

    with pyassimp.load(path) as scene:
        # Collect data
        mesh_instances, _all_nodes = collect_mesh_instances(scene)
        total_nodes = count_nodes(scene)

        print(f"Meshes: {len(scene.meshes)}")
        print(f"Materials: {len(scene.materials)}")
        print(f"Textures: {len(scene.textures) if scene.textures else 0}")
        print(f"Animations: {len(scene.animations) if scene.animations else 0}")

        if show_summary:
            print_summary(scene, mesh_instances, total_nodes)
        if show_textures:
            print_texture_inventory(scene)
        if show_tree:
            print_node_tree(scene)
        if show_meshes:
            print_meshes(scene, mesh_instances)
        if show_materials:
            print_materials(scene)
        if show_instances:
            print_instance_transforms(scene, mesh_instances)


if __name__ == '__main__':
    main()
