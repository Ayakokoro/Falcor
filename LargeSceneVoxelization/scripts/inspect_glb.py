import json
import struct
import sys
from collections import defaultdict
from pathlib import Path


def load_glb_json(glb_path):
    with open(glb_path, "rb") as f:
        # GLB Header
        header = f.read(12)

        if len(header) != 12:
            raise RuntimeError("Invalid GLB file: header too short")

        magic, version, length = struct.unpack("<4sII", header)

        if magic != b"glTF":
            raise RuntimeError("Not a valid GLB file")

        if version != 2:
            raise RuntimeError(f"Unsupported GLB version: {version}")

        # Read chunks
        while f.tell() < length:
            chunk_header = f.read(8)

            if len(chunk_header) < 8:
                break

            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            chunk_data = f.read(chunk_length)

            # JSON chunk type = 0x4E4F534A ("JSON")
            if chunk_type == 0x4E4F534A:
                text = chunk_data.decode("utf-8").rstrip("\x00 \t\r\n")
                return json.loads(text)

    raise RuntimeError("No JSON chunk found in GLB")


def mesh_name(meshes, mesh_index):
    if mesh_index is None:
        return "-"

    mesh = meshes[mesh_index]

    return mesh.get("name", f"mesh_{mesh_index}")


def node_name(nodes, node_index):
    node = nodes[node_index]
    return node.get("name", f"node_{node_index}")


def print_scene_tree(data):
    nodes = data.get("nodes", [])
    meshes = data.get("meshes", [])
    scenes = data.get("scenes", [])

    if not scenes:
        print("No scenes found.")
        return

    default_scene_index = data.get("scene", 0)

    print("\n" + "=" * 80)
    print("SCENE GRAPH")
    print("=" * 80)

    def recurse(node_index, prefix="", is_last=True):
        node = nodes[node_index]

        branch = "└── " if is_last else "├── "

        name = node.get("name", f"node_{node_index}")
        mesh_index = node.get("mesh")

        extra = ""

        if mesh_index is not None:
            mname = mesh_name(meshes, mesh_index)
            extra = f"  [mesh {mesh_index}: {mname}]"

        print(prefix + branch + f"Node {node_index}: {name}" + extra)

        children = node.get("children", [])

        new_prefix = prefix + ("    " if is_last else "│   ")

        for i, child in enumerate(children):
            recurse(
                child,
                new_prefix,
                i == len(children) - 1
            )

    for scene_index, scene in enumerate(scenes):
        scene_name = scene.get("name", f"scene_{scene_index}")

        marker = " (DEFAULT)" if scene_index == default_scene_index else ""

        print(f"\nScene {scene_index}: {scene_name}{marker}")

        roots = scene.get("nodes", [])

        for i, root in enumerate(roots):
            recurse(
                root,
                "",
                i == len(roots) - 1
            )


def analyze_mesh_sharing(data):
    nodes = data.get("nodes", [])
    meshes = data.get("meshes", [])

    mesh_users = defaultdict(list)

    for node_index, node in enumerate(nodes):
        mesh_index = node.get("mesh")

        if mesh_index is not None:
            mesh_users[mesh_index].append(node_index)

    print("\n" + "=" * 80)
    print("NODE -> MESH")
    print("=" * 80)

    for node_index, node in enumerate(nodes):
        mesh_index = node.get("mesh")

        if mesh_index is None:
            continue

        nname = node.get("name", f"node_{node_index}")
        mname = mesh_name(meshes, mesh_index)

        users = mesh_users[mesh_index]

        shared = ""

        if len(users) > 1:
            shared = f"  <-- SHARED x{len(users)}"

        print(
            f"Node {node_index:4d} "
            f"{nname:<40} "
            f"-> Mesh {mesh_index:4d} "
            f"{mname}"
            f"{shared}"
        )

    print("\n" + "=" * 80)
    print("MESH SHARING SUMMARY")
    print("=" * 80)

    sorted_meshes = sorted(
        mesh_users.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    for mesh_index, users in sorted_meshes:
        mname = mesh_name(meshes, mesh_index)

        print(
            f"Mesh {mesh_index:4d}: "
            f"{mname:<40} "
            f"referenced by {len(users):4d} node(s)"
        )

    shared_meshes = {
        mesh_index: users
        for mesh_index, users in mesh_users.items()
        if len(users) > 1
    }

    print("\n" + "=" * 80)
    print("SHARED / INSTANCED MESHES")
    print("=" * 80)

    if not shared_meshes:
        print("No shared meshes found.")
        print("Each node appears to reference its own mesh.")
        return

    for mesh_index, users in sorted(
        shared_meshes.items(),
        key=lambda x: len(x[1]),
        reverse=True
    ):
        mname = mesh_name(meshes, mesh_index)

        print()
        print(
            f"Mesh {mesh_index}: {mname}"
            f"  -> shared by {len(users)} nodes"
        )

        for node_index in users[:20]:
            print(
                f"    Node {node_index}: "
                f"{node_name(nodes, node_index)}"
            )

        if len(users) > 20:
            print(
                f"    ... and {len(users) - 20} more"
            )


def print_statistics(data):
    nodes = data.get("nodes", [])
    meshes = data.get("meshes", [])
    materials = data.get("materials", [])
    textures = data.get("textures", [])
    images = data.get("images", [])

    nodes_with_mesh = [
        node for node in nodes
        if "mesh" in node
    ]

    referenced_meshes = {
        node["mesh"]
        for node in nodes_with_mesh
    }

    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)

    print(f"Nodes:                {len(nodes)}")
    print(f"Nodes with mesh:      {len(nodes_with_mesh)}")
    print(f"Mesh definitions:     {len(meshes)}")
    print(f"Referenced meshes:    {len(referenced_meshes)}")
    print(f"Materials:            {len(materials)}")
    print(f"Textures:             {len(textures)}")
    print(f"Images:               {len(images)}")

    if nodes_with_mesh:
        ratio = len(nodes_with_mesh) / max(len(referenced_meshes), 1)

        print()
        print(
            "Average nodes per referenced mesh: "
            f"{ratio:.2f}"
        )


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("    python inspect_glb.py your_file.glb")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    try:
        data = load_glb_json(path)

        print(f"\nGLB: {path}")
        print_statistics(data)
        print_scene_tree(data)
        analyze_mesh_sharing(data)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
