"""Exercise both CPU pipelines, the unchanged VPW3 layout and its VBM1 companion."""
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def run(executable, model, output, cache, *options, succeeds=True):
    result = subprocess.run(
        [str(executable), str(model), "-o", str(output), "-r", "2", "-s", "8",
         "--tmp-dir", str(cache), *options], capture_output=True, text=True)
    if (result.returncode == 0) != succeeds:
        raise AssertionError(result.stdout + result.stderr)
    return result


def check_companion(voxel):
    source = voxel.read_bytes()
    data = Path(str(voxel) + ".vbm").read_bytes()
    magic, version, header, stride, resolution, samples, nodes, flags, size, checksum = struct.unpack_from("<8I2Q", data)
    assert (magic, version, header, stride, resolution, samples, flags) == (0x314D4256, 1, 48, 68, 4, 32, 0)
    assert len(data) == header + nodes * stride and size == len(source)
    hashed = 14695981039346656037
    for byte in source:
        hashed = ((hashed ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    assert checksum == hashed, "Companion fingerprint differs from the bytes actually written"

    # VPW3's existing x64 layout. Do not silently change its GPU-shared ABI.
    assert source[-4:] == b"VPW3"
    assert struct.unpack_from("<Q", source, 40)[0] == nodes
    depth = struct.unpack_from("<I", source, 56)[0]
    counts = struct.unpack_from(f"<{depth + 1}I", source, 60)
    assert sum(counts) == nodes and len(counts) > 1
    octree_offset = 60 + 4 * len(counts)
    voxel_offset = octree_offset + nodes * 12
    assert len(source) == voxel_offset + 480 * nodes + 4
    supported = 0
    records = []
    for node in range(nodes):
        _, _, index = struct.unpack_from("<3I", source, octree_offset + 12 * node)
        assert index == node
        *masks, metadata = struct.unpack_from("<17I", data, header + stride * index)
        source_lobes = 0
        for original in range(8):
            weight, nx, ny, nz = struct.unpack_from("<4f", source, voxel_offset + 480 * index + 44 * original)
            assert all(math.isfinite(v) for v in (weight, nx, ny, nz))
            if weight > 0 and nx * nx + ny * ny + nz * nz > 0:
                source_lobes |= 1 << original
        assert metadata & 0xFF == source_lobes, "Lobe compaction differs from the material bake"
        count = source_lobes.bit_count()
        valid = (metadata >> 8) & 0xFF
        assert metadata >> 16 == 0 and valid >> count == 0
        allocated = 0
        for k in range(count):
            begin, end = 32 * k // count, 32 * (k + 1) // count
            bits = ((1 << (end - begin)) - 1) << begin
            if valid & (1 << k):
                allocated |= bits
                supported += 1
                assert any(mask & bits for mask in masks)
        assert all(mask & ~allocated == 0 for mask in masks)
        assert struct.unpack_from("<f", source, voxel_offset + 480 * index + 476)[0] == 0
        records.append((tuple(masks), metadata))
    assert supported >= nodes, "Some populated fixture nodes have no sampled lobe"
    return records


def main():
    executable = Path(sys.argv[1]).resolve()
    work = Path(sys.argv[2]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pipeline-", dir=work) as tmp:
        root = Path(tmp)
        model = root / "cube.obj"
        model.write_text("""v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
vn 0 0 -1
vn 0 0 1
vn 0 -1 0
vn 0 1 0
vn -1 0 0
vn 1 0 0
f 1//1 4//1 3//1 2//1
f 5//2 6//2 7//2 8//2
f 1//3 2//3 6//3 5//3
f 4//4 8//4 7//4 3//4
f 1//5 5//5 8//5 4//5
f 2//6 3//6 7//6 6//6
""", encoding="ascii")
        outputs = []
        for name, options in [("disk1", ["-t", "1"]), ("disk2", ["-t", "2"]),
                              ("memory", ["--in-memory", "-t", "1"])]:
            voxel = root / (name + ".bin")
            run(executable, model, voxel, root / name, "--visibility-bitmask", *options)
            outputs.append(check_companion(voxel))
        for index, name in [(1, "disk2"), (2, "memory")]:
            differing = [i for i, (a, b) in enumerate(zip(outputs[0], outputs[index])) if a != b]
            assert not differing, f"Fixture mask records differ: disk1 vs {name}, nodes {differing}"

        legacy = root / "legacy.bin"
        run(executable, model, legacy, root / "legacy", "--in-memory", "-t", "1")
        assert legacy.read_bytes()[-4:] == b"VPW3" and not Path(str(legacy) + ".vbm").exists()
        old, new = legacy.read_bytes(), (root / "memory.bin").read_bytes()
        node_count = struct.unpack_from("<Q", new, 40)[0]
        data_offset = len(new) - 4 - 480 * node_count
        for node in range(node_count):
            begin = data_offset + 480 * node
            assert old[begin:begin + 476] == new[begin:begin + 476], "Mask bake changed material, ellipsoid or SH data"
        broken = root / "broken.bin"
        Path(str(broken) + ".vbm").mkdir()
        result = run(executable, model, broken, root / "broken", "--in-memory",
                     "--visibility-bitmask", "-t", "1", succeeds=False)
        assert "Cannot open visibility-mask output" in result.stderr
        assert "Voxelization completed" not in result.stdout
    print("Visibility pipeline tests passed (disk 1/2 workers, in-memory, legacy, failed output).")


if __name__ == "__main__":
    main()
