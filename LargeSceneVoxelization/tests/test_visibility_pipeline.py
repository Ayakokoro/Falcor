"""Legacy layout/content regression and opt-in material/mask integration."""
from pathlib import Path
import struct
import subprocess
import sys
import shutil
import uuid
from contextlib import contextmanager


@contextmanager
def scratch_directory(work):
    # Use inherited workspace permissions (Python 3.13's Windows mkdtemp ACL
    # excludes restricted-token build workers).
    root = work / ('compatibility-' + uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        assert root.resolve().parent == work.resolve()
        shutil.rmtree(root)


def fingerprint(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return len(data), value


def legacy(path):
    data = bytearray(path.read_bytes())
    # GridData has four alignment bytes before its size_t; they are not data.
    data[36:40] = bytes(4)
    nodes = struct.unpack_from('<Q', data, 40)[0]
    depth = struct.unpack_from('<I', data, 56)[0]
    counts = struct.unpack_from(f'<{depth + 1}I', data, 60)
    assert sum(counts) == nodes
    node_start = 60 + 4 * len(counts)
    voxel_start = node_start + nodes * 12
    assert len(data) == voxel_start + nodes * 420, 'Legacy 4-lobe ABI changed'
    meta = dict(line.split('=', 1) for line in Path(str(path) + '.meta').read_text().splitlines() if '=' in line)
    assert meta['version'] == '2' and meta['projectionEncoding'] == 'octahedral'
    return bytes(data), nodes, depth, counts, node_start, voxel_start


def extension(path, expected_lods, with_masks=True):
    _, nodes, depth, counts, node_start, _ = legacy(path)
    source = path.read_bytes()
    material = Path(str(path) + '.vmat').read_bytes()
    h = struct.unpack_from('<8I4Q4I', material)
    assert h[:8] == (0x31544d56, 1, 80, 360, nodes, 8, depth, expected_lods)
    assert h[8:10] == fingerprint(source) and h[12:] == (0, 0, 0, 0)
    assert len(material) == 80 + nodes * 360
    masks = Path(str(path) + '.vbm').read_bytes() if with_masks else None
    if masks:
        assert h[10:12] == fingerprint(masks), 'Material/mask pairing differs'
        mh = struct.unpack_from('<8I2Q', masks)
        assert mh[:8] == (0x314d4256, 1, 48, 68, 4, 32, nodes, 0)
        assert mh[8:] == fingerprint(source)
        assert len(masks) == 48 + 68 * nodes
    else:
        assert h[10:12] == (0, 0)
    first = 0
    for level, count in enumerate(counts):
        for node in range(first, first + count):
            index = struct.unpack_from('<I', source, node_start + node * 12 + 8)[0]
            record = 80 + index * 360
            baked = struct.unpack_from('<I', material, record + 356)[0]
            assert baked == int(level >= depth - expected_lods), 'Unsupported LOD advertised as baked'
            source_lobes = 0
            for k in range(8):
                w, nx, ny, nz = struct.unpack_from('<4f', material, record + k * 44)
                if w > 0 and nx*nx + ny*ny + nz*nz > 0:
                    source_lobes |= 1 << k
            if masks:
                *words, metadata = struct.unpack_from('<17I', masks, 48 + index * 68)
                assert metadata & 255 == source_lobes
                if not baked:
                    assert metadata == 0 and not any(words)
        first += count
    return material[80:], masks[48:] if masks else None


def main():
    executable = Path(sys.argv[1]).resolve()
    work = Path(sys.argv[2]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    with scratch_directory(work) as temp:
        root = Path(temp)
        model = root / 'cube.obj'
        model.write_text('''v 0 0 0
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
''', encoding='ascii')

        def run(name, *options, succeeds=True):
            output = root / (name + '.bin')
            command = [str(executable), str(model), '-o', str(output), '-r', '4', '-s', '8',
                       '--tmp-dir', str(root / (name + '-cache')), *options]
            result = subprocess.run(command, capture_output=True, text=True)
            assert (result.returncode == 0) == succeeds, result.stdout + result.stderr
            return output

        for lod_mode in ['approximate', 'brute-force']:
            options = ['--lod-levels', '1', '--lod-mode', lod_mode, '-t', '1']
            old = run(lod_mode + '-old', *options)
            new = run(lod_mode + '-visible', *options, '--visible-material')
            mask = run(lod_mode + '-mask', *options, '--visibility-bitmask')
            assert legacy(old)[0] == legacy(new)[0] == legacy(mask)[0], 'Opt-in bake changed legacy content'
            lods = int(lod_mode == 'brute-force')
            assert extension(new, lods, False)[0] == extension(mask, lods)[0]
            assert not Path(str(old) + '.vmat').exists()

        outputs = []
        for name, options in [('disk1', ['-t', '1']), ('disk2', ['-t', '2']),
                              ('memory', ['--in-memory', '-t', '1'])]:
            path = run(name, '--visibility-bitmask', *options)
            outputs.append(extension(path, 0))
        assert outputs[0] == outputs[1] == outputs[2], 'Material/mask bake depends on scheduling or pipeline'

        # Replacing a bake in legacy mode must invalidate its optional extensions.
        old = run('memory', '--in-memory', '-t', '1')
        assert not Path(str(old) + '.vmat').exists() and not Path(str(old) + '.vbm').exists()
        Path(str(root / 'broken.bin') + '.vbm').mkdir()
        run('broken', '--visibility-bitmask', succeeds=False)
    print('Visibility pipeline: legacy compatibility, LODs, paired masks, determinism and output failure passed.')


if __name__ == '__main__':
    main()
