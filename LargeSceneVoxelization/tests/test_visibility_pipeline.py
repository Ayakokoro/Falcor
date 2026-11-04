"""Legacy layout/content regression and opt-in material/mask integration."""
from pathlib import Path
import struct
import subprocess
import sys
import shutil
import uuid
import re
import csv
import math
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


def comparable_memory_legacy(path):
    data, nodes, _, _, _, voxel_start = legacy(path)
    data = bytearray(data)
    # The existing in-memory path leaves Ellipsoid uninitialized for unbaked
    # (area == 0) parents. Compare all defined fields without making allocator
    # contents part of the instrumentation regression contract.
    for node in range(nodes):
        start = voxel_start + node * 420
        if struct.unpack_from('<f', data, start + 176)[0] == 0:
            data[start + 180:start + 228] = bytes(48)
    return data


def check_inspector(inspector, model, path, cache, root):
    """Compare re-analysis with the actual serialized production bake at two LODs."""
    data, _, depth, _, node_start, _ = legacy(path)
    materials = Path(str(path) + '.vmat').read_bytes()
    masks = Path(str(path) + '.vbm').read_bytes()
    for lod in (0, 1):
        output = root / f'inspect-lod{lod}'
        base = [str(inspector), '--scene', str(model), '--bin', str(path),
                '--tmp-dir', str(cache), '--lod', str(lod), '--cell', '0', '0', '0',
                '--visibility-bitmask', '--sample-frequency', '8', '--samples-per-polygon', '16',
                '--spherical-resolution', '8', '--block-count', '4', '--block-size', '4',
                '--ndf-resolution', '8', '--threads', '1']
        run = subprocess.run(base + ['--mode', 'all', '--out', str(output)], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        summary = dict(line.split('=', 1) for line in (output / 'summary.txt').read_text().splitlines() if '=' in line)
        assert summary['materialModel'] == 'visible8' and summary['internalVisibility'] == 'bitmask'
        node = 0
        for _ in range(depth - lod):
            child, mask, _ = struct.unpack_from('<3I', data, node_start + node * 12)
            assert mask & 1, 'Fixture must contain cell 0,0,0'
            node = child
        index = struct.unpack_from('<I', data, node_start + node * 12 + 8)[0]
        for k in range(8):
            fields = struct.unpack_from('<11f', materials, 80 + index * 360 + k * 44)
            observed = []
            for name in ('weight', 'normal', 'roughness', 'diffuse', 'specular'):
                observed.extend(map(float, summary[f'visibleLobe[{k}].{name}'].split(',')))
            assert len(observed) == 11
            assert all(math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-7) for a, b in zip(fields, observed)), (lod, k, fields, observed)
        area, baked = struct.unpack_from('<fI', materials, 80 + index * 360 + 352)
        assert math.isclose(float(summary['visibleArea']), area, rel_tol=2e-6)
        assert int(summary['visibleBaked']) == baked == 1
        words = struct.unpack_from('<17I', masks, 48 + index * 68)
        assert [int(summary[f'mask[{d}]']) for d in range(16)] + [int(summary['mask.metadata'])] == list(words)

        def range_line(file):
            with file.open('rb') as stream:
                assert stream.readline() == b'P5\n'
                return stream.readline()

        for name, reference_suffix in [('splitting_error', 'gt'), ('ndf', 'gt'),
                                        ('spherical_primitive', 'exact'), ('spherical_polygons', 'exact'),
                                        ('spherical_total', 'exact')]:
            assert range_line(output / f'{name}_{reference_suffix}.pgm') == range_line(output / f'{name}_approx.pgm')
        with (output / 'splitting_error.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert all(math.isfinite(float(row[c])) and float(row[c]) >= 0 for row in rows for c in 'rgba')
        assert all(math.isclose(float(row['b']), abs(float(row['r']) - float(row['g'])), abs_tol=2e-6, rel_tol=2e-5) for row in rows)
        off_output = root / f'inspect-off-lod{lod}'
        run = subprocess.run(base + ['--mode', 'splitting', '--internal-visibility', 'off', '--out', str(off_output)], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        with (off_output / 'splitting_error.csv').open() as stream:
            off_rows = list(csv.DictReader(stream))
        assert len(rows) == len(off_rows)
        for masked, off in zip(rows, off_rows):
            assert masked['r'] == off['r'], 'Changing approximation mode changed the reference samples'
            assert float(masked['g']) <= float(off['g']) + 2e-6, 'Bitmask increased the surface response'


def main():
    executable = Path(sys.argv[1]).resolve()
    work = Path(sys.argv[2]).resolve()
    inspector = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
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

        logs = {}

        def run(name, *options, succeeds=True):
            output = root / (name + '.bin')
            command = [str(executable), str(model), '-o', str(output), '-r', '4', '-s', '8',
                       '--tmp-dir', str(root / (name + '-cache')), *options]
            result = subprocess.run(command, capture_output=True, text=True)
            assert (result.returncode == 0) == succeeds, result.stdout + result.stderr
            logs[name] = result.stdout
            return output

        for lod_mode in ['approximate', 'brute-force']:
            options = ['--lod-levels', '1', '--lod-mode', lod_mode, '-t', '1']
            old = run(lod_mode + '-old', *options)
            new = run(lod_mode + '-visible', *options, '--visible-material')
            mask = run(lod_mode + '-mask', *options, '--visibility-bitmask', '--keep-temp')
            assert legacy(old)[0] == legacy(new)[0] == legacy(mask)[0], 'Opt-in bake changed legacy content'
            lods = int(lod_mode == 'brute-force')
            assert extension(new, lods, False)[0] == extension(mask, lods)[0]
            assert not Path(str(old) + '.vmat').exists()
            if inspector and lod_mode == 'brute-force':
                check_inspector(inspector, model, mask, root / (lod_mode + '-mask-cache'), root)

            timed_name = lod_mode + '-timed'
            timed = run(timed_name, *options, '--visibility-bitmask', '--timings')
            assert legacy(timed)[0] == legacy(mask)[0], 'Timing changed legacy bake data'
            assert extension(timed, lods) == extension(mask, lods), 'Timing changed extension payloads'
            assert '[Timing] Load scene wall=' in logs[lod_mode + '-old']
            assert 'summed worker elapsed=' not in logs[lod_mode + '-old'], 'Detailed timing must be opt-in'
            assert 'Clip clip workers' in logs[timed_name]
            if lods:
                for stage in ['Clip LOD 1', 'Merge LOD 1', 'Analyze LOD 1']:
                    assert f'[Timing] {stage} wall=' in logs[timed_name]

            # Check real work accounting per level, not timing thresholds that
            # depend on machine load. Cube leaves and coarse nodes are solid.
            _, _, _, counts, _, _ = legacy(timed)
            totals = re.findall(r'analyzed nodes=(\d+), polygons=(\d+), solid=(\d+), primitive samples=(\d+)', logs[timed_name])
            expected_counts = [counts[-1]] + ([counts[-2]] if lods else [])
            assert [int(row[0]) for row in totals] == expected_counts
            for nodes, polygons, solid, samples in totals:
                assert int(nodes) == int(solid) and int(polygons) >= int(nodes)
                assert int(samples) == int(solid) * 16 * 4 * 8
            for section in logs[timed_name].split('    task totals:')[1:]:
                task_values = re.findall(r'^      .+: ([\d.]+) s \(([\d.]+)%\)', section, re.MULTILINE)
                assert len(task_values) == 17  # 16 disjoint tasks + unaccounted time
                assert all(float(seconds) >= 0 for seconds, _ in task_values)
                assert abs(sum(float(percent) for _, percent in task_values) - 100) < 0.01

        outputs = []
        for name, options in [('disk1', ['-t', '1']), ('disk2', ['-t', '2']),
                              ('memory', ['--in-memory', '-t', '1'])]:
            path = run(name, '--visibility-bitmask', *options)
            outputs.append(extension(path, 0))
        assert outputs[0] == outputs[1] == outputs[2], 'Material/mask bake depends on scheduling or pipeline'

        for name, options in [('disk2', ['-t', '2']), ('memory', ['--in-memory', '-t', '1'])]:
            timed = run(name + '-timed', '--visibility-bitmask', '--timings', *options)
            untimed = root / (name + '.bin')
            if name == 'memory':
                assert comparable_memory_legacy(timed) == comparable_memory_legacy(untimed)
            else:
                assert legacy(timed)[0] == legacy(untimed)[0]
            assert extension(timed, 0) == outputs[0], 'Profiling changed threaded/in-memory results'
            assert 'worker 0: wall=' in logs[name + '-timed']
            assert ('worker 1: wall=' in logs[name + '-timed']) == (name == 'disk2')

        # Replacing a bake in legacy mode must invalidate its optional extensions.
        old = run('memory', '--in-memory', '-t', '1')
        assert not Path(str(old) + '.vmat').exists() and not Path(str(old) + '.vbm').exists()
        Path(str(root / 'broken.bin') + '.vbm').mkdir()
        run('broken', '--visibility-bitmask', succeeds=False)
    print('Visibility pipeline: legacy compatibility, LODs, paired masks, determinism, timing and output failure passed.')
    if inspector:
        print('Inspector pipeline: serialized material/mask parity at two LODs, visibility modes and common image scales passed.')


if __name__ == '__main__':
    main()
