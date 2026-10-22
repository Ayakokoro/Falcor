"""Recording schedule checks; run with Python alone, without Falcor or a GPU."""

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("record_video", ROOT / "scripts/RecordVideo.py")
recording = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recording)


class Pass:
    def __init__(self, kind, properties):
        self.type = kind
        self.properties = properties
        self.samples = []

    def reset(self):
        self.samples.clear()


class Graph:
    def __init__(self):
        self.passes = {
            "AccumulatePass": Pass("AccumulatePass", {"autoReset": True, "maxFrameCount": 1024}),
            "PathTracer": Pass("PathTracer", {"samplesPerPixel": 1, "fixedSeed": 7}),
        }

    def get_pass(self, name):
        return self.passes[name]

    def update_pass(self, name, props):
        self.passes[name] = Pass(self.passes[name].type, props)

    def addPass(self, render_pass, name):
        self.passes[name] = render_pass
        self.export = name

    def remove_pass(self, name):
        del self.passes[name]

    def add_edge(self, source, target):
        self.edge = (source, target)

    def mark_output(self, output):
        self.output = output


class Renderer:
    def __init__(self):
        self.activeGraph = Graph()
        self.scene = SimpleNamespace(camera=SimpleNamespace(position=(4, 3, 2), target=(0, 0, 0), up=(0, 1, 0), animated=True,
                                                           focalLength=21.0, frameHeight=24.0))
        self.clock = SimpleNamespace(time=0, frame=0, framerate=60, startTime=0, endTime=-1,
                                     exitTime=0, exitFrame=0, pause=lambda: None)
        self.ui = True
        self.sceneUpdateCallback = None
        self.frameCapture = SimpleNamespace(outputDir="original", baseFilename="original", captureAllOutputs=True, capture=self.capture)
        self.captures = []
        self.focal_lengths = []
        self.fail_capture = False

    def setActiveGraph(self, graph):
        self.activeGraph = graph

    def resizeFrameBuffer(self, width, height):
        self.size = (width, height)

    def renderFrame(self):
        self.activeGraph.get_pass("AccumulatePass").samples.append((self.clock.frame, self.scene.camera.position))

    def capture(self):
        if self.fail_capture:
            raise RuntimeError("capture failed")
        assert self.frameCapture.captureAllOutputs is False
        self.captures.append(list(self.activeGraph.get_pass("AccumulatePass").samples))
        self.focal_lengths.append(self.scene.camera.focalLength)
        name = f"frame.{self.activeGraph.export}.dst.{self.clock.frame}.png"
        (Path(self.frameCapture.outputDir) / name).write_bytes(b"\x00\x00\x00\x00IEND\xaeB`\x82")


class TestRecordVideo(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "scripts/record_video.example.json").read_text())
        self.work = ROOT / "build" / "record-video-tests" / uuid4().hex
        self.config.update(frame_count=3, fps=2, spp=8, samples_per_render=2, sample_pass="PathTracer",
                           resolution=[64, 48], output_dir=str(self.work), encode_video=False, quit_after=False,
                           camera_orbit=None, camera_fov_y_degrees=None,
                           camera_keyframes=[
                               {"time": 0, "position": [2, 1.5, 3], "target": [.5, .5, .5], "up": [0, 1, 0]},
                               {"time": 4, "position": [3, 1.5, 2], "target": [.5, .5, .5], "up": [0, 1, 0]},
                           ])
        self.falcor = SimpleNamespace(createPass=lambda name, props: Pass(name, props), float3=lambda *v: tuple(v))

    def tearDown(self):
        assert self.work.resolve().parent == (ROOT / "build" / "record-video-tests").resolve()
        if self.work.exists():
            shutil.rmtree(self.work)

    def test_exact_samples_at_each_pose_and_state_restored(self):
        renderer = Renderer()
        old_camera = copy.copy(renderer.scene.camera)
        with patch.dict(sys.modules, {"falcor": self.falcor}):
            output = recording.record_video(renderer, self.config)
        self.assertEqual(sorted(p.name for p in (output / "frames").iterdir()), ["000000.png", "000001.png", "000002.png"])
        self.assertEqual(len(renderer.captures), 3)
        for frame, samples in enumerate(renderer.captures):
            expected_pose = tuple(recording.camera_pose(self.config["camera_keyframes"], frame / 2)[0])
            self.assertEqual(samples, [(frame, expected_pose)] * 4)  # 4 renders x 2 SPP, no warmup samples
        self.assertEqual(renderer.scene.camera, old_camera)
        self.assertEqual(renderer.frameCapture.outputDir, "original")
        self.assertTrue(renderer.ui)
        self.assertTrue(renderer.activeGraph.get_pass("AccumulatePass").properties["autoReset"])
        self.assertEqual(renderer.activeGraph.get_pass("PathTracer").properties["fixedSeed"], 7)
        self.assertEqual(set(renderer.activeGraph.passes), {"AccumulatePass", "PathTracer"})

    def test_capture_failure_restores_graph(self):
        renderer = Renderer()
        renderer.fail_capture = True
        self.config["camera_fov_y_degrees"] = 45
        with patch.dict(sys.modules, {"falcor": self.falcor}):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                recording.record_video(renderer, self.config)
        self.assertTrue(renderer.ui)
        self.assertTrue(renderer.scene.camera.animated)
        self.assertEqual(renderer.scene.camera.focalLength, 21.0)
        self.assertEqual(set(renderer.activeGraph.passes), {"AccumulatePass", "PathTracer"})
        self.assertTrue(renderer.activeGraph.get_pass("AccumulatePass").properties["autoReset"])

    def test_invalid_spp_and_camera(self):
        self.config["spp"] = 7
        with self.assertRaisesRegex(ValueError, "divisible"):
            recording.validate_config(self.config)
        keys = self.config["camera_keyframes"]
        self.assertEqual(recording.camera_pose(keys, -1)[0], keys[0]["position"])
        self.assertEqual(recording.camera_pose(keys, 100)[0], keys[-1]["position"])
        invalid = [{"time": 0, "position": [0, 0, 1], "target": [0, 0, 0], "up": [0, 1, 0]},
                   {"time": 1, "position": [0, 0, -1], "target": [0, 0, 0], "up": [0, 1, 0]}]
        with self.assertRaisesRegex(ValueError, "target"):
            recording.camera_pose(invalid, 0.5)

    def test_console_reentry_creates_one_recording_and_allows_another_run(self):
        renderer = Renderer()
        script = ROOT / "scripts/RecordVideo.py"
        code = compile(script.read_text(), str(script), "exec")
        # Console::flush() in old binaries re-executes the pending m.script()
        # command from the first renderFrame(), in the original Python globals.
        namespace = {"m": renderer, "__name__": "__main__", "__file__": str(script)}
        calls = []
        original_render = renderer.renderFrame

        def nested_render():
            original_render()
            calls.append(1)
            if len(calls) > 50:
                raise AssertionError("Console command recursively started another recording")
            namespace["__file__"] = str(script)
            exec(code, namespace)

        renderer.renderFrame = nested_render
        self.work.mkdir(parents=True)
        config_file = self.work / "input.json"
        config_file.write_text(json.dumps(self.config))
        with patch.dict(sys.modules, {"falcor": self.falcor}), patch.dict(os.environ, {"FALCOR_RECORD_CONFIG": str(config_file)}):
            exec(code, namespace)
            self.assertFalse(namespace["_FALCOR_RECORD_VIDEO_RUNNING"])
            self.assertEqual(len(list(self.work.glob("*/config.json"))), 1)
            self.assertEqual(len(renderer.captures), 3)
            # The guard must be cleared after completion, not permanently disable recording.
            namespace["__file__"] = str(script)
            exec(code, namespace)
            self.assertFalse(namespace["_FALCOR_RECORD_VIDEO_RUNNING"])
        self.assertEqual(len(list(self.work.glob("*/config.json"))), 2)
        self.assertEqual(len(renderer.captures), 6)

    def test_orbit_and_fov_are_applied_to_all_samples_then_restored(self):
        renderer = Renderer()
        orbit = {"target": [0, 1, 0], "reference_position": [0, 2, 2],
                 "elevation_degrees": 60, "radius_scale": 1.0, "turns": 1.0}
        self.config.update(camera_orbit=orbit, camera_keyframes=[], camera_fov_y_degrees=90, frame_count=4)
        with patch.dict(sys.modules, {"falcor": self.falcor}):
            recording.record_video(renderer, self.config)
        for frame, samples in enumerate(renderer.captures):
            position = samples[0][1]
            self.assertAlmostEqual(math.hypot(position[0], position[2]), 2)
            self.assertAlmostEqual(position[1], 1 + 2 * math.sqrt(3))
            self.assertEqual(samples, [(frame, position)] * 4)
            self.assertAlmostEqual(renderer.focal_lengths[frame], 12)  # 24 mm sensor, 90 degree vertical FOV
        self.assertAlmostEqual(renderer.captures[1][0][1][0], 2)  # Quarter turn from +Z towards +X.
        self.assertAlmostEqual(renderer.captures[2][0][1][2], -2)
        self.assertEqual(renderer.scene.camera.focalLength, 21.0)
        self.assertEqual(renderer.scene.camera.position, (4, 3, 2))
        first = recording.orbit_pose(orbit, 0, 2)[0]
        last = recording.orbit_pose(orbit, 2, 2)[0]
        for a, b in zip(first, last):
            self.assertAlmostEqual(a, b)

    def test_invalid_fov_or_orbit_is_rejected_before_rendering(self):
        for fov in (0, 180, float("nan")):
            with self.subTest(fov=fov):
                self.config["camera_fov_y_degrees"] = fov
                with self.assertRaisesRegex(ValueError, "camera_fov_y_degrees"):
                    recording.validate_config(self.config)
        self.config.update(camera_fov_y_degrees=None, camera_keyframes=[],
                           camera_orbit={"target": [0, 0, 0], "reference_position": [0, 2, 0], "elevation_degrees": 60})
        with self.assertRaisesRegex(ValueError, "XZ plane"):
            recording.validate_config(self.config)


if __name__ == "__main__":
    unittest.main()
