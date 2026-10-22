"""Offline video frames for Mogwai. See docs/usage/video-recording.md.

Run with m.script("E:/syc/Falcor/scripts/RecordVideo.py"), or --script.
FALCOR_RECORD_CONFIG selects a JSON config; paths inside it are repository-relative.
No numpy or Python image/video packages are required.
"""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def validate_config(config):
    for field in ("fps", "frame_count", "spp", "samples_per_render"):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if config["spp"] % config["samples_per_render"]:
        raise ValueError("spp must be divisible by samples_per_render (no silent rounding)")
    size = config["resolution"]
    if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("resolution must contain two positive integers")
    if config["encode_video"] and any(v % 2 for v in size):
        raise ValueError("H.264 yuv420p requires an even width and height")
    if type(config["warmup_frames"]) is not int or config["warmup_frames"] < 1:
        raise ValueError("warmup_frames must be at least 1")
    fov = config.get("camera_fov_y_degrees")
    if fov is not None and (not math.isfinite(fov) or not 0 < fov < 180):
        raise ValueError("camera_fov_y_degrees must be between 0 and 180, or null")
    keys = config.get("camera_keyframes", [])
    orbit = config.get("camera_orbit")
    if orbit is not None:
        if keys:
            raise ValueError("Use camera_orbit or camera_keyframes, not both")
        for name in ("target", "reference_position"):
            v = orbit[name]
            if len(v) != 3 or not all(math.isfinite(x) for x in v):
                raise ValueError(f"Orbit {name} must contain three finite numbers")
        elevation = orbit["elevation_degrees"]
        if not math.isfinite(elevation) or not -89 < elevation < 89:
            raise ValueError("Orbit elevation_degrees must be between -89 and 89 (positive looks down)")
        scale = orbit.get("radius_scale", 1.0)
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(orbit.get("turns", 1.0)):
            raise ValueError("Orbit radius_scale must be positive and turns must be finite")
        orbit_pose(orbit, 0, config["frame_count"] / config["fps"])
    last_time = -1.0
    for key in keys:
        t = key["time"]
        if not math.isfinite(t) or t < 0 or t <= last_time:
            raise ValueError("Camera keyframe times must be finite, nonnegative and increasing")
        last_time = t
        for name in ("position", "target", "up"):
            v = key[name]
            if len(v) != 3 or not all(math.isfinite(x) for x in v):
                raise ValueError(f"Camera {name} must contain three finite numbers")
        validate_pose(key["position"], key["target"], key["up"])


def validate_pose(position, target, up):
    direction = [b - a for a, b in zip(position, target)]
    cross = [direction[1] * up[2] - direction[2] * up[1],
             direction[2] * up[0] - direction[0] * up[2],
             direction[0] * up[1] - direction[1] * up[0]]
    if sum(v * v for v in cross) < 1e-12:
        raise ValueError("Camera target must differ from position; up must not be parallel to view")


def camera_pose(keys, seconds):
    """Piecewise linear interpolation; hold the first/last pose outside the range."""
    if seconds <= keys[0]["time"]:
        left = right = keys[0]
    elif seconds >= keys[-1]["time"]:
        left = right = keys[-1]
    else:
        for left, right in zip(keys, keys[1:]):
            if left["time"] <= seconds <= right["time"]:
                break
    alpha = 0 if left is right else (seconds - left["time"]) / (right["time"] - left["time"])
    pose = [[a + (b - a) * alpha for a, b in zip(left[name], right[name])]
            for name in ("position", "target", "up")]
    validate_pose(*pose)
    return pose


def orbit_pose(orbit, seconds, duration):
    """Y-up circular orbit. Elevation is measured above the horizontal plane."""
    target = orbit["target"]
    reference = orbit["reference_position"]
    dx, dz = reference[0] - target[0], reference[2] - target[2]
    radius = math.hypot(dx, dz) * orbit.get("radius_scale", 1.0)
    if radius < 1e-6:
        raise ValueError("Orbit reference_position must differ from target in the XZ plane")
    angle = math.atan2(dx, dz) + math.tau * orbit.get("turns", 1.0) * seconds / duration
    height = radius * math.tan(math.radians(orbit["elevation_degrees"]))
    position = [target[0] + radius * math.sin(angle), target[1] + height, target[2] + radius * math.cos(angle)]
    pose = [position, list(target), [0.0, 1.0, 0.0]]
    validate_pose(*pose)
    return pose


def focal_length_for_fov(frame_height, degrees):
    # Falcor uses sensor height and focal length (both in mm), not a FOV property.
    return frame_height / (2 * math.tan(math.radians(degrees) / 2))


def wait_for_png(path, timeout=120):
    """FrameCapture writes asynchronously. IEND is emitted after all PNG pixels."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with path.open("rb") as stream:
                stream.seek(-12, os.SEEK_END)
                if stream.read() == b"\x00\x00\x00\x00IEND\xaeB`\x82":
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for FrameCapture to finish: {path}")


def ffmpeg_command(config, run_dir, executable):
    return [executable, "-hide_banner", "-n", "-framerate", str(config["fps"]),
            "-start_number", "0", "-i", str(run_dir / "frames" / "%06d.png"),
            "-frames:v", str(config["frame_count"]), "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(run_dir / "video.mp4")]


def record_video(renderer, config):
    from falcor import createPass, float3

    validate_config(config)
    encoder = shutil.which(config["ffmpeg"])
    if config["encode_video"] and not encoder:
        raise RuntimeError("FFmpeg was not found. Set ffmpeg to its full path, or encode_video to false.")

    # m.script executes in Mogwai's shared globals. Keep the recording state local
    # so graph scripts cannot replace it, including their own __file__ assignment.
    if config["graph_script"]:
        renderer.script(str(repo_path(config["graph_script"])))
    if config["scene"]:
        renderer.loadScene(str(repo_path(config["scene"])))
    graph = renderer.getGraph(config["graph_name"]) if config["graph_name"] else renderer.activeGraph
    if graph is None or renderer.scene is None:
        raise RuntimeError("Load a graph and scene first, or specify graph_script and scene in the config.")
    renderer.setActiveGraph(graph)
    accumulator_name = config["accumulation_pass"]
    old_accumulation = dict(graph.get_pass(accumulator_name).properties)
    if graph.get_pass(accumulator_name).type != "AccumulatePass":
        raise ValueError("accumulation_pass must name an AccumulatePass instance")

    # samples_per_render describes the renderer, not an AccumulatePass setting.
    # PathTracer exposes samplesPerPixel; RayMarching currently emits 1 sample.
    sampler_name = config["sample_pass"]
    old_sampler = None
    if sampler_name:
        old_sampler = dict(graph.get_pass(sampler_name).properties)
        if "samplesPerPixel" not in old_sampler:
            raise ValueError(f"{sampler_name} does not expose samplesPerPixel; use sample_pass: null for RayMarching")
    elif config["samples_per_render"] != 1:
        raise ValueError("Set sample_pass to configure samples_per_render greater than 1")

    run_dir = repo_path(config["output_dir"]) / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    frames_dir = run_dir / "frames"
    captures_dir = run_dir / "captures"
    frames_dir.mkdir(parents=True)
    captures_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def report(message):
        # Console output is buffered until the command returns. Keep progress
        # on disk as well, including the stage reached before any native crash.
        with (run_dir / "recording.log").open("a", encoding="utf-8") as log:
            log.write(message + "\n")
        print(message, flush=True)

    report(f"Starting recording: {config['frame_count']} frames, {config['spp']} SPP, {config['fps']} FPS")
    capture = renderer.frameCapture
    old_capture = (capture.outputDir, capture.baseFilename, capture.captureAllOutputs)
    old_ui = renderer.ui
    old_callback = renderer.sceneUpdateCallback
    camera = renderer.scene.camera
    old_camera = (camera.position, camera.target, camera.up, camera.animated)
    old_focal_length = camera.focalLength
    clock = renderer.clock
    clock_fields = ("time", "framerate", "startTime", "endTime", "exitTime", "exitFrame")
    old_clock = {name: getattr(clock, name) for name in clock_fields}
    export_name = "RecordVideo_" + uuid4().hex[:8]
    export_added = False
    try:
        renderer.ui = False
        renderer.sceneUpdateCallback = None
        clock.pause()
        clock.startTime = 0
        clock.endTime = -1
        clock.exitTime = 0
        clock.exitFrame = 0
        clock.framerate = config["fps"]
        clock.frame = 0
        # Without an orbit or keyframes, hold the current camera. Scene animation
        # can still advance once per output frame, but never between its samples.
        camera.animated = False
        renderer.resizeFrameBuffer(*config["resolution"])
        if sampler_name:
            sampling = dict(old_sampler, samplesPerPixel=config["samples_per_render"])
            sampling.pop("fixedSeed", None)  # Repeated identical samples do not increase effective SPP.
            graph.update_pass(sampler_name, sampling)
            actual = graph.get_pass(sampler_name).properties["samplesPerPixel"]
            if actual != config["samples_per_render"]:
                raise ValueError(f"{sampler_name} accepted {actual} samples per render, not the requested value")
        graph.update_pass(accumulator_name, dict(old_accumulation, enabled=True, autoReset=False,
                                                maxFrameCount=0, precisionMode="Single"))
        accumulator = graph.get_pass(accumulator_name)
        # The input must already be tone mapped. This pass only converts linear
        # display RGB to an 8-bit sRGB PNG, also matching the requested size.
        graph.addPass(createPass("BlitPass", {"outputFormat": "RGBA8UnormSrgb"}), export_name)
        export_added = True
        graph.add_edge(config["output"], export_name + ".src")
        graph.mark_output(export_name + ".dst")
        capture.outputDir = str(captures_dir)
        capture.baseFilename = "frame"
        capture.captureAllOutputs = False  # True re-executes the graph and adds a sample!

        def set_camera(frame):
            clock.frame = frame
            pose = None
            if config.get("camera_orbit") is not None:
                pose = orbit_pose(config["camera_orbit"], frame / config["fps"], config["frame_count"] / config["fps"])
            elif config.get("camera_keyframes"):
                pose = camera_pose(config["camera_keyframes"], frame / config["fps"])
            if pose is not None:
                pos, target, up = pose
                camera.position, camera.target, camera.up = float3(*pos), float3(*target), float3(*up)
            if config.get("camera_fov_y_degrees") is not None:
                camera.focalLength = focal_length_for_fov(camera.frameHeight, config["camera_fov_y_degrees"])

        set_camera(0)
        report(f"Warming up: {config['warmup_frames']} renders")
        for _ in range(config["warmup_frames"]):
            renderer.renderFrame()
        renders = config["spp"] // config["samples_per_render"]
        report(f"Saving frames to {frames_dir}")
        for frame in range(config["frame_count"]):
            report(f"Frame {frame + 1}/{config['frame_count']}: accumulating {renders} renders")
            set_camera(frame)
            # Apply scene/camera changes and any history reset before counting samples.
            renderer.renderFrame()
            accumulator.reset()
            for _ in range(renders):
                renderer.renderFrame()
            capture.capture()
            png = captures_dir / f"frame.{export_name}.dst.{clock.frame}.png"
            wait_for_png(png)
            # Copy instead of renaming: the asynchronous writer may not yet
            # have closed its file handle even though IEND is already visible.
            shutil.copyfile(png, frames_dir / f"{frame:06d}.png")
            report(f"Recorded {frame + 1}/{config['frame_count']} ({config['spp']} SPP)")
    except Exception:
        report(traceback.format_exc())
        raise
    finally:
        if export_added:
            graph.remove_pass(export_name)
        graph.update_pass(accumulator_name, old_accumulation)
        if old_sampler is not None:
            graph.update_pass(sampler_name, old_sampler)
        capture.outputDir, capture.baseFilename, capture.captureAllOutputs = old_capture
        camera.position, camera.target, camera.up, camera.animated = old_camera
        camera.focalLength = old_focal_length
        renderer.sceneUpdateCallback = old_callback
        renderer.ui = old_ui
        for name in ("startTime", "endTime", "framerate", "time"):
            setattr(clock, name, old_clock[name])
        if old_clock["exitTime"]:
            clock.exitTime = old_clock["exitTime"]
        else:
            clock.exitFrame = old_clock["exitFrame"]
        # Leave the clock paused so inspecting the recording cannot move the scene.

    command = ffmpeg_command(config, run_dir, encoder or config["ffmpeg"])
    (run_dir / "encode-command.txt").write_text(subprocess.list2cmdline(command), encoding="utf-8")
    if config["encode_video"]:
        # Argument list, never shell=True; paths containing spaces remain safe.
        report("Encoding video; encoder output is in ffmpeg.log")
        with (run_dir / "ffmpeg.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    report(f"Recording complete: {frames_dir}")
    return run_dir


def main(renderer):
    config_path = repo_path(os.environ.get("FALCOR_RECORD_CONFIG", "scripts/record_video.example.json"))
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    record_video(renderer, config)
    if config["quit_after"]:
        from falcor import exit
        exit()


if "m" in globals():
    # Older Mogwai builds keep the console command pending during execution.
    # renderFrame() then runs m.script() again in these SAME globals. Preserve
    # this flag across script reloads, and do not let a nested entry clear it.
    if not globals().get("_FALCOR_RECORD_VIDEO_RUNNING", False):
        _FALCOR_RECORD_VIDEO_RUNNING = True
        try:
            main(m)
        finally:
            _FALCOR_RECORD_VIDEO_RUNNING = False
elif __name__ == "__main__":
    raise SystemExit("Run this script inside Mogwai, not with standalone Python. See docs/usage/video-recording.md.")
