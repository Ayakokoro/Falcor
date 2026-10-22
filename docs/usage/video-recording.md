# 按指定 SPP 录制视频

使用 [`scripts/RecordVideo.py`](../../scripts/RecordVideo.py) 在 Mogwai 中逐帧渲染，保存 PNG 序列，可选调用 FFmpeg 编码 MP4。不需要重新编译 Falcor，也不依赖 numpy、Pillow 或 Python 视频库。

## 在已经调好的场景上录制

1. 加载场景和 graph，例如 `scripts/SimplyRendering.py`。在 RayMarching 的界面中选好体素文件、显示模式、bounce 和其他参数，确认画面正常。
2. 编辑 [`scripts/record_video.example.json`](../../scripts/record_video.example.json)。其中 `graph_script`、`graph_name`、`scene` 保持 `null`，表示使用当前场景及当前 graph，不重建 RayMarching pass。
3. 配置相机。当前示例与 `visible_material_experiment` 分支一致：以 `[0, 0.5, 0]` 为中心，采用 30° 俯角环绕，FOV 沿用场景相机；输出 512×512、256 SPP、60 FPS、120 帧。其他场景需要修改 `camera_orbit` 的坐标。设 `camera_orbit: null` 可以改用 `camera_keyframes`；两者分别为 `null` 和 `[]` 时保持当前相机位置。
4. 在 Mogwai 的 Python 控制台中执行：

```python
m.script("E:/syc/Falcor/scripts/RecordVideo.py")
```

相机位置可以在控制台查看，也可以用 Scene 面板的 `Add Viewpoint` / `Save Viewpoints` 收集位置后填入 JSON：

```python
print(m.scene.camera.position)
print(m.scene.camera.target)
print(m.scene.camera.up)
```

录制期间 UI 隐藏，相机与时间由脚本控制。结束或异常时恢复累积 pass 的属性、相机位置和焦距、UI 和截图配置；窗口保留录制分辨率，时钟保持暂停。加载的新 graph/场景也保留。默认 `quit_after: false`，不会退出 Mogwai。

脚本带有防重入保护，可用于尚未修复控制台命令队列的旧版 Mogwai。如果之前从控制台执行后立即关闭、并在同一秒生成大量空目录，这是 `renderFrame()` 重复执行待处理控制台命令导致的递归问题。更新脚本后可以直接重试，无需先重新编译；`Console.cpp` 中的命令队列修复会在下次编译 Falcor 后生效。

## 配置

所有配置字段见示例 JSON；文件路径可以是绝对路径，相对路径均相对于仓库根目录，不受 Mogwai 可执行文件所在目录影响。

| 字段 | 含义 |
| --- | --- |
| `graph_script` | 需要加载的 graph `.py`；`null` 使用当前 graph |
| `graph_name` | 选择指定 graph 名称；`null` 使用 active graph |
| `scene` | 场景文件路径；`null` 保留当前场景 |
| `output` | 视频来源输出，例如 `ToneMapper.dst`，使用 graph 中的实例名称 |
| `accumulation_pass` | 累积 pass 的实例名称，通常为 `AccumulatePass` |
| `sample_pass` | 支持 `samplesPerPixel` 的采样 pass 实例名，例如 `PathTracer`；RayMarching 填 `null` |
| `samples_per_render` | 一次渲染的 SPP；当前 RayMarching 填 `1` |
| `spp` | 每个输出视频帧的总 SPP，必须整除 `samples_per_render` |
| `fps` / `frame_count` | 视频帧率 / 总帧数；时长为 `frame_count / fps` 秒 |
| `resolution` | 最终输出宽高。固定分辨率的上游 pass 会被缩放；要改变 RayMarching 实际采样分辨率，也需调整其界面设置 |
| `warmup_frames` | 开始前预热次数，不计入最终 SPP；默认 4 |
| `camera_fov_y_degrees` | 垂直 FOV，单位为度，范围 `(0, 180)`；`null` 或省略时沿用当前相机焦距 |
| `camera_orbit` | 圆周环绕参数；设为 `null` 或省略则使用关键帧 |
| `camera_keyframes` | 时间（秒）、位置、目标、up；按时间递增，分段线性插值，范围外保持端点 |
| `output_dir` | 输出父目录；每次创建独立子目录，保留已有录制 |
| `encode_video` | `true` 时导出完成后自动调用 FFmpeg |
| `ffmpeg` | FFmpeg 可执行文件名或完整路径；需支持 `libx264` |
| `quit_after` | 成功完成后是否退出 Mogwai；无界面运行建议 `true` |

例如，每个视频帧 256 SPP、30 FPS、10 秒：

```json
"spp": 256,
"samples_per_render": 1,
"fps": 30,
"frame_count": 300
```

相机最后一个关键帧可设在 10 秒；实际采样时刻为 `0, 1/30, ..., 299/30`。相机关键帧采用位置/目标/up 的线性插值，不含运动模糊。已有的相机动画和 `sceneUpdateCallback` 在录制期间关闭；内置场景动画仍随视频帧时间推进，在同一视频帧的所有采样期间保持静止。

### 环绕俯视和 FOV

FOV 控制镜头取景范围，俯角控制相机从多高的方向看向目标，两者独立。下面是另一组采用 Greasewood 原始 target、60° 俯角和 60° FOV 的配置示例，可替换默认参数：

```json
"camera_fov_y_degrees": 60.0,
"camera_orbit": {
  "target": [-0.338859528, 1.01085567, 0.890973508],
  "reference_position": [-0.638005257, 1.40110111, 1.7617321],
  "elevation_degrees": 60.0,
  "radius_scale": 1.0,
  "turns": 1.0
},
"camera_keyframes": []
```

- `target`：环绕中心，也是镜头始终看向的位置。假定世界 Y 轴向上。
- `reference_position`：用它相对 target 的 XZ 偏移确定起始方向和水平半径。Y 坐标由俯角重新计算。
- `elevation_degrees`：高于水平面的角度；60 表示向下看 60°。正值越大越接近俯视，必须小于 89°，以避免 up 与视线重合。
- `radius_scale`：水平半径倍率；增大后相机会更远，高度也会相应增加以保持俯角。
- `turns`：整个视频转几圈；1 为一圈，-1 为反方向一圈，0 为固定方向。
- `camera_fov_y_degrees`：改成 40 会缩小取景范围，让主体更大；改成 75 会扩大取景范围。不会修改视频分辨率。

这组 Greasewood 参数的水平半径约为 `0.920711`，起始相机位置约为 `(-0.638005, 2.605574, 1.761732)`，始终看向给定的 target。默认一圈按整个视频时长均匀分配：120 帧 / 60 FPS 是 2 秒一圈；若希望 10 秒一圈，可将 `frame_count` 改为 600。轨迹按圆周公式计算，不是折线插值；结尾不重复首帧，便于循环播放。

Falcor 的 Python 相机接口使用 `focalLength`，脚本按 `frameHeight / (2 * tan(FOV_Y / 2))` 换算。若要先在 Mogwai 控制台单独预览 FOV，可以执行：

```python
import math
m.scene.camera.focalLength = m.scene.camera.frameHeight / (2 * math.tan(math.radians(60) / 2))
```

每个视频帧先渲染一次以应用相机/场景变化，再 `reset()` 累积，然后渲染 `spp / samples_per_render` 次，最后截图一次。预热和应用变化的那一次不计入最终结果。PathTracer 的 `fixedSeed` 在录制期间去掉，以免反复累积同一组样本；结束后恢复原属性。

**`output` 必须在累积 pass 下游**，例如当前脚本中的 `ToneMapper.dst`。若选 `RayMarchingPass.color` 或 `PathTracer.color`，输出会绕过累积，不能得到配置的总 SPP。输出按已完成 tone mapping 的线性 RGB 处理，再转成 8-bit sRGB PNG；直接输出原始 HDR 会被截断，不适合用这个脚本保存 HDR 数据。

## 从命令行加载 graph 并录制

复制示例 JSON，例如保存到 `scripts/my_recording.json`，设置 `graph_script`、`scene`、相机关键帧及 `quit_after: true`。PathTracer 示例可填：

```json
"graph_script": "scripts/PathTracer.py",
"scene": "Scene/your_scene.pyscene",
"sample_pass": "PathTracer",
"samples_per_render": 1
```

在仓库根目录执行 PowerShell：

```powershell
$env:FALCOR_RECORD_CONFIG = "scripts/my_recording.json"
.\build\windows-ninja-msvc\bin\Release\Mogwai.exe --headless --script scripts/RecordVideo.py
```

也可以在已打开的 Mogwai 控制台指定配置：

```python
import os
os.environ["FALCOR_RECORD_CONFIG"] = "scripts/my_recording.json"
m.script("E:/syc/Falcor/scripts/RecordVideo.py")
```

当前 `voxelization_with_diskio` 分支的 `RayMarchingPass` 支持 `voxelFile` 脚本属性。交互录制可直接使用当前调好的 graph；无人值守运行时，应在自定义 graph 脚本中创建 pass 时指定体素文件和显示模式，例如 `createPass("RayMarchingPass", {"voxelFile": "E:/syc/Falcor/resource/scene.bin", "drawMode": 0, "maxBounce": 1})`，再将此 graph 脚本填入 `graph_script`。`scripts/SimplyRendering.py` 本身没有预设体素文件，需要先在界面中选好文件才能直接录制。

## 输出文件

每次录制生成独立目录，例如 `captures/video/20261022-120000-xxxxxxxx/`：

- `frames/000000.png`、`000001.png`……：最终用于视频的连续帧。
- `config.json`：此次配置。
- `recording.log`：实时进度和 Python 异常，包含预热、各帧累积和截图完成的阶段；控制台输出要等整条命令结束才显示，录制期间可以查看此文件。
- `encode-command.txt`：对应的 FFmpeg 编码命令。`encode_video: false` 时可以稍后手动执行，PowerShell 下如果命令以带引号的 exe 路径开头，需在前面加 `&`。
- `video.mp4`：开启自动编码时生成；H.264、CRF 18、yuv420p，因此宽高需为偶数。
- `ffmpeg.log`：开启自动编码时记录编码器输出和错误信息。
- `captures/`：Falcor 原始截图。当前 `FrameCapture` 会导出所有已标记输出，所以这里可能还包含原 graph 的其他输出；最终视频只使用 `frames/` 中选定的输出。需要减少额外磁盘写入时，可在录制前取消 graph 中不需要的输出标记。`captures/` 在 Mogwai 完成写入后可自行删除。

脚本会等待每张 PNG 的写入结束标记再复制到 `frames/`，避免异步截图未完成就开始编码。若 FFmpeg 不在 PATH 中，保持 `encode_video: false` 即可先导出图片。自动编码失败时图片序列仍会保留。
