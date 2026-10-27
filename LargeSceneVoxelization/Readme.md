example use

## CPU 烘焙计时

默认打印 `[Timing]` 各阶段的实际经过时间（`wall`，单位秒）：场景/纹理加载、
Clip、Merge（叶层包含建树）、Analyze、父级生成、写文件与清理。
`brute-force` 的每个 LOD 分别计时；Merge 原有的读取、排序、写入细分计时也会保留。

在现有命令后增加 `--timings`，开启 Analyze 热点与线程统计。例如从 `LargeSceneVoxelization` 目录运行：

```powershell
.\build\Release\LargeSceneVoxelization.exe "..\Scene\Greasewood\greasewood.glb" -o "..\resource\greasewood_profile.bin" -r 512 -s 1024 --visibility-bitmask --lod-levels 4 --lod-mode brute-force --tmp-dir ".\tmp_profile" --timings 2>&1 | Tee-Object -FilePath bake-timing.log
```

详细输出包含：

- Analyze：节点多边形读取/反序列化/分配、节点复制、几何准备、纹理采样与旧材质聚合、椭球拟合。
- 投影面积：primitive 射线采样、旧可见面积光栅化、总投影面积，分别计时。
- 可见材质：准备、depth/owner 光栅化、像素材质累积、最终拟合/打包。
- Visibility mask：准备、16 方向光栅化与候选面筛选、CDF/拒绝采样/可见性查询。
- Clip 各线程的实例/三角形/多边形数量与耗时；Analyze 各线程的节点/多边形数量与耗时。
- primitive 采样总数、mask 候选次数、最终保留样本数与无效 lobe 数。

`summed worker elapsed` 是各工作线程的经过时间之和，通常大于该阶段的 `wall`。
热点百分比以这个线程累计时间为分母，各项互不重叠；它包含 I/O 等待和线程被调度出去的时间，
**不是操作系统 CPU 使用率**。`max/mean` 是最慢线程耗时除以平均线程耗时，越大通常说明负载越不均衡，
也可能受 I/O 或系统调度影响。`other / timer overhead` 包含未细分的查表、释放、进度输出及计时开销。
mask 的 `retained/candidates` 是最终保留效率：未填满而失效的 lobe，其部分接受样本不会计入 retained。

详细计时使用每线程独立统计，在方向/节点级别取时，不对每条射线或每个像素取时。
它仍有额外开销，测正常吞吐量时去掉 `--timings`。`--in-memory` 也支持，叶节点分析与近似父级
处于同一循环，阶段总时间合并报告，父级聚合在详细任务中单列。计时不写入 `.bin/.meta/.vmat/.vbm`。

## CPU 投影采样优化

以下优化默认启用，命令行参数和文件布局不变：

- 按 `-s` 缓存 Hammersley 圆盘样本的半球坐标，所有 Analyze 工作线程共享只读数据。
  `-s 1024` 的样本表约占 12 KiB；固定的 64 个 Octahedral 子方向及其基底也只初始化一次。
- 每个节点继续执行原来的射线坐标变换与 AABB 裁剪；采样顺序、样本数量、面积归一化分母保持不变。
- 椭球面积计数使用 `intersectsSegment()`，通过二次函数在线段上的最小值判断命中，
  普通路径省去求根与除法；精确落在浮点边界时回退到原 `clip()`，零长度线段按点处理。
  需要进入/离开参数的调用仍可使用原 `clip()`。

Release 使用 `/fp:fast`（其他编译器对应 `-ffast-math`），缓存中间值和重新组织命中判定会改变
浮点舍入，**投影面积不保证与旧版本逐位相同**。验证包含采样位置误差、相切/端点/退化线段、
双精度几何参考和投影面积差异；减少 `-s` 仍会单独影响采样精度。

```powershell
cmake --build build --config Release --target LargeSceneVoxelization VoxelizationInspector ProjectionTests VisibilityMaskTests
ctest --test-dir build -C Release --output-on-failure

# 可选：读取已有烘焙，均匀抽取约 1024 个叶节点，对照旧采样/仅缓存/缓存加新判定
# 这是投影内核测试，不包含场景加载、Clip、Merge 或其他 Analyze 工作。
.\build\Release\ProjectionTests.exe --benchmark "..\resource\greasewood_visible_lod0.bin"
```

## 可选：可见材质与 visibility mask

默认命令继续使用原来的 4-lobe 材质、`.bin` 布局和 version 2 `.meta`。
`--visible-material` 额外生成 8-lobe 的 `.bin.vmat`；`--visibility-bitmask`
同时生成配对的 `.bin.vbm`，并隐含开启 `--visible-material`。
原 `.bin` 仍包含旧材质，旧 renderer 可以照常读取。重新以默认模式覆盖输出时，
同名 `.vmat/.vbm` 会被移除，避免留下过期扩展。

```powershell
# 叶节点扩展；从 LargeSceneVoxelization 目录运行
.\build\Release\LargeSceneVoxelization.exe "..\Scene\Metropolis\Metropolis.fbx" -o "..\resource\Metropolis_visible.bin" -r 1024 -s 1024 --visibility-bitmask

# 额外为两层父级独立烘焙扩展
.\build\Release\LargeSceneVoxelization.exe "..\Scene\Metropolis\Metropolis.fbx" -o "..\resource\Metropolis_visible_lod2.bin" -r 1024 -s 1024 --visibility-bitmask --lod-levels 2 --lod-mode brute-force
```

Renderer 中选择文件，勾选 **Visible material (reload)** 即加载扩展，选择 **Draw Mode → Default** 查看材质着色；关闭可见材质开关后重新加载旧模式。
新模式中的 **Internal visibility** 提供 `Legacy VPA/TPA`、`Bitmask`、`Off`。
这里的 Legacy 只切换内部可见性公式，材质仍是 8-lobe 可见材质。
没有烘焙 mask 时，Bitmask 使用 `Q=1`；无效、错配或损坏的扩展会报告加载错误，
不会将其按旧结构解释。`InternalVisibility` 绘制模式可查看所选入射方向与 lobe 的 Q，
洋红色表示没有有效样本。

`approximate` 父级仍按原流程生成旧材质；其可见材质扩展仅包含叶节点。
Renderer 的新模式会限制到扩展实际覆盖的 LOD。`brute-force` 父级则从自身几何独立烘焙，
不会合并孩子的 mask。`--in-memory` 支持叶节点扩展，沿用原有的父级生成限制。

脚本也可选择新模式，`drawMode=0` 使用材质着色：

```python
createPass("RayMarchingPass", {
    "voxelFile": "E:/syc/Falcor/resource/Metropolis_visible.bin",
    "visibleMaterial": True,
    "internalVisibilityMode": 1,  # 0=VPA/TPA, 1=Bitmask, 2=Off
    "drawMode": 0,
    "maxBounce": 3,
})
```

Inspector 的 `--visible-material` 让 NDF/材质诊断使用新模型；增加 `--visibility-bitmask`
会在 `summary.txt` 写出 mask metadata 和方向位值。投影面积诊断继续使用原 Octahedral 表。
Inspector 从保留的节点 polygons 重新分析，不从 `.vmat` 反推几何。

扩展布局与内存：

- `.vmat` 使用 VMT1：80 字节头部，每节点 360 字节，包含 8 个有向材质 lobe、面积及烘焙标记。
- `.vbm` 使用 VBM1：48 字节头部，每节点 68 字节；16 个完整球面方向共享总共 32 个表面样本。
- 两者按原 `dataIndex` 对齐；校验主 `.bin` 的长度/hash，`.vmat` 还校验配对 `.vbm` 的长度/hash。
- mask 使用完整球面方向编码，和折叠正反方向的 Octahedral 面积表独立。
- Renderer 按实际材质 stride 动态分段，逐块读取扩展并上传；只驻留当前选择的材质模式。
- CPU 仍保存整棵树的结果数组；开启两个扩展额外占约 `428 × 节点数` 字节，另加每工作线程的临时投影数组。

此功能迁移保留当前分支的文件流程；实验分支已有的 VPW3/SH 主文件不能直接作为这里的输入。

```powershell
cmake --build build --config Release --target LargeSceneVoxelization VoxelizationInspector VisibilityMaskTests
ctest --test-dir build -C Release --output-on-failure
```

cmake --build build --config Release --target VoxelizationCore VoxelizationInspector LargeSceneVoxelization

# 默认按 MetalRough 解释材质；只有显式传入 --spec-gloss 才使用 SpecGloss 转换
.\build\Release\LargeSceneVoxelization.exe "..\Scene\SimplePlane\simpleplane.fbx" --spec-gloss

# 只生成最精细层级（LOD 0）；默认使用原有安全的单节点多边形上限
.\build\Release\LargeSceneVoxelization.exe "..\Scene\Old tree\OldTree.fbx" -o "..\resource\OldTree.bin" -r 128 -s 256 -t 2

# 在 approximate 模式下额外生成 2 层父节点 LOD
.\build\Release\LargeSceneVoxelization.exe "..\Scene\Old tree\OldTree.fbx" -o "..\resource\OldTree_lod2.bin" -r 128 -s 256 -t 2 -l 2 --lod-mode approximate

# 在 brute-force 模式下额外生成 2 层父节点 LOD。
# 每一层都会独立重扫全部三角形，并执行 Clip -> Merge -> Analyze。
# brute-force 使用默认的 disk-backed 流程，不要附加 --in-memory。
.\build\Release\LargeSceneVoxelization.exe "..\Scene\Old tree\OldTree.fbx" -o "..\resource\OldTree_exact_lod2.bin" -r 128 -s 256 -t 2 -l 2 --lod-mode brute-force --tmp-dir ".\tmp_oldtree_exact" --clean

# 取消单节点多边形截断（可能显著增加当前层 Merge 的内存和磁盘占用）
.\build\Release\LargeSceneVoxelization.exe "..\Scene\Old tree\OldTree.fbx" -o "..\resource\OldTree_unlimited.bin" -r 128 -s 256 -t 2 --max-polygons-per-node 0

# 每个输出 bin 会同时生成同名文本 sidecar：OldTree.bin.meta
# sidecar 记录版本、maxDepth、generatedLodLevels、lodMode、producer 等信息。

.\build\Release\LargeSceneVoxelization.exe "..\Scene\Old tree\OldTree.fbx" -o "..\resource\OldTree.bin" -r 128 -t 2 -s 256 -j 20

.\build\Release\LargeSceneVoxelization.exe "..\Scene\Old tree\OldTree.fbx" -o "..\resource\OldTree.bin" -r 256 -s 256

.\build\Release\LargeSceneVoxelization.exe "..\Scene\Metropolis\Metropolis.fbx" -o "..\resource\Metropolis_1.bin" -r 256 -s 1024

.\build\Release\LargeSceneVoxelization.exe "..\Scene\RainbowCoralReef\coral_reef.glb" -o "..\resource\Coral_reef_appx.bin" -r 512 -s 1024 --lod-levels 2 --lod-mode approximate


# 先生成并保留中间文件。为了查看节点内全部多边形，建议关闭单节点截断：
build\Release\LargeSceneVoxelization.exe "..\Scene\SimplePlane\simpleplane.fbx" -o "build\simple_plane.bin" -r 512 -s 1024 -t 4 --tmp-dir "tmp_inspect" --keep-temp --max-polygons-per-node 0
# 查看叶子节点（LOD 0）：
build\Release\VoxelizationInspector.exe --scene "..\Scene\SimplePlane\simpleplane.fbx" --bin "build\simple_plane.bin" --tmp-dir "tmp_inspect" --lod 0 --cell 263 5 107 --mode all --out "build\inspect_leaf" --dump-polygons
# 其中：
- --lod 0：最精细叶子层，实际 tree level 为 maxDepth
- --lod 1：上一层，实际 tree level 为 maxDepth - 1
- --cell x y z：目标节点在对应层级的坐标，需要替换成你的节点坐标
- --mode all：执行全部分析
- --dump-polygons：额外输出该节点所有多边形到 polygons.txt
# 只查看投影误差：
build\Release\VoxelizationInspector.exe --scene "..\Scene\SimplePlane\simpleplane.fbx" --bin "build\simple_plane.bin" --tmp-dir "tmp_inspect" --lod 0 --cell 263 5 107 --mode projection --out "build\inspect_projection"
# 查看球面函数：
build\Release\VoxelizationInspector.exe --scene "..\Scene\SimplePlane\simpleplane.fbx" --bin "build\simple_plane.bin" --tmp-dir "tmp_inspect" --lod 0 --cell 263 5 107 --mode spherical --spherical-resolution 256 --out "build\inspect_spherical"
# 查看 splitting 和 NDF：
build\Release\VoxelizationInspector.exe --scene "..\Scene\SimplePlane\simpleplane.fbx" --bin "build\simple_plane.bin" --tmp-dir "tmp_inspect" --lod 0 --cell 263 5 107 --mode splitting --block-count 8 --block-size 32 --out "build\inspect_splitting"
build\Release\VoxelizationInspector.exe --scene "..\Scene\SimplePlane\simpleplane.fbx" --bin "build\simple_plane.bin" --tmp-dir "tmp_inspect" --lod 0 --cell 263 5 107 --mode ndf --ndf-resolution 256 --out "build\inspect_ndf"
# 生成两层 brute-force LOD，并检查上一层：
build\Release\LargeSceneVoxelization.exe "..\Scene\SimplePlane\simpleplane.fbx" -o "build\simple_plane_exact.bin" -r 512 -s 1024 -t 4 -l 2 --lod-mode brute-force --tmp-dir "tmp_exact" --keep-temp --max-polygons-per-node 0
build\Release\VoxelizationInspector.exe --scene "..\Scene\SimplePlane\simpleplane.fbx" --bin "build\simple_plane_exact.bin" --tmp-dir "tmp_exact" --lod 1 --cell x y z --mode all --out "build\inspect_lod1" --dump-polygons
# 输出目录中主要包括：
projection.csv
spherical_primitive_*.csv / *.pgm
spherical_polygons_*.csv / *.pgm
spherical_total_*.csv / *.pgm
splitting_error.csv / *.pgm
ndf.csv / *.pgm
polygons.txt
summary.txt

# spherical_* maps use an N x N upper-hemisphere disk, matching the NDF map:
# the disk center is +Z, the boundary is the XY equator, and pixels outside
# the unit disk are zero. CSV files contain raw values; PGM files are display
# normalized inside the disk.
