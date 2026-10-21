# 体素双向可见性位掩码：CPU 烘焙与 GPU 渲染

本实现位于 `LargeSceneVoxelization/`，完成独立的 CPU 烘焙、序列化和查询参考实现。
配置为 **4×4 八面体方向图、每节点 32 个表面样本**。方向图占 64 字节，
另有 4 字节 lobe 元数据，共 **68 字节/节点**。所有八叉树层级都独立烘焙。

GPU 接入位于 `Source/RenderPasses/Voxelization/`。`RayMarchingPass` 在读取主文件时
加载并校验附属位掩码，提供 **旧单向模型 / 位掩码 / 关闭** 三种内部可见性模式，
同时用于直接光和间接反射。五分段更新已通过插件构建和独立 shader 编译，尚未完成真实场景运行验证。

## 使用方法与兼容性

在原有命令中添加 `--visibility-bitmask`，例如从仓库根目录运行：

```powershell
.\LargeSceneVoxelization\build\Release\LargeSceneVoxelization.exe `
    ".\Scene\example.fbx" -o ".\resource\example.bin" `
    -r 128 -s 256 -t 4 --tmp-dir ".\LargeSceneVoxelization\tmp\example" `
    --visibility-bitmask
```

输出为：

- `example.bin`：现有 VPW3 布局，包含材质、椭球和三个投影面积 SH。
- `example.bin.vbm`：新增 VBM1 文件，记录顺序与 `.bin` 的 `VoxelData` 顺序一致。

`--in-memory` 同样支持该选项。不添加选项时仍使用原有标量相关性烘焙，
不生成 `.vbm`。位掩码模式跳过旧相关性扰动投影及拟合，主文件中的
`visibilityCorrelation` 保持初始化值 **0**；这不是新方案的相关性估计。
因此，不能把这种文件里的标量交给依赖旧相关性拟合的实验路径来比较效果。

采用附属文件保持主文件兼容：没有把新增记录塞进旧 `VoxelData`，
也没有更改为 VPW4。旧主文件的布局、材质和
投影函数仍可被现有工具读取；`--validate-material` 继续只验证材质。
`--visibility-bitmask` 是烘焙选项，不能与只读材质验证模式一起使用。

两个输出不是原子事务。主文件或附属文件写入失败时，进程返回失败；
附属文件的主文件长度和指纹用于识别失败重烘焙后残留的旧文件。
不启用位掩码时不会删除已有同名附属文件，后续读取必须验证配对信息。

## 估计量与适用假设

以 lobe 内表面集合 \(A_k\)、点可见性 \(V(p,\omega)\in\{0,1\}\) 表示目标：

\[
Q_k(\omega_i,\omega_o)=
\frac{\int_{A_k}V_iV_o c_i c_o\,dA}
     {\int_{A_k}V_o c_i c_o\,dA}.
\]

当前实现采用 lobe 内固定法线近似。在入射和出射都位于该法线正半球时，
常量 \(c_i c_o\) 可以约去。对同一组等面积权重的样本，得到：

\[
\widehat Q_k(i,o)=
\frac{\operatorname{popcount}(M_i\mathbin{\&}M_o\mathbin{\&}B_k)}
     {\operatorname{popcount}(M_o\mathbin{\&}B_k)}.
\]

这里 `M[d]` 是一个方向对应的 32 位字，`B[k]` 是该 lobe 的样本位区间。
**所有方向中相同编号的 bit 都指向同一个表面点**。它不表示像素编号、
三角形编号，也不表示每个方向单独重新抽样得到的排名。

在有效且分母非零时，相同可见集合给出 1、互斥集合给出 0，且
`Q(o,o) == 1`。往所有相关方向均不可见的集合中添加面积，不会改变这个比值。
这解决了旧 `VPA/TPA` 边缘可见率随隐藏后层面积增加而下降的问题。

这仍是压缩近似：lobe 法线变化、材质与遮挡的相关性、固定视角材质拟合误差
都没有因此消失。对单个 lobe 的几何可见性乘以其聚合 BRDF，还需要 lobe 内
材质与可见性近似独立。有限样本的比值也不保证无偏。

## 每个节点的烘焙流程

实现入口为 [AnalyzePhase.h](../../LargeSceneVoxelization/AnalyzePhase.h) 中的
`analyzeNodeData()`，新算法集中在
[VisibilityMaskBake.h](../../LargeSceneVoxelization/VisibilityMaskBake.h)。

1. 完成原有材质采样、主方向深度投影、可见材质聚合和 lobe 权重拟合。
   使用已有 `lobePairs`，保留正反两面的分组；不重新按几何法线划分材质 lobe。
2. 按原始 ABSDF 索引从小到大压缩有效 lobe。有效条件使用 `ABSDFLobe::isValid()`，
   与 `SurfaceBRDF.init()` 的有效 lobe 压缩顺序对应。
3. 将已经完成材质拟合的深度/owner 临时空间复用于 16 个固定方向，
   每个方向沿用现有 `PolygonRange::rasterizeDepth()` 的 64×64 投影。
4. 收集至少拥有一个首层像素的候选多边形。按
   `(instanceIdx, meshID, triangleID, materialID)` 排序，避免 shard 合并顺序改变采样 CDF。
5. 为每个 lobe 建立其候选多边形的面积 CDF。按面积选择多边形及其扇形三角形，
   再用平方根重心采样在三角形内部均匀取点。
6. 对同一个候选点查询全部 16 个方向，得到一个 16 位可见性签名。
   只有在该 lobe 至少一个正半球采样方向可见的点才被接受。
7. 将接受点分配给固定 sample bit，并将它的方向签名转置写入 16 个方向字。
   点坐标、owner、面积 CDF 和深度图都不写入输出。

设候选多边形集合为 \(E_k\)。面积采样加拒绝采样得到的是
\(E_k\cap\bigcup_{d:n_k\cdot\omega_d>10^{-5}}\{p:V_d(p)=1\}\) 上的均匀面积样本。
这里的可见性和候选集合均由有限分辨率投影近似。没有把不同视图的可见像素
等权混合成表面样本，因此不会引入这种混合所带来的重复计数和投影余弦采样偏差。

每个 lobe 的候选尝试上限为 `256 * 分配样本数`，每节点最多 8192 次。
随机数状态只来自节点 cell、level 和原始 lobe 编号；没有共享随机数状态。
输入几何和排序一致时结果可复现。当前立方体回归覆盖了两条管线及 1/2 个裁剪线程；
这不保证任意场景在不同浮点实现、不同裁剪结果或共面深度竞争下逐位一致。

若达到预算仍未填满某个 lobe，该 lobe 的全部样本位会清零，其有效标志也清零。
不会复制少数点假装获得了完整的独立样本。本版没有实现像素提议分布的
重要性重采样后备路径。

## 深度重投影

新路径复用现有深度图和首层 polygon owner，但使用独立的
`VisibilityMaskBake::visible()` 查询。旧标量拟合的重投影函数保持原行为。

目标像素中心与查询点的投影位置一般不同。新查询使用 **目标首层面片的平面**，
把像素中心的深度校正到查询位置，再比较样本点深度。
深度容差为节点边长单位下的 `1e-5`，不使用旧路径最大达 `0.05` 的斜率容差。
若该首层面片确实更近，还会检查查询位置是否落在该面片的投影内部，
避免把像素中心的遮挡直接扩展到面片以外。

空像素、目标面片没有覆盖查询位置、或目标平面接近投影切线时，按“未观察到遮挡”
处理为可见。超出投影平面的点返回不可见；正常节点几何应处于平面范围内。
这是倾向偏亮的光栅近似，可避免单个无遮挡平面的边缘因空 texel 被错误压暗。
它无法发现像素中心没有记录到的细小遮挡，也没有搜索被当前 owner 遮住的第二层。
小于深度容差的层间距、掠射方向和亚像素几何仍需要更高精度参考验证。

## 方向图、样本分区和查询接口

方向编号为 `d = y * 4 + x`，方向取八面体 texel 中心：

```text
omega[d] = OctDecode(((x + 0.5) / 4, (y + 0.5) / 4))
```

使用与 [MathHelpers.slang](../../Source/Falcor/Utils/Math/MathHelpers.slang)
的 `ndir_to_oct_unorm()` / `oct_to_ndir_unorm()` 相同的非等面积八面体约定，
世界轴固定，中心为 +Z，四角对应 -Z。两半球分别存储，不能用偶对称 SH 的方式
把相反方向视为同一可见集合。**等面积要求针对表面样本，不针对方向 texel。**

当前只实现最近 texel 查询，不插值整数掩码。对于 `K` 个有效材质 lobe，
紧凑索引 `k` 占用：

```text
begin = floor(32 * k / K)
end   = floor(32 * (k + 1) / K)
B[k]  = bits [begin, end)
```

K=2 时每个 lobe 16 点，K=8 时每个 lobe 只有 4 点。
K=0 表示没有可供查询的材质 lobe，记录为零。

[VisibilityMask.h](../../LargeSceneVoxelization/VisibilityMask.h) 提供 CPU 参考：

```cpp
uint32_t i = VisibilityMask::directionIndex(wi);
uint32_t o = VisibilityMask::directionIndex(wo);
auto q = VisibilityMask::query(data, compactLobe, i, o);
// q.value, q.outgoingSamples, q.supported
```

调用者负责判断实际入射/出射方向与 lobe 法线的正半球关系，以及 BRDF 中的余弦。
无效 lobe、无效方向或分母为零时，返回 `value=1, supported=false`。
有效查询会同时返回分母样本数。这个偏亮后备值不表示测得了完全可见。
`validLobes` 只说明采样分区完整，并不保证每个出射方向都有样本。

烘焙日志输出候选次数、有效存储样本数、无效 lobe 数，以及没有任何出射样本的
正半球方向数。最后一项可能来自有限采样，也可能来自那个方向确实没有可见面积；
它是后续评估权重模型和后备策略的诊断信息。

## VBM1 二进制格式

纯整数载荷定义在 [VisibilityMaskFormat.h](../../LargeSceneVoxelization/VisibilityMaskFormat.h)，
CPU 烘焙器和渲染器共用 [VisibilityMaskFile.h](../../LargeSceneVoxelization/VisibilityMaskFile.h) 的文件读取与校验。
两者只依赖 C++17 标准库，不向 Falcor 引入独立项目的 `Types.h` 或 GLM 类型。
采用 little endian，文件长度严格为 `48 + nodeCount * 68` 字节。

| 文件头偏移 | 类型 | 内容 |
| --- | --- | --- |
| 0 | uint32 | magic = `0x314d4256`，字节为 `VBM1` |
| 4 | uint32 | version = 1 |
| 8 | uint32 | headerBytes = 48 |
| 12 | uint32 | recordBytes = 68 |
| 16 | uint32 | resolution = 4 |
| 20 | uint32 | sampleCount = 32 |
| 24 | uint32 | nodeCount，包含所有 BFS 层级 |
| 28 | uint32 | flags = 0，保留 |
| 32 | uint64 | 主 `.bin` 的字节数 |
| 40 | uint64 | 主 `.bin` 所有字节的 FNV-1a 64 位指纹，包括 VPW3 尾标记 |

每个记录的布局：

| 记录偏移 | 类型 | 内容 |
| --- | --- | --- |
| 0 | uint32[16] | 按 `y*4+x` 排列的方向掩码 |
| 64 | uint32 | metadata |

`metadata` 的 0..7 位表示原始 ABSDF 有效 lobe 集合；8..15 位表示压缩后
已完整采样的 lobe 集合；16..31 位必须为零。对无效采样 lobe，其分区内的所有方向位必须为零。

附属记录通过主文件 `OctreeNode.dataIndex` 索引。本次两条管线都保持它与 BFS 索引一致。
父节点直接分析自己的多边形，不能 OR 或平均子节点掩码，因为 bit 身份不同。

写入主文件后，对同一批内存字节计算指纹，不重读整个主文件。
`VisibilityMaskFile::read()` 要求调用者提供对应主文件的节点数、实际字节长度和指纹，
验证版本、固定参数、文件长度、配对信息和 lobe 位合法性。渲染器还逐节点对照主文件
的 `ABSDFLobe::IsValid()` 验证 `sourceLobes`；CPU 集成测试也检查这项对应关系。
FNV-1a 用于发现误配，不是密码学保证。

## 存储与预处理成本

- 持久数据增加 `68 * 节点数 + 48` 字节；一百万节点约增加 64.85 MiB。
  这是一个方向到 32 位值的二维函数，加一个元数据字，没有逐点坐标或逐 lobe 方向图。
- 位掩码路径的深度与 owner 存储为 `16 * 64 * 64 * 8 = 512 KiB/分析线程`，
  复用材质阶段空间。另有候选集合、面积 CDF、已有材质样本及少量统计内存。
- 原有主方向材质投影与 SH 估计继续执行。新路径增加 16 个固定方向投影，
  同时省去每个旧主方向的四个扰动投影以及相关性拟合。
- 拒绝采样每节点最多 8192 个候选，每候选查询 16 个方向；通常填满 32 个样本后就停止。
  面片排序和主文件指纹也有 CPU 成本。低接受率节点可能更慢，不能据此宣称全场景预处理加速。
- GPU 每次方向对求值读取两个方向 uint 和一个 metadata uint，所有 lobe 共用这三个字，
  然后逐 lobe 做位运算和 popcount。不读取整个方向图，不进行逐三角形追踪。
  GPU 位掩码载荷仍为 68 字节/节点。尚无 GPU 性能测量。

## 自动验证

独立 CPU 项目增加了可选 `LSV_BUILD_TESTS`。例如在已有构建目录中启用：

```powershell
cmake -S LargeSceneVoxelization -B LargeSceneVoxelization/build -DLSV_BUILD_TESTS=ON
cmake --build LargeSceneVoxelization/build --config Release --target LargeSceneVoxelization VisibilityMaskTests
ctest --test-dir LargeSceneVoxelization/build -C Release --output-on-failure -R VisibilityMask
```

`VisibilityMaskTests` 不依赖 GPU。它覆盖 bit 分区、八面体 texel 往返、相同/互斥/部分重叠集合、
1/8/32/256 层隐藏面片、倾斜薄层间隔、无遮挡投影边缘、实际挡板产生的不同方向遮挡、
面积比例采样、亚像素无效 lobe，以及序列化、截断和主文件误配拒绝。

安装 Python 且构建为 64 位时，还会注册 `VisibilityMaskPipeline`。
它生成小型 OBJ 立方体，验证磁盘管线 1/2 个线程与内存管线、所有层级的记录索引、
材质 lobe 压缩、主文件指纹、旧 VPW3 ABI、不启用选项的行为及输出失败返回值。
还检查在同一内存管线中启用新模式不会改写材质、椭球或三个 SH 字段。

当前验证限于这些合成场景；尚未验证大型真实场景的角度量化误差、无支持方向比例、
真实烘焙耗时或最终图像。4×4 方向分辨率和最多八个 lobe 分享 32 点都是明显的质量限制。

## GPU 使用与模式切换

将同次烘焙得到的 `.bin` 和 `.bin.vbm` 放在一起，在 `RayMarchingPass` 中选择 `.bin`
并点击 `Read`。文件列表仅列出 `.bin`，附属文件自动读取。将 `Draw Mode` 设为 `Default`
查看着色结果，用 `Internal Visibility` 下拉框切换：

| 模式 | 内部可见性因子 | 数据要求 |
| --- | --- | --- |
| `Legacy one-way (VPA/TPA)`，默认 | 原来的 `calcInternalVisibility(l)` | VPW3 主文件即可 |
| `Bitmask` | 每个 lobe 独立的条件可见性 `Q_k(l,v)` | 与主文件匹配的 VBM1 附属文件 |
| `Off` | 1 | VPW3 主文件即可 |

这里的“关闭”只关闭体素内部的几何衰减。Coverage、null transmission、体素间阴影，
以及材质 GGX 自身的微表面几何项保持原行为。`visibilityCorrelation` 标量不参与这三种
模式，因此位掩码烘焙中将它置零不影响旧单向模式的比较。

模式是常量缓冲区中的统一参数。成功加载的位掩码在 Legacy/Off 模式下仍保留在 GPU，
便于切换比较；正常着色在这两种模式下不读取它。启用像素拾取时，所选像素的诊断会单独查询位掩码。
切换模式会通知累积 pass 刷新，并重置本 pass 的随机帧序号。

## GPU 加载与寻址

[RayMarchingPass.cpp](../../Source/RenderPasses/Voxelization/RayMarchingPass.cpp) 在读取主文件
时对读到的原始字节顺序计算指纹，包括头部、层级计数、节点、材质载荷和 VPW3 标记。
不存在附属文件时省去指纹计算；存在时必须通过完整配对检查。

校验成功后，以 `StructuredBuffer<uint>` 上传，不改 `VoxelData`、`TEBSDF` 或
`PrepareShadingData.cs.slang` 的存储布局。最多五个可见性 buffer 使用与材质/椭球相同的
`gbSplit0/gbSplit1/gbSplit2/gbSplit3`，由命中节点的 `OctreeNode.dataIndex` 寻址：

```text
segment = index < split0 ? 0 : index < split1 ? 1 : index < split2 ? 2 : index < split3 ? 3 : 4
localIndex = index - (segment == 0 ? 0 : split[segment - 1])
base = localIndex * 17
Mi   = visibilityBuffer[base + incidentDirection]
Mo   = visibilityBuffer[base + outgoingDirection]
meta = visibilityBuffer[base + 16]
```

每段按 `VoxelData`、`TEBSDF`、`Ellipsoid`、位掩码记录中的最大 stride 限制在 2 GiB 以下；
超过五个分段容量的输入会被拒绝，并报告所需分段数。当前最大 stride 为 `TEBSDF` 的
696 字节，五段最多容纳 15,427,325 个节点；实际还受可用显存限制。
内部 LOD 节点同样检查 `dataIndex` 范围，使用自己独立
烘焙的记录。上传时直接使用 68 字节记录内存，不额外展开成 16 张图或逐 lobe buffer。

主文件读取成功、GPU 资源准备完成后才替换当前整组数据。主文件读取失败时保留先前数据，
UI 显示加载失败；主文件有效而附属文件缺失/拒绝时，可继续用三种模式，但 Bitmask 采用
`Q=1, supported=false` 的回退。切换到没有位掩码的新文件会清除之前的位掩码绑定，
空资源槽绑定一个 4 字节占位 buffer；可用性分支在读数据前返回。

## GPU 求值与概率抵消

[VisibilityMask.slang](../../Source/RenderPasses/Voxelization/VisibilityMask.slang) 实现方向映射、
样本分区及 popcount 比值。使用最近八面体 texel，方向是世界空间中从表面向外的 `l`、`v`；
不把出射方向取反，不插值整数掩码。零/非有限方向、无效 lobe、lobe 数不一致或出射样本数
为零时返回 `Q=1, supported=false`。`K=1` 时使用不涉及移位 32 位的分区表达式。

[Shading.slang](../../Source/RenderPasses/Voxelization/Shading.slang) 新增
`SurfaceBRDF.evalVisible(l,v,h,pair)`，实现：

\[
F_{\mathrm{vis}}(l,v)=\sum_k
\frac{a_k\max(n_k\cdot v,0)}{\sum_j a_j\max(n_j\cdot v,0)}
\,f_k(l,v)\,Q_k(l,v).
\]

`f_k` 对应原 `lobes[k].eval()`，已经包含入射余弦和材质微表面项。
分母仍包含所有视线正面的材质瓣；不会按 `Q`、采样支持状态或受光瓣重新归一化。
实际入射方向在 lobe 背面时贡献为零，几何查询的偏亮回退不改变这一点。

[RayMarching.ps.slang](../../Source/RenderPasses/Voxelization/RayMarching.ps.slang) 的
`evalVisibleSurface(bsdf,nodeIndex,l,v,h)` 统一选择三种模式，本身不含 Coverage。
Bitmask 路径不会额外乘旧 `VPA/TPA` 或标量相关性函数。两个实际积分位置为：

```text
直接光：外部阴影可见性 * lightColor * Coverage(v) * F_vis(l,v) / lightPdf
反射吞吐量：throughput *= F_vis(l,v) / surfacePdf
```

反射分支已经以 `Coverage(v)` 概率被选中，分子和事件概率中的 Coverage 抵消。
null 分支仍以 `1-Coverage` 概率选中，其吞吐量因子保持 1。材质 lobe 选择、方向采样、
`SurfaceBRDF.evalPdf()`、直接光混合采样及其原有 PDF 处理均不变。

## GPU 诊断与验证范围

`Draw Mode = InternalVisibility` 在 Bitmask 模式下显示指定紧凑 lobe 的几何 `Q_k(l,v)`。
`Visibility Incident (world)` 指定非零入射方向，`v` 使用当前相机方向；不会固定查询
恒为 1 的 `Q(v,v)`。`Visibility Lobe (compact)` 选择压缩后的索引 0..7。
灰度表示有效的 Q，洋红色表示无支持，包括所选索引超过实际 lobe 数的情况。
这是几何比值视图，不包含材质半球截断、余弦、Coverage 或光照强度。

像素拾取、`Display NDF` 和所选体素的同步回读由编译选项
`VOXELIZATION_ENABLE_PIXEL_INSPECTION` 控制，默认 **OFF**。关闭时，CPU 不创建、清除、
绑定或回读拾取缓冲，shader 不声明该 UAV 或执行拾取写入，界面隐藏相关控件。
全屏 `InternalVisibility` 可视化仍然可用。需要拾取诊断时，在已有构建目录启用并重新构建插件：

```powershell
cmake -S . -B build/windows-ninja-msvc -DVOXELIZATION_ENABLE_PIXEL_INSPECTION=ON
cmake --build build/windows-ninja-msvc --config Release --target Voxelization
```

将选项改为 `OFF` 并重新构建可恢复默认路径；仅修改 shader 内的默认宏不会改变 CPU 回读行为。
启用后仍采用每帧同步回读，并非异步回读。

启用拾取并点击像素后，UI 显示所选 lobe 的 Q、出射样本数 `No`、支持状态和材质 lobe 总数。
Legacy 显示指定入射方向的 `VPA/TPA`，Off 显示因子 1。选中像素的回读记录从两个
`float4` 扩为三个，沿用已有回读流程；没有增加每像素或每节点的诊断存储。

最初接入时做了源代码静态检查：核对共享常量/记录布局、CPU 与 shader 的资源和参数名、
分段寻址、主文件身份校验、lobe 顺序、两处求值及 Coverage/PDF 抵消关系。
后续五分段更新已通过 Release `Voxelization.dll` 构建，并使用随工程提供的 `slangc`
独立编译 `PrepareShadingData.cs.slang`、`RayMarching.ps.slang` 和 `DisplayNDF.ps.slang`。
这不代表完成了真实场景加载、所有 shader 配置、GPU 性能或最终图像验证。
