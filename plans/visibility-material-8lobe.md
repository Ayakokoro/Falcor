# 视线方向可见性材质项（8-lobe 双面聚合）

> 对应 `plans/plan.md`「包含对视线可见性的材质推导」一节的完整落地实现。
> 范围：**材质项 $\hat{f}_{\text{vis}}$**（8-lobe 双面聚合）+ **可见性项 $\hat{V}_{\text{vis}}$**（双向 Smith 近似）。

## 1. 目标与范围

- **材质项 $\hat{f}_{\text{vis}}$**：把「折叠后的 4 双面 lobe」改为「全球面 8 双面 lobe」，用深度缓冲光栅化得到逐面片可见性 $V_k^*$，按可见投影面积加权聚合材质。
- **可见性项 $\hat{V}_{\text{vis}}$**：plan 式 (5) 的双向 Smith 近似，渲染端用 TPA/VPA 的 SH 重建实时计算。

数学对应（plan 式 (3)）：

$$
\hat{f}_v(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \approx \hat{f}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \cdot \hat{V}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o)
$$

其中材质项代入 plan 299 行的固定主视角假设 $V(\mathbf{p},\omega_o)\approx V(\mathbf{p},\mathbf{n}_{0,k})\equiv V_k^*(\mathbf{p})$：

$$
\bar\theta_k,\ \tilde n_k = \mathbb{E}_{\mathbf{p}\sim\mu_k}[\theta(\mathbf{p}),n(\mathbf{p})],\qquad
\mu_k \propto V_k^*(\mathbf{p})\,\langle n(\mathbf{p})\cdot n_{0,k}\rangle\,\mathrm{d}\mathbf{p}
$$

$n_{0,k}$ = 该 lobe 的面积加权平均法线（归一化前就是 `ABSDFLobe.normal` 的累加量）。

## 2. 8-lobe 划分（旋转后的主导轴 + 材质双面化）

上半球沿用原 4 扇区主导轴 `NormalIndex`（`|x|` vs `|z|` 取主导、再取符号 → `+X/−X/+Z/−Z`），下半球镜像到高 4 位：

```slang
inline uint NormalIndex8(float3 n)
{
    uint hemi = (n.y < 0.0f) ? 4u : 0u;
    float3 up = (hemi != 0u) ? -n : n;
    return hemi + NormalIndex(up);
}
```

- `k ∈ {0..3}`：上半球 4 个扇区；`k ∈ {4..7}`：下半球 4 个镜像扇区。
- `lobe[4+s]` 的代表法线 $\tilde n_{4+s}\approx -\tilde n_s$。

**材质双面化**：每个面片同时向 `+n` 和 `-n` 两个 lobe 贡献（材质相同、法线相反），
使正反面各自有独立 lobe、独立可见性。这是「方向相关可见性」的关键——
墙正面 lobe 会被海报遮挡，背面 lobe 完整，两者不再混淆。

## 3. 逐面片可见性（深度缓冲）

### 复用 vs 新建

| 组件 | 位置 | 复用/新建 |
|---|---|---|
| `orthonormal_basis` | `Projection.slang` | 复用 |
| `edgeFunc` / `pointInTriangle` / bbox clamp | `Projection.slang` | 复用 |
| `PROJ_RES=64` 及其投影平面参数 | `Projection.slang` | 复用（分辨率定为 64） |
| `calcVisibleProjCellsRaster` | `Polygon.slang` | **不复用**（coverage 语义，无深度） |
| `rasterizeTriDepth` | `Projection.slang` | **新建**（重心插值深度 + 最前表面保留） |
| `PolygonRange::rasterizeDepth` | `Polygon.slang` | **新建**（节点级逐面片深度光栅化） |

关键点：现有 `calcVisibleProjCellsRaster` 每像素只存 **1 bit 覆盖位**（并集），给不出「哪块面片在最前」。可见性 $V_k^*$ 必须新建 depth 语义。

### 深度方向约定（z-buffer）

`rasterizeDepth` 里深度取负：`dv[k] = -dot(p, direction)`，配合 `rasterizeTriDepth` 的
`if (d < depthArr[idx])` 最小深度比较，保证「离观察者越近 = 深度值越小 = 被 min 选中」。
这是修过一个反向 bug 后的正确约定。

### 数据结构（消除面片数上限）

- **局部（每线程）**：只有 `depthArr[4096]` + `idArr[4096]`，**与面片数无关**（约 33KB/线程，会溢出到 local memory，离线一次性可接受）。
- **全局 scratch（每面片状态）**：仅 `lobeOfBuffer`（`uint × maxPolygonCount`，打包存正反两个 lobe 索引）。

> 教训：Slang 线程局部数组必须编译期定长，任何 O(面片数) 的线程局部存储都会被迫设上限（早期 `MAX_NODE_POLYGONS=256` 是错的）。每面片状态必须放全局 scratch，面片数只出现在循环里、不出现在数组大小里。

### 可见性获取：直接遍历像素，不回到面片维度

深度光栅化 `rasterizeDepth` 把 `idArr[p]` 填成「每像素 → 最前表面全局面片号」，可见性信息**已经完整在 `idArr` 里**。聚合阶段直接遍历 4096 个像素：

- 早期方案（已废弃）：`rasterizeDepth → markVisible → 再遍历面片判 visible`，Pass 2 要 `8 × count` 次遍历，大节点下极慢。
- 现方案：`rasterizeDepth → 遍历 4096 像素聚合`，Pass 2 固定 `8 × 4096` 次遍历，与面片数无关。

## 4. 数据流（`AnalyzePolygon.cs.slang::calc`，单线程逐 lobe）

```
Pass 1（每面片仅算法线 sampleNormal）:
    normal = sampleNormal(poly)
    k0 = NormalIndex8(normal);  k1 = NormalIndex8(-normal)
    lobeOfBuffer[global] = k0 | (k1 << 4)      // 打包正反两个 lobe
    n0Sum[k0] += area·normal;  n0Area[k0] += area
    n0Sum[k1] += area·(-normal); n0Area[k1] += area
    → n0[k] = normalize(n0Sum[k])

总表面积: data.ABSDF.area = Σ calcArea()   // solid 判定 + ellipsoid 拟合用

Pass 2（逐 lobe k = 0..7）:
    clearDepth → rasterizeDepth(n0[k])     // idArr[p] = 最前表面面片 global id
    vpa_k = 0
    for p in 0..4095:
        global = idArr[p]
        if global == 0xFFFFFFFF: continue
        vpa_k += pixelArea                // VPA(n0k)：所有覆盖像素，不管属于哪个 lobe
        k0 = packed & 0xF;  k1 = (packed >> 4) & 0xF
        if k == k0: sign = +1
        elif k == k1: sign = -1           // 背面侧
        else: continue                    // 该像素面片不属于本 lobe
        input = sampleMaterial（完整材质）
        if sign < 0: input.normal = -input.normal
        input.projArea = pixelArea
        ABSDF.accumulate(input)

    lobes[k].normalizeSelf(vpa_k)          // per-lobe：w_k = VPA_k / VPA(n0k)
```

复杂度对比：

| | 遍历 | 采样 |
|---|---|---|
| 早期（回到面片维度） | Pass 2 = 8 × count | ≤ 2 × count |
| 现方案（遍历像素） | Pass 2 = 8 × 4096（常数） | count（Pass1 法线）+ 8 × 覆盖像素 |

大节点（count 十几万）下 Pass 2 遍历从 `8×十几万` 降到 `8×4096`，数量级提升。

## 5. 可见性项（双向 Smith，渲染端）

### 公式（plan 式 (5) / 348 行）

$$
\hat{V}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \approx \frac{1+\Lambda(\boldsymbol{\omega}_o)}{1+\Lambda(\boldsymbol{\omega}_i)+\Lambda(\boldsymbol{\omega}_o)},\qquad
\Lambda(\boldsymbol{\omega}) = \frac{TPA(\boldsymbol{\omega}) - VPA(\boldsymbol{\omega})}{VPA(\boldsymbol{\omega})}
$$

### 代码映射（`Shading.slang::TEBSDF`）

| 数学 | 代码 |
|---|---|
| $TPA(\omega)$ | `totalProjAreaFunc.calc(ω)`（烘焙好的 SH） |
| $VPA(\omega)$ | `polygonsProjAreaFunc.calc(ω)`（烘焙好的 SH） |
| $\Lambda(\omega)$ | `calcLambda(ω)`（新增） |
| $\hat{V}_{vis}$ | `calcVisibility(wi, wo)`（新增） |
| `eval` 调用 | `coverage * surface.eval(l,v,h) * calcVisibility(l, v)` |

`calcInternalVisibility` 保留（debug 可视化 `RayMarching.ps.slang:488` 仍用，且是 $\Lambda_o=0$ 的退化形式）。

### 烘焙侧零改动

`Estimate()` 已用 Lebedev 采样烘焙 `totalProjAreaFunc`（无遮挡 TPA）和
`polygonsProjAreaFunc`（可见 VPA），可见性项纯粹是两者的比值组合，**不需要新 SH 系数**。

### 退化行为（对应 plan 351-373 行）

- $\Lambda_o=0$（视线无阻挡）→ $\hat V = 1/(1+\Lambda_i)$，退化为单向，海报正视场景消除 50% 暗斑。
- $\Lambda_i=0$ → $\hat V = 1$。
- $VPA\to0$（全遮挡）→ `calcLambda` 返回 `1e6`，$\hat V\to0$。

## 6. 文件改动清单

| 文件 | 改动 |
|---|---|
| `Voxel/ABSDF.slang` | `LOBE_COUNT 4→8`；新增 `NormalIndex8`；`ABSDFInput` 加 `projArea`；聚合权重改 `projArea`；`normalizeSelf` 分母改可见投影面积；`accumulate` 双面（按法线符号归 lobe） |
| `Math/Projection.slang` | 新增 `rasterizeTriDepth`（重心插值深度 + 最前表面保留，含退化防护） |
| `Math/Polygon.slang` | 新增 `PolygonRange::rasterizeDepth`（节点局部坐标，深度取负，输出全局面片号） |
| `AnalyzePolygon.cs.slang` | 新增 `buildTriangleData`/`sampleNormal`/`sampleMaterial`；重写 `calc()`：Pass1 双面定 lobe + Pass2 逐 lobe 光栅化后遍历像素聚合 + per-lobe 归一化 |
| `Shading.slang` | `LobeBRDF` 单面消费（前向重归一采样）；`SurfaceBRDF` 前向 lobe 采样/重归一 pdf；新增 `calcLambda`/`calcVisibility`，`eval` 改用双向可见性 |
| `VoxelizationPass.h/.cpp` | 分配并绑定 `lobeOfBuffer` scratch |

## 7. 关键正确性约定

1. **双面材质 + 单面 lobe 消费**：
   - 聚合端（`AnalyzePolygon.calc`）：每个面片同时贡献 `+n`/`-n` 两个 lobe，各自独立可见性。
   - 渲染端（`SurfaceBRDF`）：`eval` 用未重归一权重 $w_k$、背向 lobe 由 `v.z<=0` 自归零；`evalPdf`/`sample` 用前向 lobe 重归一权重，pdf 与采样分布一致（MIS）。

2. **per-lobe 归一化（修复能量减半 bug）**：`w_k = VPA_k / VPA(n_{0,k})`，分母是「沿该 lobe 主视角的总可见投影面积」，**不是**所有 8 个 lobe 的 VPA 之和。否则双面化后 `w(+Z)=w(-Z)=0.5`，从任何方向看都损失一半能量。

3. **scratch 缓冲无竞争**：`localHead` 是 batch 内局部索引（`flushCurrent` 归零），索引上界 = `maxPolygonCount`，非 `totalPolygonCount`。

4. **遮挡来自全节点**：深度光栅化针对全节点所有面片，符合 plan 里 $V$ 定义在体素表面 $A$ 上的语义。

5. **解包 lobe 判定不可省**：同一大面片沿不同 $n_{0,k}$ 方向都可能成为前表面，不判 `k==k0/k1` 会在多个 lobe 里重复累加。

6. **同 lobe 内重复采样不影响结果**（不做去重）：面片覆盖 N 个像素 → 采样 N 次，材质相同、权重都是像素面积，`diffuse /= weight` 归一化后值不变。

7. **`projArea` 语义**：离散像素面积 `pixelArea`，与 SH 拟合的 `calcVisibleProjAreaRaster` 同度量（node-local 投影像素面积）。

## 8. 已知限制 / 后续

- 深度缓冲分辨率固定 4096（复用 `PROJ_RES=64`），可见性判定精度受其限制；不达标可降 32。
- 单节点面片数仍受 host 端 `kSafePerNodePolygonLimit=128000` cap（已有安全上限，非本次引入）。
- 波瓣可见权重 $w_k$ 仍用「平均法线方向 $n_{0,k}$ 的投影面积」近似方向相关的 $w_k(\omega_o)$（plan 299 行固定主视角假设），非真正的方向相关权重。
- 后续若要做精确 $w_k(\omega_o)$：需给每 lobe 存「沿 $n_{0,k}$ 的可见面积标量 + 方向相关 SH 权重」。
