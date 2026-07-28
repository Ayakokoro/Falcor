# SceneVoxelization 磁盘化流水线架构设计

## 一、当前架构分析

### 现有流程（全内存）

```
Phase 0: Load        Phase 1: Clip         Phase 2: BFS       Phase 3: Analyze      Phase 4: Write
┌──────────┐       ┌──────────────┐      ┌─────────────┐     ┌──────────────┐      ┌──────────┐
│ 加载 FBX  │  ──▶  │ 层次裁剪      │ ──▶  │ BFS 排序     │ ──▶ │ 逆向 BFS 分析 │ ──▶  │ 写出结果  │
│ (实例化)  │       │ (叶子存多边形) │      │ (构建八叉树) │     │ (ABSDF+椭球) │      │ (二进制)  │
└──────────┘       └──────────────┘      └─────────────┘     └──────────────┘      └──────────┘
                          │                      │                    │
                    mNodePolygonMap        polygonArrays           gBuffer
                    (unordered_map)        (vector<vector<Polygon>>)
```

**内存瓶颈：Phase 1 结束后，所有叶节点的裁剪多边形同时驻留在 `polygonArrays` 中。**
对于大规模场景（数亿三角形），内存直接耗尽。

### 核心问题
1. 裁剪出的多边形全部存于内存（`mNodePolygonMap` → `polygonArrays`）
2. 裁剪（Clip）和分析（Analyze）耦合在同一函数中，无法流水线化
3. 主裁剪循环单线程（`SceneVoxelization::clipHierarchical`），虽有 `PolygonGenerator::clipHierarchicalAll` 的多线程路径但未用于 instanced 路径

---

## 二、新架构设计

### 整体思路

将裁剪和分析**完全解耦**，中间通过磁盘文件桥接：

```
Phase 0          Phase 1                Phase 2             Phase 3               Phase 4
Load             Clip (MT)             Merge               Analyze               Write
┌────────┐     ┌──────────┐         ┌──────────┐        ┌──────────────┐       ┌────────┐
│ 加载FBX │ ──▶ │ 多线程裁剪 │ ─────▶ │ 分片合并  │ ────▶ │ 逐叶分析       │ ────▶ │ 写出    │
│ 实例化  │     │ 写分片文件 │  N个   │ 按叶组织  │       │ ABSDF+椭球    │       │ 八叉树  │
└────────┘     └──────────┘   shard └──────────┘       │ SH+聚合       │       └────────┘
                    │           文件       │             └──────────────┘
                    │                     │                    │
              /tmp/voxel_clip/     /tmp/voxel_merge/   从 merged 文件
              thread_000.bin       leaves.idx           流式读入
              thread_001.bin       polygons.dat
              ...
```

### 临时文件目录结构

所有中间文件存放在 `LargeSceneVoxelization/tmp/` 下（相对于工作目录），便于调试和断点续跑：

```
LargeSceneVoxelization/tmp/
├── clip/                          # Phase 1 输出：每个线程一个 shard
│   ├── thread_0000.bin
│   ├── thread_0001.bin
│   └── ...
├── merge/                         # Phase 2 输出：合并后的全局文件
│   ├── leaves.idx                 # 索引文件：nodeKey -> (offset, count)
│   └── polygons.dat               # 数据文件：按 nodeKey 排序的 per-leaf 多边形块
└── analyze/                       # Phase 3 可选：per-leaf 分析中间结果
    └── (预留)
```

---

## 三、各阶段详细设计

### Phase 0: 场景加载（保持不变）

与现有实现一致：
- `SceneLoader::loadMeshInstances()` 加载 FBX，提取唯一 mesh + 实例变换矩阵
- `setupGrid()` 根据所有实例的世界空间包围盒计算 GridData
- `loadTextures()` 加载材质纹理的 MIP 链

**不变的部分：**
```cpp
InstancedScene scene;
loader.loadMeshInstances(fbxPath, scene);
setupGrid(scene);
loadTextures(scene.materials);
```

### Phase 1: 多线程裁剪 → 写分片文件

#### 线程划分策略

按**实例**（instance）划分，而非按三角形数。原因：
- 每个实例的三角形需要 `localToVoxel()` 变换，变换矩阵不同
- 按实例划分可以避免线程间共享变换状态
- 负载均衡：按实例数取模分配到 N 个线程

```
线程分配:
  thread_0: instance[0], instance[N], instance[2N], ...
  thread_1: instance[1], instance[N+1], instance[2N+1], ...
  ...
```

如果实例数差异巨大（某些 mesh 三角形极多），可进一步按三角形粒度做 work stealing：
- 先按实例分配
- 若某线程提前完成，从全局原子计数器取下一个未处理的实例

#### 每个线程的处理流程

```cpp
void clipWorker(int threadId,
    const InstancedScene& scene,
    const GridData& grid,
    uint maxDepth,
    const std::vector<uint>& assignedInstances)
{
    // 打开当前线程的输出文件
    std::ofstream out(tmpDir + "/clip/thread_" + pad4(threadId) + ".bin",
                      std::ios::binary);

    float3 invVoxelSize = 1.0f / grid.voxelSize;

    for (uint instIdx : assignedInstances) {
        const MeshInstance& inst = scene.instances[instIdx];
        const MeshGeometry& mesh = scene.meshes[inst.meshID];
        const glm::mat4& worldM = inst.transform;
        uint matID = mesh.materialID;

        for (uint localTid = 0; localTid < mesh.triangles.size(); localTid++) {
            uint3 idx = mesh.triangles[localTid];

            // 实时变换三角形到体素空间
            Triangle tri;
            tri.vertices[0] = localToVoxel(mesh.positions[idx.x], worldM, grid.gridMin, invVoxelSize);
            tri.vertices[1] = localToVoxel(mesh.positions[idx.y], worldM, grid.gridMin, invVoxelSize);
            tri.vertices[2] = localToVoxel(mesh.positions[idx.z], worldM, grid.gridMin, invVoxelSize);
            tri.uvs[0] = mesh.texCoords[idx.x];
            tri.uvs[1] = mesh.texCoords[idx.y];
            tri.uvs[2] = mesh.texCoords[idx.z];
            tri.normals[0] = transformNormal(mesh.normals[idx.x], worldM);
            tri.normals[1] = transformNormal(mesh.normals[idx.y], worldM);
            tri.normals[2] = transformNormal(mesh.normals[idx.z], worldM);
            tri.buildTBN();

            // 层次裁剪，只存叶子层
            clipAndWrite(tri, tri.calcAABBInt(),
                        inst.meshID, matID, localTid, instIdx,
                        int3(0,0,0), 0, maxDepth,
                        out);
        }
    }
    out.close();
}
```

#### Shard 文件格式

每个 thread shard 文件是**流式追加写入**的二进制序列，无需排序：

```
┌────────────────────────────────────────────────────┐
│ Header                                              │
│   magic:       uint32  (0x564F5843 = "VOXC")       │
│   version:     uint32  (1)                          │
│   threadId:    uint32                               │
│   entryCount:  uint64  (记录总数，写完 header 后回填) │
│   reserved:    uint64                               │
├────────────────────────────────────────────────────┤
│ Entry 0                                             │
│   nodeKey:     uint64  (level<<32 | cell编码)       │
│   polyCount:   uint32                               │
│   polygon[0..polyCount-1]: SerializedPolygon        │
├────────────────────────────────────────────────────┤
│ Entry 1                                             │
│   ...                                               │
├────────────────────────────────────────────────────┤
│ ...                                                 │
└────────────────────────────────────────────────────┘
```

**SerializedPolygon 结构**（紧凑二进制，非 POD 直接 dump）：
```cpp
struct SerializedPolygon {
    uint32_t triMeshID;       // TriangleRef::meshID
    uint32_t triTriangleID;   // TriangleRef::triangleID
    uint32_t triMaterialID;   // TriangleRef::materialID
    uint32_t triInstanceIdx;  // TriangleRef::instanceIdx
    uint32_t vertexCount;     // 实际顶点数 (3..MAX_VERTEX_COUNT)
    float normalX, normalY, normalZ;  // polygon normal
    // 然后是 vertexCount 个 float3:
    //   float vx, vy, vz;   (每个顶点，在 [0,1] 归一化空间)
};
// 总大小 = 4*5 + 4 + 3*4 + vertexCount*3*4
// 对于 6 顶点 = 20 + 4 + 12 + 72 = 108 bytes
```

> **设计决策：** 不直接用 `sizeof(Polygon)` dump，因为 Polygon 含 `TriangleRef` 和 `float3 vertices[MAX_VERTEX_COUNT]`，直接 dump 会有 padding 和对齐问题，且浪费空间（MAX_VERTEX_COUNT=6 时，未使用的顶点槽位也会被写入）。序列化版本只写实际使用的数据。

#### 为什么按叶子写入而非按层次

当前代码只在 `level == maxDepth` 时存多边形（`SceneVoxelization.h:444`）。
保持这一决策：
- 非叶子节点只需要聚合后的 VoxelData（ABSDF + 椭球 + SH），不需要存储原始多边形
- 大大减少磁盘写入量和后续 merge 的数据量
- 简化 merge 阶段的索引结构

---

### Phase 2: 分片合并 → per-leaf 连续存储

#### 合并策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **A. 全内存哈希表** | 简单，一次遍历 | 需要内存容纳所有多边形 | 小场景 |
| **B. 外部排序合并** | 内存可控 | 实现复杂，需要多轮 I/O | 超大规模 |
| **C. 哈希分桶合并（推荐）** | 内存可控，只需两轮扫描 | 需要额外桶文件 | 通用 |

#### 推荐方案：哈希分桶合并（Hash-Bucket Merge）

**第一轮 — 分桶：**
1. 遍历所有 thread shard 文件
2. 对每条 entry，根据 `nodeKey` 的低 N 位（如 N=10，即 1024 个桶）决定目标桶
3. 追加写入对应桶文件：`/tmp/voxel_merge/bucket_XXXX.bin`

```
Round 1: Scatter
  thread_0000.bin ──┐
  thread_0001.bin ──┤
  ...               ├──▶ hash(nodeKey) & 0x3FF ──▶ bucket_0000.bin
  thread_N.bin ─────┘                              bucket_0001.bin
                                                    ...
                                                    bucket_1023.bin
```

**第二轮 — 桶内排序 + 写出：**
1. 逐桶处理（每个桶的数据量应能完全载入内存）
2. 将桶内所有 entry 读入内存，按 `nodeKey` 排序
3. 合并同一 `nodeKey` 的多个 entry（来自不同 thread shard 的同一叶节点多边形）
4. 顺序追加写入全局 `polygons.dat`，同时记录 `(nodeKey, offset, count)` 到 `leaves.idx`

```
Round 2: Sort & Merge
  bucket_0000.bin ──▶ 读入内存 ──▶ 按 nodeKey 排序 ──▶ 合并同 nodeKey ──▶ polygons.dat
  bucket_0001.bin ──▶ 读入内存 ──▶ ...                                   │
  ...                                                               leaves.idx
  bucket_1023.bin ──▶ ...
```

**桶数量选择：**
- 目标：每桶数据量 < 可用内存的 1/4（留空间给排序和合并）
- 设场景有 5000 万 polygons，每个 serialized ~100 bytes → 总共 ~5 GB
- 1024 个桶 → 每桶约 5 MB，完全可放入内存
- 极大场景（5 亿 polygons → 50 GB）：8192 个桶 → 每桶约 6 MB

#### 合并后的文件格式

**`leaves.idx`** — 索引文件（按 nodeKey 升序排列）：
```
┌─────────────────────────────────────────┐
│ Header                                   │
│   magic:        uint32  (0x49444C56)    │
│   version:      uint32  (1)             │
│   leafCount:    uint64                  │
│   maxDepth:     uint32                  │
│   reserved:     uint32                  │
├─────────────────────────────────────────┤
│ LeafIndex[0]                             │
│   nodeKey:      uint64                  │
│   dataOffset:   uint64  (在 polygons.dat 中的偏移) │
│   polyCount:    uint32                  │
│   padding:      uint32                  │
├─────────────────────────────────────────┤
│ LeafIndex[1]                             │
│   ...                                    │
└─────────────────────────────────────────┘
```
每条 LeafIndex 固定 24 bytes，可二分查找/顺序扫描。

**`polygons.dat`** — 数据文件（按 nodeKey 顺序连续存放）：
```
┌────────────────────────────────────────────┐
│ [nodeKey_0 的多边形块]                      │
│   polyCount:   uint32                      │
│   polygon[0..polyCount-1]: SerializedPolygon│
├────────────────────────────────────────────┤
│ [nodeKey_1 的多边形块]                      │
│   ...                                       │
└────────────────────────────────────────────┘
```

#### 备选优化：桶文件直接作为 per-leaf 文件

如果叶节点总数不大（< 10 万），可以在 Round 2 直接输出为 per-leaf 独立文件：
```
/tmp/voxel_merge/leaves/
├── leaf_000000.bin   # nodeKey 最小
├── leaf_000001.bin
└── ...
```
然后 `leaves.idx` 只存文件名映射。这种方式下 Phase 3 读取更简单，但小文件过多时有文件系统压力。

**建议：默认使用单文件 + 索引方式。** 预留独立文件模式作为配置选项（通过 `--leaf-files` 标志切换）。

---

### Phase 3: 逐叶分析（流式读取）

#### 流程

```
for each entry in leaves.idx:
    1. 从 polygons.dat 读取该叶子的所有多边形（反序列化）
    2. 执行现有 leaf 分析逻辑：
       ├── 重建原始三角形（从 triRef + scene mesh 数据）
       ├── UV 插值 + 纹理采样 → ABSDF 累积
       ├── 椭球拟合（Ellipsoid::fit）
       └── SH 投影面积估计（Estimate）
    3. 将 VoxelData 写入中间缓冲（gBuffer[bfsIndex]）
    4. 释放多边形内存
```

**关键优化 — 三角形重建缓存：**
每个多边形的 `triRef` 指向原始三角形。在分析阶段需要重复重建三角形（`localToVoxel` 变换）。可以：
- 预计算所有实例的 voxel-space 三角形（如果显存/内存允许）
- 或者在分析时按需重建（每个 mesh 只变换一次，所有引用该 mesh 的 polygon 共享）

**建议：在 Phase 1 时将变换后的三角形也写入 shard**（增大 shard 体积但避免 Phase 3 重复变换）。权衡后，保持当前按需重建方式，因为变换计算量远小于 I/O 瓶颈。

#### Phase 3 内存模型

不再需要 `polygonArrays`（`vector<vector<Polygon>>`），改为：
- 流式处理：一次只在内存中保留**一个叶节点**的多边形
- `gBuffer` 仍需保留（所有节点的 VoxelData，但远小于原始多边形数据）
- `mOctreeNodes` 和 `mBFSOrder` 需在分析前从 merge 阶段构建

#### BFS 结构构建

在 Phase 2 merge 阶段同步构建 BFS order 和 OctreeNode 结构：

```cpp
// Phase 2 的最后一步：从 leaves.idx 的 nodeKey 集合重建八叉树结构
void buildOctreeFromLeaves(
    const std::vector<LeafIndex>& leaves,  // sorted by nodeKey
    uint maxDepth,
    std::vector<BFSNodeInfo>& bfsOrder,
    std::vector<OctreeNode>& octreeNodes,
    std::vector<uint32_t>& levelNodeCounts)
{
    // 1. 收集所有被占用的节点（叶子 + 祖先）
    std::unordered_set<uint64_t> occupiedNodes;
    for (auto& leaf : leaves) {
        uint64_t key = leaf.nodeKey;
        while (true) {
            occupiedNodes.insert(key);
            uint level = key >> 32;
            if (level == 0) break;
            int3 cell = cellFromKey(key);
            int3 parentCell = cell / 2;
            key = makeNodeKey(level - 1, parentCell);
        }
    }

    // 2. BFS 遍历构建顺序
    // （与现有 finalizeBFS 的 BFS 部分相同，但用 occupiedNodes 而非 mOccupiedNodes）
    
    // 3. 构建 OctreeNode（childBase, childMask, dataIndex）
    // 叶子节点在 gBuffer 中的索引由 leaves.idx 的顺序决定
}
```

---

### Phase 4: 父节点聚合 + 写出

与现有一致：
- 逆向 BFS：叶子 → 根
- 叶子：直接使用 Phase 3 填充好的 `gBuffer[bfsIndex]`
- 非叶子：从子节点的 VoxelData 聚合（ABSDF lobe 面积加权，SH 系数求和，椭球从极值点拟合）
- 最终写出网格数据、八叉树节点、VoxelData 到输出文件

---

## 四、数据流总览

```
                    Phase 1                  Phase 2                    Phase 3
                 ┌─────────────┐       ┌──────────────┐         ┌──────────────┐
scene.instances  │  Thread 0   │ ───▶  │              │         │              │
    .meshes      │  shard_0    │       │   Bucket     │         │  For each    │
    .materials   │             │       │   Merge      │         │  leaf:       │
       │         │  Thread 1   │ ───▶  │              │         │  read polys  │
       │         │  shard_1    │       │  ┌─────────┐ │         │  analyze     │
       ├────────▶│             │       │  │leaves.idx│ │ ──────▶ │  → gBuffer   │
       │         │  Thread 2   │ ───▶  │  ├─────────┤ │         │              │
       │         │  shard_2    │       │  │polygons. │ │         └──────┬───────┘
       │         │             │       │  │dat       │ │                │
       │         │  ...        │ ───▶  │  └─────────┘ │                ▼
       │         └─────────────┘       └──────────────┘         Phase 4
       │              N shards          leaves.idx 按         逆向 BFS 聚合
       │           (每个 ~数百MB)        nodeKey 排序          写出最终文件
       │
       ▼
   /tmp/voxel_clip/               /tmp/voxel_merge/          输出: .bin
```

**内存峰值对比：**

| 阶段 | 当前架构 | 新架构 |
|------|---------|--------|
| Phase 1 (Clip) | mNodePolygonMap 全部叶节点多边形 | 每线程一个 batch buffer (~64K entries)，定期 flush 到磁盘 |
| Phase 2 (Merge) | polygonArrays (全量) | 单桶数据 (~5-50 MB) |
| Phase 3 (Analyze) | polygonArrays + gBuffer | 单叶多边形 + gBuffer |

---

## 五、实现步骤

### Step 1: 定义序列化格式
- 新增 `PolygonSerializer.h`：`SerializePolygon()` / `DeserializePolygon()` 函数
- 定义 `LeafIndex` 结构体和 leaves.idx 的读写
- 定义 shard 文件 header 结构和读写

### Step 2: 实现 Phase 1 — 多线程裁剪写 shard
- 新增 `ClipPhase.h` / `ClipPhase.cpp`
- `ClipPhase::execute(scene, grid, maxDepth, tmpDir, numThreads)`
- 每个线程：分配实例 → 层次裁剪 → 写 shard 文件
- 保持现有的 `clipHierarchical` 递归裁剪逻辑，只改存储目标（memory → file）

### Step 3: 实现 Phase 2 — 合并
- 新增 `MergePhase.h` / `MergePhase.cpp`
- `MergePhase::execute(tmpDir, maxDepth)`
- 第一轮：Scatter 到桶文件
- 第二轮：每桶排序合并 → 写出 `leaves.idx` + `polygons.dat`
- 构建 OctreeNode 和 BFS order

### Step 4: 实现 Phase 3 — 流式分析
- 修改 `SceneVoxelization::process()` 的分析部分
- 从 `leaves.idx` 流式读取，每次处理一个叶节点
- ABSDF 累积、椭球拟合、Estimate 逻辑保持不变
- gBuffer 按 BFS 索引填充

### Step 5: Phase 4 — 聚合 + 写出
- 逆向 BFS 聚合逻辑保持不变
- `writeOutput()` 保持不变

### Step 6: 清理与配置
- 新增 CLI 参数：`--tmp-dir`（默认 `./tmp/`）、`--num-threads`（默认 CPU 核数）、`--keep-temp`（保留中间文件）
- 添加错误处理和断点续跑支持（检查已有 shard/merge 文件）

---

## 六、线程安全与错误处理

### 线程安全
- **Phase 1**：每个线程写独立文件，零竞争；无需锁
- **Phase 2**：单线程顺序处理桶；桶内排序无竞争
- **Phase 3**：可单线程流式处理，也可多个叶子并行（读同一 `polygons.dat` 的不同 offset 区间，只读不写）

### 错误处理
- 每个 shard 文件写完后校验（文件大小 vs header 声明）
- Phase 2 检查所有 shard 是否存在、header 是否合法
- Phase 3 遇到损坏的多边形时跳过并警告，不中断整个流程

---

## 七、可选的进一步优化

### 1. 压缩 shard 文件
- 使用 zstd 流式压缩每个 shard，显著减少磁盘 I/O
- 在 Merge 阶段解压读取

### 2. Phase 3 并行化
- 多个线程各自读取 `polygons.dat` 的不同区间，并行分析不同叶节点
- 需要预先分配好 gBuffer 的各节点槽位

### 3. 断点续跑
- 检查 `/tmp/voxel_clip/` 下的 shard 文件，如果数量和线程数匹配则跳过 Phase 1
- 检查 `/tmp/voxel_merge/leaves.idx`，如存在则跳过 Phase 2
- 通过 `--resume` 标志启用

### 4. 替代存储：SQLite / RocksDB
- 对于超大规模场景（百万级叶节点），可用嵌入式 KV 存储替代文件系统
- RocksDB 提供高效的排序扫描，天然支持 nodeKey → polygons 的映射
- 代价：引入额外依赖；收益：简化 merge 逻辑、获得压缩能力
