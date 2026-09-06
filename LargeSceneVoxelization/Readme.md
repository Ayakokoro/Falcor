example use

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

# 实例化输出
.\build\Release\LargeSceneVoxelization.exe "..\Scene\Metropolis\Metropolis_2.glb" --instanced --output-dir "..\resource\Metropolis_2_instanced" -r 512 -s 1024 --lod-levels  --lod-mode approximate


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
