example use

cmake --build . --config Release

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
