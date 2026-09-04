# RoboCup RGB-D TSDF 物体建库与抓取位姿工具

本项目实现离线 RGB-D 序列校验、Open3D TSDF 重建、对象坐标系设置、人工抓取位姿标注、
FoundationPose 在线适配、坐标组合和验收。所有长度均为**米**，变换统一命名为
`T_dst_src`，含义是把 `src` 坐标中的点变换到 `dst` 坐标。

BundleSDF 已移到仓库根目录的独立 `BundleSDF/`，其实验资产位于 `bundlesdf_benchmark/`。
`robot_grasp` 主管线不导入、不修改也不依赖 BundleSDF。

## 目录结构

```text
robocup-grasp/
  src/robot_grasp/       # 可安装的核心 Python 包
  configs/               # 受版本控制的运行配置
  examples/              # 小型、可提交的格式示例
  tests/                 # 单元测试
  docs/                  # 设计、标定与集成说明
  data/
    raw/                 # 原始采集或下载数据
    interim/             # 已校验/已转换的 RGB-D 序列
    processed/           # 清洗后的可复用数据
  models/
    objects/             # 验收后可部署的对象 Mesh 与抓取定义
    weights/             # FoundationPose 等后端权重
  outputs/
    reconstruction/      # 重建运行结果
    inference/           # 位姿估计与抓取组合结果
    acceptance/          # 验收报告
  scripts/               # 相机、机器人 SDK 和批处理适配脚本
  requirements/          # 基础与 Open3D 依赖清单
```

推荐数据流为 `data/raw -> data/interim -> data/processed -> outputs/reconstruction ->
models/objects`。详细放置规则见 [`data/README.md`](data/README.md)、
[`models/README.md`](models/README.md) 和 [`outputs/README.md`](outputs/README.md)。

## 环境

在仓库根目录进行可编辑安装：

```bash
python -m pip install -e .
```

重建、Mesh 坐标系转换、查看器和 Mesh 尺寸验收额外需要 Open3D：

```bash
python -m pip install -e ".[open3d]"
```

也可使用 `requirements/lock.txt` 构建 P0-A Linux x86_64 可复现基线（Python 3.10.12；包含运行、Open3D
和测试依赖的完整解析版本）：

```bash
python -m pip install -r requirements/lock.txt
python -m pip install -e . --no-deps --no-build-isolation
```

`requirements/base.txt` 和 `requirements/open3d.txt` 仍保留为允许小版本更新的通用依赖清单。
锁定文件只安装依赖，不安装本项目本身；完成上述两步后统一使用 `robot-grasp` 命令。该基线由
`.github/workflows/p0-a.yml` 在每次 push 和 Pull Request 中自动验证。

FoundationPose 是外部可选依赖，项目不会下载仓库、权重或 CUDA 依赖。应按所用
FoundationPose 版本的文档单独安装，再填写
[`configs/foundationpose.example.json`](configs/foundationpose.example.json)。

## 1. 采集目录与校验

启用 mask 融合时，每帧的 RGB、深度、mask 和 pose 必须使用完全相同的文件 stem；配置
`use_mask: false` 时可以省略整个 `masks/` 目录，深度有效比例按整张图像统计：

```text
sequence/
  rgb/000001.png
  depth/000001.png
  masks/000001.png
  poses/000001.json
  intrinsics.json
  metadata.json
```

`poses/000001.json` 保存具名的 4x4 `T_base_camera`。`intrinsics.json` 必须包含
`width, height, fx, fy, cx, cy`。`metadata.json` 必须包含：

```json
{
  "depth_scale": 1000.0,
  "object_id": "cup_01",
  "coordinate_frames": {
    "base": "fixed reconstruction frame in meters",
    "camera": "optical frame; poses are T_base_camera"
  }
}
```

`depth_scale` 是每米对应的原始深度单位数：毫米图为 `1000.0`，已经以米存储的浮点深度为
`1.0`。不能依靠隐含单位。相机位姿可由任何 SDK 采集，但采集端应适配
`CameraPoseProvider.get_T_base_camera()`，核心代码不依赖具体机器人 SDK。

严格校验会检查配对、图像尺寸、有效深度比例、4x4 末行、旋转正交性与行列式。默认要求 mask：

```bash
robot-grasp validate-sequence \
  --input examples/minimal_sequence
```

无 mask 序列可显式使用 `--no-mask`。

仓库中的最小序列使用文本 PPM/PGM，不含大型二进制，仅用于格式和 dry-run 校验，不足以生成有意义的
三维模型。真实重建应从多个视角采集清晰、对齐的 RGB-D、mask 和 `T_base_camera`。

### 导入 BOP 真实 RGB-D 数据

BOP scenewise 数据包含 RGB、16 位深度、内参、对象姿态和实例 mask。导入器以 BOP 对象模型系作为
固定 `base`，将 BOP 毫米平移转为米，并保存
`T_base_camera = inverse(T_camera_object)`：

```bash
robot-grasp import-bop \
  --dataset data/raw/bop/lm \
  --split test \
  --scene 1 \
  --object-id 1 \
  --frame-step 4 \
  --output data/interim/lm_ape_sequence
```

导入使用 `mask_visib`，且要求所选帧内参和深度单位一致。输出完成后会自动运行严格序列校验并写入
`import_report.json`。

LINEMOD BOP19 子集适合校验数据格式、变换方向和 FoundationPose 推理，但其早期 RGB-D 深度在
`mask_visib` 边缘仍可能包含背景长尾离群点。它不应在未做尺寸验收时直接作为毫米级建库基准；
使用真实序列时必须将 TSDF 包围盒与已知模型或卡尺尺寸比较，不通过时改用对齐质量更好的本机采集或显式的
边缘/深度离群过滤配置。

### 导入 HouseCat6D 杯子基准

已下载数据位于 `data/raw/housecat6d/`，下载来源、SHA-256 和内容检查记录在
`download_manifest.json`。当前第一轮对象为 `val_scene1` 中的
`cup-plastic_green_flowers`（实例 ID 4）。695 帧数值闭环确认了原始标注语义：

```text
T_camera_object = inverse(T_world_camera) @ T_world_object
```

导入器把 HouseCat6D 原始对象/模型系作为严格序列的固定 `base`，因此
`T_base_camera = inverse(T_camera_object)`。这只是离线基准坐标，不是机器人底座。可分别导入原始深度和
数据集提供的理想 `depth_gt`：

```bash
robot-grasp import-housecat6d \
  --dataset data/raw/housecat6d/dataset \
  --scene val_scene1 \
  --object cup-plastic_green_flowers \
  --depth-source depth \
  --frame-step 5 \
  --output data/interim/housecat6d/val_scene1_cup_raw

robot-grasp import-housecat6d \
  --dataset data/raw/housecat6d/dataset \
  --scene val_scene1 \
  --object cup-plastic_green_flowers \
  --depth-source depth_gt \
  --frame-step 5 \
  --output data/interim/housecat6d/val_scene1_cup_depth_gt
```

两者都显式保存 `depth_scale=1000.0`，导入完成后自动执行严格序列校验。标签 pickle 使用受限读取器，
不会加载任意 Python 全局对象。

## 2. Open3D TSDF 重建

```bash
robot-grasp reconstruct \
  --input data/processed/cup_01/v1 \
  --output outputs/reconstruction/cup_01/v1 \
  --config configs/reconstruction.yaml
```

默认参数为 `voxel_length=0.0025`、`sdf_trunc=0.01`、`depth_min=0.1`、
`depth_max=1.5`、`use_mask=true`。融合前先用序列的显式 `depth_scale` 转成米，范围外及 mask
外的深度置零。Open3D `integrate()` 需要 world-to-camera 外参，因此代码明确传入：

```text
T_camera_base = inverse(T_base_camera)
```

输出包括：

- `fused.ply`：带估计法向的融合点云。
- `mesh_high.ply`：移除退化面和小连通分量、计算法向的高分辨率 Mesh。
- `mesh_collision.obj`：按 `collision_target_triangles` 做二次曲面简化的碰撞 Mesh。
- `reconstruction_report.json`：帧数、有效深度比例、包围盒、顶点/三角形数，以及代码/依赖版本、输入
  SHA-256 manifest、配置快照和运行时间。
- `reconstruction_config.yaml`：可直接再次传给 `--config` 的完整重建参数。

管线不会自动填补大孔洞。应通过增加有效视角和改善 mask 修复缺失表面。输出目录只要已经存在就默认拒绝；
显式 `--overwrite` 后只覆写已知输出文件，不清理目录中的其他文件。

### HouseCat6D 已执行重建结果

第一轮几何基准使用 139 帧 `depth_gt`、1 mm 体素和 4 mm 截断距离：

```bash
robot-grasp reconstruct \
  --input data/interim/housecat6d/val_scene1_cup_depth_gt \
  --output outputs/reconstruction/housecat6d_cup/depth_gt_highres \
  --config configs/reconstruction.housecat6d_depth_gt.yaml
```

重建对象在模型坐标轴下为 `111.346 x 84.913 x 80.978 mm`，对比官方 Mesh 的
`111.668 x 85.478 x 80.356 mm`，逐轴误差为 `0.322 / 0.565 / 0.622 mm`。这验证的是理想深度下的
管线、单位和坐标方向，不代表真实 RGB-D 传感器达到了毫米精度。

同一对象的原始 depth 在默认配置下得到约 `188.679 x 108.876 x 112.552 mm`；3 px mask 腐蚀和
1%-99% 深度分位裁剪仍未消除长尾误差。因此原始 depth 结果被保留为传感器噪声压力测试并判失败，未使用
激进裁剪把它包装成通过。

为判断这是否只是透明/反光杯子的材料问题，又在同一 `val_scene1` 对不透明的 box、can 和 remote 各导入
139 帧并进行了成对重建。raw 与 `depth_gt` 使用同一导入器、mask、姿态和默认 TSDF 配置；另用 1 mm
体素复核 `depth_gt`：

| 对象 | CAD AABB (mm) | raw 默认 TSDF (mm) | raw 最大误差 | `depth_gt` 高分辨率最大误差 |
| --- | --- | --- | ---: | ---: |
| box | 138.713 x 145.786 x 49.364 | 395.995 x 235.000 x 175.718 | 257.282 mm | 3.460 mm |
| can | 67.490 x 118.617 x 67.481 | 604.623 x 173.342 x 261.054 | 537.133 mm | 0.252 mm |
| remote | 184.989 x 28.254 x 53.561 | 240.000 x 120.292 x 158.355 | 104.794 mm | 4.080 mm |

完整数据位于本地生成的 `outputs/reconstruction/housecat6d_opaque/benchmark_report.json`（该目录被
`.gitignore` 忽略，当前 checkout 不包含该运行产物）。
三个不透明对象的 raw depth 同样严重失败，而且 raw 相对 `depth_gt` 的中位偏差分别约为
`+12/+27/+23 mm`。相反，高分辨率 `depth_gt` 全部进入 5 mm 门槛。这否定了“仅杯子材料导致失败”，
也不支持导入器单位或位姿方向错误是主因。HouseCat6D raw depth 与其 mask、姿态和 CAD 参考的组合不适合
直接做度量 TSDF 建库；真实 raw-depth 建库应转向目标相机本机采集，并用独立卡尺重新验收。

## 3. 设置对象坐标系

准备一个 JSON 4x4 `T_object_model`，它把当前重建模型坐标转换到对象坐标：

```bash
robot-grasp set-object-frame \
  --mesh outputs/reconstruction/cup_01/v1/mesh_high.ply \
  --transform examples/T_object_model.json \
  --object-id cup_01 \
  --output models/objects/cup_01/v1
```

输出目录保存 `mesh_object.ply`、`object_frame.json` 和一份 `model_original.*` 原始 Mesh 副本；输入
Mesh 永不原地覆盖。

杯子的推荐对象坐标系是：杯底中心为原点，杯轴为 `+Z`，杯柄方向为 `+X`。这样人工抓取、
FoundationPose 模型和卡尺的 `x/y/z` 尺寸比较共享同一个可解释坐标系。

HouseCat6D 杯子已经使用
[`configs/housecat6d_cup_object_frame.json`](configs/housecat6d_cup_object_frame.json) 转换：模型
`+Y` 为杯轴，模型 `-X` 为杯柄方向。可重复执行：

```bash
robot-grasp set-object-frame \
  --mesh outputs/reconstruction/housecat6d_cup/depth_gt_highres/mesh_high.ply \
  --transform configs/housecat6d_cup_object_frame.json \
  --object-id housecat6d_cup \
  --output models/objects/housecat6d_cup/v1
```

`models/objects/housecat6d_cup/v1/grasps.json` 包含 `handle_side`、`body_opposite_handle` 和
`top_backup` 三个机器生成的格式种子，并非用户人工标注。它们只能用于测试 CRUD、查看器和变换组合；在用户
通过 GUI 逐个确认，并完成真实夹爪碰撞、机器人可达性和执行验证前，不得作为有效抓取数据。

## 4. 抓取候选 CRUD 与标注查看器

[`grasps.schema.json`](src/robot_grasp/grasps.schema.json) 定义 `grasps.json`。每个候选包含 `id`、
`T_object_grasp`、`pregrasp_offset`、`gripper_width`、`approach_distance`、`priority`、
`enabled`、`symmetry_class` 和 `notes`。`pregrasp_offset` 是抓取坐标系中的 `[x,y,z]` 米制偏移。

创建并增加候选：

```bash
robot-grasp grasps add \
  --grasps models/objects/cup_01/v1/grasps.json \
  --object-id cup_01 \
  --id side_handle \
  --transform examples/T_object_grasp.json \
  --pregrasp-offset 0 0 -0.10 \
  --gripper-width 0.06 \
  --approach-distance 0.10 \
  --priority 10 \
  --symmetry-class cup_handle_fixed
```

查看、更新和删除：

```bash
robot-grasp grasps list --grasps models/objects/cup_01/v1/grasps.json
robot-grasp grasps show --grasps models/objects/cup_01/v1/grasps.json --id side_handle
robot-grasp grasps update --grasps models/objects/cup_01/v1/grasps.json --id side_handle --priority 20
robot-grasp grasps delete --grasps models/objects/cup_01/v1/grasps.json --id side_handle
```

Open3D 最小查看器不连接机器人：

```bash
robot-grasp view-grasps \
  --mesh models/objects/cup_01/v1/mesh_object.ply \
  --grasps models/objects/cup_01/v1/grasps.json
```

对象原点和当前 TCP 均显示坐标轴。按键：`W/S` 沿对象 `Y` 平移，`A/D` 沿 `X`，`Q/E` 沿
`Z`；`J/L`、`I/K`、`U/O` 分别绕当前抓取坐标的 `X/Y/Z` 旋转；`N/P` 切换候选，`V`
校验并保存，`Delete` 删除当前候选。步长可用 `--translation-step`（米）和
`--rotation-step`（度）配置。

## 5. 在线位姿接口与组合 dry-run

在线边界位于 [`interfaces.py`](src/robot_grasp/interfaces.py)：

- `CameraPoseProvider` 返回 `T_base_camera`，由具体机器人/标定系统在项目外适配。
- `ObjectPoseEstimator` 返回 `T_camera_object`。
- `GraspCandidateProvider` 加载 `T_object_grasp`。
- `GraspSelector` 过滤 `enabled`，按 `priority` 选择，并可接收可达性回调。

FoundationPose 使用示例：

```python
from robot_grasp.foundationpose_adapter import FoundationPoseAdapter, FoundationPoseConfig

config = FoundationPoseConfig.from_json("/data/foundationpose.json")
estimator = FoundationPoseAdapter(config)
T_camera_object = estimator.estimate(rgb, raw_depth, mask=object_mask)
```

适配器延迟导入 FoundationPose、trimesh 和 nvdiffrast。配置必须给出 FoundationPose 目录、对象
坐标 Mesh、内参、mask（或调用时传入）和 `depth_scale`。标准 FoundationPose `register()` 返回
`T_camera_object`；如果外部封装返回 `T_object_camera`，设置 `output_convention` 后适配器会取逆。
缺目录或依赖时会给出安装路径提示，不会返回随机或占位位姿。

### HouseCat6D FoundationPose 无 GT-pose 泄漏评测

本机配置见
[`configs/foundationpose.housecat6d_cup.json`](configs/foundationpose.housecat6d_cup.json)。以下命令从
五个相隔 150 原始帧的视角分别做独立 `register()`，没有使用上一帧位姿或 GT pose 初始化：

```bash
robot-grasp evaluate-foundationpose \
  --config configs/foundationpose.housecat6d_cup.json \
  --sequence data/interim/housecat6d/val_scene1_cup_raw \
  --object-frame configs/housecat6d_cup_object_frame.json \
  --output outputs/inference/housecat6d_cup/five_view_registration \
  --frame-step 30 \
  --max-frames 5
```

估计器输入只有 RGB、raw depth、内参、GT 可见实例 mask 和重建 Mesh。序列校验会读取 pose 文件，但
每一帧 GT pose 只在 `estimate()` 返回后用于计算误差，不作为估计器参数或初始化；自动化测试也检查
适配器调用中没有 GT pose。2026-08-27
实际运行的平移误差为均值 `6.129 mm`、中位数 `5.232 mm`、最大 `8.978 mm`，旋转误差为均值
`0.786 deg`、最大 `0.884 deg`。将预测转换到共同 `T_base_object` 后，最大两两差为
`8.878 mm / 1.569 deg`。这是 **GT-mask 条件下** 的离线位姿评测，不是在线分割或真实机器人验证。

不连接机器人即可验算完整变换方向：

```bash
robot-grasp compose-grasp \
  --camera-pose examples/T_base_camera.json \
  --object-pose examples/T_camera_object.json \
  --grasps examples/grasps.json
```

输出为机器可读 JSON，并严格计算：

```text
T_base_grasp = T_base_camera @ T_camera_object @ T_object_grasp
```

使用五视角评测产生的 `000300` 预测做已执行 dry-run：

```bash
robot-grasp compose-grasp \
  --camera-pose outputs/inference/housecat6d_cup/five_view_registration/compose_inputs/000300_camera_pose.json \
  --object-pose outputs/inference/housecat6d_cup/five_view_registration/compose_inputs/000300_object_pose.json \
  --grasps models/objects/housecat6d_cup/v1/grasps.json
```

该命令成功选择 `handle_side` 并输出 `T_base_grasp`；这里的 `base` 仍是离线 HouseCat6D 基准系。

## 6. 验收

卡尺文件使用对象坐标系轴向尺寸和米制单位，格式见
[`examples/caliper.json`](examples/caliper.json)。FoundationPose 重复运行结果格式见
[`examples/object_pose_samples.json`](examples/object_pose_samples.json)。执行：

```bash
robot-grasp accept \
  --mesh models/objects/cup_01/v1/mesh_object.ply \
  --caliper examples/caliper.json \
  --poses examples/object_pose_samples.json \
  --config configs/acceptance.yaml \
  --output outputs/acceptance/cup_01/v1
```

Mesh 使用对象坐标系轴对齐包围盒与卡尺 `x/y/z` 对比。位姿重复性报告所有样本对中的平移距离和
旋转测地角，并以最坏两两偏差对比 `pose_translation_mm`、`pose_rotation_deg`；Mesh 对比
`mesh_error_mm`。结果写入 `acceptance_report.json`，全通过退出 `0`，阈值失败退出 `1`，输入或依赖
错误退出 `2`。

HouseCat6D 正式验收命令如下，其中尺寸参考来自官方 Mesh，不是真实卡尺：

```bash
robot-grasp accept \
  --mesh models/objects/housecat6d_cup/v1/mesh_object.ply \
  --caliper configs/housecat6d_cup_official_dimensions.json \
  --poses outputs/inference/housecat6d_cup/five_view_registration/acceptance_pose_samples.json \
  --config configs/acceptance.yaml \
  --output outputs/acceptance/housecat6d_cup/five_view
```

实际报告为失败并返回退出码 `1`：Mesh 最大尺寸误差 `0.622 mm` 通过，旋转重复性
`1.569 deg` 通过，平移重复性 `8.878 mm` 超过 `5 mm` 门槛。报告使用具名的
`T_base_object` 样本，因为不同相机视角下的 `T_camera_object` 本来就不应彼此相同。

## HOI4D / Wild6D 复杂度升级条件

当前下载状态应记录在本地生成的 `data/raw/hoi4d/download_manifest.json`（数据目录被忽略，当前 checkout
不包含该 manifest）：HOI4D Mug 子集含 300 帧
RGB、raw/prior depth、内参与逐帧相机外参，但这个镜像子集不含对象 mask、精确实例 Mesh或对象 6D GT。
因此它目前只能作为动态交互数据资产，不能直接宣称已验证对象 TSDF、FoundationPose 绝对误差或整条抓取
管线。进入同等级验收前必须补齐并核对：

1. 逐帧目标实例 mask，且明确遮挡区域语义。
2. 与该 Mug 实例匹配、单位明确的 Mesh。
3. 可保留到估计后再读取的对象 6D GT，或独立高精度测量参考。
4. 动态片段中对象静止区间或逐帧对象运动补偿，不能把移动对象直接融合进静态 TSDF。

Wild6D 当前仅有官方工具和类别级模板资产，完整测试数据尚未落盘。它是类别级、真实场景位姿估计数据，
类别均值 Mesh 不等同于待建库实例的高精度 Mesh；在取得实例尺度、深度、mask 和可核验姿态前，不作为
HouseCat6D 几何基准的替代品。

## 测试

```bash
pytest -q tests
python -m compileall -q src/robot_grasp
```

单元测试覆盖单位矩阵、已知旋转和平移、逆变换往返、非法旋转、容易写反的组合方向、序列配对与
图像尺寸、抓取 CRUD/选择、输出覆盖保护、具名位姿重复性，以及 FoundationPose 评测不向估计器传入 GT
pose。真实 RGB-D 建库精度、机器人可达性和抓取执行必须在目标硬件与本机采集数据上另行验收；接口实现和
离线数据集评测都不等同于真实机器人验证。
