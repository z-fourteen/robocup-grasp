# TSDF 重建现状与任务规划

日期：2026-09-04

## 1. 结论

当前 TSDF 部分属于**离线 PoC 已跑通，但还不能作为真实相机建库生产基线**的阶段。

已经得到证据支持的能力：

- RGB-D 序列格式、深度单位和位姿方向能够被严格校验。
- Open3D TSDF 的米制转换和外参方向正确。
- 在理想 `depth_gt` 下，可以得到毫米级的局部表面几何。

尚未得到证据支持的能力：

- 真实 RGB-D 相机能够生成尺寸和表面均可靠的对象 Mesh。
- 深度噪声、RGB-depth 对齐和手眼标定误差满足任务精度。
- 重建 Mesh 表面完整，collision Mesh 可安全用于规划。
- 重建结果能够稳定支持后续 FoundationPose 和机器人抓取。

TSDF 在当前管线中的定位是**离线对象模型建库**。除非运行时确实需要持续融合场景，否则不应把在线 TSDF 作为下一阶段的首要任务。

## 2. 当前实现

核心实现位于 [`src/robot_grasp/reconstruction.py`](../src/robot_grasp/reconstruction.py)，CLI 入口是：

```bash
robot-grasp reconstruct \
  --input data/processed/cup_01/v1 \
  --output outputs/reconstruction/cup_01/v1 \
  --config configs/reconstruction.yaml
```

当前处理流程如下：

1. 使用 [`validate_sequence()`](../src/robot_grasp/sequence.py) 校验 RGB、depth、pose（以及 `use_mask=true` 时的 mask）的文件 stem、图像尺寸、深度有效率、内参和 4x4 位姿。
2. 根据 `metadata.json` 中的 `depth_scale` 将原始深度转换为米。
3. 按深度范围、mask 和逐帧深度分位过滤无效像素。
4. 将序列中的 `T_base_camera` 求逆为 `T_camera_base`，传给 Open3D `integrate()`。
5. 提取融合点云和三角网格。
6. 删除重复、退化、非流形面，并按连通分量清理小组件。
7. 生成高分辨率 Mesh、简化后的碰撞 Mesh、点云和 JSON/YAML 报告。

当前 [`ReconstructionConfig`](../src/robot_grasp/config.py) 的默认参数为：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `voxel_length` | 2.5 mm | TSDF 体素边长 |
| `sdf_trunc` | 10 mm | TSDF 截断距离 |
| `depth_min` / `depth_max` | 0.1 / 1.5 m | 深度有效范围 |
| `use_mask` | `true` | 是否将 mask 用于融合 |
| `min_component_triangles` | 100 | 小连通分量的最小三角形数 |
| `collision_target_triangles` | 5000 | 碰撞 Mesh 目标三角形数 |

## 3. 已有验证结果

历史实验记录在 [`docs/pipeline_review_and_roadmap_20260827.md`](pipeline_review_and_roadmap_20260827.md)。

### 3.1 理想深度 `depth_gt`

HouseCat6D `val_scene1` 杯子使用 139 帧、1 mm 体素和 4 mm 截断距离：

- 重建 AABB 与官方 Mesh 的逐轴误差为 `0.322 / 0.565 / 0.622 mm`。
- 重建表面到官方 Mesh 的平均距离约 `0.358 mm`，P95 约 `0.698 mm`。
- 官方 Mesh 到重建表面的 P95 约 `5.603 mm`。
- 重建表面积约为参考 Mesh 的 66%。

这证明了单位、位姿方向和 TSDF 融合流程在理想深度条件下基本正确，但也说明遮挡区域、内壁或底部存在缺失；该结果不能代表完整 CAD 级模型。

### 3.2 原始深度 raw depth

原始深度结果明显失败：

- 杯子重建尺寸约为 `188.679 x 108.876 x 112.552 mm`，明显偏离参考尺寸。
- box、can、remote 的最大尺寸误差约为 `257 / 537 / 105 mm`。
- raw depth 相对 `depth_gt` 存在约 `+7～27 mm` 的系统偏差，P95 误差仍为几十毫米。
- 3 px mask 腐蚀和 1%～99% 深度分位裁剪没有解决问题。

结论是：主要瓶颈是深度尺度/偏置、视角相关误差和长尾噪声，而不是单纯的 mask 边缘泄漏。不能通过继续调节腐蚀像素或分位阈值把该结果包装为通过。

## 4. 当前代码与工程缺口

P0-A 已经修正输入契约、统计口径、运行报告和合成测试。当前仍待处理的事项是：

1. 所有帧基本等权融合，没有坏帧拒绝、置信度、重投影一致性、ICP 残差或位姿质量检查。
2. 深度处理仅包含范围、mask 和分位裁剪，缺少尺度/偏置校正、跳变检测、局部滤波和飞点检测。
3. Mesh 清理只做拓扑清理和 AABB 级观察，没有双向表面距离、覆盖率、法向、孔洞和碰撞简化误差指标。
4. 没有 RGB-depth 配准质量、时间同步、相机序列号和姿态来源等元数据校验。
5. 当前 checkout 只有数据和输出目录骨架，数据集与运行产物被忽略，历史结果不能直接在当前目录复现。
6. 仍没有真实相机数据的 TSDF 验收结果；当前 35 项测试是合成基线，不替代真实深度、标定和手眼验证。

## 5. 分阶段任务规划

### P0-A：建立可复现基线

工作项：

- 固定 Python、Open3D 和相关依赖版本。
- 增加合成 RGB-D 集成测试：已知几何、已知深度和已知位姿，验证单位、外参方向、mask、空结果和 Mesh 尺寸。
- 统一有效深度比例统计口径，修正 `use_mask` 的契约语义。
- 在重建报告中记录代码版本、依赖版本、输入 manifest、配置快照和运行时间。

完成标准：合成数据可以在 CI 中稳定重建，关键几何误差有固定阈值，失败原因可定位。

### P0-B：真实相机采集与标定

工作项：

- 选定目标 RGB-D 相机、机器人、安装方式和夹爪。
- 采集一个静态杯子的多视角 RGB-D、内参、时间戳和机器人位姿。
- 明确 RGB/depth 是否硬件同步、是否已完成 depth-to-color registration。
- 完成深度尺度、偏置、平面误差和手眼标定的独立验证。
- 扩展序列 metadata，记录相机序列号、同步状态、配准状态和姿态来源。

完成标准：数据能够转换为严格序列；每个 `T_base_camera` 的来源、时间戳、坐标轴约定和验证误差可追溯。

### P0-C：深度预处理与帧质量控制

建议新增可插拔 `DepthPreprocessor`，按顺序实现：

- 深度尺度和偏置校正；
- 深度跳变、飞点和局部离群检测；
- 中值、双边或引导滤波；
- mask 连通区域筛选；
- 逐帧质量评分；
- 坏帧拒绝或降权。

首轮先实现帧拒绝和质量报告。Open3D 当前融合接口没有直接暴露可靠的逐帧权重机制；复杂权重策略应在基线结果后再决定是否更换 TSDF 后端。

完成标准：每帧都有有效像素、深度分布、质量分数、拒绝原因和融合状态；原始数据保持只读。

### P0-D：TSDF 几何质量验收

在现有验收模块基础上增加：

- 双向 Chamfer 或点到面距离；
- P50、P90、P95、P99 和 Hausdorff 距离；
- 表面覆盖率和法向一致性；
- 孔洞、边界边和连通分量统计；
- high-resolution Mesh 与 collision Mesh 的简化误差和保守性。

真实建库的首轮门槛建议为：

- AABB 最大轴误差不超过 5 mm；
- 双向表面距离 P95 不超过 5 mm；
- 不存在影响规划的远距离孤立表面；
- high-resolution 和 collision Mesh 均通过质量报告。

### P1：下游衔接与独立评测

只有真实 TSDF 通过后，再进行对象坐标系转换、FoundationPose 和抓取规划衔接：

- 使用独立场景或独立采集批次评测，避免同轨迹闭环。
- 至少 100 帧，分别报告 GT mask 与预测 mask 结果。
- 区分固定视角重复性、跨视角一致性和相对 GT 的绝对精度。
- 将真实手眼标定误差纳入最终位姿误差预算。

## 6. 推荐执行顺序

1. [已完成] 修正 TSDF 输入契约和报告统计口径。
2. [已完成] 增加合成重建测试并固定运行环境。
3. 完成真实相机、深度和手眼标定。
4. 实现深度预处理和逐帧质量报告。
5. 用真实数据达到尺寸和表面质量门槛。
6. 再将 Mesh 交给 FoundationPose、IK 和碰撞规划。

在上述步骤完成前，不建议继续通过增大帧数、调小体素或反复修改 mask 腐蚀参数来优化 raw depth 结果。每个“完成”结论都必须对应可复现产物、明确指标和通过门槛。

## 7. P0-A 执行记录

本轮已落地以下基线工作：

- 新增 `requirements/lock.txt` 和 `.python-version`，固定 P0-A Linux x86_64 验证环境为 Python 3.10.12、Open3D 0.19.0 及其运行/测试传递依赖版本。
- `validate_sequence()` 和 `reconstruct_sequence()` 现在显式传递 `use_mask`：启用时 mask 必须存在，禁用时允许省略 `masks/`，且有效深度比例分别按 mask 像素或整图像统计。
- 重建报告新增代码版本/Git commit、依赖版本、输入文件 SHA-256 manifest、配置快照和 UTC 运行时间。
- 新增合成 RGB-D TSDF 集成测试，覆盖已知 80 mm 立方体的单位/位姿方向、mask 与无 mask 融合、报告上下文和全零深度空结果。
- 项目 pytest 配置屏蔽无关的 ROS 自动插件，使 `pytest -q` 在当前环境可直接稳定运行。
- 新增 `.github/workflows/p0-a.yml`，在干净的 Ubuntu 22.04 / Python 3.10.12 环境安装锁定依赖、运行完整测试集并校验提交的最小序列。

当前验证结果：本地锁定环境 `.venv/bin/python -m pytest -q` 为 35 项通过，最小序列 CLI 校验通过；CI workflow 已纳入同样的安装和验证步骤。
该结果只证明可复现合成基线，不替代 P0-B 的真实相机、深度标定和手眼验证。
