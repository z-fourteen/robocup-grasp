# Grasp 环境与能力分析

日期：2026-09-04

## 1. 范围与结论

本文只分析本仓库内的 grasp 部分：抓取候选数据契约、候选编辑、位姿组合、查看器、测试和 Python 环境。

本轮不分析 ROS、相机驱动、机械臂、夹爪驱动、BundleSDF 或 FoundationPose 外部环境。

当前项目的 grasp 能力可以定义为：**离线抓取候选管理和坐标组合 dry-run 已完成；可执行抓取计划尚未实现。**

## 2. 当前 Python 环境

项目定义在 `pyproject.toml` 中：

| 类别 | 内容 |
|---|---|
| Python | `>=3.10` |
| 核心依赖 | `numpy`、`Pillow`、`PyYAML`、`jsonschema` |
| Open3D | 可选依赖 `.[open3d]`，用于查看器、Mesh 和重建功能 |
| 当前 `.venv` | Python 3.10.12、NumPy 1.26.4、Open3D 0.19.0、Pillow 12.3.0、PyYAML 6.0.3、jsonschema 4.26.0 |
| 当前未声明/未安装 | `torch`、`cv2`、`trimesh` |

当前 grasp CRUD 和 `compose-grasp` 不依赖 Open3D；`view-grasps` 需要 Open3D 及可用的图形窗口。仓库本身没有 GraspNet 推理模块或 GraspNet 依赖声明，因此不能把本项目的 grasp 候选管理当作 GraspNet 推理环境。

## 3. 已实现能力

### 3.1 候选数据契约

契约位于 [`src/robot_grasp/grasps.schema.json`](../src/robot_grasp/grasps.schema.json)，候选字段包括：

- `id`
- `T_object_grasp`
- `pregrasp_offset`
- `gripper_width`
- `approach_distance`
- `priority`
- `enabled`
- `symmetry_class`
- `notes`

全局约定为：

- 长度单位是米。
- `T_dst_src` 把 `src` 坐标系中的点转换到 `dst` 坐标系。
- `T_object_grasp` 是对象坐标系到抓取坐标系的位姿变换。
- `pregrasp_offset` 文档约定为抓取坐标系中的 `[x, y, z]` 偏移。

`validate_grasps()` 除 JSON Schema 外还检查：

- 抓取 ID 不重复。
- `T_object_grasp` 是有限值、右手系、正交的 4x4 刚体变换。
- 所有长度字段是有限数值。
- `gripper_width` 和 `approach_distance` 不小于零。

### 3.2 CRUD 和查看器

实现位于 [`src/robot_grasp/grasps.py`](../src/robot_grasp/grasps.py) 和 [`src/robot_grasp/cli.py`](../src/robot_grasp/cli.py)。当前支持：

```bash
robot-grasp grasps list --grasps examples/grasps.json
robot-grasp grasps show --grasps examples/grasps.json --id side_handle
robot-grasp grasps add --grasps <grasps.json> --object-id <object> \
  --id <id> --transform <T_object_grasp.json>
robot-grasp grasps update --grasps <grasps.json> --id <id> --priority 20
robot-grasp grasps delete --grasps <grasps.json> --id <id>
```

Open3D 查看器位于 [`src/robot_grasp/viewer.py`](../src/robot_grasp/viewer.py)。它显示对象坐标轴和当前抓取坐标轴，并支持平移、旋转、切换、保存和删除候选。它不连接机器人，也不加载真实夹爪碰撞几何。

### 3.3 坐标组合

`compose-grasp` 从三个 JSON 输入读取：

- `T_base_camera`
- `T_camera_object`
- `T_object_grasp`

组合实现位于 [`src/robot_grasp/transforms.py`](../src/robot_grasp/transforms.py)：

```text
T_base_grasp = T_base_camera @ T_camera_object @ T_object_grasp
```

命令会过滤禁用候选，并按 `priority` 选择一个候选，然后输出 `selected_grasp_id` 和 `T_base_grasp`。

## 4. 验证结果

使用仓库自己的虚拟环境执行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

结果（截至 2026-09-04）：

```text
34 passed
```

已验证内容包括：

- 4x4 变换和逆变换。
- 容易写反的抓取组合方向。
- 非法旋转矩阵拒绝。
- 抓取候选新增、更新、删除和重复 ID 拒绝。
- 禁用候选、优先级和可达性回调的选择行为。
- `compose-grasp` CLI 输出。

示例 dry-run：

```bash
.venv/bin/robot-grasp compose-grasp \
  --camera-pose examples/T_base_camera.json \
  --object-pose examples/T_camera_object.json \
  --grasps examples/grasps.json
```

该命令能够输出合法的 `T_base_grasp`。这只证明文件读取、候选选择和矩阵方向正确，不证明候选可达、无碰撞或能够完成抓取。

## 5. 主要缺口

### 5.1 `compose-grasp` 不是可执行计划

当前命令只输出 `T_base_grasp`，没有输出或使用：

- `T_base_pregrasp`
- `T_base_retreat`
- 夹爪打开/闭合目标
- 接近和撤退速度
- 力或闭合状态
- 碰撞、IK 和规划结果

如果 `pregrasp_offset` 确实是在抓取坐标系中表达，后续通常应按以下方式计算：

```text
T_base_pregrasp = T_base_grasp @ Trans(pregrasp_offset)
```

但当前代码没有实现这一计算，也没有 retreat 偏移字段。

### 5.2 候选选择没有执行约束

[`GraspSelector`](../src/robot_grasp/interfaces.py) 目前先过滤 `enabled`，再按 `priority` 取最大值。可选的 reachability 回调接收原始候选字典，而不是组合后的基座位姿；代码没有碰撞、IK、路径和桌面净空检查。

因此高优先级候选可能仍然不可执行。

### 5.3 字段语义约束不足

Schema 没有约束 `pregrasp_offset` 与 `approach_distance` 的关系。例如以下不一致数据当前仍会通过：

```text
pregrasp_offset = [0, 0, 0]
approach_distance = 0.1
```

`gripper_width` 也没有与任何具体夹爪的最小/最大开口关联；例如任意很大的宽度仍可通过通用 schema 校验。

此外，候选没有记录抓取坐标轴的明确含义、TCP 偏置、夹爪型号或接触区域。

### 5.4 资产和模型输出不完整

当前 `models/objects/` 只有目录占位文件，没有经过验收的对象 Mesh 与 `grasps.json` 部署包。`examples/grasps.json` 是格式示例，不能作为真实抓取数据。

本项目也没有统一的模型推理输出到 `grasps.json` 的转换契约，候选来源、置信度、时间戳和模型版本没有被记录。

### 5.5 查看器和验收范围有限

查看器只显示坐标轴，不显示夹爪实体、指尖接触区、接近路径或碰撞状态。现有验收模块主要面向 Mesh 尺寸和位姿重复性，不包含抓取成功率或候选执行验收。

## 6. 建议的下一步

建议只在本仓库内按以下顺序推进：

1. 定义 `GraspPlan` 数据结构和 schema，输出 pregrasp、grasp、retreat、夹爪参数及状态。
2. 明确 `pregrasp_offset` 的坐标语义，决定它与 `approach_distance` 是互相校验还是保留一个字段。
3. 让候选评估回调接收组合后的计划，记录每个候选的接受或拒绝原因。
4. 增加候选的抓取坐标轴、TCP 偏置、夹爪约束和来源元数据。
5. 为本地变换、偏移组合、候选筛选和无可执行候选增加测试。
6. 扩展查看器，至少显示夹爪简化几何和 pregrasp-to-grasp 路径。
7. 只有上述契约稳定后，再增加具体推理后端或外部执行适配。

## 7. 当前状态摘要

| 项目 | 状态 |
|---|---|
| 抓取候选 JSON schema | 已实现并有单元测试 |
| 候选 CRUD | 已实现并有单元测试 |
| 离线坐标组合 | 已实现并有 CLI 测试 |
| 抓取坐标轴和长度单位约定 | 已有基础约定，仍需补充夹爪/TCP 语义 |
| pregrasp/retreat 计划 | 未实现 |
| IK、碰撞和路径筛选 | 未实现 |
| 真实夹爪控制参数 | 未实现 |
| 可部署对象抓取资产 | 当前 checkout 中不存在 |
| 实物抓取验收 | 未实现 |
