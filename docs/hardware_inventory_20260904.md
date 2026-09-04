# 实机硬件清单

日期：2026-09-04

## RealSense

本机使用 librealsense `2.58.x` 工具和系统 Python `3.10.12` 的 `pyrealsense2` 完成只读枚举。
检测到三台 USB 3.2 设备：

| 设备 | 序列号 | 固件 | 深度尺度（m/raw） | 角色建议 |
|---|---|---|---:|---|
| Intel RealSense D455 | `260722304986` | `5.17.3.10` | `0.0010000000475` | 主 RGB-D 建库候选 |
| Intel RealSense D405 | `260322272099` | `5.15.1.55` | `0.00009999999747` | 近距离候选 |
| Intel RealSense D405 | `260322273941` | `5.15.1.55` | `0.00009999999747` | 近距离候选 |

R1 bringup 当前配置的活动 profile 为：D455（head）RGB/depth `640x480 @ 15 FPS`，两台 D405
（left/right）RGB/depth `640x480 @ 30 FPS`，格式均为 RGB8/Z16。RealSense SDK 读取到的活动内参和
深度尺度已写入每次 `probe_realsense.py --output` 的 JSON profile；不要用不同分辨率的内参替代。
D405 单次探测的 RGB/depth 时间差约 `0.31 ms`（启动样本可能有约 `100 ms` 瞬态）；D455 启动样本
观察到约 `136 ms` 瞬态，稳定帧约 `0.005 ms`，因此采集程序必须做 warm-up、记录逐帧时间差并将
启动瞬态排除在同步统计之外。

D455 本轮 `640x480@15` 实际采集会话的 color 内参为 `fx=386.9253`、`fy=386.3872`、
`cx=316.4272`、`cy=248.7384`；depth 内参为 `fx=fy=390.4359`、`cx=325.0951`、`cy=234.2213`。
内参会随活动 profile 和设备校准状态记录在 `camera_profile.json`，生产采集应以同一会话输出为准。

RealSense SDK 能提供设备信息、内参、深度尺度和设备内部 depth-to-color 外参，但不能提供相机相对
机器人 base/tool 的手眼变换。相机实际安装方式和 `T_tool_camera`/`T_base_camera` 仍需现场确认和独立标定。

## 机器人与夹爪

机器人描述和 bringup launch 显示为 R1 移动底盘，左右机械臂型号分别为 AR5L08/AR5R08，左右地址
分别为 `192.168.11.60`、`192.168.11.61`，默认 6 关节，控制端口 `8090`、实时端口 `10001`、
控制模式 `2`。当前配置文件记录的 Zenoh `device_group_name` 为 `vla`；Joint_angle 的 payload
语义、采集使用的手臂和末端 frame 仍需运行时确认。

夹爪通过 `/dev/ttyUSB0`/`/dev/ttyUSB1` 与 Zenoh 话题交互，但型号、序列号、TCP 偏置和开口标定
尚未提供。因此当前不能把采集结果提升为包含机器人位姿的严格实机序列。

本次检查时 ROS 2 没有运行节点，`/left_arm/joint_states`、`/right_arm/joint_states` 和 `/joint_states`
均无发布者；因此没有从机器人侧读取到关节样本或 TF。后续采集必须在 bringup/状态发布运行后，将同一
时钟域的关节状态与 RealSense JSONL 按时间戳配对，再通过 FK 和手眼标定计算 `T_base_camera`。

## 相机安装关系

`robot_bringup/load_robot_description.launch.py` 加载 `r1_with_arm.xacro`。其中 head 相机支架由
`head_Link -> camera_H_link` 固定关节描述；左右 D405 由 `left_moduan_link -> L_camera_d405_link`
和 `right_moduan_link -> R_camera_d405_link` 固定关节描述。RealSense launch 使用 `head_camera`、
`left_camera`、`right_camera` namespace/name，但显式设置 `publish_tf: False`，所以 SDK 能读取的
只是相机内部 depth-to-color 外参，不能替代相机到机器人坐标系的手眼变换。物理序列号与上述 URDF
安装关系仍需现场逐台确认。

完整机器配置保存在 [`configs/hardware.rpp.yaml`](../configs/hardware.rpp.yaml)。
