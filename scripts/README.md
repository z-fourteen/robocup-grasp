# 集成脚本

本目录用于放置调用核心包的薄脚本，例如相机采集、机器人 SDK 适配、批量导入和部署启动脚本。
可复用的校验、变换和抓取逻辑应放在 `src/robot_grasp/`，不要复制到脚本中。

## 当前实机适配

`probe_realsense.py` 通过 `pyrealsense2` 读取连接设备的型号、序列号、固件、流内参、深度尺度、
RGB/depth 时间戳和设备内参外参：

```bash
/usr/bin/python3 scripts/probe_realsense.py
/usr/bin/python3 scripts/probe_realsense.py --serial 260722304986 \
  --output outputs/realsense_d455_profile.json
```

当前机器检测到的设备清单和机器人/Zenoh 配置见
[`configs/hardware.rpp.yaml`](../configs/hardware.rpp.yaml)。由于同一主机连接了多台 RealSense，
采集命令必须显式指定 `--serial`，不能依赖设备枚举顺序。

`capture_realsense.py` 只保存原始 RGB/depth 和时间戳 JSONL，不伪造机器人位姿：

```bash
/usr/bin/python3 scripts/capture_realsense.py \
  --serial 260722304986 \
  --output data/raw/cup_01/2026-09-04_run01 \
  --frames 30
```

生成的 `realsense_frames.jsonl` 需要与同一时钟域的机器人姿态、FK 和手眼标定结果合并，之后再用
`robot-grasp capture-sequence` 转换为严格序列。D455 首帧可能出现启动瞬态，报告中的时间差统计必须
在多帧采集后再用于同步验收。

机器人侧可使用 `record_zenoh_json.py` 记录 RPP 文档中定义的 JSON 话题：

```bash
/usr/bin/python3 scripts/record_zenoh_json.py \
  --device-group-name <device_group_name> \
  --side right \
  --output data/raw/cup_01/2026-09-04_run01/robot_zenoh.jsonl \
  --duration 30
```

该脚本默认记录 `Joint_angle`、`Relative_pose` 和 `Gripper_status`。它只保留原始 JSON 和消息时间戳，
不把 `Joint_angle.angle[8]` 猜测为 FK 位姿。Python 依赖是可选的，见
[`requirements/zenoh.txt`](../requirements/zenoh.txt)；fastcdr payload 需要另外提供消息类型解码器。

## 中立帧清单转换

厂商采集适配器应输出一行一个 JSON 对象的 manifest，至少包含 `stem`、`rgb`、`depth`、
`T_base_camera`、`rgb_timestamp_ns`、`depth_timestamp_ns`、`pose_timestamp_ns` 和 `pose_source`。
转换命令为：

```bash
robot-grasp capture-sequence \
  --manifest /path/to/frames.jsonl \
  --config /path/to/capture_profile.yaml \
  --output data/interim/cup_01/run01
```
