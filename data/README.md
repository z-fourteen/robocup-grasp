# 数据目录

本目录只存放 `robot_grasp` 管线的数据，不存 BundleSDF 数据。大型数据默认不纳入 Git。

- `raw/`：相机原始采集或原样下载的数据，只读保存，不在这里做清洗。
- `interim/`：已导入并通过格式校验的 RGB-D 序列，例如 BOP 转换结果。
- `processed/`：去噪、裁剪、重新标注等可复用的派生数据。

建议按对象和采集批次命名，例如：

```text
raw/cup_01/2026-08-27_run01/
interim/cup_01/2026-08-27_run01/
processed/cup_01/v1/
```

每个 RGB-D 序列内部仍使用 `rgb/`、`depth/`、`masks/`、`poses/`、
`intrinsics.json` 和 `metadata.json` 的固定格式。需要版本化大型数据时，应接入 DVC、对象存储或
NAS，不应直接提交到 Git。
