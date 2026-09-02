# 模型目录

- `objects/<object_id>/`：可部署的对象资产，例如 `mesh_object.ply`、
  `mesh_collision.obj`、`object_frame.json` 和 `grasps.json`。
- `weights/<backend>/`：FoundationPose 等后端的下载权重。

`outputs/reconstruction/` 中的实验结果通过验收后，再复制为一个明确版本的对象包，例如
`models/objects/cup_01/v1/`。模型和权重默认不纳入 Git；如需团队共享，使用制品库或 DVC。
