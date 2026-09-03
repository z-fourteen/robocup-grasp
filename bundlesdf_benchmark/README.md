# BundleSDF Benchmark Workspace

本目录仅用于独立的 BundleSDF 环境和实验，不属于 `robot_grasp` Python 包。

```text
bundlesdf_benchmark/
  data/       # benchmark 输入和下载缓存
  models/     # LoFTR 等 BundleSDF 权重
  outputs/    # benchmark 结果
  docker/     # 独立构建环境
```

外部 BundleSDF 源码位于仓库根目录 `BundleSDF/`。构建镜像：

```bash
docker build -f bundlesdf_benchmark/docker/Dockerfile -t bundlesdf:local .
```

运行时分别挂载源码和实验目录：

```bash
docker run --gpus all --rm -it \
  -v "$PWD/BundleSDF:/workspace/BundleSDF" \
  -v "$PWD/bundlesdf_benchmark:/workspace/bundlesdf_benchmark" \
  bundlesdf:local
```
