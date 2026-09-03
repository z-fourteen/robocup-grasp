FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    MAX_JOBS=4 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      freeglut3-dev \
      git \
      libblas-dev \
      libboost-all-dev \
      libeigen3-dev \
      libgl1-mesa-dev \
      libglu1-mesa-dev \
      liblapack-dev \
      libopencv-dev \
      libpcl-dev \
      libyaml-cpp-dev \
      libzmq3-dev \
      ninja-build \
      pkg-config \
      pybind11-dev \
      && rm -rf /var/lib/apt/lists/*

# Match BundleSDF's current official Dockerfile where possible. NumPy remains
# at 1.26 because several upstream geometry packages are not NumPy 2 clean.
RUN python -m pip install --no-cache-dir --timeout 1000 --retries 10 \
      "Cython>=0.29.37" \
      dearpygui \
      einops \
      imageio \
      joblib \
      kornia \
      matplotlib \
      numpy==1.26.4 \
      open3d==0.19.0 \
      opencv-python-headless==4.11.0.86 \
      pandas \
      pillow \
      pyglet==1.5.31 \
      pyrender \
      rtree \
      ruamel.yaml \
      scikit-image \
      scikit-learn \
      scipy \
      transformations \
      trimesh \
      tqdm \
      yacs

# Source builds are required for PyTorch 2.7/CUDA 12.8 and sm_120.
RUN python -m pip install --no-cache-dir --timeout 1000 --retries 10 \
      "https://github.com/facebookresearch/pytorch3d/archive/refs/heads/main.zip"
RUN python -m pip install --no-cache-dir --timeout 1000 --retries 10 --no-build-isolation \
      "https://github.com/NVIDIAGameWorks/kaolin/archive/refs/tags/v0.18.0.zip"

# BundleTrack uses the boost::shared_ptr API from PCL 1.10. Ubuntu 22.04's
# PCL 1.12 changed this API to std::shared_ptr and cannot compile the official
# BundleSDF sources unchanged.
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -L --retry 5 --retry-all-errors \
      -o /tmp/pcl-1.10.0.tar.gz \
      https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-1.10.0.tar.gz && \
    tar -xzf /tmp/pcl-1.10.0.tar.gz -C /tmp && \
    cmake -S /tmp/pcl-pcl-1.10.0 -B /tmp/pcl-pcl-1.10.0/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_CXX_FLAGS=-std=c++17 \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_apps=OFF \
      -DBUILD_CUDA=OFF \
      -DBUILD_examples=OFF \
      -DBUILD_global_tests=OFF \
      -DBUILD_GPU=OFF \
      -DBUILD_simulation=OFF \
      -DBUILD_tools=OFF \
      -DPCL_ONLY_CORE_POINT_TYPES=ON \
      -DWITH_VTK=OFF && \
    cmake --build /tmp/pcl-pcl-1.10.0/build -j4 && \
    cmake --install /tmp/pcl-pcl-1.10.0/build && \
    rm -rf /tmp/pcl-1.10.0.tar.gz /tmp/pcl-pcl-1.10.0 /var/lib/apt/lists/*

# Match the versions used by BundleSDF's official Dockerfile. The Ubuntu
# OpenCV package is CPU-only and does not provide cudafeatures2d/cudaoptflow.
RUN python -m pip uninstall -y opencv-python-headless && \
    python -m pip install --no-cache-dir --timeout 1000 --retries 10 \
      pybind11==2.13.0 kornia==0.4.1

RUN git clone --depth 1 --branch 4.11.0 https://github.com/opencv/opencv.git /tmp/opencv && \
    git clone --depth 1 --branch 4.11.0 https://github.com/opencv/opencv_contrib.git /tmp/opencv_contrib && \
    cmake -S /tmp/opencv -B /tmp/opencv/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
      -DOPENCV_ENABLE_NONFREE=ON \
      -DBUILD_LIST=core,imgproc,imgcodecs,highgui,features2d,calib3d,rgbd,cudev,cudaarithm,cudafilters,cudaimgproc,cudafeatures2d,cudaoptflow,python3 \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_opencv_cudacodec=OFF \
      -DBUILD_opencv_python2=OFF \
      -DBUILD_opencv_python3=ON \
      -DCUDA_ARCH_BIN=12.0 \
      -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DENABLE_PRECOMPILED_HEADERS=OFF \
      -DPYTHON3_EXECUTABLE=/opt/conda/bin/python \
      -DPYTHON3_INCLUDE_DIR=/opt/conda/include/python3.11 \
      -DPYTHON3_LIBRARY=/opt/conda/lib/libpython3.11.so \
      -DPYTHON3_NUMPY_INCLUDE_DIRS=/opt/conda/lib/python3.11/site-packages/numpy/core/include \
      -DPYTHON3_PACKAGES_PATH=/opt/conda/lib/python3.11/site-packages \
      -DWITH_CUDA=ON \
      -DWITH_CUDNN=OFF \
      -DWITH_FFMPEG=OFF \
      -DWITH_GTK=OFF \
      -DWITH_OPENGL=OFF \
      -DWITH_QT=OFF && \
    cmake --build /tmp/opencv/build -j4 && \
    cmake --install /tmp/opencv/build && \
    rm -rf /tmp/opencv /tmp/opencv_contrib

# BundleTrack includes pybind11 headers in its main C++ target, outside the
# pybind11_add_module target, so expose the pip-installed headers globally.
ENV CPLUS_INCLUDE_PATH=/opt/conda/lib/python3.11/site-packages/pybind11/include

WORKDIR /workspace/BundleSDF
