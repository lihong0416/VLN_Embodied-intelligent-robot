# syntax=docker/dockerfile:1
# ------------------------------------------------------------
# ROS2 Humble + Camera(RealSense/Orbbec runtime deps) + Voice + LLM + Torch(GPU) 一体化镜像
# 目标：容器内直接跑 rpp_control_cli.py / llm_cam_ros2_nav.py / cam_nav_map_shihe.py / main_llm_nav.py
#
# 基础镜像选用 CUDA 12.1 + Ubuntu 22.04（方便 YOLO/torch 走 GPU）
# ------------------------------------------------------------

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-lc"]

WORKDIR /app

# ---- 1) System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg2 lsb-release \
      python3 python3-pip python3-dev python3-venv \
      git \
      build-essential cmake pkg-config \
      ffmpeg \
      portaudio19-dev \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      libusb-1.0-0 libusb-1.0-0-dev libudev1 libudev-dev \
      udev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# ---- 2) Install ROS2 Humble (ros-base) ----
RUN mkdir -p /etc/apt/keyrings \
 && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
    | gpg --dearmor -o /etc/apt/keyrings/ros-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-ros-base \
      python3-colcon-common-extensions \
      python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*

# ---- 3) (Optional) RealSense runtime ----
# 说明：RealSense 更推荐“宿主机装 librealsense”，容器只跑 pyrealsense2。
# 如果你希望容器也装 librealsense2，可把下面段落保留；如果构建失败，可注释掉。
RUN set -eux; \
    mkdir -p /etc/apt/keyrings; \
    curl -sSL https://librealsense.intel.com/Debian/librealsense.pgpkey \
      | gpg --dearmor -o /etc/apt/keyrings/librealsense.gpg; \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.gpg] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/librealsense.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends librealsense2-utils librealsense2-dev; \
    rm -rf /var/lib/apt/lists/*

# ---- 4) Python deps ----
RUN pip install \
      requests \
      numpy \
      opencv-python \
      sounddevice \
      scipy \
      pyyaml \
      faster-whisper \
      pyrealsense2

# ---- 5) Torch (GPU) ----
# 默认按 CUDA 12.1 安装（如需改版本，可调整 index-url）
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio

# ---- 6) Copy project ----
COPY . /app

# ---- 7) ROS env for runtime ----
# 让多数脚本不必手动 source；脚本里也有自动 re-exec 逻辑
ENV ROS_DISTRO=humble \
    ROS_DOMAIN_ID=0 \
    ROS_LOCALHOST_ONLY=0

# 可选：如果 Orbbec SDK 需要额外 so 路径，可在这里追加（按你实际目录调整）
# ENV LD_LIBRARY_PATH=/app/pyorbbecsdk-2-main/lib:${LD_LIBRARY_PATH}

# ---- 8) Entrypoint：自动 source ROS ----
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'source /opt/ros/humble/setup.bash' \
'exec "$@"' \
> /ros_entrypoint.sh \
 && chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]

# 默认进 bash，你也可以 docker run ... python3 llm_cam_ros2_nav.py 直接跑
CMD ["bash"]
