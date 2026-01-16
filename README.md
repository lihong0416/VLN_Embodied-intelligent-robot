# ROS2 + 相机 + 语音 + LLM 一体化容器

这套仓库主要做两件事：

1) **语音/文字指令 → LLM(Qwen via Ollama) → 结构化 JSON → AGV API 字符串 →（可选）TCP 下发机器人**  
   - 入口：`main_voice.py`（连机器人） / `agv_voice_cli.py`（本地语音演示） / `agv_nul_cli.py`（本地文字演示）

2) **RGB-D 相机 → 场景/可通行信息 → LLM 输出动作 → ROS2 `/cmd_vel` 执行**  
   - 入口：`llm_cam_ros2_nav.py`（RealSense / dummy / custom，相机+LLM+ROS2 控制）
   - 入口：`rpp_control_cli.py`（交互式底盘控制台 + 自动启动 rpp_ros_driver）
   - 入口：`cam_nav_map_shihe.py`（Orbbec RGB-D→局部栅格→自动建议动作，只打印不发 ROS）
   - 入口：`main_llm_nav.py`（Orbbec + YOLO-3D + LLM 语义导航 demo，只打印）

> 你现在选择的是：**ROS2 + 相机也全塞进容器（重但一体化）**。  
> 这个方案会把：ROS2 Humble、常用视觉/语音/LLM 依赖、RealSense/Orbbec 运行库、Torch（GPU 版）都装进镜像。  
> 但请注意：**底盘串口 / 相机 USB / X11 GUI / DDS 网络发现** 都需要容器启动参数配合。

---

## 1. 需要准备的目录/文件

### 1.1 代码目录
 `.py` 都在根目录即可。除此以外，如果要跑 Orbbec/YOLO 的脚本，还需要把下面这些目录也放在项目根目录：

- `pyorbbecsdk-2-main/`（Orbbec SDK Python 工程，里面有 `examples/utils.py`）
- `YOLO-3D-main/`（YOLO-3D 工程，里面有 `detection_model.py` 等）
- `机器人_project_有图/`（工程里引用的一些模块/资源）

### 1.2 必要配置/资源文件（NLU 逻辑需要）
以下文件名在 `config_markers.py` / `prompts_nlu.py` 等里会被读取（或用于点位维护）：

- `maps.json`
- `markers.json`
- `actions_config.json`
- `current_map.json`（可选）
- `system_prompt*.txt`（提示词）
- `examples_nlu.json`（few-shot 示例）

> ⚠️ 注意：现在的 `prompts_nlu.py` 里 `EXAMPLES_FILE` 写的是 `package.json`；如果实际文件叫 `examples_nlu.json`，改一行就行。

### 1.3 离线 ASR 模型目录
为了容器离线也能语音识别，建议把 faster-whisper 模型目录也放到项目根目录，例如：

- `faster-whisper-small/`（目录里要能递归找到 `model.bin`）

`asr_client.py` 会扫描该目录来定位 `model.bin`。

---

## 2. Docker 镜像构建

### 2.1 前置条件（GPU + ROS2 + 相机）
- Linux 主机已安装 **Docker**
- 若要 GPU：安装 **NVIDIA Driver + NVIDIA Container Toolkit**
- 串口/相机设备在宿主机能正常识别：
  - 串口一般是 `/dev/ttyUSB0`
  - 相机 USB 在 `/dev/bus/usb/`

### 2.2 构建镜像
在项目根目录执行：

```bash
#下载相机SDK
git@github.com:orbbec/pyorbbecsdk.git

Open PowerShell with administrator privileges, then use the cd command to enter the directory where the obsensor_metadata_win10.ps1 script is located;
Execute .\obsensor_metadata_win10.ps1 -op install_all to complete the registration.

cd pyorbbecsdk/scripts
sudo chmod +x ./install_udev_rules.sh
sudo ./install_udev_rules.sh
sudo udevadm control --reload && sudo udevadm trigger

docker build -t agv-ros2-cam:latest .

#下载touch版本：
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -u -p $CONDA_DIR \
 && rm -f /tmp/miniconda.sh
 
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r  --channel https://repo.anaconda.com/pkgs/msys2
conda update --force conda
conda create -n VLN python=3.12 -y \
    && echo "conda activate aeq" >> ~/.bashrc
pip install requirements.txt
```

> 如果没有 GPU 或不需要 YOLO/torch，可在 Dockerfile 里把 torch 那段注释掉，构建会更快更稳。

---

## 3. 启动容器

下面给你一个“尽量一把梭”的启动方式（适合：ROS2 + `/cmd_vel` + 相机 USB + OpenCV 窗口显示）：

### 3.1 Linux
```bash
# 允许 root 容器访问你的 X11（只对本机有效）
xhost +local:root

docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev:/dev \
  agv-ros2-cam:latest bash
```

说明：
- `--network host`：Ollama（在宿主机）才能用 `localhost:11434` 访问；ROS2 DDS 发现也更容易成功
- `--privileged -v /dev:/dev`：让容器访问串口 + USB 相机
- `-v /tmp/.X11-unix ...`：让容器能弹 OpenCV 窗口

> 如果你不需要 GUI，把 X11 相关的 3 行删掉即可。

### 3.2 机器人 driver（rpp_ws）如何进容器？
`rpp_control_cli.py` 会 `source ~/rpp_ws/install/setup.bash`。你有两种方式：

**方式A：运行时挂载（推荐）**
```bash
# 宿主机上有 rpp_ws
docker run ... -v $HOME/rpp_ws:/root/rpp_ws agv-ros2-cam:latest bash
```

**方式B：把 rpp_ws 放到项目根目录，一起 COPY 进镜像**
- 你需要把 `rpp_ws/` 放在 Docker build context 中（项目根目录下），并自己在镜像里 `colcon build`（此仓库默认不自动 build，因为每个人 workspace 不一样）。

---

## 4. 容器内运行示例

### 4.1 文字 NLU 演示（不连机器人）
```bash
python3 agv_nul_cli.py
```

### 4.2 语音 NLU 演示（不连机器人）
```bash
python3 agv_voice_cli.py
```

### 4.3 真机语音控制（TCP 下发机器人）
```bash
python3 main_voice.py
```
你需要在 `main_voice.py` 里改机器人 IP/端口，或后续我帮你做成读取 `config.yml`。

### 4.4 ROS2 底盘控制台（自动启动 driver + 20Hz /cmd_vel）
```bash
python3 rpp_control_cli.py
```

### 4.5 LLM + 相机 → ROS2 /cmd_vel（RealSense）
```bash
python3 llm_cam_ros2_nav.py --camera realsense --model qwen2.5:7b --ollama-url http://localhost:11434/api/chat
```

### 4.6 LLM + dummy 相机（先验证 ROS2 + LLM 链路）
```bash
python3 llm_cam_ros2_nav.py --camera dummy
```

### 4.7 Orbbec RGB-D → 局部栅格 → 自动建议动作（只打印）
```bash
python3 cam_nav_map_shihe.py
```

### 4.8 Orbbec + YOLO-3D + LLM 语义导航 demo（只打印）
```bash
python3 main_llm_nav.py
```

---

## 5. config.yml
仓库提供了 `config.yml`，用于集中管理：机器人 IP、Ollama 地址、相机类型、ROS2 话题、ASR 模型目录等。  
**当前代码还是以硬编码为主**，如果你希望“所有脚本优先读取 config.yml”，我可以按最小改动给你补上。

---

## 6. 常见问题

1) **Ollama 调用失败**  
   - 宿主机确认：`curl http://127.0.0.1:11434/api/tags` 能返回
   - 容器必须 `--network host`，并使用 `http://localhost:11434/api/chat`

2) **ROS2 发现不到机器人 / /cmd_vel 没订阅者**  
   - 先用 `--network host`
   - 确认同网段、`ROS_DOMAIN_ID` 一致、路由器 multicast 不屏蔽  
   - 必要时在 `llm_cam_ros2_nav.py` 用 `--dds-peer <机器人IP>`（CycloneDDS 单播发现）

3) **串口打不开 /dev/ttyUSB0**  
   - 用 `--privileged -v /dev:/dev`
   - 宿主机先确认 `ls -l /dev/ttyUSB0`

4) **相机打不开（RealSense/Orbbec）**  
   - 宿主机先验证相机能用（`realsense-viewer` / Orbbec 示例）
   - 容器启动一定要带 USB 设备映射（`--privileged -v /dev:/dev`）

5) **OpenCV 窗口不显示**  
   - Linux GUI 环境下用 X11 映射（上面 3.1 的启动命令）
