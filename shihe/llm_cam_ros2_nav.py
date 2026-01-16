#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llm_cam_ros2_nav.py  (大模型端：相机 + 语音/文本 + Ollama/Qwen + ROS2 /cmd_vel)

目标：
1) 从 RGB-D 相机读取 (BGR, depth_meters)
2) 提取简单可通行信息：前/左/右可用距离（米）
3) (可选) 语音转文本 / 或直接键盘输入
4) 调用 Ollama(Qwen) 输出严格 JSON 动作：
   - {"type":"move","meters":0.5,"speed":0.15}
   - {"type":"turn","deg":90,"wz":0.4}
   - {"type":"stop"}
   - {"type":"plan","steps":[ ...同上... ]}
5) 执行动作：连续发布 /cmd_vel (20Hz) 持续 duration，结束自动停，支持继续下一条
6) 输入 exit/quit/q 退出：自动停车

重要前提（不做执行器的情况下）：
- 机器人端必须已经运行 rpp_ros_driver（订阅 /cmd_vel）
- 大模型端必须能通过 ROS2 DDS 发现到机器人（同网段、同 ROS_DOMAIN_ID、允许 multicast）

运行示例（Linux/WSL2）：
  source /opt/ros/humble/setup.bash
  export ROS_DOMAIN_ID=0
  unset ROS_LOCALHOST_ONLY
  python3 llm_cam_ros2_nav.py --model qwen2.5:7b --camera dummy
"""

import argparse
import base64
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

# ---------------------------
# 0) ROS 环境自动 re-exec（Linux/WSL 可用）
# ---------------------------
ROS_ENV_BASH = (
    "source /opt/ros/humble/setup.bash && "
    "export ROS_DOMAIN_ID=0 && "
    "unset ROS_LOCALHOST_ONLY && "
)

def _need_reexec_with_ros_env() -> bool:
    # 没有 AMENT_PREFIX_PATH 通常意味着没 source ROS 环境
    ap = os.environ.get("AMENT_PREFIX_PATH", "")
    return ("/opt/ros/humble" not in ap)

def _reexec_with_ros_env(argv: List[str]) -> None:
    this = os.path.abspath(argv[0])
    args = " ".join(shlex.quote(a) for a in argv[1:])
    cmd = ROS_ENV_BASH + f"python3 {shlex.quote(this)} {args}"
    os.execvp("bash", ["bash", "-lc", cmd])

# 先解析参数（为了可以在 rclpy.init 前设置 DDS 配置）
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b", help="Ollama 模型名，例如 qwen2.5:7b / qwen2.5-vl:7b")
    ap.add_argument("--ollama-url", default="http://localhost:11434/api/chat", help="Ollama /api/chat 地址")
    ap.add_argument("--camera", default="dummy", choices=["dummy", "realsense", "custom"], help="相机类型")
    ap.add_argument("--send-image", action="store_true", default=False, help="把 RGB 图编码后发给模型（需要 VL 模型支持）")
    ap.add_argument("--topic", default="/cmd_vel")
    ap.add_argument("--hz", type=float, default=20.0, help="发布 /cmd_vel 的频率")
    ap.add_argument("--voice", action="store_true", default=False, help="启用语音识别（需要 sounddevice + faster-whisper）")
    ap.add_argument("--record-seconds", type=float, default=3.5, help="每次录音时长（秒）")
    ap.add_argument("--min-front", type=float, default=0.55, help="安全前方距离阈值（米），用于提示模型")
    ap.add_argument("--default-speed", type=float, default=0.15, help="默认前进速度 m/s")
    ap.add_argument("--default-wz", type=float, default=0.40, help="默认角速度 rad/s")
    ap.add_argument("--dds-peer", default="", help="（可选）指定对端 DDS peer（例如机器人IP）用于某些网络下发现失败时")
    return ap.parse_args()

ARGS = parse_args()

# （可选）某些 WiFi/路由下 multicast 不通，可以试试 CycloneDDS + unicast peer
def maybe_set_dds_peer(peer_ip: str):
    if not peer_ip:
        return
    # 仅做一个“尽量能用”的配置：要求你机器上 rmw_cyclonedds_cpp 可用
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    xml = f"""
<CycloneDDS>
  <Domain>
    <General>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <Peers>
        <Peer address="{peer_ip}"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
""".strip()
    os.environ["CYCLONEDDS_URI"] = xml

maybe_set_dds_peer(ARGS.dds_peer)

# Linux/WSL：没 source 就自动 reexec
if sys.platform.startswith("linux") and _need_reexec_with_ros_env():
    _reexec_with_ros_env(sys.argv)

# 现在再导入 ROS2
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
except Exception as e:
    print("[ERROR] 无法导入 ROS2 Python (rclpy)。你需要在大模型端安装/配置 ROS2 humble。")
    print("原始错误：", repr(e))
    sys.exit(1)

# ---------------------------
# 1) 相机适配层（你只需要改 custom 这一块）
# ---------------------------

class CameraBase:
    def start(self): ...
    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回：
          bgr: uint8 [H,W,3]
          depth_m: float32 [H,W]，单位米，缺失用 0 或 NaN
        """
        raise NotImplementedError
    def stop(self): ...

class DummyCamera(CameraBase):
    def start(self):
        self.h, self.w = 480, 640
    def read(self):
        bgr = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # 做个假的深度：中间远一点，两侧近一点
        depth = np.ones((self.h, self.w), dtype=np.float32) * 1.2
        depth[:, : self.w//4] = 0.6
        depth[:, -self.w//4:] = 0.7
        return bgr, depth
    def stop(self): pass

class RealSenseCamera(CameraBase):
    def __init__(self):
        self.pipeline = None
        self.align = None

    def start(self):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense: empty frame")

        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * float(self.depth_scale)
        bgr = np.asanyarray(color_frame.get_data()).copy()  # already bgr8
        return bgr, depth

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()

class CustomCamera(CameraBase):
    """
    ✅ 你把这里替换成“你相机SDK已经搞好的那套取帧代码”即可。

    你只要保证 read() 最终返回：
      - bgr: uint8 HxWx3
      - depth_m: float32 HxW (米)
    """
    def start(self):
        # TODO: 初始化你自己的 RGB-D pipeline
        raise NotImplementedError("请把 CustomCamera.start/read 替换成你的相机SDK实现")

    def read(self):
        raise NotImplementedError

    def stop(self):
        pass

def make_camera(name: str) -> CameraBase:
    if name == "dummy":
        return DummyCamera()
    if name == "realsense":
        return RealSenseCamera()
    if name == "custom":
        return CustomCamera()
    raise ValueError(name)

# ---------------------------
# 2) 深度→可通行特征（给 LLM 用）
# ---------------------------

def robust_min_depth(depth_m: np.ndarray, q: float = 0.10) -> float:
    """取某个 ROI 深度的 q 分位数，抗噪声。"""
    d = depth_m.copy()
    d = d[np.isfinite(d)]
    d = d[d > 0.05]
    if d.size == 0:
        return 0.0
    return float(np.quantile(d, q))

def sense_clearance(depth_m: np.ndarray) -> Dict[str, float]:
    """
    把深度图切成 3 个 ROI：左/中/右，输出可用距离（米）。
    你后面也可以扩展：上方障碍/地面过滤/占据栅格等。
    """
    h, w = depth_m.shape[:2]
    y1, y2 = int(h * 0.35), int(h * 0.85)  # 主要看前方中下区域
    left = depth_m[y1:y2, : w//3]
    mid  = depth_m[y1:y2, w//3: 2*w//3]
    right= depth_m[y1:y2, 2*w//3:]

    return {
        "left_m": robust_min_depth(left, 0.10),
        "front_m": robust_min_depth(mid, 0.10),
        "right_m": robust_min_depth(right, 0.10),
    }

# ---------------------------
# 3) Ollama/Qwen：强制 JSON 输出
# ---------------------------

SYSTEM_PROMPT = """你是机器人运动控制器，只输出 JSON，不要输出任何多余文字。
你必须从以下动作类型里选择，并严格遵守字段：
1) 直行：{"type":"move","meters":<float>,"speed":<float>}
2) 转向：{"type":"turn","deg":<float>,"wz":<float>}
3) 停止：{"type":"stop"}
4) 计划：{"type":"plan","steps":[<action1>,<action2>,...]}

规则：
- meters>0 表示前进，meters<0 表示后退
- deg>0 表示左转，deg<0 表示右转
- speed 和 wz 必须为正数
- 如果前方距离不足（front_m 很小），优先转向或停止
"""

def bgr_to_b64jpg(bgr: np.ndarray, quality: int = 75) -> str:
    import cv2
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("encode jpg failed")
    return base64.b64encode(enc.tobytes()).decode("utf-8")

def call_ollama_json(user_text: str, sense: Dict[str, float], b64img: Optional[str]) -> Dict[str, Any]:
    # 给模型的 user prompt（尽量短、结构化）
    hint = ""
    if sense.get("front_m", 0.0) < ARGS.min_front and sense.get("front_m", 0.0) > 0:
        hint = "注意：前方距离较近，优先转向或停止。"

    user_prompt = f"""用户指令：{user_text}

当前深度感知（单位米）：
- front_m={sense.get("front_m",0.0):.3f}
- left_m={sense.get("left_m",0.0):.3f}
- right_m={sense.get("right_m",0.0):.3f}
{hint}

请输出一个动作 JSON。"""

    payload: Dict[str, Any] = {
        "model": ARGS.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
    }

    # 如果是 VL 模型（例如 qwen2.5-vl），可以带图（Ollama 的 message 支持 images）
    if ARGS.send_image and b64img:
        payload["messages"][-1]["images"] = [b64img]

    resp = requests.post(ARGS.ollama_url, json=payload, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama 调用失败: {resp.status_code}, {resp.text}")

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return parse_llm_json(content)

def parse_llm_json(text: str) -> Dict[str, Any]:
    # 1) 直接 json loads
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) 从文本里抠第一段 {...} 或 [...]
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if not m:
        raise ValueError(f"模型输出不是 JSON：{text[:200]}")
    s = m.group(1)
    obj = json.loads(s)
    # 如果返回 list，当作 plan
    if isinstance(obj, list):
        return {"type": "plan", "steps": obj}
    if not isinstance(obj, dict):
        raise ValueError("JSON 不是 dict")
    return obj

# ---------------------------
# 4) ROS2 /cmd_vel 连续发布器（避免 0.5s watchdog 刹车）
# ---------------------------

@dataclass
class Vel:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

class CmdVelStreamer(Node):
    def __init__(self, topic="/cmd_vel", hz=20.0):
        super().__init__("llm_cmdvel_streamer")
        self.pub = self.create_publisher(Twist, topic, 10)
        self.topic = topic
        self.hz = float(hz)
        self._vel = Vel()
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_vel(self, vx: float, vy: float, wz: float):
        with self._lock:
            self._vel = Vel(float(vx), float(vy), float(wz))

    def stop_robot(self):
        self.set_vel(0.0, 0.0, 0.0)

    def get_sub_count(self) -> int:
        try:
            return int(self.pub.get_subscription_count())
        except Exception:
            return 0

    def _publish_once(self, vx: float, vy: float, wz: float):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self.pub.publish(msg)

    def _loop(self):
        dt = 1.0 / self.hz
        while self._running and rclpy.ok():
            with self._lock:
                v = self._vel
            self._publish_once(v.vx, v.vy, v.wz)
            time.sleep(dt)

    def shutdown(self):
        # 多发几次 0，确保停稳
        for _ in range(10):
            self._publish_once(0, 0, 0)
            time.sleep(0.03)
        self._running = False
        self._thread.join(timeout=1.0)

# ---------------------------
# 5) 执行动作（move/turn/plan）
# ---------------------------

def execute_move(streamer: CmdVelStreamer, meters: float, speed: float):
    speed = abs(float(speed))
    meters = float(meters)
    if speed <= 0:
        raise ValueError("speed 必须 > 0")
    duration = abs(meters) / speed
    vx = speed if meters >= 0 else -speed
    streamer.set_vel(vx, 0.0, 0.0)
    time.sleep(duration)
    streamer.stop_robot()
    time.sleep(0.25)

def execute_turn(streamer: CmdVelStreamer, deg: float, wz: float):
    wz = abs(float(wz))
    deg = float(deg)
    if wz <= 0:
        raise ValueError("wz 必须 > 0")
    rad = abs(deg) * np.pi / 180.0
    duration = rad / wz
    wz_cmd = wz if deg >= 0 else -wz
    streamer.set_vel(0.0, 0.0, wz_cmd)
    time.sleep(duration)
    streamer.stop_robot()
    time.sleep(0.25)

def execute_action(streamer: CmdVelStreamer, act: Dict[str, Any], default_speed: float, default_wz: float):
    t = str(act.get("type", "")).lower().strip()

    if t == "stop":
        streamer.stop_robot()
        return

    if t == "move":
        meters = float(act.get("meters", 0.0))
        speed = float(act.get("speed", default_speed))
        execute_move(streamer, meters, speed)
        return

    if t == "turn":
        deg = float(act.get("deg", 0.0))
        wz = float(act.get("wz", default_wz))
        execute_turn(streamer, deg, wz)
        return

    if t == "plan":
        steps = act.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("plan.steps 必须是 list")
        for i, s in enumerate(steps):
            if not isinstance(s, dict):
                raise ValueError(f"plan.steps[{i}] 不是 dict")
            execute_action(streamer, s, default_speed, default_wz)
        return

    raise ValueError(f"未知动作类型: {t}")

# ---------------------------
# 6) 可选：语音识别
# ---------------------------

class VoiceASR:
    def __init__(self):
        # 延迟导入，避免无依赖时报错
        import sounddevice as sd
        from scipy.io.wavfile import write as wav_write
        from faster_whisper import WhisperModel
        self.sd = sd
        self.wav_write = wav_write
        self.WhisperModel = WhisperModel

        # 你可以改成自己的模型路径/大小
        model_size = os.environ.get("ASR_MODEL", "small")
        device = os.environ.get("ASR_DEVICE", "cpu")
        compute_type = os.environ.get("ASR_COMPUTE_TYPE", "int8")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def record_and_transcribe(self, seconds: float = 3.5, samplerate: int = 16000) -> str:
        import tempfile
        self.sd.default.samplerate = samplerate
        self.sd.default.channels = 1
        audio = self.sd.rec(int(seconds * samplerate), dtype="float32")
        self.sd.wait()

        # 写 wav 临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        self.wav_write(path, samplerate, (audio * 32767).astype(np.int16))

        try:
            segments, info = self.model.transcribe(path, language="zh")
            text = "".join(seg.text for seg in segments).strip()
            return text
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

# ---------------------------
# 7) 主循环
# ---------------------------

def main():
    print("\n[INFO] 启动：LLM + RGB-D + ROS2 /cmd_vel 控制端（大模型端）")

    # 1) init ROS
    rclpy.init(args=None)
    streamer = CmdVelStreamer(topic=ARGS.topic, hz=ARGS.hz)

    # 2) 相机
    cam = make_camera(ARGS.camera)
    try:
        cam.start()
    except Exception as e:
        print("[WARN] 相机启动失败：", repr(e))
        print("       你可以先用 --camera dummy 验证 ROS+LLM 控制链路")
        cam = DummyCamera()
        cam.start()

    # 3) 可选语音
    asr = None
    if ARGS.voice:
        try:
            asr = VoiceASR()
            print("[INFO] 语音识别已启用。每次按回车开始录音。")
        except Exception as e:
            print("[WARN] 语音识别依赖不完整，已降级为键盘输入。错误：", repr(e))
            asr = None

    # 4) 等待机器人端订阅（如果网络能发现得到）
    print("[INFO] 等待 /cmd_vel 订阅者（机器人端 rpp_ros_driver）...")
    t0 = time.time()
    while time.time() - t0 < 8.0:
        cnt = streamer.get_sub_count()
        if cnt > 0:
            print(f"[OK] 已发现订阅者数量={cnt}，可以发控制了！")
            break
        time.sleep(0.2)
    else:
        print("[WARN] 未检测到 /cmd_vel 订阅者。")
        print("       可能原因：机器人端 driver 没启动，或 DDS 网络发现失败（同网段/防火墙/multicast）")
        print("       你仍然可以继续测试 LLM 输出和本地发布，但车未必会动。")

    print("\n用法：")
    print(" - 直接输入：'向前走0.5米' / '左转30度' / '后退0.2米' / '停止'")
    print(" - 输入 q/quit/exit 退出（会停车）\n")

    try:
        while True:
            # 读一帧相机
            bgr, depth_m = cam.read()
            sense = sense_clearance(depth_m)

            # 获取用户指令
            if asr:
                cmdline = input("按回车录音（或直接输入文字，q退出）> ").strip()
                if cmdline.lower() in ("q", "quit", "exit"):
                    break
                if cmdline == "":
                    text = asr.record_and_transcribe(seconds=ARGS.record_seconds)
                    print("[ASR]", text)
                    user_text = text
                else:
                    user_text = cmdline
            else:
                user_text = input("输入指令（q退出）> ").strip()
                if user_text.lower() in ("q", "quit", "exit"):
                    break

            if not user_text:
                continue

            # 可选：发图给 VL 模型
            b64img = None
            if ARGS.send_image:
                try:
                    b64img = bgr_to_b64jpg(bgr)
                except Exception as e:
                    print("[WARN] 图像编码失败，改为不发图：", repr(e))
                    b64img = None

            # 调用 LLM 得到动作 JSON
            try:
                act = call_ollama_json(user_text, sense, b64img)
                print("[LLM]", act)
            except Exception as e:
                print("[ERROR] LLM 调用/解析失败：", repr(e))
                continue

            # 执行动作（每条自动停，可继续下一条）
            try:
                execute_action(streamer, act, ARGS.default_speed, ARGS.default_wz)
                print("[OK] done\n")
            except Exception as e:
                print("[ERROR] 动作执行失败：", repr(e))
                streamer.stop_robot()

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C")
    finally:
        try:
            streamer.stop_robot()
            streamer.shutdown()
        except Exception:
            pass
        try:
            cam.stop()
        except Exception:
            pass
        try:
            streamer.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        print("[INFO] 退出完成（已停车）")

if __name__ == "__main__":
    main()
