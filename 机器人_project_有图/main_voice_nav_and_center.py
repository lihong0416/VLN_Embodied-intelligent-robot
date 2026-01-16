# main_voice_nav_and_center.py
"""
有图版：文本指令 → 底盘导航 → 到达后开启 RGB-D 相机自动找目标并转向居中

关键点：
- “去哪儿”的解析和发指令，完全沿用你原来的 main.py / main_voice.py 逻辑；
- 这里只是多做一步：到达点位后，开深度相机 + YOLO 找目标物体，并通过 joy_control 左右转让它到视野中心。
- 不做任何“兜底改 marker”的操作：如果大模型说当前地图没有 B，就不发送导航指令。

使用方式：
  cd D:\\机器人_无图_project\\机器人_project_有图
  conda activate pytorch_train
  python main_voice_nav_and_center.py

指令示例：
  去B点拿一个瓶子
  去A点看一下人

解析策略：
  - 先从整句里用简单关键词找“要拿什么”（瓶子/椅子/人）→ YOLO 类别名；
  - 再把整句按“拿/取”切开：
      导航子句：前半截（例如“去B点”）
      取物子句：后半截（例如“拿一个瓶子”）
    导航子句交给 Qwen → JSON → validate_command → build_robot_api_string → send_command_and_receive_response；
    取物子句只用来判断 YOLO 要找的类别。
"""

import os
import sys
import time
from typing import Optional, Tuple, List

# 允许 OpenMP 重复加载（PyTorch + 其它库时会用到）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import json as _json

# ========= 导入“有图”项目中的导航相关模块 =========

from model_client import call_qwen_local_and_get_json
from command_logic import validate_command, build_robot_api_string
from config_markers import get_current_map_name, set_current_map_name
import main as agv_main

# 和 main.py / main_voice.py 保持一致
HOST = "192.168.10.10"
PORT = 31001

# 复用 main.py 里的函数逻辑
send_command_and_receive_response = agv_main.send_command_and_receive_response
exec_joy_control_macro = agv_main.exec_joy_control_macro
normalize_turn_from_text = agv_main.normalize_turn_from_text
apply_marker_side_effects_after_robot = agv_main.apply_marker_side_effects_after_robot

# ========= 为 RGB-D + YOLO 添加路径，并导入相关模块 =========

# 本文件在 机器人_project_有图 目录，下一级是项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORB_DIR = os.path.join(PROJECT_ROOT, "pyorbbecsdk-2-main")
ORB_EXAMPLES_DIR = os.path.join(ORB_DIR, "examples")
YOLO_DIR = os.path.join(PROJECT_ROOT, "YOLO-3D-main")

for p in [ORB_DIR, ORB_EXAMPLES_DIR, YOLO_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from pyorbbecsdk import AlignFilter  # 类型提示用
from utils import frame_to_bgr_image
from detection_model import ObjectDetector
from main_RGB_D import (
    init_orbbec_pipeline,
    compute_depth_map_in_meters,
    get_distance_for_bbox,
)

# ========= 一些参数 =========

ESC_KEY = 27

CENTER_TOLERANCE_RATIO = 0.05    # 水平偏差占画面宽度比例 < 5% 视为“居中”
MAX_ANGULAR_VEL = 0.6            # joy_control 自动微调最大角速度
BASE_SCAN_VEL = 0.3              # 没目标时原地扫描角速度
STEP_DURATION_S = 0.4            # 每次 joy_control 持续时间
MAX_CENTERING_TIME_S = 60.0      # 找目标 + 对齐 最大时间（秒）

# ========= 中文 → YOLO 类别名映射 =========

ZH_TO_YOLO_CLASS = {
    "水瓶": "bottle",
    "矿泉水": "bottle",
    "瓶子": "bottle",
    "瓶": "bottle",
    "杯子": "cup",
    "杯": "cup",
    "人": "person",
    "行人": "person",
    "椅子": "chair",
    "椅": "chair",
    "书": "book",
    "手机": "cell phone",
    "电脑": "laptop",
}


def extract_target_class_from_text(text: str) -> Optional[str]:
    """
    从中文里抠出“要找什么东西” → YOLO 类别名。
    示例：
        “去B点拿一个瓶子” → "bottle"
        “去A点看一下人”   → "person"
    """
    if not text:
        return None
    low = text.lower()
    # 先匹配长的，避免“水瓶”被“瓶”先吃掉
    for zh, cls in sorted(ZH_TO_YOLO_CLASS.items(), key=lambda kv: -len(kv[0])):
        if zh in low:
            return cls
    return None


def split_nav_and_object(text: str) -> Tuple[str, str]:
    """
    根据“拿/取”把整句粗略切成：
      - 导航子句 nav_text：前半段（例如 “去B点”）
      - 取物子句 obj_text：后半段（例如 “拿一个瓶子”）
    如果没有找到“拿/取”，就认为整句都是导航，obj_text 为空。
    """
    if not text:
        return "", ""
    for kw in ["拿", "取"]:
        idx = text.find(kw)
        if idx != -1:
            nav = text[:idx]
            obj = text[idx:]
            return nav.strip(), obj.strip()
    return text.strip(), ""


# ========= 从 YOLO 检测结果中挑一个要对齐的目标 =========

def select_target_detection(
    detections: List,
    class_names: List[str],
    img_w: int,
    target_class: Optional[str],
) -> Optional[Tuple[Tuple[float, float, float, float], str]]:
    """
    detections 每个元素：(bbox, score, class_id, obj_id)
    返回：(bbox, class_name) 或 None
    """
    if not detections:
        return None

    center_x = img_w / 2.0
    candidates: List[Tuple[Tuple[float, float, float, float], str, float, float]] = []

    for det in detections:
        bbox, score, class_id, obj_id = det
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = 0.5 * (x1 + x2)
        offset = abs(cx - center_x)
        cname = class_names[int(class_id)]
        candidates.append((bbox, cname, float(score), offset))

    # 1) 如果指定了 target_class，就在该类别里找“最接近中心”的
    if target_class is not None:
        filtered = [d for d in candidates if d[1] == target_class]
        if filtered:
            filtered.sort(key=lambda x: x[3])
            bbox, cname, score, _ = filtered[0]
            return bbox, cname

    # 2) 否则就在所有目标里选“最接近中心”的
    candidates.sort(key=lambda x: x[3])
    bbox, cname, score, _ = candidates[0]
    return bbox, cname


# ========= RGB-D 主逻辑：开相机 + YOLO，自动找目标并左右转对齐 =========

def center_object_with_rgbd(
    target_class: Optional[str],
    max_time_s: float = MAX_CENTERING_TIME_S,
):
    """
    到达点位后调用：
    - 开 Orbbec RGB-D + YOLO；
    - 当 target_class 给了的时候：优先找这个类别；
    - 没有指定类别的时候：找画面中“最接近中心”的任意目标；
    - 检测不到目标时：原地缓慢旋转扫描；
    - 一旦目标在水平上基本居中，就停止。

    会真实往底盘发 joy_control 指令（左右转）。
    """

    print("\n[RGB-D] 启动 RGB-D + YOLO 自动对齐逻辑...")

    # 1. 初始化 Orbbec pipeline
    pipeline, config, align_filter = init_orbbec_pipeline()
    print("[RGB-D] Orbbec Pipeline 已启动。")

    # 2. 初始化 YOLO 检测器
    detector = ObjectDetector(
        model_size="medium",
        conf_thres=0.3,
        iou_thres=0.45,
        classes=None,
        device="cuda",   # 如需 CPU，可以改成 "cpu"
    )
    class_names = detector.get_class_names()
    print("[RGB-D] YOLO 检测器已初始化。类别：", class_names)

    start_time = time.time()
    scan_direction = 1  # 1 先向左转，-1 向右

    try:
        while True:
            now = time.time()
            if now - start_time > max_time_s:
                print("[RGB-D] 已超过最大居中时间，停止。")
                break

            frames = pipeline.wait_for_frames(100)
            if not frames:
                continue

            # 对齐：深度→彩色（保持和 main_RGB_D.py 一致）
            frames = align_filter.process(frames)
            if not frames:
                continue
            frames = frames.as_frame_set()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = frame_to_bgr_image(color_frame)
            depth_m = compute_depth_map_in_meters(depth_frame)
            if depth_m is None:
                continue

            h_img, w_img, _ = color_image.shape
            center_x = w_img / 2.0

            # YOLO 检测
            annotated_image, detections = detector.detect(color_image, track=True)

            # 选一个要对齐的目标
            target = select_target_detection(detections, class_names, w_img, target_class)

            if target is None:
                # 没检测到任何目标：原地缓慢旋转
                print("[RGB-D] 暂时没有检测到目标，原地缓慢旋转搜索...")
                ang_vel = BASE_SCAN_VEL * scan_direction

                cmd_turn = {
                    "action": "joy_control",
                    "linear_velocity": 0.0,
                    "angular_velocity": float(ang_vel),
                    "duration_s": STEP_DURATION_S,
                }
                exec_joy_control_macro(cmd_turn, HOST, PORT)

                # 每隔一段时间反向一次，避免一直朝一边转
                if int(now - start_time) % 10 == 0:
                    scan_direction *= -1

            else:
                bbox, class_name = target
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                distance_m = get_distance_for_bbox(depth_m, bbox)
                if distance_m is not None:
                    distance_text = f"{distance_m:.2f}m"
                else:
                    distance_text = "未知距离"

                # 画框 + 中心点 + 文字
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated_image, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                cv2.circle(annotated_image, (int(center_x), int(h_img / 2)), 4, (0, 0, 255), -1)

                label = f"{class_name} {distance_text}"
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # 水平偏差
                offset_px = cx - center_x
                offset_ratio = offset_px / center_x  # 左负右正

                print(
                    f"[RGB-D] 目标: {class_name}, 距离: {distance_text}, "
                    f"水平偏差: {offset_px:.1f}px ({offset_ratio:.3f})"
                )

                # 若偏差足够小，认为居中
                if abs(offset_ratio) <= CENTER_TOLERANCE_RATIO:
                    print("[RGB-D] 目标已经基本居中，停止自动对齐。")
                    cv2.putText(
                        annotated_image,
                        "CENTERED",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("RGB-D Auto Center", annotated_image)
                    cv2.waitKey(1000)
                    break
                else:
                    # 控制策略：目标在右 → 右转（角速度<0），目标在左 → 左转（角速度>0）
                    ang_vel = -float(offset_ratio) * MAX_ANGULAR_VEL
                    ang_vel = max(min(ang_vel, MAX_ANGULAR_VEL), -MAX_ANGULAR_VEL)

                    print(f"[RGB-D] 发送微调转动：angular_velocity={ang_vel:.3f} rad/s")
                    cmd_turn = {
                        "action": "joy_control",
                        "linear_velocity": 0.0,
                        "angular_velocity": float(ang_vel),
                        "duration_s": STEP_DURATION_S,
                    }
                    exec_joy_control_macro(cmd_turn, HOST, PORT)

            # 显示调试画面（q/ESC 退出自动对齐）
            cv2.imshow("RGB-D Auto Center", annotated_image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ESC_KEY):
                print("[RGB-D] 手动退出自动对齐。")
                break

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[RGB-D] 已停止相机与窗口。")


# ========= 文本 → 导航指令 → 到达后自动打开 RGB-D 对齐 =========

def handle_user_text(user_text: str):
    """
    单次处理用户的一条中文指令：
    1. 用 split_nav_and_object 把“去哪儿”和“拿什么”分开；
    2. 导航子句 nav_text 交给 Qwen → JSON → validate_command → build_robot_api_string → send_command_and_receive_response；
    3. 取物子句 / 整句用 extract_target_class_from_text 提取 YOLO 类别；
    4. 如果：
        - action == move_to_marker
        - 且提取到了 target_class
       则在导航后开启 RGB-D 找这个物体并居中。
    """
    if not user_text:
        return

    # 1) 分出“导航部分”和“拿东西部分”
    nav_text, obj_text = split_nav_and_object(user_text)
    if not nav_text:
        nav_text = user_text

    # 2) 从整句里解析目标物体类别（只决定 YOLO 找什么）
    target_class = extract_target_class_from_text(user_text)
    if target_class:
        print(f"[解析] 推测目标物体类别（YOLO 类别）：{target_class}")
    else:
        print("[解析] 文本中未解析出具体目标物体类别，本次只做导航。")

    print(f"\n[NLU] 导航子句发送给 Qwen 的内容：『{nav_text}』")

    # 3) 调 Qwen → JSON 指令（完全交给你的 prompts_nlu 逻辑）
    cmd = call_qwen_local_and_get_json(nav_text)

    print("\n[原始 JSON 指令（模型直接输出）]")
    print(_json.dumps(cmd, ensure_ascii=False, indent=2))

    # 4) 根据原始整句纠正 joy_control 左/右转方向（虽然导航一般不会用到）
    cmd = normalize_turn_from_text(cmd, user_text)

    # 5) 严格校验 / 补全（marker 不在地图里就让它报错，不做兜底）
    cmd = validate_command(cmd)

    action = cmd.get("action")

    # set_map：切换当前地图（只动内存）
    if action == "set_map":
        set_current_map_name(cmd["map_name"])
        print(f"[地图切换] 当前地图已切换为 {cmd['map_name']} (floor={cmd['floor']})")

    print("\n[校验 / 补全后的 JSON 指令]")
    print(_json.dumps(cmd, ensure_ascii=False, indent=2))

    # ask_user_clarification：只提示，不下发
    if action == "ask_user_clarification":
        print("\n[无法安全执行，不会下发指令]")
        print("原因：", cmd.get("reason", "模型未给出原因"))
        print("\n" + "=" * 60 + "\n")
        return

    # joy_control：走宏指令逻辑
    if action == "joy_control":
        response = exec_joy_control_macro(cmd, HOST, PORT)
        if response is not None:
            print("\n[机器人返回的最后一次响应 JSON]")
            print(_json.dumps(response, ensure_ascii=False, indent=4))
        else:
            print("\n[机器人响应无效或解析失败]")
        print("\n" + "=" * 60 + "\n")
        return

    # 其它动作（包括 move_to_marker）：一个 action 对应一个 API
    api_str = build_robot_api_string(cmd)

    print("\n[将要发送给机器人的 API 文本指令是]")
    print(api_str)

    response = send_command_and_receive_response(api_str, HOST, PORT)
    if response is not None:
        print("\n[机器人返回的响应 JSON]")
        print(_json.dumps(response, ensure_ascii=False, indent=4))
    else:
        print("\n[机器人响应无效或解析失败]")

    # 同步更新 maps.json（插入/删除点位）
    apply_marker_side_effects_after_robot(cmd, response)

    print("\n" + "=" * 60 + "\n")

    # 6) 如果本次是导航 + 有目标物体，就在“到达后”开 RGB-D 自动对齐
    if action == "move_to_marker" and target_class is not None:
        print("[提示] 本次指令包含“导航 + 要找的物体”，将在到达后开启 RGB-D 自动寻找并对齐。")
        input("请等待机器人到达目标点位，确认到达后按回车继续... ")
        center_object_with_rgbd(target_class=target_class, max_time_s=MAX_CENTERING_TIME_S)
    else:
        if action == "move_to_marker":
            print("[提示] 本次是纯导航指令（未解析出具体目标物体），不会自动开启 RGB-D。")


def main():
    print("=== 有图版：文本指令 → 底盘导航 → 到达后 RGB-D 自动对齐 Demo ===")
    try:
        cur_map = get_current_map_name()
        print(f"当前地图：{cur_map}")
    except Exception:
        print("当前地图：<获取失败，请检查 config_markers.py / maps.json>")

    print("\n使用说明：")
    print("  1. 在终端输入中文指令，例如：")
    print("       去B点拿一个瓶子")
    print("       去A点看一下人")
    print("  2. 若指令包含“去某点位 + 提到某个物体”，将先导航，再自动开 RGB-D 找该物体并对齐。")
    print("  3. 输入 q / quit / exit 回车退出。\n")

    while True:
        user_text = input("请输入一条中文指令（q 退出）：").strip()
        if not user_text:
            continue
        if user_text.lower() in {"q", "quit", "exit"}:
            print("退出。")
            break

        handle_user_text(user_text)


if __name__ == "__main__":
    main()
