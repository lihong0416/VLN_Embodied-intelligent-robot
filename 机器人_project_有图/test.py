# main_voice_nav_and_center_debug.py
"""
验证版：文本 → 导航指令 + RGB-D 自动对齐（只打印，不发给机器人）

功能：
1. 终端输入中文指令，例如：
     去B点拿一个瓶子
     去A点看一下人
2. 将整句拆成两部分：
   - 导航子句 nav_text：例如 “去B点”
   - 取物子句 obj_text：例如 “拿一个瓶子”
3. 导航子句丢给本地 Qwen：
   - call_qwen_local_and_get_json(nav_text)
   - normalize_turn_from_text(...)
   - validate_command(...)
   - build_robot_api_string(...)
   - 不走 TCP，只打印“将要发送的 /api/... 串”
4. 用整句文本解析 YOLO 要找的类别（瓶子/人/椅子等），只决定 RGB-D 阶段要对齐什么目标。
5. 若 action == move_to_marker 且解析出目标类别：
   - 开 Orbbec 深度相机 + YOLO
   - 阶段一（align）：只转向对齐到视野中心；
   - 阶段二（approach）：在已经居中的前提下，只根据距离 > / <= 0.5m 决定是否前进；
   - 每隔 CONTROL_INTERVAL_S 秒最多生成一次控制指令；
   - 深度测距使用最近 5 次的历史：只要 5 次里有一次有效就继续，没有就中断。
   - 所有指令只打印，不发 TCP。
"""

import os
import sys
import time
from typing import Optional, Tuple, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import json as _json

# ================== 导入“有图项目”里的模块 ==================

from model_client import call_qwen_local_and_get_json
from command_logic import validate_command, build_robot_api_string
from config_markers import get_current_map_name, set_current_map_name
import main as agv_main  # 用里面的 normalize_turn_from_text、STEP_DURATION、MAX_JOY_DURATION

normalize_turn_from_text = agv_main.normalize_turn_from_text
STEP_DURATION = agv_main.STEP_DURATION
MAX_JOY_DURATION = agv_main.MAX_JOY_DURATION

# ================== 相机 & YOLO 所在路径 ==================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # D:\机器人_无图_project

ORB_DIR = os.path.join(PROJECT_ROOT, "pyorbbecsdk-2-main")
ORB_EXAMPLES_DIR = os.path.join(ORB_DIR, "examples")
YOLO_DIR = os.path.join(PROJECT_ROOT, "YOLO-3D-main")

for p in [ORB_DIR, ORB_EXAMPLES_DIR, YOLO_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# ================== Orbbec SDK & YOLO 导入 ==================

from pyorbbecsdk import AlignFilter  # 类型提示用
from utils import frame_to_bgr_image
from detection_model import ObjectDetector
from main_RGB_D import (
    init_orbbec_pipeline,
    compute_depth_map_in_meters,
    get_distance_for_bbox,
)

ESC_KEY = 27

# ================== 控制参数 ==================

CENTER_TOLERANCE_RATIO = 0.05   # 水平偏差阈值（占画面宽度比例）
MAX_ANGULAR_VEL = 0.6           # 最大角速度
BASE_SCAN_VEL = 0.3             # 扫描时角速度
STEP_DURATION_S = 0.4
MAX_CENTERING_TIME_S = 60.0

CONTROL_INTERVAL_S = 3.0        # 每 3 秒最多发一次控制指令
MIN_DISTANCE_M = 0.5            # 希望最终停在的距离
FORWARD_SPEED = 0.15            # 前进线速度（m/s）
FORWARD_DURATION_MAX = 2.0      # 单次前进最长时长

# ================== 中文 → YOLO 类别映射 ==================

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
    if not text:
        return None
    low = text.lower()
    for zh, cls in sorted(ZH_TO_YOLO_CLASS.items(), key=lambda kv: -len(kv[0])):
        if zh in low:
            return cls
    return None


def split_nav_and_object(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    for kw in ["拿", "取"]:
        idx = text.find(kw)
        if idx != -1:
            nav = text[:idx]
            obj = text[idx:]
            return nav.strip(), obj.strip()
    return text.strip(), ""


# ================== 只打印、不发 TCP 的 joy_control 宏指令 ==================

def debug_exec_joy_control_macro(cmd: dict):
    """
    模拟 main.py 里的 exec_joy_control_macro：
    - 将 linear/angular_velocity 四舍五入到小数点后三位
    - 计算步数
    - 构造 /api/joy_control?... 串
    - 只打印，不发 TCP
    """
    # 拷贝一份，避免改到外面的 cmd
    cmd = dict(cmd)

    lin_raw = float(cmd.get("linear_velocity", 0.0))
    ang_raw = float(cmd.get("angular_velocity", 0.0))

    lin = round(lin_raw, 3)
    ang = round(ang_raw, 3)

    cmd["linear_velocity"] = lin
    cmd["angular_velocity"] = ang

    duration = float(cmd.get("duration_s", STEP_DURATION))
    if duration <= 0:
        duration = STEP_DURATION
    if duration > MAX_JOY_DURATION:
        print(f"[JOY-DEBUG] duration_s={duration:.2f}s 超过限制，截断为 {MAX_JOY_DURATION:.2f}s")
        duration = MAX_JOY_DURATION

    steps = max(1, int(round(duration / STEP_DURATION)))

    api_str = build_robot_api_string(cmd)

    print("\n[joy_control 宏指令【模拟发送】]")
    print(f"  · 线速度 linear_velocity = {lin:.3f} m/s")
    print(f"  · 角速度 angular_velocity = {ang:.3f} rad/s")
    print(f"  · 总时长 duration_s ≈ {duration:.2f} s")
    print(f"  · 单步时长 STEP_DURATION = {STEP_DURATION:.2f} s")
    print(f"  · 预计会向机器人发送 {steps} 次同样的 API：")
    print(f"    {api_str}")

    for i in range(steps):
        print(f"[JOY-DEBUG] Step {i + 1}/{steps} -> {api_str}")
    print("[JOY-DEBUG] (仅打印，不通过 TCP 发送)\n")


# ================== 从 YOLO 检测结果中选一个目标 ==================

def select_target_detection(
    detections: List,
    class_names,
    img_w: int,
    target_class: Optional[str],
) -> Optional[Tuple[Tuple[float, float, float, float], str]]:
    if not detections:
        return None

    center_x = img_w / 2.0
    candidates: List[Tuple[Tuple[float, float, float, float], str, float, float]] = []

    for det in detections:
        bbox, score, class_id, obj_id = det
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = 0.5 * (x1 + x2)
        offset = abs(cx - center_x)

        cls_idx = int(class_id)
        if isinstance(class_names, dict):
            cname = class_names.get(cls_idx, str(cls_idx))
        else:
            try:
                cname = class_names[cls_idx]
            except Exception:
                cname = str(cls_idx)

        candidates.append((bbox, cname, float(score), offset))

    if target_class is not None:
        filtered = [d for d in candidates if d[1] == target_class]
        if filtered:
            filtered.sort(key=lambda x: x[3])
            bbox, cname, score, _ = filtered[0]
            return bbox, cname

    candidates.sort(key=lambda x: x[3])
    bbox, cname, score, _ = candidates[0]
    return bbox, cname


# ================== RGB-D 自动对齐（两阶段 + 最近5次测距，仅打印 joy_control） ==================

def center_object_with_rgbd_debug(
    target_class: Optional[str],
    max_time_s: float = MAX_CENTERING_TIME_S,
):
    """
    两阶段控制：
    1) align 阶段：只旋转，让目标进视野中心（不考虑距离）。
    2) approach 阶段：不再旋转，只根据“最近5次深度测距”的结果来判断是否前进或结束：
        - 若最近5次里至少有一次有效 → 用最近一次有效距离判断：
              <= MIN_DISTANCE_M  -> 完成
              >  MIN_DISTANCE_M  -> 建议前进
        - 若最近5次全部 None       -> 深度长期不可用，任务中断。
    所有指令只打印，不发 TCP。
    """

    print("\n[RGB-D-DEBUG] 启动 RGB-D + YOLO 自动对齐（两阶段，仅打印 joy_control 指令）...")

    pipeline, config, align_filter = init_orbbec_pipeline()
    print("[RGB-D-DEBUG] Orbbec Pipeline 已启动。")

    detector = ObjectDetector(
        model_size="medium",
        conf_thres=0.3,
        iou_thres=0.45,
        classes=None,
        device="cuda",
    )
    class_names = detector.get_class_names()
    print("[RGB-D-DEBUG] YOLO 检测器已初始化。")

    start_time = time.time()
    last_control_time = start_time - CONTROL_INTERVAL_S
    scan_direction = 1
    mode = "align"      # 或 "approach"

    recent_distances: List[Optional[float]] = []  # 最近 5 次测距（可以是 None）

    try:
        while True:
            now = time.time()
            if now - start_time > max_time_s:
                print("[RGB-D-DEBUG] 已超过最大对齐时间，停止。")
                break

            frames = pipeline.wait_for_frames(100)
            if not frames:
                continue

            frames = align_filter.process(frames)
            if not frames:
                continue
            frames = frames.as_frame_set()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("[RGB-D-DEBUG] Failed to convert color frame to image")
                continue

            depth_m = compute_depth_map_in_meters(depth_frame)
            if depth_m is None:
                # 这一帧深度全 None，就先跳过距离计算，但不立刻终止
                pass

            h_img, w_img, _ = color_image.shape
            center_x = w_img / 2.0

            annotated_image, detections = detector.detect(color_image, track=True)
            if annotated_image is None:
                annotated_image = color_image

            target = select_target_detection(detections, class_names, w_img, target_class)
            can_control_now = (now - last_control_time) >= CONTROL_INTERVAL_S

            if target is None:
                print(f"[RGB-D-DEBUG] 当前模式: {mode}，暂未检测到目标。")

                if mode == "align" and can_control_now:
                    print("[RGB-D-DEBUG] align 阶段：建议原地缓慢旋转搜索...")
                    ang_vel = BASE_SCAN_VEL * scan_direction
                    cmd_turn = {
                        "action": "joy_control",
                        "linear_velocity": 0.0,
                        "angular_velocity": float(ang_vel),
                        "duration_s": STEP_DURATION_S,
                    }
                    debug_exec_joy_control_macro(cmd_turn)
                    last_control_time = now

                    if int(now - start_time) % 10 == 0:
                        scan_direction *= -1

                elif mode == "approach":
                    print("[RGB-D-DEBUG] approach 阶段目标丢失，不再建议移动，结束。")
                    cv2.imshow("RGB-D Centering Debug", annotated_image)
                    cv2.waitKey(1000)
                    break

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

                # 维护最近 5 次测距（可以是 None）
                recent_distances.append(distance_m)
                if len(recent_distances) > 5:
                    recent_distances.pop(0)

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

                offset_px = cx - center_x
                offset_ratio = offset_px / center_x

                print(
                    f"[RGB-D-DEBUG] 模式: {mode} | 目标: {class_name}, 距离: {distance_text}, "
                    f"水平偏差: {offset_px:.1f}px ({offset_ratio:.3f})"
                )

                # ========== 阶段一：只做对齐 ==========
                if mode == "align":
                    if abs(offset_ratio) <= CENTER_TOLERANCE_RATIO:
                        cv2.putText(
                            annotated_image,
                            "CENTERED (ALIGN DONE)",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                        )
                        print("[RGB-D-DEBUG] align 阶段：目标已到视野中心，切换到 approach 阶段。")
                        mode = "approach"
                        last_control_time = now
                        # 不在当前帧马上前进，下一个控制周期再根据“最近5次测距”决定
                    else:
                        if can_control_now:
                            ang_vel = -float(offset_ratio) * MAX_ANGULAR_VEL
                            ang_vel = max(min(ang_vel, MAX_ANGULAR_VEL), -MAX_ANGULAR_VEL)
                            print(
                                f"[RGB-D-DEBUG] align 阶段：建议发送微调转动 "
                                f"angular_velocity={ang_vel:.3f} rad/s"
                            )
                            cmd_turn = {
                                "action": "joy_control",
                                "linear_velocity": 0.0,
                                "angular_velocity": float(ang_vel),
                                "duration_s": STEP_DURATION_S,
                            }
                            debug_exec_joy_control_macro(cmd_turn)
                            last_control_time = now

                # ========== 阶段二：只调距离，不再转向 ==========
                elif mode == "approach":
                    cv2.putText(
                        annotated_image,
                        "APPROACH",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 0),
                        2,
                    )

                    # 用最近 5 次测距来决定是否有可用深度
                    valid_dists = [d for d in recent_distances if d is not None]

                    if not valid_dists:
                        # 最近 N 次（N<=5）都测不出深度
                        if len(recent_distances) >= 5:
                            print(
                                "[RGB-D-DEBUG] approach 阶段：最近 5 次深度均无效，"
                                "认为深度传感异常，任务中断。"
                            )
                            cv2.imshow("RGB-D Centering Debug", annotated_image)
                            cv2.waitKey(1000)
                            break
                        else:
                            print(
                                "[RGB-D-DEBUG] approach 阶段：当前/最近深度暂不可用，"
                                "继续观测，不发送前进指令。"
                            )
                    else:
                        # 取最近一次有效距离
                        distance_use = valid_dists[-1]

                        # 距离足够近：任务完成
                        if distance_use <= MIN_DISTANCE_M:
                            print(
                                f"[RGB-D-DEBUG] approach 阶段：目标已居中且距离 {distance_use:.2f}m "
                                f"<= {MIN_DISTANCE_M}m，任务完成。"
                            )
                            cv2.imshow("RGB-D Centering Debug", annotated_image)
                            cv2.waitKey(1000)
                            break

                        # 距离偏大：按控制周期建议前进
                        if can_control_now:
                            extra = distance_use - MIN_DISTANCE_M
                            t = extra / FORWARD_SPEED
                            if t <= 0:
                                t = STEP_DURATION_S
                            if t > FORWARD_DURATION_MAX:
                                t = FORWARD_DURATION_MAX

                            print(
                                f"[RGB-D-DEBUG] approach 阶段：最近有效距离 {distance_use:.2f}m > "
                                f"{MIN_DISTANCE_M}m，建议前进约 {extra:.2f}m，"
                                f"估算 duration_s ≈ {t:.2f} s"
                            )

                            cmd_forward = {
                                "action": "joy_control",
                                "linear_velocity": float(FORWARD_SPEED),
                                "angular_velocity": 0.0,
                                "duration_s": float(t),
                            }
                            debug_exec_joy_control_macro(cmd_forward)
                            last_control_time = now

            cv2.imshow("RGB-D Centering Debug", annotated_image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ESC_KEY):
                print("[RGB-D-DEBUG] 手动退出自动对齐。")
                break

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[RGB-D-DEBUG] 相机与窗口已关闭。")


# ================== 文本 → JSON → API 串（只打印） ==================

def handle_user_text_debug(user_text: str):
    if not user_text:
        return

    nav_text, obj_text = split_nav_and_object(user_text)
    if not nav_text:
        nav_text = user_text

    target_class = extract_target_class_from_text(user_text)
    if target_class:
        print(f"[解析] 推测目标物体类别（YOLO 类别）：{target_class}")
    else:
        print("[解析] 文本中未解析出具体目标物体类别，本次只验证导航指令。")

    print(f"\n[NLU] 导航子句发送给 Qwen 的内容：『{nav_text}』")

    cmd = call_qwen_local_and_get_json(nav_text)

    print("\n[原始 JSON 指令（模型直接输出）]")
    print(_json.dumps(cmd, ensure_ascii=False, indent=2))

    cmd = normalize_turn_from_text(cmd, user_text)

    cmd = validate_command(cmd)
    action = cmd.get("action")

    if action == "set_map":
        set_current_map_name(cmd["map_name"])
        print(f"[地图切换] 当前地图已切换为 {cmd['map_name']} (floor={cmd['floor']})")

    print("\n[校验 / 补全后的 JSON 指令]")
    print(_json.dumps(cmd, ensure_ascii=False, indent=2))

    if action == "ask_user_clarification":
        print("\n[无法安全执行，本次仅提示，不构造 API 串]")
        print("原因：", cmd.get("reason", "模型未给出原因"))
        print("\n" + "=" * 60 + "\n")
        return

    if action == "joy_control":
        debug_exec_joy_control_macro(cmd)
        print("\n" + "=" * 60 + "\n")
        return

    api_str = build_robot_api_string(cmd)

    print("\n[【模拟】将要发送给机器人的 API 文本指令是]")
    print(api_str)
    print("[说明] 上面这条指令本脚本不会通过 TCP 发送，只做打印验证。")

    print("\n" + "=" * 60 + "\n")

    if action == "move_to_marker" and target_class is not None:
        print("[提示] 本次指令包含“导航 + 要找的物体”。")
        input("假设机器人已经导航到目标点位，准备好后按回车开启 RGB-D 自动对齐（仅打印指令）... ")
        center_object_with_rgbd_debug(target_class=target_class, max_time_s=MAX_CENTERING_TIME_S)
    else:
        if action == "move_to_marker":
            print("[提示] 本次是纯导航指令（未解析出具体目标物体），不会开启 RGB-D。")


def main():
    print("=== 文本 → 指令 + RGB-D 自动对齐【验证版】(不发 TCP) ===")
    try:
        cur_map = get_current_map_name()
        print(f"当前地图：{cur_map}")
    except Exception:
        print("当前地图：<获取失败，请检查 config_markers.py / maps.json>")

    print("\n使用说明：")
    print("  1. 在终端输入中文指令，例如：")
    print("       去B点拿一个瓶子")
    print("       去A点看一下人")
    print("  2. 脚本只会打印 JSON 指令和 /api/... 文本，不会连机器人。")
    print("  3. 如果包含“导航 + 物体”，会在你按回车后开启相机，用相机画面来模拟应当发送的 joy_control。")
    print("  4. 输入 q / quit / exit 回车退出。\n")

    while True:
        user_text = input("请输入一条中文指令（q 退出）：").strip()
        if not user_text:
            continue
        if user_text.lower() in {"q", "quit", "exit"}:
            print("退出。")
            break

        handle_user_text_debug(user_text)


if __name__ == "__main__":
    main()
