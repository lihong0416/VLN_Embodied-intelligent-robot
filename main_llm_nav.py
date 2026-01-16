# main_llm_nav.py
"""
RGB-D + YOLO + LLM 语义导航 Demo（无终端距离打印，LLM 每 3 秒决策一次）

功能：
1. 打开 Orbbec RGB-D 相机，跑 YOLO 检测、测距、BEV 小地图。
2. 按 z 键，在终端输入一句导航指令（例如“去椅子旁边的苹果附近”）。
3. 将当前场景 objects + relations 打包成 JSON，连同指令和历史动作一起发给本地 Qwen。
4. Qwen 按导航 System Prompt 只返回“一小步动作”，例如：
   - 先左转 12 度
   - 再向前走 0.8 米
   - 或者认为已经到达，停止。

目前只打印 LLM 的 JSON，不真正控制机器人，方便你调试。
后续你可以把 step 里的 turn_deg / forward_m 映射到 /api/joy_control。
"""

import os
import sys
import time
import json
from typing import List, Tuple, Optional, Dict, Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import requests

# 这三个目录名与现在工程结构一致：
ORB_DIR = "pyorbbecsdk-2-main"
ORB_EXAMPLES_DIR = os.path.join(ORB_DIR, "examples")
YOLO_DIR = "YOLO-3D-main"
AGV_DIR = "机器人_project_有图"   # 里面有 config_markers.py、model_client.py 等

for p in [ORB_DIR, ORB_EXAMPLES_DIR, YOLO_DIR, AGV_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ----------------- Orbbec SDK & YOLO / BEV 导入 -----------------
from pyorbbecsdk import (
    Pipeline,
    Config,
    OBSensorType,
    OBFormat,
    AlignFilter,
    OBStreamType,
)

# 来自 pyorbbecsdk-2-main/examples/utils.py
from utils import frame_to_bgr_image

# 来自 YOLO-3D-main/detection_model.py
from detection_model import ObjectDetector

# 来自 YOLO-3D-main/bbox3d_utils.py
from bbox3d_utils import BirdEyeView

# Ollama + Qwen 配置（从 机器人_project_有图/config_markers.py 读取）
from config_markers import MODEL_NAME, OLLAMA_URL

ESC_KEY = 27

# 鸟瞰图近远映射范围（米）
BEV_NEAR_M = 0.2
BEV_FAR_M = 6.0

# 深度可视化的裁剪范围（米）
DEPTH_VIS_NEAR = 0.2
DEPTH_VIS_FAR = 6.0

# ================== 深度/几何工具 ==================


def compute_depth_map_in_meters(depth_frame) -> Optional[np.ndarray]:
    """
    将 Orbbec 的 depth frame 转成以米为单位的浮点深度图。
    根据 depth_scale 判断原始单位（mm 或 m），统一转为米。
    """
    depth_format = depth_frame.get_format()
    if depth_format != OBFormat.Y16:
        print("Depth format is not Y16, got:", depth_format)
        return None

    width = depth_frame.get_width()
    height = depth_frame.get_height()

    scale_raw = depth_frame.get_depth_scale()
    if not scale_raw:
        scale_raw = 1.0
    scale_raw = float(scale_raw)

    if scale_raw >= 1.0:
        # 通常 1 表示原始是 mm，这里转成米
        scale_m = scale_raw / 1000.0
    else:
        # 有的设备直接给米
        scale_m = scale_raw

    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((height, width))
    depth_m = depth_data.astype(np.float32) * scale_m

    # 小于等于 0 的值视为无效
    depth_m[depth_m <= 0] = 0.0
    return depth_m


def get_distance_at_pixel(
    depth_m: np.ndarray,
    cx: int,
    cy: int,
    region_half_size: int = 5,
) -> Optional[float]:
    """
    在某个像素附近取一个 patch，计算中位数深度，返回距离（米）。
    这是通用测距接口。
    """
    h, w = depth_m.shape

    # 防止越界
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    x1 = max(0, cx - region_half_size)
    x2 = min(w, cx + region_half_size)
    y1 = max(0, cy - region_half_size)
    y2 = min(h, cy + region_half_size)

    patch = depth_m[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    valid = patch[patch > 0]
    if valid.size == 0:
        return None

    distance = float(np.median(valid))
    if distance <= 0.0:
        return None
    return distance


def get_distance_for_bbox(
    depth_m: np.ndarray,
    bbox: Tuple[float, float, float, float],
    region_half_size: int = 5,
) -> Optional[float]:
    """
    针对一个 2D 检测框，取中心点附近 patch 的中位数深度，返回距离（米）。
    """
    x1, y1, x2, y2 = bbox
    cx = int(0.5 * (x1 + x2))
    cy = int(0.5 * (y1 + y2))
    return get_distance_at_pixel(depth_m, cx, cy, region_half_size=region_half_size)


def normalize_depth_for_bev(distance_m: Optional[float]) -> float:
    """
    把真实距离（米）映射到 [0, 1]，用于 BirdEyeView。
    """
    if distance_m is None or distance_m <= 0:
        return 1.0
    d_clamp = max(BEV_NEAR_M, min(BEV_FAR_M, float(distance_m)))
    return float((d_clamp - BEV_NEAR_M) / (BEV_FAR_M - BEV_NEAR_M))


def pixel_to_yaw_deg(cx: float, img_w: int, hfov_deg: float = 70.0) -> float:
    """
    把目标中心点的水平像素位置，大致换算成相对机器人朝向的角度（度）。

    约定：
        - 图像中心为 0°
        - 目标在“机器人左边”时返回正角度（需要左转）
        - 目标在“机器人右边”时返回负角度（需要右转）
    """
    center_x = img_w / 2.0
    half_w = img_w / 2.0
    norm = (cx - center_x) / max(half_w, 1e-6)  # 右正左负
    yaw = -norm * (hfov_deg / 2.0)  # 左正右负
    return float(yaw)


# ================== LLM 导航提示词 & 调用 ==================

NAV_SYSTEM_PROMPT = """
你是一个室内移动机器人的“视觉导航决策助手”。

输入：
- 用户的一句中文指令，例如 “去椅子那里”、“去椅子附近” 等。
- 当前一帧 RGB-D 场景的 JSON 描述 scene_state，其中：
  - scene_state["objects"] 是一个列表，每个元素包含：
    - id: 物体 ID（整数）
    - class: 物体类别名，例如 "person", "chair", "apple"
    - distance_m: 距离机器人直线距离（米）
    - bearing_deg: 方位角（度），>0 在左边，<0 在右边，0 在正前方
    - center_norm: [x_norm, y_norm]，物体在图像中的归一化位置
  - scene_state["relations"] 是物体之间的关系：
    - type 可以是 "near"、"left_of"、"right_of" 等。

- history_steps：本轮导航已经执行过的动作列表（可能为空数组）。

你的任务：
- 理解用户意图，基于 objects + relations 选择一个或多个目标物体，
  例如“椅子旁边的人附近”就需要找到 class="person" 且与某个 class="person" 存在 near 关系的物体。
- 只规划“下一小步动作”，不要一次性给出完整路径。
- 尽量保证安全，避免朝特别近的障碍物直接大步前进。

严格要求你只输出一个 JSON 对象，不能有任何多余文字或解释。
JSON 结构必须是：

{
  "goal_description": string,          // 你理解到的当前目标的文字描述
  "chosen_object_id": int | null,      // 当前选择追踪的目标物体 id，找不到就填 null
  "step": {
    "type": "turn" | "forward" | "stop",
    "turn_deg": float,                 // 当 type="turn" 时有效，左转为正，右转为负
    "forward_m": float,                // 当 type="forward" 时有效，向前为正，向后为负
    "comment": string                  // 对这一小步动作的简短解释
  },
  "need_next_step": bool               // 是否还需要后续步骤；到达目标附近时为 false
}

规划原则：
- 如果找不到与指令匹配的目标，可以先做一个搜索动作：
  比如 chosen_object_id=null，step.type="turn"，turn_deg=20 左右，need_next_step=true。
- bearing_deg 约定：
  - > 5 度在机器人左侧，<-5 度在右侧。
  - 在 [-5, 5] 度之间视为已经大致正对目标。
- 当目标距离 distance_m 小于 0.6 米时，认为已经到达目标附近：
  - 输出 step.type="stop"，need_next_step=false。
- 单步转向一般 5~25 度之间，单步前进 0.3~1.5 米之间，根据 distance_m 自行估计。
"""


def build_scene_state_from_objects(
    objects_for_llm: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    根据 objects_for_llm 列表构建 relations，并组合成 scene_state。
    objects_for_llm 的每个元素至少包含：
        id, class, distance_m, bearing_deg, center_norm
    """
    relations: List[Dict[str, Any]] = []

    # 简单关系：left_of / right_of / near
    for i in range(len(objects_for_llm)):
        for j in range(i + 1, len(objects_for_llm)):
            a = objects_for_llm[i]
            b = objects_for_llm[j]

            ax, ay = a["center_norm"]
            bx, by = b["center_norm"]

            # left/right：比较 x
            if ax < bx - 0.05:
                relations.append({"a": a["id"], "b": b["id"], "type": "left_of"})
                relations.append({"a": b["id"], "b": a["id"], "type": "right_of"})
            elif ax > bx + 0.05:
                relations.append({"a": a["id"], "b": b["id"], "type": "right_of"})
                relations.append({"a": b["id"], "b": a["id"], "type": "left_of"})

            # near：像素距离 + 深度差都不大
            dx = ax - bx
            dy = ay - by
            pixel_dist = (dx * dx + dy * dy) ** 0.5
            depth_diff = abs(a["distance_m"] - b["distance_m"])
            if pixel_dist < 0.12 and depth_diff < 0.6:
                relations.append({"a": a["id"], "b": b["id"], "type": "near"})
                relations.append({"a": b["id"], "b": a["id"], "type": "near"})

    scene_state = {
        "objects": objects_for_llm,
        "relations": relations,
    }
    return scene_state


def build_nav_user_prompt(
    user_command: str,
    scene_state: Dict[str, Any],
    history_steps: List[Dict[str, Any]],
) -> str:
    """
    构造发给 Qwen 的 user prompt（不含 system 部分）。
    """
    scene_json = json.dumps(scene_state, ensure_ascii=False, indent=2)
    history_json = json.dumps(history_steps, ensure_ascii=False, indent=2)

    parts = []
    parts.append("下面是当前一帧 RGB-D 场景的结构化描述 JSON：")
    parts.append(scene_json)
    parts.append("")
    parts.append("下面是到目前为止已经执行过的动作列表（可能为空数组）：")
    parts.append(history_json)
    parts.append("")
    parts.append("请根据上面的信息，规划下一小步动作，并用 JSON 格式回答。")
    parts.append(f"用户指令：{user_command}")
    parts.append("助手：")

    return "\n".join(parts)


def call_qwen_nav(
    user_command: str,
    scene_state: Dict[str, Any],
    history_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    调用本地 Ollama + Qwen，做一次导航决策，返回 JSON dict。
    """
    user_prompt = build_nav_user_prompt(user_command, scene_state, history_steps)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": NAV_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama 调用失败: {resp.status_code}, {resp.text}")

    data = resp.json()
    json_str = data["message"]["content"]
    return json.loads(json_str)


# ================== Orbbec 初始化 ==================


def init_orbbec_pipeline() -> Tuple[Pipeline, Config, AlignFilter]:
    """
    初始化 Orbbec Pipeline：启用彩色 + 深度，并开启对齐到彩色。
    """
    pipeline = Pipeline()
    config = Config()

    # 启用彩色流
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    # 启用深度流
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    # 开启帧同步（尽量让 color / depth 时间对齐）
    try:
        pipeline.enable_frame_sync()
    except Exception as e:
        print("Warning: enable_frame_sync failed:", e)

    # 启动 pipeline
    pipeline.start(config)

    # 对齐：深度 → 彩色坐标
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    return pipeline, config, align_filter


# ================== 主循环：RGB-D + YOLO + LLM 导航 ==================


def main_llm_nav():
    # 1. 初始化 Orbbec pipeline
    pipeline, config, align_filter = init_orbbec_pipeline()
    print("Orbbec RGB-D pipeline started.")

    # 2. 初始化 YOLO 检测器
    detector = ObjectDetector(
        model_size="medium",
        conf_thres=0.3,
        iou_thres=0.45,
        classes=None,
        device="cuda",  # 如果没有 GPU 可以改成 "cpu"
    )
    class_names = detector.get_class_names()
    print("YOLO detector initialized with classes:", class_names)

    # 3. 初始化 BirdEyeView（俯视小地图）
    bev = BirdEyeView(size=(400, 400), scale=60, camera_height=1.2)
    bev.reset()

    # 固定几个“纯深度测距点”的相对位置（0~1）
    depth_probe_points = [
        (0.5, 0.5),  # 中心
        (0.3, 0.5),  # 左中
        (0.7, 0.5),  # 右中
        (0.5, 0.3),  # 上中
        (0.5, 0.7),  # 下中
    ]

    # LLM 导航相关状态
    current_goal_text: Optional[str] = None
    history_steps: List[Dict[str, Any]] = []
    last_nav_call_time = 0.0
    nav_call_interval = 3.0  # ★ LLM 导航：每 3 秒调用一次

    print("按 z 键可以在终端输入一条导航指令；按 x 取消当前导航；按 q 或 ESC 退出。")

    try:
        while True:
            # 1) 取一帧 RGB-D
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

            # 彩色图
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                continue

            # 深度图（米）
            depth_m = compute_depth_map_in_meters(depth_frame)
            if depth_m is None:
                continue

            h_img, w_img, _ = color_image.shape

            # 2) 全局最近障碍 & 正前方距离（只用于画面显示，不再终端打印）
            global_min_distance = None
            valid_depth = depth_m[depth_m > 0]
            if valid_depth.size > 0:
                global_min_distance = float(valid_depth.min())

            forward_distance = None
            y1 = int(h_img * 0.3)
            y2 = int(h_img * 0.7)
            x1 = int(w_img * 0.33)
            x2 = int(w_img * 0.66)
            roi = depth_m[y1:y2, x1:x2]
            valid_roi = roi[roi > 0]
            if valid_roi.size > 0:
                forward_distance = float(valid_roi.min())

            # 3) YOLO 检测（每帧运行，用于实时可视化和 LLM）
            annotated_image, detections = detector.detect(color_image, track=True)

            # 4) BirdEyeView 清空重新画
            bev.reset()

            # 用于画左上角“最近的几个目标”
            summary_infos: List[Tuple[float, str, int]] = []

            # 给 LLM 用的 objects 列表
            objects_for_llm: List[Dict[str, Any]] = []

            # 5) 遍历所有检测框：测距 + 画框 + BEV + 构造 objects_for_llm
            for det in detections:
                bbox, score, class_id, obj_id = det
                oid = int(obj_id) if obj_id is not None else -1

                x1b, y1b, x2b, y2b = [int(v) for v in bbox]
                cx = 0.5 * (x1b + x2b)
                cy = 0.5 * (y1b + y2b)

                # 距离（米）
                distance_m = get_distance_for_bbox(depth_m, bbox, region_half_size=5)
                if distance_m is None:
                    continue

                class_name = class_names[int(class_id)]

                # ---- 在图像上画距离小圆点 + 文本 ----
                cv2.circle(
                    annotated_image,
                    (int(cx), int(cy)),
                    4,
                    (0, 255, 255),
                    -1,
                )
                text = f"ID:{oid} {class_name} {float(score):.2f} {distance_m:.2f}m"
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                x_text = x1b
                y_text = max(0, y1b - 5)
                cv2.rectangle(
                    annotated_image,
                    (x_text - 2, y_text - th - 2),
                    (x_text + tw + 2, y_text + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    annotated_image,
                    text,
                    (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # ---- 汇总信息，用于左上角排序显示 ----
                summary_infos.append((distance_m, class_name, oid))

                # ---- BirdEyeView ----
                depth_norm = normalize_depth_for_bev(distance_m)
                box_3d = {
                    "class_name": class_name,
                    "bbox_2d": bbox,
                    "depth_value": depth_norm,
                    "object_id": oid,
                    "score": float(score),
                }
                bev.draw_box(box_3d)

                # ---- 给 LLM 用的 objects ----
                bearing = pixel_to_yaw_deg(cx, w_img)
                obj_llm = {
                    "id": oid,
                    "class": class_name,
                    "distance_m": round(float(distance_m), 2),
                    "bearing_deg": round(float(bearing), 1),
                    "center_norm": [
                        round(cx / max(w_img, 1), 3),
                        round(cy / max(h_img, 1), 3),
                    ],
                }
                objects_for_llm.append(obj_llm)

            # 6) 左上角文字：Nearest / Forward + 最近的几个目标（只画，不打 log）
            y_offset = 20
            if global_min_distance is not None:
                txt = f"Nearest obstacle: {global_min_distance:.2f}m"
                cv2.putText(
                    annotated_image,
                    txt,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                y_offset += 25

            if forward_distance is not None:
                txt = f"Forward distance: {forward_distance:.2f}m"
                cv2.putText(
                    annotated_image,
                    txt,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )
                y_offset += 25

            # 排序显示前 5 个目标
            summary_infos.sort(key=lambda x: x[0])
            for idx, (dist, cname, oid) in enumerate(summary_infos[:5], start=1):
                txt = f"{idx}. ID:{oid} {cname}: {dist:.2f}m"
                cv2.putText(
                    annotated_image,
                    txt,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                y_offset += 20

            # 7) 几个固定深度采样点（不依赖 YOLO）
            for (ux, uy) in depth_probe_points:
                px = int(ux * w_img)
                py = int(uy * h_img)
                dist = get_distance_at_pixel(depth_m, px, py, region_half_size=5)
                if dist is None:
                    continue
                cv2.circle(annotated_image, (px, py), 4, (0, 255, 255), -1)
                txt = f"{dist:.2f}m"
                (tw, th), _ = cv2.getTextSize(
                    txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                cv2.rectangle(
                    annotated_image,
                    (px - 2, py - th - 2),
                    (px + tw + 2, py + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    annotated_image,
                    txt,
                    (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

            # 8) 构造给 LLM 的 scene_state
            scene_state = build_scene_state_from_objects(objects_for_llm)

            # 9) 合成一张叠加 BEV 的图像
            overlay = annotated_image.copy()
            bev_img = bev.get_image()
            bh, bw, _ = bev_img.shape
            scale = 0.7
            bev_small = cv2.resize(
                bev_img,
                (int(bw * scale), int(bh * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            sh, sw, _ = bev_small.shape
            overlay[h_img - sh : h_img, w_img - sw : w_img] = bev_small

            # 深度热力图
            depth_vis = depth_m.copy()
            depth_vis[depth_vis <= 0] = DEPTH_VIS_FAR
            depth_vis = np.clip(depth_vis, DEPTH_VIS_NEAR, DEPTH_VIS_FAR)
            depth_norm = (depth_vis - DEPTH_VIS_NEAR) / (DEPTH_VIS_FAR - DEPTH_VIS_NEAR)
            depth_u8 = (depth_norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

            # 显示窗口
            cv2.imshow("RGB-D Navigation Perception (LLM Nav)", overlay)
            cv2.imshow("Depth Heatmap", depth_color)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ESC_KEY):
                break

            # 按 z 键：在终端输入一条新的导航指令
            if key == ord("z"):
                try:
                    user_text = input("\n请输入一条导航指令：").strip()
                except EOFError:
                    user_text = ""
                if user_text:
                    current_goal_text = user_text
                    history_steps = []
                    print(f"[Goal] 新的导航指令：{current_goal_text}")
                else:
                    print("[Goal] 空指令，忽略。")

            # 按 x 键：取消当前导航
            if key == ord("x"):
                current_goal_text = None
                history_steps = []
                print("[Goal] 取消当前导航目标。")

            # 只有在有导航目标时才调用 LLM（并限制频率为每 3 秒一次）
            if (
                current_goal_text
                and (time.time() - last_nav_call_time) > nav_call_interval
            ):
                try:
                    nav_cmd = call_qwen_nav(
                        user_command=current_goal_text,
                        scene_state=scene_state,
                        history_steps=history_steps,
                    )
                    print("\n[LLM 下一步指令]", json.dumps(nav_cmd, ensure_ascii=False))
                    step = nav_cmd.get("step", {})
                    history_steps.append(step)
                    last_nav_call_time = time.time()

                    if not nav_cmd.get("need_next_step", True):
                        print("[LLM] 认为已到达目标附近，本轮导航结束。")
                        current_goal_text = None
                        history_steps = []
                except Exception as e:
                    print("[LLM] 调用失败：", e)

    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
        print("Pipeline stopped and windows closed.")


if __name__ == "__main__":
    main_llm_nav()
