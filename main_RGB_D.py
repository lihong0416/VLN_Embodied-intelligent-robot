import os
import sys
import time
from typing import List, Tuple, Optional

# 允许 OpenMP 重复加载（PyTorch + 其它库时会用到）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_DIR = os.path.join(CUR_DIR, "YOLO-3D-main")
if os.path.isdir(YOLO_DIR) and YOLO_DIR not in sys.path:
    sys.path.insert(0, YOLO_DIR)
from detection_model import ObjectDetector

# ================== 手动加入两个项目路径（相对路径写法） ==================
ORB_DIR = r"pyorbbecsdk-2-main"
ORB_EXAMPLES_DIR = r"pyorbbecsdk-2-main/examples"
YOLO_DIR = r"YOLO-3D-main"

for p in [ORB_DIR, ORB_EXAMPLES_DIR, YOLO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ================== Orbbec SDK & YOLO / BEV 导入 ==================
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

ESC_KEY = 27

# 鸟瞰图近远映射范围（米）
BEV_NEAR_M = 0.2
BEV_FAR_M = 6.0


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

    # 原始 scale，有的设备是 1（表示 mm），也可能是 0.001（本来就是米）
    scale_raw = depth_frame.get_depth_scale()
    if not scale_raw:
        scale_raw = 1.0
    scale_raw = float(scale_raw)

    # 如果 scale >= 1，基本可以认为单位是“毫米”，转成米要 ÷1000
    if scale_raw >= 1.0:
        scale_m = scale_raw / 1000.0  # mm → m
    else:
        scale_m = scale_raw  # 已经是米的缩放

    depth_data = np.frombuffer(
        depth_frame.get_data(), dtype=np.uint16
    ).reshape(height, width)

    # 单位：米
    depth_m = depth_data.astype(np.float32) * scale_m

    # 保留 >0 的值，其它置 0
    depth_m = np.where(depth_m > 0, depth_m, 0.0)

    return depth_m


def get_distance_for_bbox(
    depth_m: np.ndarray,
    bbox: List[float],
    region_half_size: int = 5,
) -> Optional[float]:
    """
    给定深度图（米）和 2D 框，从框中心附近取一个小区域的中位数深度作为物体距离。
    """
    h, w = depth_m.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # 防止越界
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    return get_distance_at_pixel(depth_m, cx, cy, region_half_size)


def get_distance_at_pixel(
    depth_m: np.ndarray,
    cx: int,
    cy: int,
    region_half_size: int = 5,
) -> Optional[float]:
    """
    在某个像素附近取一个 patch，计算中位数深度，返回距离（米）。
    这个是“与 YOLO 无关”的通用测距接口。
    """
    h, w = depth_m.shape

    # 防止越界
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    x1r = max(0, cx - region_half_size)
    x2r = min(w, cx + region_half_size)
    y1r = max(0, cy - region_half_size)
    y2r = min(h, cy + region_half_size)

    patch = depth_m[y1r:y2r, x1r:x2r]
    if patch.size == 0:
        return None

    valid = patch[patch > 0]
    if valid.size == 0:
        return None

    distance = float(np.median(valid))
    if distance <= 0.0:
        return None
    return distance


def normalize_depth_for_bev(distance_m: Optional[float]) -> float:
    """
    将真实距离（米）映射到 [0, 1]，供 BirdEyeView 使用。
    这里简单地把 [BEV_NEAR_M, BEV_FAR_M] 线性映射到 [0, 1]。
    """
    if distance_m is None or distance_m <= 0:
        return 0.5
    d = max(BEV_NEAR_M, min(BEV_FAR_M, distance_m))
    return (d - BEV_NEAR_M) / (BEV_FAR_M - BEV_NEAR_M)


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


def main():
    # 1. 初始化 Orbbec pipeline
    pipeline, config, align_filter = init_orbbec_pipeline()
    print("Orbbec RGB-D pipeline started.")

    # 2. 初始化 YOLO 检测器（先用 CPU）
    detector = ObjectDetector(
        model_size="medium",
        conf_thres=0.3,
        iou_thres=0.45,
        classes=None,
        device="cuda",  #  CPU or CUDA
    )
    class_names = detector.get_class_names()
    print("YOLO detector initialized with classes:", class_names)

    # 3. 初始化 BirdEyeView（俯视小地图）
    bev = BirdEyeView(size=(400, 400), scale=60, camera_height=1.2)
    bev.reset()

    print("Press 'q' or ESC to quit.")

    # 终端打印节流：一秒一次
    last_print_time = 0.0

    # 定义一些“深度测距点”（相对坐标：0~1）
    # 比如中间、左中、右中、上中、下中这 5 个点
    depth_probe_points = [
        (0.5, 0.5),   # 屏幕正中心
        (0.33, 0.5),  # 中心左侧
        (0.66, 0.5),  # 中心右侧
        (0.5, 0.33),  # 中心偏上
        (0.5, 0.66),  # 中心偏下
    ]

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if not frames:
                continue

            # 对齐：深度→彩色
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
                print("Failed to convert color frame to image")
                continue

            # 深度图（米）
            depth_m = compute_depth_map_in_meters(depth_frame)
            if depth_m is None:
                continue

            # ======== A. 计算全局最近障碍距离（不依赖 YOLO） ========
            valid = depth_m[depth_m > 0]
            if valid.size > 0:
                global_min_distance = float(valid.min())
            else:
                global_min_distance = None

            # ======== B. 计算正前方 ROI 最近距离 ========
            h_d, w_d = depth_m.shape
            roi = depth_m[int(h_d * 0.3): int(h_d * 0.7),
                          int(w_d * 0.33): int(w_d * 0.66)]
            valid_roi = roi[roi > 0]
            if valid_roi.size > 0:
                forward_distance = float(valid_roi.min())
            else:
                forward_distance = None

            # 4. YOLO 检测
            annotated_image, detections = detector.detect(color_image, track=True)
            h_img, w_img, _ = annotated_image.shape

            # 5. 每帧重置 BEV 背景 + 坐标
            bev.reset()

            # 用于在画面左上角 & 终端打印
            summary_infos = []

            # ======== C. YOLO 识别到的目标：带 ID + 类别 + 距离 ========
            for det in detections:
                bbox, score, class_id, obj_id = det

                # 有的检测器会返回 obj_id=None，这里做个安全的 ID
                oid = int(obj_id) if obj_id is not None else -1

                x1, y1, x2, y2 = [int(v) for v in bbox]

                # 距离（米）
                distance_m = get_distance_for_bbox(depth_m, bbox, region_half_size=6)

                # 类别名
                if isinstance(class_names, dict):
                    class_name = class_names.get(int(class_id), str(class_id))
                else:
                    try:
                        class_name = class_names[int(class_id)]
                    except Exception:
                        class_name = str(class_id)

                if distance_m is None:
                    continue

                # 记录信息（排序 & 打印用）
                summary_infos.append(
                    (distance_m, class_name, oid)
                )

                # ========= 在框里画“ID+类别+置信度+距离(m)” =========
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(annotated_image, (cx, cy), 3, (0, 255, 255), -1)

                label = f"ID:{oid} {class_name} {score:.2f} {distance_m:.2f}m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # 把文字放在框内部上方一点
                y_text = max(y1 + th + 4, 0)
                y_rect_top = y_text - th - 4
                y_rect_bottom = y_text
                y_rect_top = max(0, y_rect_top)
                y_rect_bottom = min(annotated_image.shape[0] - 1, y_rect_bottom)

                cv2.rectangle(
                    annotated_image,
                    (x1, y_rect_top),
                    (x1 + tw + 4, y_rect_bottom),
                    (0, 0, 0),
                    -1,
                )

                cv2.putText(
                    annotated_image,
                    label,
                    (x1 + 2, y_rect_bottom - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                # ==================================================

                # 6. BEV 中绘制
                depth_norm = normalize_depth_for_bev(distance_m)
                box_3d = {
                    "class_name": class_name,
                    "bbox_2d": bbox,
                    "depth_value": depth_norm,
                    "object_id": oid,
                    "score": float(score),
                }
                bev.draw_box(box_3d)

            # ========== D. 在画面左上角叠加“全局 & 正前方距离” ==========
            y_offset = 20
            if global_min_distance is not None:
                text_g = f"Nearest obstacle: {global_min_distance:.2f}m"
                cv2.putText(
                    annotated_image,
                    text_g,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                y_offset += 22

            if forward_distance is not None:
                text_f = f"Forward distance: {forward_distance:.2f}m"
                cv2.putText(
                    annotated_image,
                    text_f,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
                y_offset += 22

            # ========== E. YOLO 目标列表（和之前一样） ==========
            summary_infos.sort(key=lambda x: x[0])
            for i, (dist, cname, oid) in enumerate(summary_infos[:5]):
                text_obj = f"{i+1}. ID:{oid} {cname}: {dist:.2f}m"
                cv2.putText(
                    annotated_image,
                    text_obj,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                y_offset += 18

            # ========== F. 深度测距点（不依赖 YOLO，只显示距离） ==========
            for (ux, uy) in depth_probe_points:
                px = int(w_img * ux)
                py = int(h_img * uy)
                distance_probe = get_distance_at_pixel(depth_m, px, py, region_half_size=6)
                if distance_probe is None:
                    continue

                # 画一个小圆点 + 距离文字（没有 ID、没有类别）
                cv2.circle(annotated_image, (px, py), 4, (0, 255, 255), -1)
                text_d = f"{distance_probe:.2f}m"
                (tw, th), _ = cv2.getTextSize(text_d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                x_text = min(px + 5, w_img - tw - 2)
                y_text = max(py - 5, th + 2)

                cv2.rectangle(
                    annotated_image,
                    (x_text - 2, y_text - th - 2),
                    (x_text + tw + 2, y_text + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    annotated_image,
                    text_d,
                    (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            # === 终端打印：每秒一次（全局 + 正前方 + 目标） ===
            now = time.time()
            if now - last_print_time >= 1.0:
                last_print_time = now
                msg_parts = []
                if global_min_distance is not None:
                    msg_parts.append(f"Nearest={global_min_distance:.2f}m")
                if forward_distance is not None:
                    msg_parts.append(f"Forward={forward_distance:.2f}m")
                if summary_infos:
                    obj_parts = [
                        f"ID:{oid} {cname} {dist:.2f}m"
                        for dist, cname, oid in summary_infos[:3]
                    ]
                    msg_parts.append("Objs: " + " | ".join(obj_parts))
                if msg_parts:
                    print("[Depth]", " || ".join(msg_parts))
                else:
                    print("[Depth] no valid depth")

            # 8. 右下角贴 BEV 小图
            bev_img = bev.get_image()
            target_bev_size = 260
            bev_img_resized = cv2.resize(
                bev_img, (target_bev_size, target_bev_size), interpolation=cv2.INTER_LINEAR
            )

            h, w, _ = annotated_image.shape
            x_start = w - target_bev_size - 10
            y_start = h - target_bev_size - 10
            x_start = max(0, x_start)
            y_start = max(0, y_start)

            roi_img = annotated_image[y_start:y_start + target_bev_size,
                                      x_start:x_start + target_bev_size]

            h_roi, w_roi, _ = roi_img.shape
            bev_crop = bev_img_resized[:h_roi, :w_roi]
            annotated_image[y_start:y_start + h_roi, x_start:x_start + w_roi] = bev_crop

            # 9. 深度热力图窗口（整幅深度 Heatmap）
            if valid.size > 0:
                max_depth_show = min(float(valid.max()), BEV_FAR_M)
                if max_depth_show > 0:
                    depth_clip = np.clip(depth_m, 0, max_depth_show)
                    depth_8u = cv2.convertScaleAbs(
                        depth_clip, alpha=255.0 / max_depth_show
                    )
                    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                    cv2.imshow("Depth Heatmap", depth_color)

            # 10. 显示主 RGB-D 结果
            cv2.imshow("RGB-D Navigation Perception", annotated_image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ESC_KEY):
                break

    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
        print("Pipeline stopped and windows closed.")


if __name__ == "__main__":
    main()
