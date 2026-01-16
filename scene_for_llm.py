# scene_for_llm.py
from typing import List, Dict, Any
import math
import numpy as np

def pixel_to_yaw_deg(cx: float, img_w: int, hfov_deg: float = 70.0) -> float:
    """
    把目标中心点的水平像素位置，大致换算成相对机器人朝向的角度（度）。
    约定：左边为正角度（需要左转），右边为负角度。
    """
    center_x = img_w / 2.0
    half_w = img_w / 2.0
    # [-1, 1]，右正左负
    norm = (cx - center_x) / max(half_w, 1e-6)
    # 我们想要“左正右负”，所以取个反号
    yaw = -norm * (hfov_deg / 2.0)
    return float(yaw)


def build_scene_state(
    detections: List,             # YOLO 返回的 [bbox, score, class_id, obj_id] 列表
    class_names: Dict[int, str],  # detector.get_class_names()
    depth_m: np.ndarray,          # 深度图（米）
    img_w: int,
    img_h: int,
    get_distance_for_bbox_func,   # 传入 main_RGB_D 里的那个函数
) -> Dict[str, Any]:
    """
    把当前一帧变成给大模型看的“场景 JSON”。

    返回示例结构：
    {
      "global": {
        "image_size": [1080, 1920],
      },
      "objects": [
        {
          "id": 35,
          "track_id": 35,
          "class": "person",
          "distance_m": 3.47,
          "bearing_deg": 12.3,
          "center_norm": [0.55, 0.41]
        },
        ...
      ],
      "relations": [
        {"a": 101, "b": 94, "type": "near"},
        {"a": 101, "b": 94, "type": "right_of"},
        ...
      ]
    }
    """
    objects = []
    for det in detections:
        bbox, score, class_id, obj_id = det
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        distance = get_distance_for_bbox_func(depth_m, bbox)
        if distance is None or distance <= 0:
            continue

        bearing = pixel_to_yaw_deg(cx, img_w)

        obj = {
            "track_id": int(obj_id),
            "id": int(obj_id),  # 给大模型用的简化 id
            "class": class_names[int(class_id)],
            "score": float(score),
            "distance_m": round(float(distance), 2),
            "bearing_deg": round(float(bearing), 1),
            "center_norm": [
                round(cx / max(img_w, 1), 3),
                round(cy / max(img_h, 1), 3),
            ],
        }
        objects.append(obj)

    # 简单关系：谁在谁左边 / 右边，谁离谁很近（基于像素 + 距离）
    relations = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a = objects[i]
            b = objects[j]
            # left / right: 比较 center_norm 的 x
            if a["center_norm"][0] < b["center_norm"][0] - 0.05:
                relations.append({"a": a["id"], "b": b["id"], "type": "left_of"})
                relations.append({"a": b["id"], "b": a["id"], "type": "right_of"})
            elif a["center_norm"][0] > b["center_norm"][0] + 0.05:
                relations.append({"a": a["id"], "b": b["id"], "type": "right_of"})
                relations.append({"a": b["id"], "b": a["id"], "type": "left_of"})

            # near: 像素距离和深度差都不大就算“旁边”
            dx = a["center_norm"][0] - b["center_norm"][0]
            dy = a["center_norm"][1] - b["center_norm"][1]
            pixel_dist = math.hypot(dx, dy)
            depth_diff = abs(a["distance_m"] - b["distance_m"])
            if pixel_dist < 0.12 and depth_diff < 0.6:
                relations.append({"a": a["id"], "b": b["id"], "type": "near"})
                relations.append({"a": b["id"], "b": a["id"], "type": "near"})

    scene = {
        "global": {
            "image_size": [img_w, img_h],
        },
        "objects": objects,
        "relations": relations,
    }
    return scene
