r"""
cam_nav_map_shihe.py
-------------------------------------------------------------
大模型端/PC端：Orbbec RGB-D -> 深度转米 -> 局部2D栅格(XZ) -> 自动建议动作(只打印，不发ROS)
并可选融合到全局2D栅格(用于可视化)。

目录假设（按你说的）：
D:\机器人_无图_project
├─ pyorbbecsdk-2-main\
│    └─ examples\utils.py   (frame_to_bgr_image 在这里)
├─ main_RGB_D.py            (你说可参考/可用)
├─ global_map.py            (你上传的全局地图类，作为回退)
└─ shihe\
     ├─ cam_nav_map_shihe.py   (本文件)
     └─ global_map_shihe.py    (你现在在用的版本)

运行：
python cam_nav_map_shihe.py
按 q / ESC 退出
"""

import os
import sys
import time
import math
from typing import Tuple, Optional

import numpy as np
import cv2

# ================== 0) 路径设置：确保能 import 到 examples/utils.py 和 global_map ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # ...\shihe
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                       # D:\机器人_无图_project

ORB_DIR = os.path.join(PROJECT_ROOT, "pyorbbecsdk-2-main")
ORB_EXAMPLES_DIR = os.path.join(ORB_DIR, "examples")

for p in [PROJECT_ROOT, CURRENT_DIR, ORB_DIR, ORB_EXAMPLES_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Orbbec SDK
from pyorbbecsdk import (
    Pipeline, Config,
    OBSensorType, OBFormat,
    AlignFilter, OBStreamType
)

# 只从 examples/utils.py 用这个（你路径里是存在的）
from utils import frame_to_bgr_image

# 全局地图（优先用你 shihe 目录的版本；没有则回退根目录 global_map.py）
try:
    from global_map_shihe import Pose2D, GlobalGridMap
except Exception:
    from global_map import Pose2D, GlobalGridMap


# ================== 1) 深度帧 -> 米单位深度图 (H,W) ==================
def compute_depth_map_in_meters(depth_frame) -> Optional[np.ndarray]:
    """
    将 Orbbec depth frame 转成以米为单位的 float 深度图 (H, W)。
    兼容你之前遇到的 “1D shape” 问题：用 width/height reshape。
    """
    depth_format = depth_frame.get_format()
    if depth_format != OBFormat.Y16:
        # 你之前遇到过 RLE，这里先不硬解码（否则更容易越修越炸）
        print("[WARN] Depth format is not Y16, got:", depth_format, " -> skip")
        return None

    width = depth_frame.get_width()
    height = depth_frame.get_height()

    scale_raw = depth_frame.get_depth_scale()
    if not scale_raw:
        scale_raw = 1.0
    scale_raw = float(scale_raw)

    # 常见情况：scale_raw>=1 表示深度单位是“毫米”，转米需 /1000
    if scale_raw >= 1.0:
        scale_m = scale_raw / 1000.0
    else:
        scale_m = scale_raw

    depth_u16 = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(height, width)
    depth_m = depth_u16.astype(np.float32) * scale_m
    depth_m = np.where(depth_m > 0, depth_m, 0.0)
    return depth_m


# ================== 2) 初始化 Orbbec：彩色 + 深度 + 对齐 ==================
def _try_get_y16_profile(profile_list):
    """
    尝试拿到 Y16 深度 profile（避免默认 profile 变成 RLE）。
    不同版本 SDK 可能没有 get_video_stream_profile，做兼容。
    """
    if hasattr(profile_list, "get_video_stream_profile"):
        # 常用分辨率/帧率组合（你也可以自己改）
        candidates = [
            (640, 480, OBFormat.Y16, 30),
            (640, 400, OBFormat.Y16, 30),
            (848, 480, OBFormat.Y16, 30),
            (1280, 720, OBFormat.Y16, 30),
        ]
        for (w, h, fmt, fps) in candidates:
            try:
                return profile_list.get_video_stream_profile(w, h, fmt, fps)
            except Exception:
                pass
    return None


def init_orbbec_pipeline() -> Tuple[Pipeline, Config, AlignFilter]:
    pipeline = Pipeline()
    config = Config()

    # color
    color_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    # depth（优先拿 Y16，拿不到就用默认）
    depth_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = _try_get_y16_profile(depth_list)
    if depth_profile is None:
        depth_profile = depth_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    # frame sync
    try:
        pipeline.enable_frame_sync()
    except Exception as e:
        print("[WARN] enable_frame_sync failed:", e)

    pipeline.start(config)

    # align depth -> color
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    return pipeline, config, align_filter


# ================== 3) 深度图 -> 点云 -> 局部栅格(XZ) ==================
def depth_to_point_cloud(depth_m: np.ndarray,
                         fx: float, fy: float, cx: float, cy: float,
                         min_depth: float, max_depth: float,
                         stride: int = 4) -> np.ndarray:
    """深度图 -> 相机坐标系点云 (N,3). 相机坐标：x右, y下, z前"""
    H, W = depth_m.shape
    us = np.arange(0, W, stride)
    vs = np.arange(0, H, stride)
    uu, vv = np.meshgrid(us, vs)

    z = depth_m[vv, uu]
    mask = (z > min_depth) & (z < max_depth) & np.isfinite(z)
    uu = uu[mask].astype(np.float32)
    vv = vv[mask].astype(np.float32)
    z = z[mask].astype(np.float32)

    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def build_occupancy_grid(pts: np.ndarray,
                         grid_size_x: float = 3.0,
                         grid_size_z: float = 4.0,
                         resolution: float = 0.05,
                         height_min: float = -0.25,
                         height_max: float = 1.00) -> Tuple[np.ndarray, dict]:
    """
    局部栅格：0=unknown, 1=free(粗略射线), 2=occupied
    X范围 [-grid_size_x, grid_size_x], Z范围 [0, grid_size_z]
    """
    W = int((2 * grid_size_x) / resolution)
    H = int((grid_size_z) / resolution)
    grid = np.zeros((H, W), dtype=np.uint8)

    meta = {"H": H, "W": W, "grid_size_x": grid_size_x, "grid_size_z": grid_size_z, "res": resolution}

    if pts.shape[0] == 0:
        return grid, meta

    # 过滤高度（相机坐标y向下）
    mask_h = (pts[:, 1] > height_min) & (pts[:, 1] < height_max)
    pts2 = pts[mask_h]
    if pts2.shape[0] == 0:
        return grid, meta

    x = pts2[:, 0]
    z = pts2[:, 2]

    # 过滤范围
    mask_r = (x > -grid_size_x) & (x < grid_size_x) & (z > 0) & (z < grid_size_z)
    x = x[mask_r]
    z = z[mask_r]
    if x.size == 0:
        return grid, meta

    ix = ((x + grid_size_x) / (2 * grid_size_x) * W).astype(np.int32)
    iz = (z / grid_size_z * H).astype(np.int32)
    ix = np.clip(ix, 0, W - 1)
    iz = np.clip(iz, 0, H - 1)

    grid[iz, ix] = 2  # occupied

    # 粗略 free：从底部(机器人)到障碍前，标成free（按列）
    for col in range(W):
        occ = np.where(grid[:, col] == 2)[0]
        if occ.size == 0:
            continue
        first_occ = int(occ.min())
        if first_occ > 0:
            grid[:first_occ, col] = 1

    return grid, meta


def select_nav_goal_from_grid(grid: np.ndarray, meta: dict, center_margin_cols: int = 6) -> Optional[Tuple[float, float]]:
    """
    简单选目标：只看中间几列，在这些列里找“最远的 free 格子”，返回 (x,z) 米
    """
    H, W = meta["H"], meta["W"]
    gx, gz = meta["grid_size_x"], meta["grid_size_z"]

    c = W // 2
    c0 = max(0, c - center_margin_cols)
    c1 = min(W, c + center_margin_cols + 1)

    best = None
    for col in range(c0, c1):
        free_idx = np.where(grid[:, col] == 1)[0]
        if free_idx.size == 0:
            continue
        iz = int(free_idx[-1])
        if (best is None) or (iz > best[0]):
            best = (iz, col)

    if best is None:
        return None

    iz, ix = best
    x = (ix + 0.5) / W * (2 * gx) - gx
    z = (iz + 0.5) / H * gz
    return x, z


# ================== 4) 兼容 append_pose 两种签名 ==================
def append_pose_compat(global_map, pose: Pose2D):
    """
    你现在的 global_map_shihe.py 很可能是 append_pose(pose)；
    也可能有人写的是 append_pose(x, y)。
    这里做兼容，避免你再被这个坑反复搞。
    """
    try:
        global_map.append_pose(pose)  # ✅ 你当前版本大概率是这个
    except TypeError:
        global_map.append_pose(pose.x, pose.y)  # 兼容旧版


# ================== 5) 主循环 ==================
def main():
    pipeline, config, align_filter = init_orbbec_pipeline()
    print("[INFO] Orbbec Pipeline started.")
    print("[INFO] Start RGB-D mapping loop.")
    print("      自动指令：每帧从局部栅格选 nav_goal，并打印 turn/forward 建议（不发ROS）。")
    print("      按 q / ESC 退出。\n")

    # 全局地图（40m x 40m，5cm分辨率）
    global_map = GlobalGridMap(size_x=40.0, size_y=40.0, res=0.05)
    pose = Pose2D(0.0, 0.0, 0.0)

    # 局部栅格参数
    min_depth = 0.20
    max_depth = 6.00
    height_min = -0.25
    height_max = 1.00
    grid_size_x = 3.0
    grid_size_z = 4.0
    resolution = 0.05
    stride = 4

    # 内参：先用“可跑的近似”，成功后你再换成 profile.get_intrinsic()
    fx = fy = 600.0
    cx = cy = None

    last_print = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue

            aligned = None
            try:
                aligned = align_filter.process(frames)
            except Exception:
                aligned = None
            if aligned:
                frames = aligned

            if hasattr(frames, "as_frame_set"):
                frames = frames.as_frame_set()

            # 兼容取帧
            color_frame = frames.get_color_frame() if hasattr(frames, "get_color_frame") else None
            depth_frame = frames.get_depth_frame() if hasattr(frames, "get_depth_frame") else None

            if (not color_frame) or (not depth_frame):
                continue

            color_img = frame_to_bgr_image(color_frame)
            if color_img is None:
                continue

            depth_m = compute_depth_map_in_meters(depth_frame)
            if depth_m is None:
                continue

            H, W = depth_m.shape
            if cx is None:
                cx = W * 0.5
                cy = H * 0.5

            # 点云 + 局部栅格
            pts = depth_to_point_cloud(depth_m, fx, fy, cx, cy, min_depth, max_depth, stride=stride)
            local_grid, meta = build_occupancy_grid(
                pts,
                grid_size_x=grid_size_x, grid_size_z=grid_size_z,
                resolution=resolution,
                height_min=height_min, height_max=height_max
            )

            # 融合到全局图（当前 pose 不动；后续你接 /odom 或“执行器积分”就能动起来）
            try:
                global_map.fuse_local_grid(local_grid, meta, pose)
            except Exception as e:
                # 如果你 global_map_shihe 的 fuse_local_grid 签名不同，这里不会让整个程序崩
                print("[WARN] fuse_local_grid failed:", repr(e))

            append_pose_compat(global_map, pose)

            # 选目标 & 打印建议
            nav_goal = select_nav_goal_from_grid(local_grid, meta, center_margin_cols=6)
            if nav_goal is not None:
                gx, gz = nav_goal

                # 约定：输出“左转为正”，而相机坐标 x>0 是右侧，所以取负号
                turn_deg = -math.degrees(math.atan2(gx, gz))
                forward_m = float(gz)

                if time.time() - last_print > 0.25:
                    print(f"[AUTO] suggest: turn {turn_deg:+.1f} deg, forward {forward_m:.2f} m   (goal x={gx:+.2f}, z={gz:.2f})")
                    last_print = time.time()

            # 可视化：rgb / depth / local / global
            depth_vis = np.clip(depth_m, 0.0, max_depth)
            depth_vis = (depth_vis / max_depth * 255.0).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # local grid 可视化：0 unknown 黑，1 free 灰，2 occ 白
            local_vis = np.zeros((meta["H"], meta["W"], 3), dtype=np.uint8)
            local_vis[local_grid == 1] = (120, 120, 120)
            local_vis[local_grid == 2] = (255, 255, 255)

            # 标机器人在底部中间
            rr = meta["H"] - 2
            rc = meta["W"] // 2
            cv2.circle(local_vis, (rc, rr), 3, (0, 255, 0), -1)

            # 标目标
            if nav_goal is not None:
                gx, gz = nav_goal
                ix = int((gx + meta["grid_size_x"]) / (2 * meta["grid_size_x"]) * meta["W"])
                iz = int((gz / meta["grid_size_z"]) * meta["H"])
                ix = max(0, min(meta["W"] - 1, ix))
                iz = max(0, min(meta["H"] - 1, iz))
                cv2.circle(local_vis, (ix, iz), 4, (0, 0, 255), -1)

            # global map render（如果返回单通道，也能显示）
            try:
                global_vis = global_map.render(scale=4)
            except Exception:
                global_vis = np.zeros((400, 400, 3), dtype=np.uint8)

            cv2.imshow("rgb", color_img)
            cv2.imshow("depth", depth_vis)
            cv2.imshow("local_grid_xz", cv2.resize(local_vis, (400, 400), interpolation=cv2.INTER_NEAREST))
            cv2.imshow("global_map", global_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[INFO] Camera stopped, exit.")


if __name__ == "__main__":
    main()
