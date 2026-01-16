# -*- coding: utf-8 -*-
"""
global_map.py  (动态环境版 + 论文风格渲染)
---------------------------------
全局 2D 栅格地图（不依赖 Habitat），支持“桌椅被挪动后地图自动更新”，
并提供类似论文截图的可视化效果。

核心思路：
- 维护一个 occ_logodds[H, W]（占据对数概率）：
    > 0 倾向障碍； < 0 倾向 free；≈0 不确定。
- 每一帧局部栅格 local_grid:
    - local_grid == 2（障碍观测）：该格 log-odds += L_occ
    - local_grid == 1（free 观测）：该格 log-odds -= L_free
- 每次更新后，根据 log-odds 大小重新生成 grid:
    - log-odds >  L_occ_th   ->  障碍 (2)
    - log-odds < -L_free_th  ->  free (1)
    - 否则保持 unknown (0)

这样：
- 某处原来是桌子，多次看到障碍 -> log-odds 很大，稳定为障碍。
- 桌子被挪走，多次看到 free -> log-odds 慢慢被拉回甚至变成负，最终从障碍变 free。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2


# ===================== 1. 位姿 =====================

@dataclass
class Pose2D:
    """
    世界坐标系中的 2D 位姿：
    - x: 向前（world X）
    - y: 向左（world Y）
    - theta: 朝向，弧度，0 表示朝 +X 方向，逆时针为正
    """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.theta

    def local_to_world_xy(self, xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """
        局部 (x_local, z_local) -> 世界 (X_world, Y_world)

        局部坐标：
          x_local: 右为正
          z_local: 前为正
        世界坐标：
          X_world: 前
          Y_world: 左
        """
        vx = zs.astype(np.float32)
        vy = (-xs).astype(np.float32)
        v_local = np.stack([vx, vy], axis=-1)  # (N, 2) = (forward, left)

        c = math.cos(self.theta)
        s = math.sin(self.theta)
        R = np.array([[c, -s],
                      [s,  c]], dtype=np.float32)

        world = v_local @ R.T
        world[:, 0] += self.x
        world[:, 1] += self.y
        return world

    def world_to_local_xz(self, world_xy: np.ndarray) -> np.ndarray:
        """
        世界 (X, Y) -> 局部 (x_local, z_local)
        """
        dx = world_xy[:, 0] - self.x
        dy = world_xy[:, 1] - self.y
        v = np.stack([dx, dy], axis=-1)

        c = math.cos(-self.theta)
        s = math.sin(-self.theta)
        R_inv = np.array([[c, -s],
                          [s,  c]], dtype=np.float32)
        local_forward_left = v @ R_inv.T

        z_local = local_forward_left[:, 0]
        x_local = -local_forward_left[:, 1]
        return np.stack([x_local, z_local], axis=-1)

    def update_from_cmd(self, v_forward: float, v_yaw: float, dt: float):
        """
        简单里程计积分更新（demo 用）：
        v_forward: 前进速度（m/s）
        v_yaw:     角速度  （rad/s）
        dt:        时间步长（s）
        """
        self.theta += v_yaw * dt

        dx = v_forward * dt * math.cos(self.theta)
        dy = v_forward * dt * math.sin(self.theta)
        self.x += dx
        self.y += dy


# ===================== 2. 全局占据栅格（动态环境版） =====================

@dataclass
class GlobalGridMap:
    """
    动态环境下的 2D 全局占据栅格：
      - 尺寸 size_x * size_y（米）
      - 分辨率 res（米/格）
      - 状态由 occ_logodds 控制，可随观测变化更新：
          grid:
            0: unknown
            1: free
            2: obstacle
    """

    size_x: float = 20.0
    size_y: float = 20.0
    res: float = 0.05

    # 占据对数概率：正 -> 更像障碍；负 -> 更像 free；0 -> 不确定
    occ_logodds: np.ndarray = field(init=False)
    grid: np.ndarray = field(init=False)

    origin_x: float = field(init=False)
    origin_y: float = field(init=False)
    H: int = field(init=False)
    W: int = field(init=False)

    # 轨迹（世界坐标）
    traj_world: List[Tuple[float, float]] = field(default_factory=list)

    # log-odds 更新参数
    L_occ: float = 0.85    # 一次“障碍”观测的增量
    L_free: float = 0.4    # 一次“free”观测的减量
    L_min: float = -4.0    # 下限（避免数值爆炸）
    L_max: float = 4.0     # 上限
    L_occ_th: float = 0.7  # 判定为障碍的阈值
    L_free_th: float = 0.7 # 判定为 free 的阈值

    def __post_init__(self):
        self.W = int(round(self.size_x / self.res))
        self.H = int(round(self.size_y / self.res))

        self.occ_logodds = np.zeros((self.H, self.W), dtype=np.float32)
        self.grid = np.zeros((self.H, self.W), dtype=np.int8)

        self.origin_x = -self.size_x / 2.0
        self.origin_y = -self.size_y / 2.0

    # ------------ 坐标转换 ------------

    def world_to_grid(self, X: float, Y: float) -> Tuple[int, int]:
        gx = (X - self.origin_x) / self.res
        gy = (Y - self.origin_y) / self.res
        ix = int(math.floor(gx))
        iy = int(math.floor(gy))
        return ix, iy

    def grid_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        X = self.origin_x + (ix + 0.5) * self.res
        Y = self.origin_y + (iy + 0.5) * self.res
        return X, Y

    # ------------ 地图更新（关键：支持桌椅移动） ------------

    def fuse_local_grid(
        self,
        local_grid: np.ndarray,
        local_meta: Dict,
        pose: Pose2D,
    ):
        """
        把一帧“局部栅格”融合进全局地图，并通过 log-odds 更新实现动态更新。

        local_grid: (H_loc, W_loc)  0 unknown, 1 free, 2 obstacle
        local_meta: {
            "W": W_loc, "H": H_loc,
            "grid_size_x": grid_size_x,
            "grid_size_z": grid_size_z,
            "res": resolution,
        }
        pose: 当前机器人位姿，负责把局部坐标变成世界坐标。
        """
        H_loc, W_loc = local_grid.shape
        grid_size_x = float(local_meta["grid_size_x"])
        grid_size_z = float(local_meta["grid_size_z"])

        # 局部栅格每个 cell 的中心坐标（局部 x,z）
        xs = (np.arange(W_loc) + 0.5) / W_loc * (2 * grid_size_x) - grid_size_x
        zs = (np.arange(H_loc) + 0.5) / H_loc * grid_size_z
        xs_mesh, zs_mesh = np.meshgrid(xs, zs)  # (H_loc, W_loc)

        xs_flat = xs_mesh.reshape(-1)
        zs_flat = zs_mesh.reshape(-1)
        vals_flat = local_grid.reshape(-1)

        # 只用 free(1) 或 obstacle(2) 的格子（unknown=0 不更新）
        mask = vals_flat > 0
        if not np.any(mask):
            return

        xs_flat = xs_flat[mask].astype(np.float32)
        zs_flat = zs_flat[mask].astype(np.float32)
        vals_flat = vals_flat[mask].astype(np.int8)

        # 局部 -> 世界
        world_xy = pose.local_to_world_xy(xs_flat, zs_flat)
        Xs = world_xy[:, 0]
        Ys = world_xy[:, 1]

        gx = ((Xs - self.origin_x) / self.res).astype(np.int32)
        gy = ((Ys - self.origin_y) / self.res).astype(np.int32)

        in_mask = (gx >= 0) & (gx < self.W) & (gy >= 0) & (gy < self.H)
        gx = gx[in_mask]
        gy = gy[in_mask]
        vals_flat = vals_flat[in_mask]

        if gx.size == 0:
            return

        # --- log-odds 更新 ---
        for ix, iy, v in zip(gx, gy, vals_flat):
            if v == 2:       # 观测到障碍
                self.occ_logodds[iy, ix] += self.L_occ
            elif v == 1:     # 观测到 free
                self.occ_logodds[iy, ix] -= self.L_free

        # 限制范围
        np.clip(self.occ_logodds, self.L_min, self.L_max, out=self.occ_logodds)

        # 根据 log-odds 重新生成 grid
        self.grid[:] = 0
        self.grid[self.occ_logodds > self.L_occ_th] = 2
        self.grid[self.occ_logodds < -self.L_free_th] = 1

    # ------------ 轨迹记录 & 可视化 ------------

    def append_pose(self, pose: Pose2D):
        self.traj_world.append((pose.x, pose.y))

    def render(
        self,
        pose: Optional[Pose2D] = None,
        nav_goal_world: Optional[Tuple[float, float]] = None,
        frontier_points: Optional[List[Tuple[float, float]]] = None,
        scale: int = 4,
    ) -> np.ndarray:
        """
        论文风格的全局地图渲染：

          grid 语义：
            0: unknown
            1: free
            2: obstacle

        颜色约定（接近你论文截图的效果）：
          unknown: 白色背景
          free:    淡绿色 (180, 255, 180)
          obstacle: 深灰/黑 (40, 40, 40)
          轨迹:    亮绿色实线 (0, 255, 0)
          机器人:  蓝色实心圆 (255, 0, 0)
          nav_goal:黄色实心圆 (0, 255, 255)
          frontier_points: 蓝色空心圆（模拟论文里的很多蓝圈）
        """
        vis = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # 默认背景：unknown -> 白色
        vis[:, :] = (255, 255, 255)

        # 障碍：深灰 / 黑色
        vis[self.grid == 2] = (40, 40, 40)

        # free：淡绿色
        vis[self.grid == 1] = (180, 255, 180)

        # 轨迹（亮绿色线）
        if len(self.traj_world) >= 2:
            pts = []
            for X, Y in self.traj_world:
                ix, iy = self.world_to_grid(X, Y)
                if 0 <= ix < self.W and 0 <= iy < self.H:
                    pts.append((ix, iy))
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    cv2.line(vis, pts[i], pts[i + 1], (0, 255, 0), 2)

        # 机器人当前位置（蓝色点）
        if pose is not None:
            ix, iy = self.world_to_grid(pose.x, pose.y)
            if 0 <= ix < self.W and 0 <= iy < self.H:
                cv2.circle(vis, (ix, iy), 4, (255, 0, 0), -1)

        # 当前目标点（黄色点）
        if nav_goal_world is not None:
            gx, gy = nav_goal_world
            ix, iy = self.world_to_grid(gx, gy)
            if 0 <= ix < self.W and 0 <= iy < self.H:
                cv2.circle(vis, (ix, iy), 4, (0, 255, 255), -1)

        # frontier / 候选点（蓝色空心圆）
        if frontier_points is not None:
            for (fx, fy) in frontier_points:
                ix, iy = self.world_to_grid(fx, fy)
                if 0 <= ix < self.W and 0 <= iy < self.H:
                    cv2.circle(vis, (ix, iy), 5, (255, 0, 0), 2)

        if scale != 1:
            vis = cv2.resize(
                vis,
                (self.W * scale, self.H * scale),
                interpolation=cv2.INTER_NEAREST,
            )
        return vis

    # ------------ 示例：全局 frontier 目标点选择 ------------

    def choose_nav_goal_frontier(
        self,
        pose: Pose2D,
        max_range_m: float = 8.0,
    ) -> Optional[Tuple[float, float]]:
        """
        非论文完整版，只是一个 demo：
        - 在机器人周围 max_range_m 内找 "frontier cell":
            当前 unknown，邻居包含 free。
        - 选一个距离最近且略偏前方的，返回 (X_goal, Y_goal)。

        这层以后可以和 LLM / 语义融合成 VLFM 那套 value map。
        """
        ix0, iy0 = self.world_to_grid(pose.x, pose.y)
        if not (0 <= ix0 < self.W and 0 <= iy0 < self.H):
            return None

        max_cells = int(max_range_m / self.res)
        candidates = []

        for dy in range(-max_cells, max_cells + 1):
            for dx in range(-max_cells, max_cells + 1):
                ix = ix0 + dx
                iy = iy0 + dy
                if not (0 <= ix < self.W and 0 <= iy < self.H):
                    continue

                if self.grid[iy, ix] != 0:
                    continue  # 不是 unknown，就不是 frontier

                # 邻域有 free 则是 frontier
                is_frontier = False
                for yy in range(max(0, iy - 1), min(self.H, iy + 2)):
                    for xx in range(max(0, ix - 1), min(self.W, ix + 2)):
                        if self.grid[yy, xx] == 1:
                            is_frontier = True
                            break
                    if is_frontier:
                        break

                if not is_frontier:
                    continue

                Xc, Yc = self.grid_to_world(ix, iy)
                dist = math.hypot(Xc - pose.x, Yc - pose.y)
                forward_bias = Xc  # 越“靠前”越好

                candidates.append((dist, -forward_bias, Xc, Yc))

        if not candidates:
            return None

        candidates.sort(key=lambda t: (t[0], t[1]))
        _, _, X_goal, Y_goal = candidates[0]
        return X_goal, Y_goal
