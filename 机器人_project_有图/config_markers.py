# config_markers.py
"""
基础配置：
- 本地大模型名字、Ollama 接口
- 多地图配置（从 maps.json 加载）
- 允许的动作类型（从 actions_config.json 加载）
- 当前地图状态 & 工具函数
- 点位的持久化增删：直接改写 maps.json
"""

import json
from pathlib import Path
from typing import List

# ====== 大模型 & Ollama 配置 ======

# 本地 Qwen 模型名（和你 ollama pull 的名字一致）
MODEL_NAME = "qwen2.5:3b"

# Ollama 默认 HTTP 接口
OLLAMA_URL = "http://localhost:11434/api/chat"


# ====== 多地图配置：从 JSON 文件加载 ======

BASE_DIR = Path(__file__).resolve().parent
MAPS_FILE = BASE_DIR / "maps.json"               # 地图配置
ACTIONS_FILE = BASE_DIR / "actions_config.json"  # 动作配置

# 加载地图
if not MAPS_FILE.exists():
    raise RuntimeError(f"找不到地图配置文件：{MAPS_FILE}")

with MAPS_FILE.open("r", encoding="utf-8") as f:
    MAPS: dict = json.load(f)

# 当前使用的地图（进程内全局状态）
CURRENT_MAP_NAME = "map1"


# ====== 允许的动作类型：从 JSON 文件加载 ======

if not ACTIONS_FILE.exists():
    raise RuntimeError(f"找不到动作配置文件：{ACTIONS_FILE}")

with ACTIONS_FILE.open("r", encoding="utf-8") as f:
    _actions_cfg = json.load(f)

if "allowed_actions" not in _actions_cfg or not isinstance(_actions_cfg["allowed_actions"], list):
    raise ValueError("actions_config.json 中必须包含 allowed_actions 列表")

# 转成 set，给 validate_command 用来校验 action
ALLOWED_ACTIONS = set(_actions_cfg["allowed_actions"])


# ====== 一些基础工具函数 ======

def get_current_map_name() -> str:
    """获取当前地图名"""
    return CURRENT_MAP_NAME


def set_current_map_name(new_name: str) -> None:
    """切换当前地图（只改内存，不发给机器人）"""
    global CURRENT_MAP_NAME
    if new_name not in MAPS:
        raise ValueError(f"未知 map_name: {new_name}，必须是 {list(MAPS.keys())} 之一")
    CURRENT_MAP_NAME = new_name


def get_current_markers():
    """获取当前地图的点位列表"""
    return MAPS[CURRENT_MAP_NAME]["markers"]


def get_maps_brief():
    """
    返回所有地图的简要信息，用于 prompt 展示：
    { "map1": {"desc": "...", "floor": 1}, "map2": {...} }
    """
    return {
        name: {"desc": cfg.get("desc", ""), "floor": cfg.get("floor")}
        for name, cfg in MAPS.items()
    }


def get_current_floor() -> int:
    """获取当前地图的楼层（直接从 MAPS 配置中读）"""
    return int(MAPS[CURRENT_MAP_NAME]["floor"])


# ====== 点位持久化：真正修改 maps.json ======

def _save_maps_to_file() -> None:
    """把内存中的 MAPS 写回 maps.json"""
    with MAPS_FILE.open("w", encoding="utf-8") as f:
        json.dump(MAPS, f, ensure_ascii=False, indent=2)


def marker_exists_in_current_map(marker_id: str) -> bool:
    """当前地图中是否已存在给定 marker_id"""
    marker_id = str(marker_id).strip()
    return any(m["marker_id"] == marker_id for m in get_current_markers())


def add_marker_to_current_map(
    marker_id: str,
    aliases: List[str] | None = None,
    floor: int | None = None,
) -> None:
    """
    在当前地图中新增一个 marker，并写入 maps.json。
    - marker_id: 点位 id（如 "C"）
    - aliases: 别名列表，不传则自动生成 ["C点", "点C"]
    - floor: 楼层，不传则用当前地图的 floor
    """
    marker_id = str(marker_id).strip()
    if not marker_id:
        raise ValueError("add_marker_to_current_map: marker_id 不能为空")

    markers = get_current_markers()
    if any(m["marker_id"] == marker_id for m in markers):
        raise ValueError(f"add_marker_to_current_map: 地图 {CURRENT_MAP_NAME} 中已存在 marker_id={marker_id}")

    if aliases is None:
        aliases = [f"{marker_id}点", f"点{marker_id}"]
    aliases = [str(a).strip() for a in aliases if str(a).strip()]

    if not aliases:
        aliases = [marker_id]

    if floor is None:
        floor = get_current_floor()

    new_marker = {
        "marker_id": marker_id,
        "aliases": aliases,
        "floor": floor,
    }
    markers.append(new_marker)
    _save_maps_to_file()
    print(f"[config_markers] 已在地图 {CURRENT_MAP_NAME} 中新增点位: {new_marker}")


def remove_marker_from_current_map(marker_id: str) -> bool:
    """
    从当前地图删除一个 marker，并写入 maps.json。
    返回 True 表示确实删除了一个，False 表示没找到该点位。
    """
    marker_id = str(marker_id).strip()
    if not marker_id:
        raise ValueError("remove_marker_from_current_map: marker_id 不能为空")

    markers = get_current_markers()
    idx_to_del = -1
    for i, m in enumerate(markers):
        if m.get("marker_id") == marker_id:
            idx_to_del = i
            break

    if idx_to_del == -1:
        print(f"[config_markers] 当前地图 {CURRENT_MAP_NAME} 中没有 marker_id={marker_id}，无需删除")
        return False

    deleted = markers.pop(idx_to_del)
    _save_maps_to_file()
    print(f"[config_markers] 已从地图 {CURRENT_MAP_NAME} 删除点位: {deleted}")
    return True
