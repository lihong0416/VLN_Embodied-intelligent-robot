# command_logic.py
"""
负责两件事：
1. 校验模型输出的 JSON 是否符合预期（字段、取值、安全范围等）。
2. 把合法的 JSON 指令映射为 AGV 的文本 API 命令字符串。
"""

from typing import Dict, Any
from config_markers import (
    get_current_markers,
    ALLOWED_ACTIONS,
    MAPS,
)


def validate_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    """
    对模型给出的 JSON 做一层安全检查&补充。
    """
    if "action" not in cmd:
        raise ValueError("JSON 里没有 action 字段")

    action = cmd["action"]
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"不支持的 action: {action}")

    # 当前地图的 marker 集合（静态 + 运行时修改后的）
    marker_ids = {m["marker_id"] for m in get_current_markers()}

    # 1）去单个点
    if action == "move_to_marker":
        marker = cmd.get("marker")
        if marker not in marker_ids:
            raise ValueError(f"非法 marker: {marker}（不在当前地图的点位列表中）")

    # 2）多点巡游
    if action == "cruise_markers":
        mks = cmd.get("markers", [])
        if not mks or any(m not in marker_ids for m in mks):
            raise ValueError(f"非法 markers: {mks}（必须都在当前地图点位列表中）")
        count = cmd.get("count", 1)
        if not isinstance(count, int) or count == 0:
            raise ValueError("count 必须是非 0 整数（正数=巡游次数，-1=无限巡游）")

    # 3）遥控速度，顺便做个限幅
    if action == "joy_control":
        lin = float(cmd.get("linear_velocity", 0.0))
        ang = float(cmd.get("angular_velocity", 0.0))
        lin = max(min(lin, 0.6), -0.6)
        ang = max(min(ang, 1.0), -1.0)
        cmd["linear_velocity"] = lin
        cmd["angular_velocity"] = ang

    # 4）灯光颜色 0~100 限制一下
    if action == "set_led_color":
        for k in ["r", "g", "b"]:
            v = int(cmd.get(k, 0))
            v = max(min(v, 100), 0)
            cmd[k] = v

    # 5）切换地图 set_map
    if action == "set_map":
        map_name = cmd.get("map_name")
        if map_name not in MAPS:
            raise ValueError(f"非法 map_name: {map_name}（必须是 {list(MAPS.keys())} 之一）")
        # 如果 floor 没写，或者写错了，就用配置里的
        cfg = MAPS[map_name]
        cfg_floor = cfg["floor"]
        floor = cmd.get("floor")
        if floor is None or floor != cfg_floor:
            cmd["floor"] = cfg_floor

    # 6）在当前位置新增 marker（只检查名称非空，是否已存在交给上层逻辑）
    if action == "insert_marker_here":
        name = str(cmd.get("marker", "")).strip()
        if not name:
            raise ValueError("insert_marker_here 需要提供非空的 marker 名称")
        cmd["marker"] = name
        # 可选参数：类型、编号，给默认值
        mtype = int(cmd.get("marker_type", 0))
        mnum = int(cmd.get("marker_num", 0))
        cmd["marker_type"] = mtype
        cmd["marker_num"] = mnum

    # 7）查询 marker 列表
    if action == "query_marker_list":
        floor = cmd.get("floor", None)
        if floor is not None:
            try:
                cmd["floor"] = int(floor)
            except Exception:
                raise ValueError("query_marker_list 的 floor 必须是整数（或不填）")

    # 8）删除 marker（必须当前地图存在）
    if action == "delete_marker":
        name = cmd.get("marker")
        if name not in marker_ids:
            raise ValueError(f"delete_marker: 当前地图不存在点位 {name}")

    # 9）统计 marker 个数（不需要额外字段）
    if action == "count_markers":
        pass

    if action == "request_human_detection":
        freq = float(cmd.get("frequency", 1.0))
        # 做一下安全限制：<=0 就回退成 1.0
        if freq <= 0:
            freq = 1.0
        cmd["frequency"] = freq

    # 其他 action 默认不做额外检查，有需要可以继续加
    return cmd


def build_robot_api_string(cmd: Dict[str, Any]) -> str | None:
    """
    把 JSON 指令映射成 AGV 的文本 API 指令字符串。
    当前只是打印使用，不真正发给机器人。
    """
    action = cmd["action"]

    # 需要澄清：不下发指令，由上层去提示用户
    if action == "ask_user_clarification":
        return None

    if action == "move_to_marker":
        marker = cmd["marker"]
        dist = cmd.get("options", {}).get("distance_tolerance", 0.5)
        return f"/api/move?marker={marker}&distance_tolerance={dist}"

    if action == "cruise_markers":
        markers = ",".join(cmd["markers"])
        count = cmd.get("count", 1)
        dist = cmd.get("distance_tolerance", 0.5)
        return f"/api/move?markers={markers}&count={count}&distance_tolerance={dist}"

    if action == "cancel_move":
        return "/api/move/cancel"

    if action == "estop":
        flag = "true" if cmd.get("flag", True) else "false"
        return f"/api/estop?flag={flag}"

    if action == "joy_control":
        lin = cmd["linear_velocity"]
        ang = cmd["angular_velocity"]
        return f"/api/joy_control?angular_velocity={ang}&linear_velocity={lin}"

    if action == "set_map":
        map_name = cmd["map_name"]
        floor = cmd["floor"]
        return f"/api/map/set_current_map?map_name={map_name}&floor={floor}"

    if action == "set_led_color":
        r, g, b = cmd["r"], cmd["g"], cmd["b"]
        return f"/api/LED/set_color?r={r}&g={g}&b={b}"

    # ====== 下面四个 URL 需要你对照 AGV API 手册改成真实路径 ======
    if action == "insert_marker_here":
        name = cmd["marker"]
        mtype = cmd.get("marker_type", 0)
        mnum = cmd.get("marker_num", 0)
        # TODO: 根据手册实际接口改
        return f"/api/markers/insert_here?name={name}&type={mtype}&num={mnum}"

    if action == "query_marker_list":
        floor = cmd.get("floor", None)
        if floor is None:
            return "/api/markers/list"
        else:
            return f"/api/markers/list?floor={floor}"

    if action == "delete_marker":
        name = cmd["marker"]
        # TODO: 根据手册实际接口改
        return f"/api/markers/delete?name={name}"

    if action == "count_markers":
        # TODO: 根据手册实际接口改
        return "/api/markers/count"

    if action == "request_human_detection":
        freq = cmd.get("frequency", 1.0)
        return f"/api/request_data?topic=human_detection&frequency={freq}"


    # 理论上不会走到这里
    return None
