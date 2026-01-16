# agv_nul_cli.py
"""
命令行 demo（本地版，不连机器人）：
- 输入一条中文指令
- 调用本地 Qwen（经 Ollama）
- 校验 JSON
- 本地更新地图状态（切图 / 新增点位 / 删除点位，并写 maps.json）
- 打印出要发送给 AGV 的 API 串
- 如果是带 duration_s 的 joy_control，会额外打印“宏指令拆解”信息
"""

import json

from model_client import call_qwen_local_and_get_json
from command_logic import validate_command, build_robot_api_string
from config_markers import (
    get_current_map_name,
    set_current_map_name,
    add_marker_to_current_map,
    remove_marker_from_current_map,
    marker_exists_in_current_map,
)

# 文档里一个 joy_control 命令大约持续 0.5s，你可以按需要调整
STEP_DURATION = 0.5


# ===== 根据中文指令纠正左/右转方向 =====

def normalize_turn_from_text(cmd: dict, user_text: str) -> dict:
    """
    对于 action=joy_control 的指令，根据原始中文强制规范：
      - “左转” → linear_velocity = 0.0, angular_velocity > 0
      - “右转” → linear_velocity = 0.0, angular_velocity < 0
    避免大模型把左右都学成同一个符号。
    """
    if cmd.get("action") != "joy_control":
        return cmd

    text = user_text.replace(" ", "")
    has_left = ("左转" in text) or ("向左转" in text) or ("左" in text and "转" in text)
    has_right = ("右转" in text) or ("向右转" in text) or ("右" in text and "转" in text)

    # 只看到“左转”而没看到“右”
    if has_left and not has_right:
        cmd["linear_velocity"] = 0.0
        ang = float(cmd.get("angular_velocity", 0.5) or 0.5)
        if ang == 0.0:
            ang = 0.5
        cmd["angular_velocity"] = abs(ang)
        return cmd

    # 只看到“右转”而没看到“左”
    if has_right and not has_left:
        cmd["linear_velocity"] = 0.0
        ang = float(cmd.get("angular_velocity", -0.5) or -0.5)
        if ang == 0.0:
            ang = -0.5
        cmd["angular_velocity"] = -abs(ang)
        return cmd

    return cmd


# ===== 点位相关：本地修改 maps.json =====

def apply_local_marker_side_effects(cmd: dict) -> None:
    """
    在“本地 demo 模式”下，对点位相关指令做真正的 maps.json 增删。

    注意：这里相当于假定机器人一定执行成功。
    将来如果你接了真实 HTTP / TCP 接口，建议改成：
    - 先调 AGV API
    - AGV 返回 success 之后，再调用这些函数写 maps.json
    """
    action = cmd.get("action")

    # 在当前位置新增点位
    if action == "insert_marker_here":
        name = cmd["marker"]

        if marker_exists_in_current_map(name):
            print(f"[本地] 当前地图已存在点位 {name}，不重复写入 maps.json")
            return

        # 不传 aliases / floor，则使用 config_markers 里的默认逻辑
        add_marker_to_current_map(name)
        print(f"[本地] 已在 maps.json 的地图 {get_current_map_name()} 中新增点位 {name}")

    # 删除现有点位
    elif action == "delete_marker":
        name = cmd["marker"]
        removed = remove_marker_from_current_map(name)
        if removed:
            print(f"[本地] 已在 maps.json 的地图 {get_current_map_name()} 中删除点位 {name}")
        else:
            print(f"[本地] maps.json 中没有点位 {name}，无需删除")


# ===== joy_control 宏指令：只做“拆解说明”，不真实发给机器人 =====

def print_joy_control_macro_info(cmd: dict, api_str: str) -> None:
    """
    对带 duration_s 的 joy_control 做一个“宏指令拆解”的打印，
    帮你直观理解：
      - 线速度 / 角速度
      - 总时长
      - 需要发多少次 /api/joy_control
    """
    lin = float(cmd.get("linear_velocity", 0.0))
    ang = float(cmd.get("angular_velocity", 0.0))
    duration = float(cmd.get("duration_s", STEP_DURATION))

    if duration <= 0:
        duration = STEP_DURATION

    # 为了防止 LLM 给太大值，这里简单限制一下（只是 demo）
    max_duration = 5.0  # 最多连续 5 秒
    if duration > max_duration:
        duration = max_duration

    steps = max(1, int(round(duration / STEP_DURATION)))

# ===== 主入口 =====

def main():
    print("=== 本地 Qwen + Ollama Demo：中文指令 → JSON → AGV API 串 ===")
    print("说明：")
    print("输入 q 退出。\n")

    while True:
        current_map = get_current_map_name()
        print(f"[当前地图] {current_map}")
        user_text = input("请输入一条指令：").strip()

        if user_text.lower() in {"q", "quit", "exit"}:
            print("退出。")
            break

        if not user_text:
            continue

        # 1. 调大模型
        try:
            cmd = call_qwen_local_and_get_json(user_text)
        except Exception as e:
            print(f"[调用大模型失败] {e}")
            continue

        # 1.5 根据原始中文，修正左/右转的角速度符号
        cmd = normalize_turn_from_text(cmd, user_text)

        # 2. 做安全校验 / 补默认值
        try:
            cmd = validate_command(cmd)
        except Exception as e:
            print(f"[指令校验失败] {e}")
            continue

        action = cmd.get("action")

        # 3. 先更新本地“地图状态”：切图 / 增删点位
        if action == "set_map":
            set_current_map_name(cmd["map_name"])
            print(f"[本地] 已切换当前地图为 {cmd['map_name']} (floor={cmd['floor']})")
        elif action in ("insert_marker_here", "delete_marker"):
            apply_local_marker_side_effects(cmd)

        # 4. 转成 AGV 的 API 文本指令（只做展示）
        api_str = build_robot_api_string(cmd)

        print("\n[解析后的 JSON 指令]")
        print(json.dumps(cmd, ensure_ascii=False, indent=2))

        print("\n[将要发送的 API 文本指令是]")
        if api_str is None:
            print("(ask_user_clarification，不下发具体 API)")
        else:
            print(api_str)

        # 5. 如果是带 duration_s 的 joy_control，额外打印宏指令拆解
        if action == "joy_control" and "duration_s" in cmd:
            print_joy_control_macro_info(cmd, api_str)

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
