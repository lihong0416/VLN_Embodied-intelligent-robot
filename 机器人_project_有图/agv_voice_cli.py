# agv_voice_cli.py
"""
语音版入口（本地 Demo）：
- 按回车开始录音，录 3 秒
- ASR 识别成中文文本
- 丢给 Qwen（Ollama）→ JSON 指令
- 根据中文指令纠正左/右转方向（normalize_turn_from_text）
- 校验 JSON
- 更新本地地图状态（切图 / 新增点位 / 删除点位，直接写 maps.json）
- 打印 AGV API 字符串（不真正发给机器人）
- 对 joy_control + duration_s 打印宏指令拆解信息
"""

import json

from asr_client import record_and_transcribe
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
    这里相当于默认 AGV 一定会执行成功。
    以后你接上真实机器人时，可以改成：收到底盘成功响应再调用这些函数。
    """
    action = cmd.get("action")

    # 在当前位置新建点位
    if action == "insert_marker_here":
        name = cmd["marker"]

        if marker_exists_in_current_map(name):
            print(f"[本地] 当前地图已存在点位 {name}，不重复写入 maps.json")
            return

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

def main():
    print("=== 语音控制 Demo：麦克风语音 → 文本 → Qwen → JSON → AGV API 串 ===")
    print("（本地演示版，不连机器人，只更新 maps.json）")
    print("操作说明：")
    print("  - 回车：开始录音（默认 3 秒）")
    print("  - 输入 q 回车：退出程序\n")

    while True:
        current_map = get_current_map_name()
        print(f"\n[当前地图] {current_map}")
        key = input("按回车开始录音，输入 q 回车退出：").strip()
        if key.lower() in {"q", "quit", "exit"}:
            print("退出。")
            break

        try:
            # 1. 录音 + 语音识别
            text = record_and_transcribe(duration=3.0)
            if not text:
                print("[ASR] 没有识别到有效语音，请再试一次。")
                continue

            print(f"[ASR 识别结果] {text}")

            # 2. 调本地 Qwen 做语义解析，得到 JSON
            cmd = call_qwen_local_and_get_json(text)

            # 2.5 根据原始中文，修正左/右转的角速度符号
            cmd = normalize_turn_from_text(cmd, text)

            # 3. 严格校验与补全
            cmd = validate_command(cmd)

            action = cmd.get("action")

            # 4. 先更新本地“地图状态”
            if action == "set_map":
                set_current_map_name(cmd["map_name"])
                print(f"[地图切换] 当前地图已切换为 {cmd['map_name']} (floor={cmd['floor']})")
            elif action in ("insert_marker_here", "delete_marker"):
                apply_local_marker_side_effects(cmd)

            # 5. 打印解析后的 JSON
            print("\n[解析后的 JSON 指令]")
            print(json.dumps(cmd, ensure_ascii=False, indent=2))

            # 6. 映射成 AGV API 字符串（当前只是打印，不真正发给机器人）
            api_str = build_robot_api_string(cmd)
            if api_str is None and cmd["action"] == "ask_user_clarification":
                print("\n[无法识别，不会下发指令]")
                print("原因：", cmd.get("reason", "模型未给出原因"))
            else:
                print("\n[将要发送的 API 文本指令是]")
                print(api_str)

            # 7. joy_control + duration_s 打印宏拆解
            if action == "joy_control" and "duration_s" in cmd:
                print_joy_control_macro_info(cmd, api_str)

            print("\n" + "=" * 60 + "\n")

        except Exception as e:
            print(f"\n[出错了] {e}\n")


if __name__ == "__main__":
    main()
