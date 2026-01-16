# main.py
"""
文本总控入口：
- 从终端读取一条中文指令
- 调本地 Qwen（Ollama）解析 -> JSON 指令
- 校验 JSON -> 映射成 AGV API 字符串
- 通过 TCP 发给机器人，打印机器人返回结果
- 对插入 / 删除点位动作，在机器人执行成功后，同步更新 maps.json
- 对 joy_control + duration_s，按 0.5s 为一个 step 连续发送多次，实现“走 N 米 / 转 N 度”的宏指令
"""

import socket
import json
import time

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
STEP_DURATION = 0.5  # 单步持续时间（秒）
MAX_JOY_DURATION = 5.0  # 单次宏指令最长持续时间（秒），防止一条指令跑太久


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


# ===== TCP 通信封装 =====

def send_command_and_receive_response(command: str, host: str, port: int):
    """
    通过 TCP 向机器人发送一条命令字符串，并接收 JSON 响应。
    返回 dict 或 None。
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))

            # 若底盘要求以换行结尾，可改为 command + "\\n"
            sock.sendall(command.encode("utf-8"))

            # 简单收一次（实际项目可按协议循环接收）
            resp = sock.recv(4096).decode("utf-8").strip()
            if not resp:
                print("[TCP] 收到空响应")
                return None

            try:
                data = json.loads(resp)
            except json.JSONDecodeError:
                print("[TCP] 收到的不是合法 JSON：", resp)
                return None

            return data

    except Exception as e:
        print(f"[TCP] 通信出错: {e}")
        return None


# ===== 点位增删：在“机器人成功执行后”同步更新 maps.json =====

def apply_marker_side_effects_after_robot(cmd: dict, response: dict | None) -> None:
    """
    在机器人确认执行成功之后，同步更新本地 maps.json。
    - 默认认为：如果响应中有 success 字段，则以它为准；
      否则视为成功（你可以按实际返回格式改这里）。
    """
    if response is None:
        return

    ok = True
    if isinstance(response, dict) and "success" in response:
        ok = bool(response["success"])

    if not ok:
        print("[本地] 机器人执行失败，不更新 maps.json")
        return

    action = cmd.get("action")

    if action == "insert_marker_here":
        name = cmd["marker"]
        if marker_exists_in_current_map(name):
            print(f"[本地] maps.json 已存在点位 {name}，不重复写入")
        else:
            add_marker_to_current_map(name)
            print(f"[本地] 已在 maps.json 的地图 {get_current_map_name()} 中新增点位 {name}")

    elif action == "delete_marker":
        name = cmd["marker"]
        removed = remove_marker_from_current_map(name)
        if removed:
            print(f"[本地] 已在 maps.json 的地图 {get_current_map_name()} 中删除点位 {name}")
        else:
            print(f"[本地] maps.json 中没有点位 {name}，无需删除")


# ===== joy_control 宏指令执行 =====

def exec_joy_control_macro(cmd: dict, host: str, port: int) -> dict | None:
    """
    执行带 duration_s 的 joy_control：
    - duration_s 表示期望持续时间（秒）
    - 按 STEP_DURATION 拆成多个小步，多次发送 joy_control API，实现“走 N 米 / 转 N 度”等
    - 返回最后一次发送的响应（dict 或 None）
    """
    lin = float(cmd.get("linear_velocity", 0.0))
    ang = float(cmd.get("angular_velocity", 0.0))
    duration = float(cmd.get("duration_s", STEP_DURATION))

    # 合理性限制
    if duration <= 0:
        duration = STEP_DURATION
    if duration > MAX_JOY_DURATION:
        print(f"[JOY] duration_s={duration:.2f}s 超过限制，截断为 {MAX_JOY_DURATION:.2f}s")
        duration = MAX_JOY_DURATION

    steps = max(1, int(round(duration / STEP_DURATION)))

    # 用 command_logic 里已有逻辑，统一构造 API 串
    api_str = build_robot_api_string(cmd)

    print("\n[joy_control 宏指令执行]")
    print(f"  · 线速度 linear_velocity = {lin:.3f} m/s")
    print(f"  · 角速度 angular_velocity = {ang:.3f} rad/s")
    print(f"  · 总时长 duration_s ≈ {duration:.2f} s")
    print(f"  · 单步时长 STEP_DURATION = {STEP_DURATION:.2f} s")
    print(f"  · 预计发送 {steps} 次同样的 API：")
    print(f"    {api_str}")

    last_response: dict | None = None
    for i in range(steps):
        print(f"[JOY] Step {i + 1}/{steps} -> {api_str}")
        last_response = send_command_and_receive_response(api_str, host, port)
        # 如果你觉得每步之间不需要 sleep，可以删掉这句；
        # 正常情况下保留 sleep，让发送频率和文档中的 0.5s 对齐。
        time.sleep(STEP_DURATION)

    return last_response


# ===== 主循环 =====

def main():
    # TODO: 按你的机器人实际 IP 和端口修改
    HOST = "192.168.10.10"
    PORT = 31001

    print("=== 机器人文本控制 Demo（main.py）===")
    print("中文指令 → Qwen → JSON → API 字符串 → TCP 下发 → 同步更新 maps.json / joy_control 宏指令")
    print("操作说明：")
    print("  - 输入 q 回车：退出程序\n")

    while True:
        current_map = get_current_map_name()
        print(f"\n[当前地图] {current_map}")
        user_text = input("请输入一条指令（q 退出）：").strip()

        if user_text.lower() in {"q", "quit", "exit"}:
            print("退出。")
            break
        if not user_text:
            continue

        try:
            # 1. 调本地 Qwen 解析自然语言 -> JSON 指令
            cmd = call_qwen_local_and_get_json(user_text)

            # 1.5 根据原始中文，修正左/右转的角速度符号（确保左>0，右<0，且原地转）
            cmd = normalize_turn_from_text(cmd, user_text)

            # 2. 严格校验与补全
            cmd = validate_command(cmd)

            action = cmd.get("action")

            # 3. 如果是切换地图，先更新当前地图（只改内存）
            if action == "set_map":
                set_current_map_name(cmd["map_name"])
                print(f"[地图切换] 当前地图已切换为 {cmd['map_name']} (floor={cmd['floor']})")

            # 4. 打印解析后的 JSON
            print("\n[解析后的 JSON 指令]")
            print(json.dumps(cmd, ensure_ascii=False, indent=2))

            # 5. ask_user_clarification：仅提示原因，不下发指令
            if action == "ask_user_clarification":
                print("\n[无法安全执行，不会下发指令]")
                print("原因：", cmd.get("reason", "模型未给出原因"))
                print("\n" + "=" * 60 + "\n")
                continue

            # 6. joy_control：走宏指令逻辑（支持 duration_s）
            if action == "joy_control":
                # 这里会内部调用 build_robot_api_string
                response = exec_joy_control_macro(cmd, HOST, PORT)

                if response is not None:
                    print("\n[机器人返回的最后一次响应 JSON]")
                    print(json.dumps(response, ensure_ascii=False, indent=4))
                else:
                    print("\n[机器人响应无效或解析失败]")

                # joy_control 不涉及点位增删，一般不需要更新 maps.json
                print("\n" + "=" * 60 + "\n")
                continue

            # 7. 其它动作：一个 action 对应一个 API
            api_str = build_robot_api_string(cmd)

            print("\n[将要发送给机器人的 API 文本指令是]")
            print(api_str)

            response = send_command_and_receive_response(api_str, HOST, PORT)
            if response is not None:
                print("\n[机器人返回的响应 JSON]")
                print(json.dumps(response, ensure_ascii=False, indent=4))
            else:
                print("\n[机器人响应无效或解析失败]")

            # 8. 若是插入 / 删除点位，且机器人执行成功，则同步更新 maps.json
            apply_marker_side_effects_after_robot(cmd, response)

            print("\n" + "=" * 60 + "\n")

        except Exception as e:
            print(f"\n[出错了] {e}\n")


if __name__ == "__main__":
    main()
