# prompts_nlu.py
"""
负责三件事：
1. 从本地 system_prompt.txt 加载 System Prompt。
2. 从 examples_nlu.json 加载若干“用户指令 -> JSON 输出”的示例。
3. 构造发给大模型的 user prompt（地图信息 + 示例 + 当前用户指令）。
"""

import json
from pathlib import Path
from config_markers import (
    get_current_markers,
    get_current_map_name,
    get_maps_brief,
)

# ===== 1. 从文件加载 System Prompt =====

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt.txt"

try:
    SYSTEM_PROMPT: str = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
except FileNotFoundError as e:
    raise RuntimeError(
        f"找不到 system_prompt.txt，期望路径：{SYSTEM_PROMPT_PATH}。\n"
        f"请确认文件存在，或者修改 prompts_nlu.py 里的 SYSTEM_PROMPT_PATH。"
    ) from e


# ===== 2. 从 JSON 加载 few-shot 示例 =====

EXAMPLES_PATH = BASE_DIR / "package.json"

def _load_examples():
    if not EXAMPLES_PATH.exists():
        print(f"[prompts_nlu] 警告：找不到 {EXAMPLES_PATH}，将不使用示例 few-shot。")
        return []

    try:
        data = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
        # 期望是 list[{"user": str, "assistant": dict}]
        if not isinstance(data, list):
            raise ValueError("examples_nlu.json 顶层必须是数组")
        for i, item in enumerate(data):
            if not isinstance(item, dict) or "user" not in item or "assistant" not in item:
                raise ValueError(f"第 {i} 个示例缺少 user/assistant 字段")
        print(f"[prompts_nlu] 已加载 {len(data)} 条示例 from {EXAMPLES_PATH}")
        return data
    except Exception as e:
        print(f"[prompts_nlu] 读取 {EXAMPLES_PATH} 失败，将不使用示例，错误: {e}")
        return []

EXAMPLES = _load_examples()


# ===== 3. 构造 user prompt =====

def build_user_prompt(user_text: str) -> str:
    """
    构造发给大模型的 user prompt，内容包括：
    - 当前有哪些地图（map_name -> desc, floor）
    - 当前正在使用哪张地图
    - 当前地图有哪些 marker_id / aliases
    - examples_nlu.json 里的示例（用户指令 + 对应 JSON）
    - 当前用户输入的自然语言指令
    """
    current_map_name = get_current_map_name()
    markers = get_current_markers()
    maps_brief = get_maps_brief()

    marker_json = json.dumps(markers, ensure_ascii=False, indent=2)
    maps_json = json.dumps(maps_brief, ensure_ascii=False, indent=2)

    # 先把地图信息写进去
    parts = [
        "当前可用的地图（map_name -> 地图信息）：",
        maps_json,
        "",
        f'当前使用的地图："{current_map_name}"',
        "",
        "当前地图的点位列表（只允许使用下面这些 marker_id）：",
        marker_json,
        "",
    ]

    # 再把 few-shot 示例写进去
    if EXAMPLES:
        parts.append("下面是一些示例（你要模仿这种输入输出关系，只输出 JSON）：\n")
        for ex in EXAMPLES:
            u = ex["user"]
            # assistant 本来就是一个 dict，这里直接转成 JSON 字符串给大模型看
            a_json = json.dumps(ex["assistant"], ensure_ascii=False)
            parts.append(f"用户：{u}\n助手：\n{a_json}\n")
    else:
        parts.append("当前没有示例 few-shot（examples_nlu.json 未配置或加载失败）。\n")

    # 最后拼接当前用户指令
    parts.append("现在开始解析新的用户指令。")
    parts.append(f"用户：{user_text}")
    parts.append("助手：")

    # 用两个换行拼起来，保持结构清晰
    prompt = "\n".join(parts)
    return prompt
