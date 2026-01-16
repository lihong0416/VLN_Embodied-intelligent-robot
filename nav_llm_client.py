# nav_llm_client.py
import json
import requests
from pathlib import Path

from config_markers import MODEL_NAME, OLLAMA_URL

NAV_SYSTEM_PROMPT = Path("system_prompt_nav.txt").read_text(encoding="utf-8")


def build_nav_user_prompt(user_command: str, scene_state: dict, history_steps: list) -> str:
    """
    history_steps: 之前已经执行过的动作列表，用于提醒大模型当前进度。
    """
    scene_json = json.dumps(scene_state, ensure_ascii=False, indent=2)
    history_json = json.dumps(history_steps, ensure_ascii=False, indent=2)

    parts = []
    parts.append("下面是当前一帧 RGB-D 场景的结构化描述 JSON：")
    parts.append(scene_json)
    parts.append("")
    parts.append("下面是到目前为止已经执行过的动作列表（可能为空数组）：")
    parts.append(history_json)
    parts.append("")
    parts.append("请根据上面的信息，规划下一小步动作，并用 JSON 格式回答。")
    parts.append(f"用户指令：{user_command}")
    parts.append("助手：")

    return "\n".join(parts)


def call_qwen_nav(user_command: str, scene_state: dict, history_steps: list) -> dict:
    """
    调用本地 Qwen，做一次“导航下一步决策”，返回 JSON dict。
    """
    user_prompt = build_nav_user_prompt(user_command, scene_state, history_steps)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": NAV_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    json_str = data["message"]["content"]
    return json.loads(json_str)
