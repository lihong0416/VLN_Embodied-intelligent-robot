# model_client.py
"""
负责：通过 HTTP 调用本地 Ollama + Qwen 模型，返回 JSON dict。
如果以后换成别的推理引擎，只改这个文件。
"""

import json
import requests
from config_markers import MODEL_NAME, OLLAMA_URL
from prompts_nlu import SYSTEM_PROMPT, build_user_prompt


def call_qwen_local_and_get_json(user_text: str) -> dict:
    """
    调用本地 Ollama Qwen 模型，要求输出 JSON。
    """
    user_content = build_user_prompt(user_text)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,      # 不要流式，直接一次性返回
        "format": "json",     # 要求返回 JSON（字符串形式）
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama 调用失败: {resp.status_code}, {resp.text}")

    data = resp.json()
    # data["message"]["content"] 就是模型输出的 JSON 字符串
    json_str = data["message"]["content"]
    cmd = json.loads(json_str)
    return cmd
