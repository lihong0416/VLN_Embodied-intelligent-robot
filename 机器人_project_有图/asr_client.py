# asr_client.py
"""
语音前端模块：
- 从麦克风录音
- 调用本地 Whisper 模型（faster-whisper）进行中文语音识别
- 返回识别出的文本
"""

import os
import tempfile

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from faster_whisper import WhisperModel

# ===== 1. 本地模型根目录 =====
# 这里就是你刚刚 download 脚本里用的那个目录
LOCAL_WHISPER_BASE = r"faster-whisper-small"


def _find_whisper_model_dir(base_dir: str) -> str:
    """
    在 base_dir 下面递归查找包含 model.bin 的目录，
    找到后返回这个目录路径，用于传给 WhisperModel。
    """
    print("[ASR] 正在扫描本地模型目录，查找 model.bin ...")
    for dirpath, dirnames, filenames in os.walk(base_dir):
        if "model.bin" in filenames:
            print("[ASR] 找到 model.bin 所在目录：", dirpath)
            return dirpath
    raise RuntimeError(f"在 {base_dir} 下没有找到 model.bin，请确认下载是否成功。")


# 自动找到真正的模型目录（里面包含 model.bin）
LOCAL_WHISPER_DIR = _find_whisper_model_dir(LOCAL_WHISPER_BASE)
print("[ASR] 使用本地 Whisper 模型目录：", LOCAL_WHISPER_DIR)

# 只用本地文件，完全离线，不访问网络
asr_model = WhisperModel(
    LOCAL_WHISPER_DIR,
    device="cpu",          # 有 GPU 想提速再改 "cuda"
    compute_type="int8",
    local_files_only=True,
)
print("[ASR] Whisper 本地模型加载完成。")


def record_audio(duration: float = 3.0, samplerate: int = 16000) -> np.ndarray:
    """
    从默认麦克风录制一段音频。
    """
    print(f"[ASR] 录音中，请在 {duration} 秒内说话...")
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = audio.squeeze()
    print("[ASR] 录音结束。")
    # 调试信息：看音量大小
    print("[ASR DEBUG] 音频最大绝对值:", float(np.max(np.abs(audio))))
    return audio


def transcribe_audio(audio: np.ndarray, samplerate: int = 16000) -> str:
    """
    使用 Whisper 模型识别一段中文语音。
    """
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_write(f.name, samplerate, audio_int16)
        tmp_path = f.name

    try:
        segments, info = asr_model.transcribe(
            tmp_path,
            language="zh",
            beam_size=5,
            vad_filter=False,
        )
        text = "".join(seg.text for seg in segments).strip()
        return text
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def record_and_transcribe(duration: float = 3.0) -> str:
    """
    一步到位：录音 + 识别，返回文本。
    """
    audio = record_audio(duration=duration)
    text = transcribe_audio(audio)
    return text
