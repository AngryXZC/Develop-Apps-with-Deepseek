"""书中个人助理示例的本地语音转文本（原 openai-whisper 的 base）→ faster-whisper。

书中：
  pip install openai-whisper
  model = whisper.load_model("base")
  model.transcribe(file)["text"]

这里用已经装好的 faster-whisper，模型仍是 base，在 CPU 上用 int8。
不调用 OpenAI 的 Whisper 接口。

默认转写第二章 tts_invoke.py 生成的中文 mp3。

运行：
  conda activate ai_code
  python 第二章/tts_invoke.py
  python 第三章/whisper_transcribe.py
"""
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")

AUDIO = Path(__file__).resolve().parent.parent / "第二章" / "tts_hello.mp3"


def transcribe(file):
    segments, _info = model.transcribe(str(file), task="transcribe")
    return "".join(segment.text for segment in segments).strip()


if __name__ == "__main__":
    if not AUDIO.is_file():
        raise FileNotFoundError(f"找不到音频，请先运行 第二章/tts_invoke.py: {AUDIO}")
    print(transcribe(AUDIO))
