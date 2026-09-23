"""书中「法语语音 → 原文转写 / 译成英文」示例的本机版。

OpenAI 用 tts-1 生成 speech_fr.mp3，再用 whisper-1：
- transcriptions：得到法语原文
- translations：译成英文

这里改成：
- edge-tts 法语音色生成 speech_fr.mp3
- faster-whisper：task="transcribe" 原文，task="translate" 英文

第一次运行会下载 Whisper small 模型。

运行：
  conda activate ai_code
  python -m pip install faster-whisper
  python 第二章/speech_translate_invoke.py
"""
import asyncio
import os
from pathlib import Path

# Hugging Face 直连在国内经常卡住；Xet 下载也会停在 0 字节。未单独设置时走镜像并关掉 Xet。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import edge_tts
from faster_whisper import WhisperModel

TEXT = "Les mathématiques sont une science fondamentale."
VOICE = "fr-FR-HenriNeural"
AUDIO = Path(__file__).resolve().parent / "speech_fr.mp3"


async def make_french_audio() -> None:
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(str(AUDIO))


def join_text(model: WhisperModel, task: str) -> tuple[str, str]:
    segments, info = model.transcribe(str(AUDIO), task=task)
    text = "".join(segment.text for segment in segments).strip()
    return text, info.language


asyncio.run(make_french_audio())
print(f"已生成: {AUDIO}")

model = WhisperModel("small", device="cpu", compute_type="int8")
french, language = join_text(model, "transcribe")
english, _ = join_text(model, "translate")

print(f"检测语言: {language}")
print(f"转写: {french}")
print(f"英译: {english}")
