"""书中语音转文本示例的国内试用版。

OpenAI 用 whisper-1。这里用阿里开源 SenseVoice Small（FunASR）在本机转写。
输入是同目录下 tts_invoke.py 生成的 tts_hello.mp3。
第一次运行会下载模型。

运行：
  conda activate ai_code
  python 第二章/tts_invoke.py
  python 第二章/stt_invoke.py
"""
from pathlib import Path

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

AUDIO = Path(__file__).resolve().parent / "tts_hello.mp3"

if not AUDIO.is_file():
    raise FileNotFoundError(f"找不到音频，请先运行 tts_invoke.py: {AUDIO}")

model = AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
result = model.generate(
    input=str(AUDIO),
    language="zh",
    use_itn=True,
)
print(rich_transcription_postprocess(result[0]["text"]))
