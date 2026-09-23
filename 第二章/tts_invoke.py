"""书中文本转语音示例的国内试用版。

DeepSeek 没有 TTS。这里用 edge-tts 调用微软 Edge 在线语音，
生成中文 mp3。适合个人练习，不适合做成正式产品大量调用。

依赖已装 edge-tts。运行：
  conda activate ai_code
  python 第二章/tts_invoke.py

输出：第二章/tts_hello.mp3
"""
import asyncio
from pathlib import Path

import edge_tts

TEXT = "莫愁前路无知己,天下谁人不识君"
VOICE = "zh-CN-XiaoxiaoNeural"
OUT = Path(__file__).resolve().parent / "tts_hello.mp3"


async def main() -> None:
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(str(OUT))
    print(f"已生成: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
