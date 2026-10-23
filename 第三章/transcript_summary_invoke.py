"""书中第 3 章「转写文稿摘要」示例（原 gpt-3.5-turbo）→ DeepSeek。

书中从 files/transcript.txt 读入视频转写，再放进一条 user 消息。
这里改读同目录的 transcript.txt，不依赖当前工作目录。
模型：deepseek-flash；密钥：环境变量 DEEPSEEK_API_CODE_KEY。

运行：
  conda activate ai_code
  python 第三章/transcript_summary_invoke.py
"""
import os
from pathlib import Path

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
    base_url="https://api.deepseek.com",
)

TRANSCRIPT = Path(__file__).resolve().parent / "transcript.txt"

with open(TRANSCRIPT, "r", encoding="utf-8") as f:
    transcript = f.read()

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "user",
            "content": f"Summarize the following video transcript:\n{transcript}",
        }
    ],
)

print(response.choices[0].message.content)
