"""书中「看图识物」示例（原 gpt-4-turbo）→ DeepSeek 改造版。

要点：
- 模型用 deepseek-flash（当前支持 Vision；deepseek-v4-pro 不支持）
- user 的 content 是「内容块数组」，不是普通字符串：
  [{"type":"text",...}, {"type":"image_url",...}]
- image_url 必须是图片文件直链（.png/.jpg 等，Content-Type 为 image/*）
  不能用百科页、百度图片详情页等 HTML 链接，否则会报 Failed to download image
- 密钥：环境变量 DEEPSEEK_API_CODE_KEY

运行（在项目根目录）：
  conda activate ai_code
  python 第二章/vision_invoke.py
"""
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
    base_url="https://api.deepseek.com",
)

# 图片直链示例（百度 BOS 上的 PNG）。网页 URL 不可用。
url = (
    "https://miaobi-lite.bj.bcebos.com/miaobi/5mao/b%27LV8xNzM2NDU0MzUyLjQ3NzgxNjY%3D%27/0.png"
)

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Give the name of the animal in the image.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                        # low：降采样更便宜；high/original/auto：更清晰
                        "detail": "auto",
                    },
                },
            ],
        }
    ],
    stream=False,
)

print(response.choices[0].message.content)
if response.usage:
    print("usage:", response.usage)
