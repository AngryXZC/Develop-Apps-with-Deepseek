"""书中本地图片 base64 视觉示例（原 gpt-4-turbo）→ DeepSeek 改造版。

与 vision_invoke.py（公网 URL）的区别：
- 把本地文件读成 base64，再以 data URL 传给 image_url
- 适合图片不能公开访问、或不想依赖外网直链的情况

注意：
- data URL 的 MIME 要和真实格式一致（本例是 PNG → image/png）
- 模型仍用 deepseek-flash
- 密钥：DEEPSEEK_API_CODE_KEY

运行（在项目根目录）：
  conda activate ai_code
  python 第二章/vision_local_invoke.py
"""
import os
from base64 import b64encode
from pathlib import Path

from openai import OpenAI

# 本文件在 第二章/，图片在项目根 图片/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "图片" / "百度图片_龙.png"


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return b64encode(image_file.read()).decode("utf-8")


client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
    base_url="https://api.deepseek.com",
)

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(f"找不到本地图片: {IMAGE_PATH}")

base64_image = encode_image(IMAGE_PATH)
# PNG 用 image/png；若是 .jpg 则改为 image/jpeg
data_url = f"data:image/png;base64,{base64_image}"

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Give the name of the animal in this image.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        }
    ],
    stream=False,
)

print(response.choices[0].message.content)
if response.usage:
    print("usage:", response.usage)
