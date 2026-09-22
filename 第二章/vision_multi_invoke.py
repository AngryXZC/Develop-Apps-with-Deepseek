"""书中多图像对比示例（原 gpt-4-turbo）→ DeepSeek 改造版。

在同一条 user.message.content 数组里放：
  1 个 text + 多个 image_url（本地图先 base64 成 data URL）

本例两张图（相对项目根）：
  图片/百度图片_龙.png
  图片/百度图片_青龙.png

运行：
  conda activate ai_code
  python 第二章/vision_multi_invoke.py
"""
import os
from base64 import b64encode
from pathlib import Path

from openai import OpenAI

# 本文件在 第二章/，图片在项目根 图片/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = PROJECT_ROOT / "图片"
IMAGE_1 = IMG_DIR / "百度图片_龙.png"
IMAGE_2 = IMG_DIR / "百度图片_青龙.png"


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return b64encode(image_file.read()).decode("utf-8")


def to_data_url(image_path: Path) -> str:
    """按后缀选择 MIME；本例均为 PNG。"""
    suffix = image_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    return f"data:{mime};base64,{encode_image(image_path)}"


client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
    base_url="https://api.deepseek.com",
)

for path in (IMAGE_1, IMAGE_2):
    if not path.is_file():
        raise FileNotFoundError(f"找不到本地图片: {path}")

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the differences between these two images?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": to_data_url(IMAGE_1)},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": to_data_url(IMAGE_2)},
                },
            ],
        }
    ],
    stream=False,
)

print(response.choices[0].message.content)
if response.usage:
    print("usage:", response.usage)
