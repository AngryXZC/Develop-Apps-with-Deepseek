"""书中第 3 章「抽视频帧再摘要」示例（原 gpt-4o）→ DeepSeek。

OpenCV 打开本地 mp4，把帧编成 JPEG 的 base64。
书中每 50 帧取 1 帧，并用 {"image": ..., "resize": 768}。
DeepSeek 视觉接口不认这种字段，改成和第二章一样的 image_url data URL。
模型：deepseek-flash；密钥：环境变量 DEEPSEEK_API_CODE_KEY。

视频：第三章/视频/sample_640x360.mp4

运行：
  conda activate ai_code
  python -m pip install opencv-python
  python 第三章/video_summary_invoke.py
"""
import base64
import os
from pathlib import Path

import cv2
from openai import OpenAI

VIDEO = Path(__file__).resolve().parent / "视频" / "test.mp4"
STEP = 50

if not VIDEO.is_file():
    raise FileNotFoundError(f"找不到视频: {VIDEO}")

video = cv2.VideoCapture(str(VIDEO))
base64_frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64_frames.append(buffer.tobytes())
video.release()

sampled = base64_frames[0::STEP]
print(f"总帧数: {len(base64_frames)}，送去摘要: {len(sampled)}")

images = [
    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(frame).decode("utf-8"),
        },
    }
    for frame in sampled
]

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "这些是一段视频里抽出的画面。请用中文写两句话摘要。",
                },
                *images,
            ],
        }
    ],
)

print(response.choices[0].message.content)
if response.usage:
    print("usage:", response.usage)
