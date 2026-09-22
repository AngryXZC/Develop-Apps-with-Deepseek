"""书中 JSON 输出示例（原 gpt-3.5-turbo-1106）→ DeepSeek 改造版。

要点：
- response_format={"type": "json_object"} 开启 JSON Output
- prompts 里必须出现 "json"，并最好给期望结构示例（否则可能空转/刷空白）
- 合理设置 max_tokens，避免 JSON 被截断
- 模型：deepseek-flash；密钥：DEEPSEEK_API_CODE_KEY

运行：
  conda activate ai_code
  python 第二章/json_invoke.py
"""
import json
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-flash",
    response_format={"type": "json_object"},
    max_tokens=512,
    messages=[
        {
            "role": "system",
            "content": "Convert the user's query in a JSON object"
        },
        {
            "role": "user",
            "content": "I am looking for blue or red shoes, leather,size 7."
        },
    ],
)

raw = response.choices[0].message.content
print(raw)

# 便于后续程序使用：解析成 dict
if raw and raw.strip():
    print(json.loads(raw))
else:
    print("空内容，可尝试改写 prompt 后重试（DeepSeek JSON 模式偶发空返回）")

if response.usage:
    print("usage:", response.usage)
