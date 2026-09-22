# import os
# from openai import OpenAI

## 发送一条消息
# client = OpenAI(
#     api_key=os.environ.get('DEEPSEEK_API_CODE_KEY'),
#     base_url="https://api.deepseek.com")

# response = client.chat.completions.create(
#     model="deepseek-flash",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False,
#     reasoning_effort="high",
#     extra_body={"thinking": {"type": "enabled"}}
# )

# print(response.choices[0].message.content)

## 发送多条消息
# import os
# from openai import OpenAI

# client = OpenAI(
#     api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
#     base_url="https://api.deepseek.com",
# )

# response = client.chat.completions.create(
#     model="deepseek-flash",
#     messages=[
#         {"role": "system", "content": "You are a helpful teacher."},
#         {
#             "role": "user",
#             "content": "Are there other measures than time complexity for an algorithm?",
#         },
#         {
#             "role": "assistant",
#             "content": "Yes, there are other measures besides time complexity for an algorithm, such as space complexity.",
#         },
#         {"role": "user", "content": "What is it?"},
#     ],
#     stream=False,
# )

# print(response.choices[0].message.content)

## 计算词元

import tiktoken

def count_message_tokens(messages, encoding_name="cl100k_base"):
    """粗算 chat messages 的输入词元数（对 DeepSeek 仅为近似）。"""
    enc = tiktoken.get_encoding(encoding_name)
    tokens_per_message = 3
    num = 0
    for message in messages:
        num += tokens_per_message
        for value in message.values():
            num += len(enc.encode(str(value)))
    num += 3  # 回复起始开销（OpenAI cookbook 经验值）
    return num

# 发送多条消息
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
    base_url="https://api.deepseek.com",
)

messages=[
        {"role": "system", "content": "You are a helpful teacher."},
        {
            "role": "user",
            "content": "Are there other measures than time complexity for an algorithm?",
        },
        {
            "role": "assistant",
            "content": "Yes, there are other measures besides time complexity for an algorithm, such as space complexity.",
        },
        {"role": "user", "content": "What is it?"},
    ]
print("约输入词元数:", count_message_tokens(messages))
response = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages,
    stream=False,
)
print(response.choices[0].message.content)
print("---------------完整返回结果-----------------")
print(response.model_dump_json(indent=2))  # 格式化 JSON
# 可选：看接口返回的真实用量（DeepSeek 若返回则更准）
usage = getattr(response, "usage", None)
if usage:
    print("---------------API 用量-----------------")
    print("API usage:", usage)
