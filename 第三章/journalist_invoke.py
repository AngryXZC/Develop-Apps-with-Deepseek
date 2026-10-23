"""书中第 3 章「记者助手」提示词示例（原 gpt-3.5-turbo）→ DeepSeek。

给模型一个角色，并把任务写具体：FACTS、TONE、LENGTH、STYLE。
角色说明和任务都放在同一条 user 消息里，对应书中的写法。
模型：deepseek-flash；密钥：环境变量 DEEPSEEK_API_CODE_KEY。

运行：
  conda activate ai_code
  python 第三章/journalist_invoke.py
"""
import os
from typing import List

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-flash"

prompt_role = """You are an assistant for journalists. 
Your task is to write articles, based on the FACTS that are given to you. You should respect the instructions: the TONE, the LENGTH, and the STYLE"""


def ask_model(messages):
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


def assist_journalist(facts: List[str], tone: str, length_words: int, style: str):
    facts = ", ".join(facts)
    prompt = f"""{prompt_role}
FACTS: {facts}
TONE: {tone}
LENGTH: {length_words} words
STYLE: {style}"""
    return ask_model([{"role": "user", "content": prompt}])


if __name__ == "__main__":
    print(
        assist_journalist(
            ["The sky is blue", "The grass is green"],
            "informal",
            100,
            "blogpost",
        )
    )
    print("---")
    print(
        assist_journalist(
            facts=[
                "A book on ChatGPT has been published last week",
                "The title is Developing Apps with GPT-4 and ChatGPT",
                "The publisher is O'Reilly.",
            ],
            tone="excited",
            length_words=50,
            style="news flash",
        )
    )
