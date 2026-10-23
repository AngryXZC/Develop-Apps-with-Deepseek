"""书中第 3 章响应服务（原 gpt-3.5-turbo）→ DeepSeek。

只根据给定事实回答问题，事实来自 DataService.search_redis 的文本列表。
模型：deepseek-flash；密钥：环境变量 DEEPSEEK_API_CODE_KEY。

运行：
  conda activate ai_code
  python 第三章/response_service.py
"""
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-flash"


class ResponseService:
    def __init__(self):
        pass

    def generate_response(self, facts, user_question):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Based on the FACTS, answer the QUESTION. "
                        f"QUESTION: {user_question}. FACTS: {facts}"
                    ),
                }
            ],
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    question = "amiibo 是什么？"
    facts = [
        "One of Link's Runes allows you to scan amiibo figures. "
        "By scanning amiibo figures related to The Legend of Zelda, "
        "you can receive useful items."
    ]
    answer = ResponseService().generate_response(facts, question)
    print(f"问题: {question}")
    print(answer)
