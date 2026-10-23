"""书中第 3 章意图识别（原 gpt-3.5-turbo）→ DeepSeek。

从用户问题里抽出关键词，不回答问题本身。
后面把这些关键词交给 data_service.DataService.search_redis。
模型：deepseek-flash；密钥：环境变量 DEEPSEEK_API_CODE_KEY。

书中用 response["choices"][0]["message"]["content"]。
当前 SDK 返回的是对象，要写成 response.choices[0].message.content。

运行：
  conda activate ai_code
  python 第三章/intent_service.py
"""
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-flash"


class IntentService:
    def __init__(self):
        pass

    def get_intent(self, user_question: str):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract the keywords from the following question: "
                        f"{user_question}. "
                        "Do not answer anything else, only the keywords."
                    ),
                }
            ],
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    question = "amiibo 是什么？"
    keywords = IntentService().get_intent(question)
    print(f"问题: {question}")
    print(f"关键词: {keywords}")
