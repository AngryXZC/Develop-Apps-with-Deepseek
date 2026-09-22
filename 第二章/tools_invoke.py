"""书中 Function Calling / tools 示例（原 gpt-3.5-turbo-0613）→ DeepSeek 改造版。

流程：
1. 定义本地函数 find_product（模拟查库）
2. 用 JSON Schema 描述函数，放进 tools
3. 第一次 create：模型决定调用 find_product，并给出 sql_query
4. 本地执行函数，把结果以 role=tool 写回 messages
5. 第二次 create：模型把结果说成自然语言

密钥：DEEPSEEK_API_CODE_KEY

运行：
  conda activate ai_code
  python 第二章/tools_invoke.py
"""
import json
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-flash"


# ---------- 1. 本地「查库」函数（示例写死结果）----------
def find_product(sql_query: str):
    # 实际项目里这里会执行 SQL；书中用固定结果演示
    print("收到 SQL:", sql_query)
    results = [
        {"name": "pen", "color": "blue", "price": 1.99},
        {"name": "pen", "color": "red", "price": 1.78},
    ]
    return results


# ---------- 2. 函数说明书（表 2-5：name / description / parameters）----------
function_find_product = {
    "name": "find_product",
    "description": "Get a list of products from a SQL query",
    "parameters": {
        "type": "object",
        "properties": {
            "sql_query": {
                "type": "string",
                "description": "A SQL query",
            }
        },
        "required": ["sql_query"],
    },
}


# ---------- 3. 第一次请求：让模型决定是否调工具 ----------
user_question = "I need the top 2 products where the price is less than 2.00"
messages = [{"role": "user", "content": user_question}]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=[{"type": "function", "function": function_find_product}],
)
response_message = response.choices[0].message
messages.append(response_message)

print("--- tool_calls ---")
print(response_message.tool_calls)


# ---------- 4. 执行本地函数，把结果塞回对话 ----------
if not response_message.tool_calls:
    print("模型未发起工具调用，直接回复:")
    print(response_message.content)
else:
    tool_call = response_message.tool_calls[0]
    function_name = tool_call.function.name

    if function_name == "find_product":
        function_args = json.loads(tool_call.function.arguments)
        products = find_product(function_args.get("sql_query"))
    else:
        products = []

    messages.append(
        {
            "role": "tool",
            "content": json.dumps(products),
            "tool_call_id": tool_call.id,
        }
    )

    # ---------- 5. 第二次请求：把工具结果转成自然语言 ----------
    second_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    print("--- 最终回复 ---")
    print(second_response.choices[0].message.content)
    if second_response.usage:
        print("usage:", second_response.usage)
