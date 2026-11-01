"""书中个人助理的状态机（原 gpt-3.5-turbo）→ DeepSeek。

从 START 出发，模型每次只走一步：分类意图、决定能否回答、追问、回答，或执行写邮件。
聊天接口无状态，位置靠不断追加并重发的 messages 记住。
停在 MORE 时留下这段记录。下一句用户的话接在后面，继续补全邮件或问题。
其他结束方式会清空记录，下一句从 START 重新开始。
模型：deepseek-flash；密钥：环境变量 DEEPSEEK_API_CODE_KEY。

动作若是「ACTION_WRITE_EMAIL | subject:…」，不能整句去对字典的键，
只取竖线前的动作名。这里不接 Gmail，只打印动作已经执行。

运行：
  conda activate ai_code
  python 第三章/state_machine.py
"""
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-flash"

# 键是状态名，值是进入该状态后发给模型的指令。
# 指令要求模型只回答下一个状态名（或最终的自然语言）。
# 没有单独的转移表：下一个状态就是模型这一轮的回答。
prompts = {
    "START": "Classify the intent of the next input. Is it: WRITE_EMAIL, QUESTION, OTHER? Only answer one word.",
    "QUESTION": "If you can answer the question: ANSWER, if you need more information: MORE, if you cannot answer: OTHER. Only answer one word.",
    "ANSWER": "Now answer the question",
    "MORE": "Now ask for more information",
    "OTHER": "Now tell me you cannot answer the question or do the action",
    "WRITE_EMAIL": (
        'If the subject or recipient or message is missing, answer "MORE". '
        "Else if you have all the information, answer "
        '"ACTION_WRITE_EMAIL | subject:subject, recipient:recipient, message:message".'
    ),
}

# 动作不是普通状态。模型答出动作名后，代码先执行，再把这里的句子发给模型，让它用自然语言收尾。
actions = {
    "ACTION_WRITE_EMAIL": "The mail has been sent. Now tell me the action is done in natural language."
}


def generate_answer(messages):
    # 接口不保存历史。messages 必须带上从 START 到当前的全部内容。
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content.strip()


def do_action(action):
    # 书中这里会去调 Gmail。本示例只打印，表示动作已被识别。
    print("正在执行: " + action)
    return "I did the action " + action


def state_of(answer):
    """把模型原文收成 prompts 或 actions 里的键。对不上就原样返回，当作结束语。"""
    text = answer.strip().strip(".")
    # 「QUESTION」这类纯状态名。
    if text in prompts:
        return text
    # 「ACTION_WRITE_EMAIL | subject:…」只认竖线前的动作名。
    action = text.split("|", 1)[0].strip()
    if action in actions:
        return action
    # 模型偶尔写成「QUESTION.」或「ANSWER the question」，取第一个词再试一次。
    word = text.split()[0].strip(".,") if text else ""
    if word in prompts:
        return word
    return text


# 跨多次 start() 保留的对话。接口本身不记这些内容。
# more_from 是进入 MORE 之前的状态，例如 WRITE_EMAIL 或 QUESTION。
# 下一句用户的话要回到这个状态，而不是丢掉前面已经说过的主题或问题。
messages = []
more_from = ""


def discussion(messages, last_step, depth=0):
    global more_from
    # 防止模型一直返回状态名，递归停不下来。
    if depth > 8:
        messages.clear()
        more_from = ""
        return "状态来回次数过多，已停止。"

    # 这一轮的回答就是下一个状态。例如当前指令是 START 的分类提示，模型应只回 QUESTION。
    answer = generate_answer(messages)
    step = state_of(answer)
    if step in prompts or step in actions:
        print(f"-> {answer}")

    if step in prompts:
        # 从 WRITE_EMAIL 或 QUESTION 进入 MORE 时，记下是谁在追问。
        if step == "MORE" and last_step in prompts and last_step != "MORE":
            more_from = last_step
        # 记下刚到达的状态，并附上该状态的指令。下一轮模型读到这条指令，再回答再下一个状态。
        messages.append({"role": "assistant", "content": step})
        messages.append({"role": "user", "content": prompts[step]})
        return discussion(messages, step, depth + 1)

    if step in actions:
        do_action(answer)
        # 动作执行完后，用 actions 里的句子再问一次，让模型说出「已经发出」。
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": actions[step]})
        return discussion(messages, step, depth + 1)

    # 回答既不是状态也不是动作，就是说给用户的那句话。
    # 上一步是 MORE：留下追问和之前的记录，等用户补充。
    # 否则这段对话结束，下一句从 START 重新开始。
    if last_step == "MORE":
        messages.append({"role": "assistant", "content": answer})
    else:
        messages.clear()
        more_from = ""
    return answer


def start(user_input):
    global messages, more_from
    if messages:
        # 上一轮停在追问。先放回那个状态的指令，再放上用户刚补充的话。
        prompt_key = more_from if more_from in prompts else "START"
        print(f"继续 {prompt_key}")
        messages.append({"role": "user", "content": prompts[prompt_key]})
        messages.append({"role": "user", "content": user_input})
        return discussion(messages, prompt_key if prompt_key != "START" else "")

    # 第一条是 START 的分类指令，第二条才是用户原话。模型据此回答 WRITE_EMAIL、QUESTION 或 OTHER。
    messages = [
        {"role": "user", "content": prompts["START"]},
        {"role": "user", "content": user_input},
    ]
    more_from = ""
    return discussion(messages, "")


if __name__ == "__main__":
    print("用户: What is the capital of France?")
    print(start("What is the capital of France?"))
    print("---")
    # 图 3-15：先只给主题，助理追问；下一句补上正文和收件人，再执行发信。
    for text in (
        "The subject of the mail is Hello",
        "The body is 'Meet me on Thursday at 4 p.m.' and the recipient is john@email.com",
    ):
        print(f"用户: {text}")
        print(start(text))
        print("---")
