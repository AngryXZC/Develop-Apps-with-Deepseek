"""书中个人助理的 Gradio 界面。

麦克风录音交给 faster-whisper 转成文字，再交给 state_machine.start。
状态机走 DeepSeek deepseek-flash，密钥是环境变量 DEEPSEEK_API_CODE_KEY。
这里不接 Gmail，写邮件只会在终端打印动作。

Gradio 6 的 sources 要写成列表。书中是 sources="microphone"。
live=True 表示录音结束后自动提交，不必再点 Submit。
没有打开 streaming，所以不会在说话过程中一段段调用模型。

运行：
  conda activate ai_code
  python 第三章/assistant_ui.py
启动后终端会打印本地地址，一般是 http://127.0.0.1:7860。
"""
import os
from urllib.request import getproxies

# macOS 的系统代理不会自动带上「不走代理」的地址。
# Gradio 启动时要访问 127.0.0.1 做自检，这个请求若走进代理会得到 503。
# 先记下系统代理，再把本机地址排除出去；只设置 NO_PROXY 时，系统代理有时会失效。
_system_proxies = getproxies()
if _system_proxies.get("http"):
    os.environ.setdefault("HTTP_PROXY", _system_proxies["http"])
    os.environ.setdefault("http_proxy", _system_proxies["http"])
if _system_proxies.get("https"):
    os.environ.setdefault("HTTPS_PROXY", _system_proxies["https"])
    os.environ.setdefault("https_proxy", _system_proxies["https"])
_bypass = [p.strip() for p in os.environ.get("NO_PROXY", "").split(",") if p.strip()]
for _host in ("localhost", "127.0.0.1", "::1"):
    if _host not in _bypass:
        _bypass.append(_host)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(_bypass)

import gradio as gr

from state_machine import start
from whisper_transcribe import transcribe


def start_chat(file):
    # 清空录音时 Gradio 传入 None，这时不调用转写和模型。
    if not file:
        return ""
    text = transcribe(file)
    # start 在 MORE 之后会接上一段录音。回答或发信结束后，下一次录音从 START 重来。
    answer = start(text)
    # 界面上带上转写结果，方便核对麦克风实际听成了什么。
    return f"转写：{text}\n\n{answer}"


demo = gr.Interface(
    fn=start_chat,
    live=True,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs="text",
)


if __name__ == "__main__":
    demo.launch()
