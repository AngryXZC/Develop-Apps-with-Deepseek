# ai_code_project

《大模型应用开发极简入门》的学习笔记与代码整理。

按书中章节练习 OpenAI 兼容 API（当前用 DeepSeek），边读边写、边跑边记。

每日小结写在 Notion：[第二章深入了解 OpenAI API](https://app.notion.com/p/3e42386ee733810ab729f2477c342af2)（父页面「大模型应用极简入门」）。仓库里不再放 `笔记/`。

## 环境

- Conda 环境：`ai_code`
- Python：3.11（环境内）
- 依赖：`openai`（OpenAI Python SDK，兼容 DeepSeek）

解释器路径（Cursor 已配置）：

`/opt/homebrew/Caskroom/miniconda/base/envs/ai_code/bin/python`

```bash
conda activate ai_code
# 在项目根目录运行第二章示例
python 第二章/api_invoke.py
python 第二章/vision_invoke.py
python 第二章/vision_local_invoke.py
python 第二章/vision_multi_invoke.py
python 第二章/json_invoke.py
python 第二章/tools_invoke.py
python 第二章/embedding_local.py
python 第二章/tts_invoke.py
python 第二章/stt_invoke.py
python 第二章/speech_translate_invoke.py
```

不要用 Homebrew 的 `pip3` 往系统 Python 装包。应在已激活的 `ai_code` 里：

```bash
python -m pip install openai tiktoken sentence-transformers funasr kaldi-native-fbank edge-tts faster-whisper
```

### 目录结构

```text
ai_code_project/
├── README.md
├── 图片/                 # 本地视觉示例用图（项目根）
│   ├── 百度图片_龙.png
│   └── 百度图片_青龙.png
└── 第二章/               # 第 2 章代码
    ├── api_invoke.py
    ├── vision_invoke.py
    ├── vision_local_invoke.py
    ├── vision_multi_invoke.py
    ├── json_invoke.py
    ├── tools_invoke.py
    ├── embedding_local.py
    ├── tts_invoke.py
    ├── stt_invoke.py
    └── speech_translate_invoke.py
```

本地图片路径由脚本用「项目根 / 图片 / …」解析（`Path(__file__).parent.parent / "图片"`），因此请在**项目根**执行 `python 第二章/...`，或从任意 cwd 运行均可（不依赖当前工作目录）。

### API Key

密钥放环境变量，不要写进代码、不要提交 git。当前代码读的是：

```bash
export DEEPSEEK_API_CODE_KEY='你的密钥'
```

macOS（zsh）长期生效：把上面这行加到 `~/.zshrc`，然后 `source ~/.zshrc`。

Cursor 用 ▶ 运行时，若读不到变量：在集成终端里先 `source ~/.zshrc`，或重启 Cursor。

密钥在 [DeepSeek 开放平台](https://platform.deepseek.com/api_keys) 创建。

## 当前进度

| 文件 | 内容 |
|------|------|
| `第二章/api_invoke.py` | `chat.completions.create`：单轮/多轮、`usage`、tiktoken 估算；可选参数见 README 表 2-3 |
| `第二章/vision_invoke.py` | 视觉：公网图片直链 + `image_url`；`deepseek-flash` |
| `第二章/vision_local_invoke.py` | 视觉：本地单图 base64 → `data:image/png;base64,...` |
| `第二章/vision_multi_invoke.py` | 视觉：本地双图对比（`图片/百度图片_龙.png` + `图片/百度图片_青龙.png`） |
| `第二章/json_invoke.py` | `response_format=json_object`：把购鞋需求转成 JSON |
| `第二章/tools_invoke.py` | Function Calling：`tools` → 本地 `find_product` → `role=tool` → 自然语言 |
| `第二章/embedding_local.py` | 本机嵌入：`BAAI/bge-m3`，`encode` 得到 1024 维向量 |
| `第二章/tts_invoke.py` | 文本转语音：`edge-tts`，音色 `zh-CN-XiaoxiaoNeural`，输出 `tts_hello.mp3` |
| `第二章/stt_invoke.py` | 语音转文本：SenseVoice Small 转写 `tts_hello.mp3` |
| `第二章/speech_translate_invoke.py` | 法语 mp3（edge-tts）→ faster-whisper 原文转写 / 译成英文 |

图像 API（生成 / 编辑 / 变体）记在 Notion，没有示例脚本。OpenAI 现用 `gpt-image-2.5-flare` 与 `gpt-image-2.5-sunburst`，书上的 DALL·E 已从 API 下线。

### 视觉输入（第二章）

书中多模态写法：`messages[].content` 为数组，同时放文字和图片。

| 要求 | 说明 |
|------|------|
| 模型 | `deepseek-flash`（支持 Vision） |
| 公网 URL | 必须是图片直链（如 `.png`/`.jpg`），不能是 wiki/百度搜索详情页 → `vision_invoke.py` |
| 本地单图 | `b64encode` 后写成 `data:image/png;base64,...` → `vision_local_invoke.py` |
| 本地多图 | 同一 `content` 数组里放多个 `image_url` → `vision_multi_invoke.py` |
| `detail` | 可选：`low` / `high` / `auto` 等，影响清晰度与费用 |

## 关键概念（备忘）

### `chat.completions.create`

真正「问模型、拿回答」靠的就是 `create`。最少两个参数：

| 参数 | 是否必需 | 含义 |
|------|----------|------|
| `model` | 必需 | 用哪个模型，如 `deepseek-flash` |
| `messages` | 必需 | 这次请求里模型能看到的整段对话 |
| 见下方「可选参数」 | 可选 | 微调行为、成本、流式等 |

### 可选参数（书表 2-3）

| 参数 | 类型 | 含义 |
|------|------|------|
| `response_format` | 对象 | `{"type":"json_object"}` 强制 JSON 输出；`messages` 里也要明确要求输出 JSON |
| `logprobs` | 布尔 | 是否返回每个输出词元的对数概率（见书第 3 章 / Cookbook） |
| `tools` | 数组 | 可用工具列表；书写时主要是 function calling（书 2.5.5） |
| `tool_choice` | 字符串或对象 | `none` 普通回答；指定 function 强制调用；`auto` 由模型选 |
| `temperature` | 浮点 0～2，默认 1 | 越高越随机；`0` 更稳（近似确定性，不保证完全相同） |
| `top_p` | 浮点，默认 1 | 核采样：只考虑概率质量累计到 `top_p` 的词元；常与 temperature 二选一调 |
| `n` | 整数，默认 1 | 一次生成几条候选；`temperature=0` 时多条往往很像 |
| `seed` | 整数 | 提高可复现性（书中注明当时仍偏试验，不保证） |
| `stream` | 布尔，默认 False | `True` 时流式返回，类似 ChatGPT 打字效果 |
| `max_tokens` | 整数 | 限制本次**输出**最长词元数，利于控成本；输入+输出仍受模型上下文上限约束 |

DeepSeek 不一定支持表中每一项，以 [DeepSeek API 文档](https://api-docs.deepseek.com/) 为准。入门优先试：`temperature`、`stream`、`max_tokens`。

```python
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_CODE_KEY"),
    base_url="https://api.deepseek.com",
)
client.chat.completions.create(model="...", messages=[...])
```

### `messages` 与无状态

API **不会**记住上一次请求。每次 `create` 都是独立调用。

多轮聊天必须自己维护历史，每次重发完整 `messages`（旧对话 + 新问题）。只发「What is it?」而没有上一句 `assistant`，模型不知道 `it` 指什么。

| role | 谁写的 | 干什么 |
|------|--------|--------|
| `system` | 开发者 | 人设/规则，通常放最前，可省略 |
| `user` | 用户 | 提问、指令 |
| `assistant` | 模型（或你模拟的历史回复） | 历史回答；不塞回去就等于没发生过 |
| `tool` | 工具结果 | 函数调用场景；入门可先不管 |

自己做聊天时的思路：

```text
第1轮：messages = [system?, user1] → 得到 reply1
第2轮：messages = [system?, user1, assistant(reply1), user2] → 得到 reply2
第3轮：再追加 assistant(reply2) + user3 …
```

### 换模型

改 `model` 字符串即可，`base_url` / `api_key` 不用动。

| model | 含义 |
|-------|------|
| `deepseek-flash` | V4.1-Flash，更快更便宜（当前示例） |
| `deepseek-v4-pro` | V4 Pro，更强更贵 |

文档：[Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)

换模型 ≠ 开/关思考。思考是额外参数（以官方 Thinking Mode 文档为准）。

`base_url`：`https://api.deepseek.com`

### tiktoken（词元估算）

书中建议：调用端点前用 [tiktoken](https://github.com/openai/tiktoken) 估算字符串有多少 token，从而估成本和是否超上下文。

```bash
python -m pip install tiktoken
```

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokens = enc.encode("Hello, 你好")
print(len(tokens))           # 词元个数
print(enc.decode(tokens))    # 解码回文本
```

聊天请求除了正文，还有 `role` 等开销，粗算时每条消息再加点固定 token。

注意：DeepSeek **不是**同一套分词器。`tiktoken` 适合练书里的概念；对 DeepSeek 计费只能近似，以官方用量/文档为准。

### 嵌入与语音

DeepSeek 只有对话（含视觉、JSON、工具调用）。嵌入、审核、语音、生图都要另找。

| 书上 | 本仓库 |
|------|--------|
| Embeddings | `embedding_local.py`：本机 `BAAI/bge-m3`，1024 维。不同模型的向量不要混着比 |
| `audio.speech`（tts-1） | `tts_invoke.py`：`edge-tts`，音色 `zh-CN-XiaoxiaoNeural` |
| `audio.transcriptions`（whisper-1） | `stt_invoke.py`：SenseVoice Small，转写中文 |
| `audio.translations`（whisper-1） | `speech_translate_invoke.py`：faster-whisper，`task="translate"` 译成英文 |

法语演示句转写与原文一致。英译实测是 “Mathematics are a fundamental science.”，书上是 is。SenseVoice 不能把语音译成英文。

审核端点没有免费平替，未写示例。旧的文本补全端点 DeepSeek 不支持，示例已删。

### 图像 API

书上三种：生成（提示词出图）、编辑（按提示词改原图）、变体（保持主题换画法）。撰写时编辑和变体只有 DALL·E 2。

现在 OpenAI 的模型是 `gpt-image-2.5-flare`（更快）和 `gpt-image-2.5-sunburst`（画质和改图更好）。`dall-e-2`、`dall-e-3` 已于 2026-05-12 下线。

国内没有三项都免费、还能直连的接口。通义万相能生成和编辑，付费。2026-08-21 硅基流动价格页上 `Kwai-Kolors/Kolors` 按张免费（要实名、有限额，用前再核对）。本机可用 Z-Image-Turbo，没有独显会很慢。详情在 Notion 的 2026-09-23 一节。
