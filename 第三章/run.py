"""书中第 3 章：把数据、意图、检索、回答串成一次提问。

这是学习用的完整演示。每次运行都会：
  1. 删掉 Redis 里已有的向量索引
  2. 重新读取 PDF，切块并做成嵌入，再写入 Docker 中的 Redis
  3. 从问题里抽出关键词，用关键词做向量检索
  4. 只根据检索到的原文回答原问题

文档没变时，实战不必每次重建索引。入库做一次即可；
之后提问只需要第 3、4 步。这里每次重建，是为了脚本单独就能跑通。

嵌入模型是本机 BAAI/bge-m3（在 DataService 里加载）。
关键词和回答走 DeepSeek deepseek-flash，密钥是环境变量 DEEPSEEK_API_CODE_KEY。
PDF 在同目录 pdf/ExplorersGuide.pdf。
Redis 是书中的 Docker 容器，默认 localhost:6379。

运行：
  conda activate ai_code
  python 第三章/run.py

启动时会看到 Hugging Face 未登录警告和 Loading weights。
权重来自本机缓存，不是重新下载。随后读 PDF、做嵌入，中间没有进度条，大约几十秒。
"""
from pathlib import Path

from data_service import DataService
from intent_service import IntentService
from response_service import ResponseService

# __file__ 是本脚本路径，不依赖当前工作目录，从项目根或第三章目录运行都可以。
PDF_PATH = Path(__file__).resolve().parent / "pdf" / "系统架构设计师第二版.pdf"


def run(question: str, file: str = str(PDF_PATH)):
    # DataService 会连上 Redis，并载入嵌入模型。模型在进程结束时释放，所以每次运行都会重新 Loading weights。
    data_service = DataService()

    # 删掉索引 embeddings-index 及其文档，避免和上一次写入的向量叠在一起。
    data_service.drop_redis_data()
    # 按约 1000 字切块，每块得到 1024 维向量。返回值是 id、vector、text 的列表。
    data = data_service.pdf_to_embeddings(file)
    # 向量转成 float32 字节，以 doc:0、doc:1… 写入 Redis，供后面的 KNN 搜索使用。
    data_service.load_data_to_redis(data)

    intent_service = IntentService()
    response_service = ResponseService()

    # 意图服务不回答问题，只抽出关键词。检索用关键词，比把整句问话直接拿去搜更贴近书中的做法。
    intents = intent_service.get_intent(question)
    # 用同一个 bge-m3 把关键词变成向量，在已写入的索引里取最相近的几段原文。
    facts = data_service.search_redis(intents)
    # 回答时仍用用户的原问题。模型只能根据 facts，不能靠记忆补全书内容。
    return response_service.generate_response(facts, question), intents


if __name__ == "__main__":
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"找不到 PDF: {PDF_PATH}")

    # 书中的测试问题：宝箱在哪里。
    question = "什么是架构设计？"
    answer, intents = run(question)
    print(f"问题: {question}")
    print(f"关键词: {intents}")
    print(answer)
