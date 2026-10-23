"""书中第 3 章信息检索：PDF 嵌入写入 Redis，再按问题做向量搜索。

DeepSeek 没有嵌入接口。查询和文档都用本机 BAAI/bge-m3（1024 维），不要和 ada-002 的向量混用。
Redis 用书中的 Docker 容器 redis/redis-stack（compose 服务 vector-db），
把 6379 映射到本机。默认连 localhost:6379，也就是这个容器，不是 Homebrew 的 redis-server。
PDF 默认读同目录下的 pdf/ExplorersGuide.pdf。

安装：
  conda activate ai_code
  python -m pip install pypdf redis

运行：
  python 第三章/data_service.py
"""
import logging
import os
from pathlib import Path

import numpy as np
import redis
from pypdf import PdfReader
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

# 这份 PDF 用了特殊字体，不装 fontTools 时 pypdf 会刷大量警告。
logging.getLogger("pypdf").setLevel(logging.ERROR)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or None

PDF_PATH = Path(__file__).resolve().parent / "pdf" / "ExplorersGuide.pdf"
INDEX_NAME = "embeddings-index"
PREFIX = "doc"
VECTOR_FIELD = "vector"


class DataService:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
        )
        self.embed_model = SentenceTransformer("BAAI/bge-m3")

    def pdf_to_embeddings(self, pdf_path: str, chunk_length: int = 1000):
        reader = PdfReader(pdf_path)
        chunks = []
        for page in reader.pages:
            text_page = page.extract_text() or ""
            chunks.extend(
                [
                    text_page[i : i + chunk_length]
                    for i in range(0, len(text_page), chunk_length)
                ]
            )
        chunks = [chunk for chunk in chunks if chunk.strip()]
        if not chunks:
            return []

        vectors = self.embed_model.encode(chunks)
        return [
            {
                "id": index,
                "vector": vectors[index].tolist(),
                "text": chunks[index],
            }
            for index in range(len(chunks))
        ]

    def load_data_to_redis(self, embeddings):
        vector_dim = len(embeddings[0]["vector"])
        fields = [
            TextField(name="text"),
            VectorField(
                VECTOR_FIELD,
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": vector_dim,
                    "DISTANCE_METRIC": "COSINE",
                    "INITIAL_CAP": len(embeddings),
                },
            ),
        ]
        try:
            self.redis_client.ft(INDEX_NAME).info()
            print("索引已存在")
        except redis.exceptions.ResponseError:
            self.redis_client.ft(INDEX_NAME).create_index(
                fields=fields,
                definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH),
            )

        for embedding in embeddings:
            key = f"{PREFIX}:{embedding['id']}"
            embedding["vector"] = np.array(embedding["vector"], dtype=np.float32).tobytes()
            self.redis_client.hset(key, mapping=embedding)
        print(f"已写入 {len(embeddings)} 条到索引 {INDEX_NAME}")

    def drop_redis_data(self, index_name: str = INDEX_NAME):
        try:
            self.redis_client.ft(index_name).dropindex(delete_documents=True)
            print(f"已删除索引 {index_name}")
        except redis.exceptions.ResponseError:
            print(f"索引 {index_name} 不存在")

    def search_redis(self, user_query: str, k: int = 5):
        embedded_query = self.embed_model.encode(user_query)
        query = (
            Query(f"*=>[KNN {k} @{VECTOR_FIELD} $vector AS vector_score]")
            .return_fields("text", "vector_score")
            .sort_by("vector_score")
            .paging(0, k)
            .dialect(2)
        )
        params_dict = {
            "vector": np.array(embedded_query, dtype=np.float32).tobytes()
        }
        results = self.redis_client.ft(INDEX_NAME).search(query, params_dict)
        return [doc["text"] for doc in results.docs]


if __name__ == "__main__":
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"找不到 PDF: {PDF_PATH}")

    service = DataService()
    rows = service.pdf_to_embeddings(str(PDF_PATH))
    print(f"块数: {len(rows)}")
    if not rows:
        raise SystemExit("PDF 没有抽出文本")
    print(f"向量维度: {len(rows[0]['vector'])}")
    service.load_data_to_redis(rows)
    question = "amiibo 是什么？"
    print(f"问题: {question}")
    for i, text in enumerate(service.search_redis(question), start=1):
        print(f"{i}. {text[:180]}")
