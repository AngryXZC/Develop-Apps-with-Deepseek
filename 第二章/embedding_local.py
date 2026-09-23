"""书中嵌入示例的本机免费版。

DeepSeek 没有嵌入接口。这里用开源模型 BAAI/bge-m3 在本地把文本变成向量。
第一次运行会下载模型（约 2GB）。

安装：
  conda activate ai_code
  python -m pip install sentence-transformers

运行：
  python 第二章/embedding_local.py
"""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
vec = model.encode("你好")
print(len(vec))  # 1024
print(vec[:8])
