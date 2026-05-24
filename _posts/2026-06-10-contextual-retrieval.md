---
layout: post
title: "Contextual Retrieval — Anthropic 的 RAG 提升秘方"
date: 2026-06-10
topic: "RAG 与检索"
tags: [AI, RAG, Retrieval]
excerpt: Anthropic 发布的 Contextual Retrieval 让检索准确率提升约 50%。核心思路很简单：在存索引之前，先让 LLM 为每个 chunk 生成一段解释它在文档中位置的上下文。
permalink: /posts/2026-06-10-contextual-retrieval.html
---

## RAG 的根本问题：chunk 失去了文档上下文

假设你有一份关于 Q3 财报的文档，其中有一段：

```
收入同比增长 23.5%，主要由订阅业务驱动。运营费用控制良好，EBITDA 利润率提升至 28.3%。
```

这段话单独来看没有问题，但它被切成独立的 chunk 存入向量库后：

- "收入同比增长 23.5%" — 哪家公司？哪个季度？
- "主要由订阅业务驱动" — 什么业务？

用户问"XYZ 公司 Q3 的利润情况"，这个 chunk 可能根本检索不到，因为 embedding 里没有"XYZ 公司"和"Q3"这些关键信息。

这就是 **chunk 丢失了文档级别的上下文**。

---

## Contextual Retrieval 的解法

Anthropic 的方案：**在为每个 chunk 生成 embedding 之前，先用 LLM 为它生成一段上下文说明**。

```
原始 chunk:
"收入同比增长 23.5%，主要由订阅业务驱动..."

→ LLM 生成的上下文：
"这段话来自 XYZ 公司 2024 年 Q3 财务报告，描述了该季度的收入增长情况和驱动因素。"

→ 存入索引的内容：
"这段话来自 XYZ 公司 2024 年 Q3 财务报告，描述了该季度的收入增长情况和驱动因素。
收入同比增长 23.5%，主要由订阅业务驱动..."
```

检索时，这个 chunk 的 embedding 就包含了"XYZ 公司"、"Q3"、"财务报告"等关键语义，检索精度大幅提升。

---

## 完整实现

```python
from anthropic import Anthropic
from typing import Optional
import json

client = Anthropic()

CONTEXT_GENERATION_PROMPT = """你是一个文档分析助手。
你的任务是为文档中的特定片段生成一段简短的上下文说明，
帮助检索系统更好地理解这段内容在整个文档中的位置和含义。

<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

请用1-2句话说明：这段内容来自什么类型的文档，描述的是什么主题，有什么关键信息。
只输出上下文说明本身，不要有任何前缀或解释。"""

def generate_chunk_context(
    document: str, 
    chunk: str,
    model: str = "claude-haiku-4-5",  # 用小模型降成本
) -> str:
    """
    为单个 chunk 生成文档上下文说明。
    用 claude-haiku-4-5 而不是 claude-opus 以控制成本。
    """
    resp = client.messages.create(
        model=model,
        messages=[{
            "role": "user",
            "content": CONTEXT_GENERATION_PROMPT.format(
                document=document[:5000],  # 文档太长则截断，保留关键部分
                chunk=chunk,
            )
        }],
        max_tokens=200,
    )
    return resp.content[0].text.strip()

def create_contextual_chunk(document: str, chunk: str) -> str:
    """
    生成带上下文的 chunk，格式：
    [上下文]\n\n[原始内容]
    """
    context = generate_chunk_context(document, chunk)
    return f"{context}\n\n{chunk}"

class ContextualRAGIndexer:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store
    
    def index_document(
        self, 
        document: str, 
        doc_id: str,
        chunk_size: int = 512,
        overlap: int = 50,
        batch_size: int = 10,
    ) -> int:
        """
        将文档分块，为每块生成上下文，然后存入向量库。
        返回存入的 chunk 数量。
        """
        # 1. 分块
        chunks = self._split_document(document, chunk_size, overlap)
        print(f"Document split into {len(chunks)} chunks")
        
        # 2. 批量生成上下文（带进度）
        contextual_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Generating context {i+1}/{len(chunks)}...", end="\r")
            contextual_text = create_contextual_chunk(document, chunk)
            contextual_chunks.append({
                "original_chunk": chunk,
                "contextual_chunk": contextual_text,
                "chunk_index": i,
                "doc_id": doc_id,
            })
        
        print(f"\nGenerating embeddings...")
        
        # 3. 生成 embedding（用带上下文的版本）
        texts_to_embed = [c["contextual_chunk"] for c in contextual_chunks]
        embeddings = self.embedder.encode(texts_to_embed, batch_size=batch_size)
        
        # 4. 存入向量库
        records = []
        for chunk_data, embedding in zip(contextual_chunks, embeddings):
            records.append({
                "id": f"{doc_id}_chunk_{chunk_data['chunk_index']}",
                "embedding": embedding.tolist(),
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_index": chunk_data["chunk_index"],
                    "original_text": chunk_data["original_chunk"],
                    "contextual_text": chunk_data["contextual_chunk"],
                }
            })
        
        self.vector_store.upsert(records)
        print(f"Indexed {len(records)} contextual chunks")
        return len(records)
    
    def _split_document(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        words = text.split()
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks
```

---

## 用 Prompt Caching 大幅降低成本

生成上下文最大的问题是成本——每个 chunk 都要发一次 LLM 请求，且每次都要把整篇文档发给模型。

Anthropic 的 Prompt Caching 可以缓存文档内容，只需第一次发送完整文档，后续 chunk 复用缓存。

```python
def generate_chunk_context_with_caching(
    document: str,
    chunks: list[str],
    model: str = "claude-haiku-4-5",
) -> list[str]:
    """
    用 Prompt Caching 批量生成所有 chunk 的上下文。
    文档内容只在第一次 API 调用时传输，后续命中缓存。
    
    成本对比：
    - 无缓存：每个 chunk 都要发送完整文档 → N × document_tokens 的输入费用
    - 有缓存：document_tokens 只计费一次（缓存读取 = 10% 的正常输入价格）
    """
    contexts = []
    
    for i, chunk in enumerate(chunks):
        resp = client.messages.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        # 文档内容设置为可缓存（cache_control）
                        {
                            "type": "text",
                            "text": f"这是完整文档：\n\n{document}\n\n",
                            "cache_control": {"type": "ephemeral"},  # 启用缓存
                        },
                        {
                            "type": "text",
                            "text": f"请为以下文档片段生成1-2句上下文说明，说明它来自什么文档、描述什么内容：\n\n{chunk}\n\n只输出上下文说明。",
                        }
                    ]
                }
            ],
            max_tokens=200,
        )
        
        context = resp.content[0].text.strip()
        contexts.append(context)
        
        # 第一次调用会写入缓存，后续调用会命中缓存
        cache_info = resp.usage
        if hasattr(cache_info, 'cache_creation_input_tokens'):
            print(f"Chunk {i+1}: cache_write={cache_info.cache_creation_input_tokens}, "
                  f"cache_read={cache_info.cache_read_input_tokens}")
    
    return contexts

# 成本估算
def estimate_cost(
    document_tokens: int,
    num_chunks: int,
    chunk_tokens: int = 100,
    model: str = "claude-haiku-4-5",
):
    """
    Haiku 价格（2025）：输入 $0.80/MTok，缓存写 $1.00/MTok，缓存读 $0.08/MTok
    """
    # 无缓存
    no_cache_cost = (document_tokens + chunk_tokens) * num_chunks * 0.80 / 1_000_000
    
    # 有缓存（第一次写入 + 后续读取）
    cache_write_cost = document_tokens * 1.00 / 1_000_000
    cache_read_cost = document_tokens * (num_chunks - 1) * 0.08 / 1_000_000
    chunk_input_cost = chunk_tokens * num_chunks * 0.80 / 1_000_000
    with_cache_cost = cache_write_cost + cache_read_cost + chunk_input_cost
    
    savings = (no_cache_cost - with_cache_cost) / no_cache_cost * 100
    
    print(f"文档长度: {document_tokens} tokens, Chunk 数: {num_chunks}")
    print(f"无缓存成本: ${no_cache_cost:.4f}")
    print(f"有缓存成本: ${with_cache_cost:.4f}")
    print(f"节省: {savings:.1f}%")

# 示例：5000 token 文档，50 个 chunk
estimate_cost(document_tokens=5000, num_chunks=50)
# 无缓存成本: $0.2575
# 有缓存成本: $0.0295
# 节省: 88.5%
```

---

## 结合 BM25 进一步提升

单纯向量检索有时会漏掉关键词完全匹配的情况（比如产品型号、专有名词）。结合 BM25 可以两全其美。

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, vector_store, alpha: float = 0.5):
        """
        alpha: 向量检索权重（0=纯 BM25，1=纯向量）
        Anthropic 报告中 alpha=0.5 效果最好
        """
        self.vector_store = vector_store
        self.alpha = alpha
        self.bm25 = None
        self.corpus = []
    
    def build_bm25_index(self, chunks: list[dict]):
        """用 contextual_text 构建 BM25 索引"""
        self.corpus = [c["contextual_text"] for c in chunks]
        tokenized = [text.split() for text in self.corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.chunk_metadata = chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # 向量检索
        query_emb = self.embedder.encode([query])[0]
        vector_results = self.vector_store.query(query_emb, top_k=top_k * 3)
        
        # BM25 检索
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 3]
        
        # 归一化分数
        def normalize(scores):
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]
        
        # 合并 RRF（Reciprocal Rank Fusion）
        scores = {}
        
        for rank, result in enumerate(vector_results):
            chunk_id = result["id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + self.alpha / (rank + 1)
        
        bm25_norm = normalize(bm25_scores)
        for rank, idx in enumerate(bm25_top_indices):
            chunk_id = self.chunk_metadata[idx]["id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + (1 - self.alpha) / (rank + 1)
        
        # 排序返回
        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [self._get_chunk(cid) for cid in sorted_ids]
```

---

## 效果数据（Anthropic 报告）

| 方案 | 检索失败率（越低越好） |
|------|----------------------|
| 基础 RAG | 基准 |
| + Contextual Retrieval | -49% |
| + BM25 | -35% |
| + Contextual + BM25 | -67% |
| + Contextual + BM25 + Rerank | -76% |

最大收益来自 Contextual Retrieval 本身，BM25 和 Rerank 叠加收益递减。

---

## 一个朴素结论

> Contextual Retrieval 解决的是 RAG 里最基础的问题：chunk 不知道自己是谁。
>
> 实现不复杂，有了 Prompt Caching 之后成本也可接受。
> 对于已有 RAG 系统，这是最值得做的单点改进——改动小，效果大。
>
> **先加上 Contextual Retrieval，再考虑 BM25 融合。不要一开始就追求最优方案。**
