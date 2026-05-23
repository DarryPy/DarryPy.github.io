---
layout: post
title: Reranker 模型选型对比 — Cohere / BGE / Voyage / Jina
date: 2026-03-22
topic: "RAG 与检索"
tags: [AI, RAG, Reranker]
excerpt: 检索出 top-50 候选后用 reranker 精排到 top-5。主流 reranker 模型横向对比和工程使用指南。
permalink: /posts/2026-03-22-reranker-selection.html
---

## 为什么要加 Reranker

向量检索 + BM25 召回的 top-K 候选**不能直接给 LLM 用**——里头会有不相关的。
加一层 reranker（精排）能再过滤掉一半噪声，让 LLM 看到的片段质量翻倍。

```
Query → Retrieval (top-50) → Reranker (top-5) → LLM
```

代价：每次 query 多 50-200ms 延迟，但召回准确率 +15-25%。

## 主流 Reranker 横评

| 模型 | 类型 | 价格/调用 | 优势 |
|---|---|---|---|
| **Cohere rerank-3** | API | $0.001/搜索 | 多语言、稳定，2026 主流首选 |
| **Voyage rerank-2** | API | $0.05/M tokens | 跟 Voyage embedding 配合好 |
| **Jina rerank-v2** | API + 开源 | 部分免费 | 中英文都好，多语言强 |
| **BGE-reranker-v2-m3** | 开源 | 自托管 | 中文场景 SOTA，可自部署 |
| **mxbai-rerank-large** | 开源 | 自托管 | 英文场景强，license 友好 |
| **GPT-4 as reranker** | LLM-as-Judge | 贵且慢 | 极致质量，小流量场景可用 |

## Cross-encoder vs Bi-encoder

- **Bi-encoder**（向量检索用的）：query 和 doc 分别编码再算相似度——快但精度有限
- **Cross-encoder**（reranker 用的）：query + doc 拼一起进同一模型，**直接输出相关性分数**——慢但准

Reranker 是 cross-encoder，**只用于二次精排小候选集**（≤ 100），不能用来全量检索。

## 选型决策

| 场景 | 推荐 |
|---|---|
| 通用英文场景 | Cohere rerank-3 |
| 中文 / 中英混合 | Jina rerank 或 BGE-reranker-v2-m3 |
| 数据合规要求自部署 | BGE / mxbai |
| 用 Voyage embedding | Voyage rerank-2（同家配合好）|
| 极致质量、小流量 | LLM-as-reranker（用 Opus）|

## 一个最小集成

```python
import cohere
co = cohere.Client(api_key="...")

def rerank(query, candidates, top_k=5):
    """candidates: List[str]，召回的文档片段"""
    response = co.rerank(
        query=query,
        documents=candidates,
        top_n=top_k,
        model="rerank-multilingual-v3.0",
    )
    return [(r.document.text, r.relevance_score) for r in response.results]
```

## 工程坑

1. **不要 rerank 超过 100 个候选**：贵且慢，召回时就要够准
2. **reranker 分数不是概率**：不要直接当置信度用，只用来排序
3. **reranker 长度限制**：每个 document 通常 512-2048 tokens 上限
4. **冷启动延迟**：第一次调用 reranker API 有冷启动，可以 warmup 一下
5. **离线 reranker 内存大**：BGE-reranker-v2-m3 加载到内存约 2-3GB，self-host 注意

## 一个朴素结论

> 没有 reranker 的 RAG 是不完整的。
> 选型上：**英文 Cohere，中文 Jina 或 BGE，自部署敏感数据用开源**。
>
> Reranker 比换 embedding 模型涨点快，是 RAG 优化第一刀。
