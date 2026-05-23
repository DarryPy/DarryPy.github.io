---
layout: post
title: Query Expansion / HyDE — RAG 检索的"假设文档"魔法
date: 2026-01-31
topic: "RAG 与检索"
tags: [AI, RAG, Query Expansion, HyDE]
excerpt: 用户 query 太短，向量检索召回烂。先让 LLM 生成"假设答案"再 embed，召回率能涨 15-20%。
permalink: /posts/2026-01-31-query-expansion-hyde.html
---

## 问题：短 query 召回烂

用户问"什么是 RAG？"——3 个字。
向量库里相关文档有几十个，但 query embedding 太短，**信号弱**，召回不准。

两种解法：扩展 query，或者扩展检索目标。

## Query Expansion（query 变长）

让 LLM 把短 query 扩展成几个相关的更具体问题：

```
原 query: 什么是 RAG?
↓ LLM 扩展
- 什么是 Retrieval-Augmented Generation?
- RAG 系统的架构组成?
- RAG 和 fine-tuning 的区别?
- RAG 应用场景?

→ 4 个 query 分别 embed 检索，结果合并
```

```python
def expand_query(query, n=4):
    prompt = f"""
基于这个用户问题，生成 {n} 个更具体、可能涉及的相关子问题：

问题：{query}

输出 JSON 数组：["子问题1", "子问题2", ...]
"""
    return llm.complete(prompt, response_format="json")

def retrieve_with_expansion(query):
    expanded = [query] + expand_query(query)
    all_results = []
    for q in expanded:
        all_results += vector_search(embed(q), k=5)
    return dedupe_and_rerank(all_results)
```

**召回 +10-15%，但检索成本 N 倍**。

## HyDE: Hypothetical Document Embeddings

2022 年提出的更优雅思路：**让 LLM 编一个"假设答案"，用假设答案的 embedding 去检索**。

```
原 query: 什么是 RAG?
↓ LLM 生成假设答案
"RAG 是 Retrieval-Augmented Generation 的缩写，是一种结合检索和生成的 AI 技术。
它先从外部知识库检索相关文档，再让大模型基于检索结果生成回答..."
↓ embed 假设答案
↓ 检索向量库
```

直觉：**检索时 query 的 embedding 跟"答案的 embedding"在向量空间里更接近答案本身**。
所以用假设答案 embed 比 query embed 召回更准。

## HyDE 代码

```python
def hyde_retrieve(query, k=10):
    # 1. 生成假设答案
    hypo = llm.complete(f"为这个问题写一段 200 字的假设答案：{query}")
    # 2. 用假设答案 embedding 检索
    return vector_search(embed(hypo), k=k)
```

实测涨幅：在零样本（zero-shot）场景下，HyDE 比直接 embed 召回准确率 **+15-25%**。

## 跟普通 query expansion 对比

| | Query Expansion | HyDE |
|---|---|---|
| LLM 调用 | 1 次（扩展）+ N 次检索 | 1 次（生成假设）+ 1 次检索 |
| 适合 | 高歧义 / 多面向问题 | 短 query / 零样本 |
| 实现 | 中等 | 简单 |
| 召回涨幅 | +10-15% | +15-25% |

实战：**HyDE 更便宜更有效**，是当前主流。

## 工程坑

### 1. 假设答案"瞎编"会污染

LLM 编的假设答案可能跟真实答案差很远，导致检索偏向错方向。

缓解：
- 用强模型生成假设（Claude / GPT 一档以上）
- temperature=0 让生成稳定
- 假设答案太长则截断

### 2. 不适合"找特定文档"的 query

```
"帮我找 2024 年 Q3 财报"
```

HyDE 会编一份"假财报内容"，但用户其实就想要 metadata 匹配。
这种 case 用 metadata filter 而不是 HyDE。

### 3. 用 reranker 做兜底

HyDE 召回准确率高，但偶尔召回奇怪结果。**配合 cross-encoder reranker** 收尾，把真正相关的精排到前面。

## 实战推荐配方

```
Query
  ├── 短 query / 零样本 → HyDE
  └── 长 query / 多面向 → Query Expansion

  ↓ 检索 top-50
  ↓
Reranker → top-5
  ↓
LLM 生成
```

## 一个朴素结论

> 不需要 fine-tune embedding，**一个 HyDE 调用就能涨 15-25% 召回**。
> 工程简单，是当前 RAG 优化的标准杀招。
