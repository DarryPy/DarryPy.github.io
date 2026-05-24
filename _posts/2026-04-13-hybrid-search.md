---
layout: post
title: Hybrid Search 实战 — BM25 + 向量混合检索
date: 2026-04-13
topic: "RAG 与检索"
tags: [AI, RAG, Hybrid Search, BM25]
excerpt: 纯向量检索的盲区是关键词。BM25 + 向量混合检索能在保留语义能力的同时召回精确匹配，2026 年生产 RAG 标配。
permalink: /posts/2026-04-13-hybrid-search.html
---

## 纯向量的盲区

向量检索擅长找"语义相似"，但有个固有弱点：**精确匹配能力差**。

- 搜"产品代号 X-2026"：向量可能给你"产品代号 X-2025"
- 搜"张三的合同"：向量找语义相近，对"张三"这种专名敏感度差
- 搜代码：变量名、API 名几乎全是专有词，向量召回烂

这些场景 BM25 / 关键词搜索反而是降维打击。

## 什么是 Hybrid Search

把两种检索结果**融合**：

```
Query
  ├── 向量检索 → top-K 语义结果
  └── BM25 检索 → top-K 关键词结果
       ↓
   分数融合 → 最终 top-K
```

融合策略最常见两种：

### 1. RRF (Reciprocal Rank Fusion)

```
score(doc) = Σ 1 / (k + rank_i(doc))
```

`rank_i` 是 doc 在第 i 路检索里的排名，`k` 一般取 60。
**不需要分数归一化**，跨检索器融合很稳。

### 2. 加权线性融合

```
score = α × vector_score + (1-α) × bm25_score
```

需要先把两个分数都归一到 [0,1]。α 经验值 0.5-0.7。

## 各家实现

| 系统 | 内置 Hybrid |
|---|---|
| Weaviate | ✅ 一行 `hybrid: { alpha: 0.5 }` |
| Qdrant | ✅ 配合 sparse vectors |
| Pinecone | ✅ 支持 sparse-dense |
| Elasticsearch | ✅ 8.x 起原生 |
| pgvector | ❌ 需要自己组合 `tsvector + vector` |
| Milvus | ✅ 2.4+ |

自己组合（pgvector 为例）：

```sql
WITH bm25 AS (
  SELECT id, ts_rank(tsv, plainto_tsquery('query')) AS r
  FROM docs
  ORDER BY tsv @@ plainto_tsquery('query') DESC
  LIMIT 50
),
vector AS (
  SELECT id, embedding <=> $1::vector AS dist
  FROM docs
  ORDER BY dist
  LIMIT 50
)
SELECT id, 
  1.0/(60 + bm25.r) + 1.0/(60 + vector.dist) AS rrf_score
FROM bm25 FULL OUTER JOIN vector USING (id)
ORDER BY rrf_score DESC
LIMIT 10;
```

## 加 Reranker 收尾

Hybrid 的常用进阶配方：

```
Query
  ├── 向量 → top-50
  └── BM25 → top-50
       ↓ 合并去重
   top-50 候选集
       ↓ Cross-encoder Reranker
   top-5 最终结果
```

Reranker（如 Cohere rerank-3 / BGE-reranker / Voyage rerank）是个更小但更精的模型，
**专门 query+candidate 配对打分**。
延迟 +50-150ms，但准确率能再涨 10-20%。

## 中文场景的额外注意

BM25 默认按空格分词，中文要用 jieba / hanlp / IK 分词器先切。
Elasticsearch 中文用 `ik_max_word` 分词；pgvector + Postgres 中文用 zhparser 扩展。

不分词的话，BM25 在中文上几乎废了。

## 什么时候 Hybrid 不值得

- 数据量 < 1000 条：怎么搜都行
- 全部是自然语言、无专名/代号：纯向量够
- 实时性要求 < 100ms：两路并行的延迟管理变难

## 调参建议

1. **先跑纯向量、纯 BM25 各一次基线**，看各自的 hit rate
2. **如果 BM25 比向量好 10%+**，说明你的数据里关键词信号强，Hybrid 收益大
3. **α 从 0.5 起调**，每次 0.1 步长
4. **建立 golden dataset**，按 hit@5 / MRR 评测
5. **加 reranker 后再调一次 α**——通常会变（reranker 已经做了大部分排序工作）

## 一个朴素观察

> 单一检索方式是 2023 年的事。
> 2026 年的生产 RAG，**默认就是 hybrid + reranker**。

不上的话，精确匹配的 case 永远是用户反馈黑洞。
