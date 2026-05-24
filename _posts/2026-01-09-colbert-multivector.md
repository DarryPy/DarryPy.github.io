---
layout: post
title: ColBERT 与 Multi-vector Retrieval — 不只一个向量
date: 2026-01-09
topic: "RAG 与检索"
tags: [AI, RAG, ColBERT, Multi-vector]
excerpt: 普通 RAG 一个文档一个向量。ColBERT 一个文档多个 token 级向量，精度更高，特别适合长文档。
permalink: /posts/2026-01-09-colbert-multivector.html
---

## 单向量的瓶颈

普通 RAG 把整段文档压成 1 个向量：

```
"RAG 是一种结合检索和生成的 AI 技术..." → 1024 维向量
```

长文档信息被压缩 → **细节丢失**。
问"RAG 中 reranker 干啥用"——文档里关于 reranker 的那一段被整段向量平均稀释了。

## ColBERT 的思路

**每个 token 一个向量**，检索时 query 和 doc 的 token 向量两两对比：

```
Query: "RAG 中 reranker 干啥"
  → [v_RAG, v_中, v_reranker, v_干啥]  (4 个 token 向量)

Doc: "RAG 包含 ... reranker 用于精排 ..."
  → [v_RAG, v_包含, ..., v_reranker, v_用于, v_精排, ...]  (N 个 token 向量)

相似度 = ∑ max_{j} cos(q_i, d_j)
        i  
        （query 每个 token 找 doc 里最相似的，加总）
```

这种"细粒度匹配"叫 **late interaction**——
比单向量的"一个对一个"信号丰富得多。

## 实测效果

在 BEIR benchmark 上：
- 单向量 dense retrieval（如 OpenAI embedding）：MRR ~0.45
- ColBERT-v2：MRR ~0.55
- **+22% 召回提升**

特别明显的场景：
- 长文档（每段信息密度不均）
- 跨段引用（信息散在多处）
- 专业术语 / 实体名（向量平均后丢失）

## 工程代价

每个 doc 存几十到几百个向量，**存储 + 检索成本 10-50x**：

| | 单向量 | ColBERT |
|---|---|---|
| 每文档向量数 | 1 | ~100 |
| 100 万文档存储 | 4GB（1024 dim f32）| 400GB |
| 检索复杂度 | O(N) | O(N × M)（M=token 数）|

直接全量 ColBERT 不现实，要做工程优化。

## 优化方法

### 1. 两阶段：单向量召回 + ColBERT 精排

```
Query
  ↓ 单向量检索 → top-1000 候选
  ↓ ColBERT 精排 → top-10
  ↓ LLM 生成
```

**单向量召回快但召回准；ColBERT 精排准但只看 1000 个**。
这种组合性能和质量都得到。

### 2. PLAID（ColBERT 加速）

PLAID 是 ColBERT 团队出的优化版：
- 倒排索引 + 量化
- 检索速度比原始 ColBERT 快 100x
- 几乎无质量损失

### 3. ColPali（视觉 ColBERT）

把 ColBERT 思路用到 PDF 截图——每页截图过 vision encoder 出多向量。
**文档 RAG 的新 SOTA**。

## 实战工具

| 工具 | 说明 |
|---|---|
| **RAGatouille** | ColBERT 的简单封装，pip install 即用 |
| **ColBERT 官方仓库** | 完整训练 + 推理 |
| **Vespa** | 向量库内置 multi-vector 支持 |
| **Qdrant** | 2024 支持 multi-vector |
| **Weaviate** | 部分支持 |

最简单：

```python
from ragatouille import RAGPretrainedModel

rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
rag.index(documents=[...], index_name="my_index")
results = rag.search("RAG 中 reranker 干啥", k=10)
```

10 行起步。

## 跟 Late Chunking 的区别

- **ColBERT**：检索时算每个 token 的相似度
- **Late Chunking**：用 long-context embedding 算 chunk 向量，每个 chunk 一个向量

两个都比单向量好。**Late Chunking 工程简单**，ColBERT **精度更高**。

## 适合什么场景

| 场景 | ColBERT |
|---|---|
| 文档量 < 10 万 | ✅ 可以直接上 |
| 文档量 100 万+ | ⚠️ 工程要做精排 + 量化 |
| 长文档 / 法律 / 学术 | ✅ 收益最大 |
| 短文档 / FAQ | ❌ 单向量够 |
| 跨段引用多 | ✅ |
| 简单事实问答 | ❌ overkill |

## 一个朴素结论

> 长文档 RAG 召回上不去？**试 ColBERT**。
> 短文档 / 简单 FAQ 不要上，单向量 + reranker 已经够。
>
> 2025-2026 RAG 的高级玩法：单向量召回 + ColBERT 精排 + LLM 生成。
