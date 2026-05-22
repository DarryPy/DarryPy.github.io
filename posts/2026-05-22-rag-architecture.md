---
layout: post
title: RAG 架构详解 — 从基础检索到 Agentic / 多模态
date: 2026-05-22
tags: [AI, RAG, 向量检索, 架构]
excerpt: 2026 年的 RAG 已经不只是"检索 + 生成"那么简单。混合检索、Agentic RAG、多模态 RAG，工程落地的真实选型路径。
permalink: /posts/2026-05-22-rag-architecture.html
---

## RAG 解决了什么问题

LLM 有几个先天痛点：

- 训练数据有截止时间，不知道最新事实
- 不知道私有/企业内的知识
- 上下文窗口虽然变大了，但每次都把全部文档塞进去**贵且慢**
- 容易"幻觉"——一本正经地编造

RAG（Retrieval-Augmented Generation）的思路简单：**先检索相关片段，再让模型基于这些片段生成答案**。
模型不再凭记忆瞎说，而是被"喂"着相关材料回答。

## 最基础的 RAG 架构

```text
┌──────────┐       ┌────────────┐       ┌──────────┐
│ 用户问题 │ ───→ │  Embedding │ ───→ │ 向量检索 │
└──────────┘       └────────────┘       └─────┬────┘
                                              │
                            ┌──────── 相关片段 ────────┐
                            ▼                          ▼
                  ┌──────────────────┐         ┌──────────┐
                  │  Prompt 组装     │ ──────→ │   LLM    │ ──→ 答案
                  └──────────────────┘         └──────────┘
```

具体到代码层面，4 个核心组件：

1. **Ingestion（入库）**：把原始文档切片 → embedding → 存进向量库
2. **Retrieval（检索）**：把用户问题向量化 → 在向量库找 top-K 相似片段
3. **Augmentation（拼接）**：把检索到的片段塞进 Prompt 里
4. **Generation（生成）**：LLM 基于增强后的 Prompt 生成回答

这就是入门版的 RAG。但 2026 年的生产 RAG，远不止这些。

## 2026 年最重要的趋势：检索才是瓶颈

业界一个广泛共识：**RAG 的瓶颈不在生成端，而在检索端**。
模型再强，喂给它的片段不准、不全，回答就废了。

这推动了几个方向的演进：

### 1. 混合检索（Hybrid Retrieval）

纯向量检索有它的盲区——它擅长语义相似，但**专有名词、产品代号、人名地名**的精确匹配反而不如关键词搜索。
现代 RAG 普遍采用：

- **稀疏检索**（BM25 / keyword）：精确命中关键词
- **稠密检索**（向量）：捕捉语义
- **重排（Reranker）**：用一个更精细的小模型把候选重排

三路融合，召回和准确率都能上一个台阶。

### 2. Agentic RAG

把"检索"从一次性步骤变成 agent 的工具调用。LLM 自己判断：
- 需要检索吗？
- 检索哪个知识库？
- 这次检索结果够不够？要不要再换关键词检索一次？

这套和 ReAct 模式的 agent 结合得很好。复杂跨文档推理、需要多次迭代的查询，Agentic RAG 比一次性 RAG 强一个数量级。

### 3. 多模态 RAG

文本以外，2026 的 RAG 还能检索：
- 表格 / 结构化数据
- 图片 embeddings
- 音频片段
- 视频帧

比如"请帮我找出第二季度财报里那张说毛利率的图"——这种问题在纯文本 RAG 上几乎无解，但多模态 RAG 可以。

### 4. 图增强（Graph-augmented RAG）

针对实体关系密集的领域（医学、法律、企业知识图谱），结合知识图谱的 RAG 比纯向量检索效果显著更好。
检索时除了拿到片段，还能拿到"这个实体和哪些实体有什么关系"。

## 工程上的几个关键选择

**切片（Chunking）策略**：定长切（512 token）/ 按段落切 / 递归切 / 按语义切——
没有银弹，需要根据文档形态实测。法律合同适合按章节，技术文档适合按 Markdown heading。

**Embedding 模型**：通用场景用 OpenAI text-embedding-3-large / Voyage / BGE；
领域强相关时考虑微调或换成专门领域模型。

**向量数据库**：Pinecone（托管省心）/ Weaviate（功能全）/ Qdrant（自部署高性能）/ Milvus（大规模）。
小规模直接用 pgvector + Postgres 也够用。

**评估**：建一套 retrieval eval（hit rate、MRR）和 end-to-end eval（answer faithfulness、relevance）。
没有 eval 就在闭眼调参。

## 一份最小可行 Stack

如果今天想搭一个生产可用的 RAG，建议起手：

| 组件 | 选型 |
|---|---|
| Embedding | OpenAI text-embedding-3-small（便宜够用） |
| 向量库 | pgvector（已有 Postgres 直接加） |
| 检索 | hybrid（pgvector + Postgres full-text） |
| Reranker | Cohere rerank-3 或 BGE-reranker |
| LLM | Claude Sonnet 4.6 / GPT-4.5 |
| Eval | LangSmith / Phoenix / Ragas |

跑起来后再针对瓶颈替换：召回不够换 embedding、推理慢换向量库、长文本无法定位加 graph。

## 常见踩坑

1. **以为"加 RAG"就能解决所有问题**——RAG 解决知识问题，不解决推理能力问题
2. **不做 eval**——上线后才发现召回烂，没法迭代
3. **chunk 太大**——一个 chunk 同时包含几个不相关主题，向量平均后什么都不像
4. **chunk 太小**——失去上下文，模型回答支离破碎
5. **忽略 prompt 部分**——再好的检索，如果 prompt 不让模型基于片段回答，照样幻觉
