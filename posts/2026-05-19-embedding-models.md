---
layout: post
title: Embedding 模型选型实战 — OpenAI / Voyage / BGE / Cohere 怎么挑
date: 2026-05-19
tags: [AI, Embedding, RAG, 向量检索]
excerpt: 主流 embedding 模型横向对比，维度、性能、价格、领域适配度，一份选型决策表。
permalink: /posts/2026-05-19-embedding-models.html
---

## Embedding 是什么、为什么重要

Embedding 就是把**文本 → 一个高维向量**。向量之间的距离（cosine / 内积）反映语义相似度。
RAG / 推荐 / 去重 / 聚类 / 异常检测——只要涉及"语义相似"，背后基本都是 embedding。

选错了 embedding，**检索召回从根上就废了**。Reranker 救不了。

## 维度、容量、价格

| 模型 | 维度 | 价格（每 1M tokens） | 备注 |
|---|---|---|---|
| OpenAI text-embedding-3-small | 1536 | $0.02 | 性价比之王 |
| OpenAI text-embedding-3-large | 3072 | $0.13 | 效果最好的通用模型之一 |
| Voyage voyage-3 | 1024 | $0.06 | 在 retrieval 任务上常超 OpenAI |
| Voyage voyage-3-large | 1024 | $0.18 | 领域微调可选 |
| Cohere embed-english-v3 | 1024 | $0.10 | 多模态 + 多语言 |
| BGE-M3 (开源) | 1024 | 自托管 | 中英文都很强，可自部署 |
| BGE-large-zh | 1024 | 自托管 | 中文场景首选开源 |

## 主流选择的几条经验

### 1. 英文为主，预算紧 → OpenAI text-embedding-3-small

便宜、API 稳定、零运维。MTEB benchmark 上的成绩不算最好但够用。
1M tokens 才 $0.02，等于**每天处理 100 万字也只要几毛钱**。

### 2. 检索效果优先 → Voyage 或 Cohere

Voyage 在 retrieval-specific 任务上表现稳定优于 OpenAI 同级。
Cohere 多语言支持非常好（涵盖 100+ 种语言）。

### 3. 中文场景 → BGE-M3 自部署 或 Cohere multilingual

OpenAI 在中文 retrieval 上能用但不是最强；
BGE 系列是智源研究院开源的中文 embedding，国内场景几乎是标配。
自托管成本：一张消费级 GPU（4090 / A6000）就能跑 BGE-M3。

### 4. 领域强相关（法律 / 医学 / 金融）→ 微调

通用 embedding 对专业术语的区分度有限。
微调成本比想的低——拿到 10k 对正负样本就能开始，BGE / Sentence-Transformers 都有现成训练脚本。

## Matryoshka：可裁剪维度

OpenAI text-embedding-3 系列引入了 Matryoshka 表示——同一个向量你可以**只取前 256 / 512 / 1536 维**，效果衰减很小。

应用场景：
- 量级巨大时只存前 256 维省 6 倍存储
- 检索时先用 256 维粗排，再用全维精排
- 不同业务方按预算自取所需

## 量化：二值 / int8

主流向量库都支持量化存储：

| 量化方式 | 存储节省 | 检索质量损失 |
|---|---|---|
| float32（默认） | 1x | 0% |
| int8 | 4x | 1-3% |
| binary（二值） | 32x | 5-15%（配合 reranker 可压回） |

千万级以上向量，**强烈建议上量化**，存储和内存都能省一个数量级。

## 一个决策清单

问自己几个问题：

1. **数据量级？** < 100 万条：什么都行；> 1000 万条：必须想清楚维度和量化
2. **语言？** 纯英文：OpenAI / Voyage；中文为主：BGE / Cohere；多语言：Cohere
3. **领域专业度？** 通用：off-the-shelf；强领域：考虑微调
4. **运维资源？** 没 GPU：托管 API；有 GPU + 数据敏感：自托管开源模型
5. **延迟敏感？** API 200-500ms，自托管 GPU 50-100ms，CPU 几百 ms

## 一个常被忽略的事

**Query 和 Document 用同一个 embedding 模型**。看着是废话，但实际工程里：

- 入库时用了 model-A
- 几个月后看到 model-B 更好，把检索查询切到 model-B 
- 然后召回烂得离谱

这是因为不同 embedding 模型的向量空间不一样，距离没法跨模型比较。
**升级 embedding = 必须全量重新入库**。所以**第一次选型要慎重**，而不是后面再换。

## 结论

通用场景 OpenAI text-embedding-3-small 就够用；
追极致检索效果用 Voyage；
中文 + 数据敏感用自部署 BGE；
专业领域考虑微调；
大数据量配合 Matryoshka + int8 / binary 量化。

选完了 90 天不要乱换。
