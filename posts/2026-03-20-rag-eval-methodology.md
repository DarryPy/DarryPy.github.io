---
layout: post
title: RAG Eval 完整方法论 — Retrieval 和 Generation 分开测
date: 2026-03-20
topic: "RAG 与检索"
tags: [AI, RAG, Eval]
excerpt: RAG 烂了到底是检索烂还是生成烂？把两端分开评估才能定位瓶颈。一份能跑通的 RAG eval 流水线。
permalink: /posts/2026-03-20-rag-eval-methodology.html
---

## RAG eval 必须分两段

RAG = 检索 + 生成。两端任一端烂，整体就烂。
**不分开测，永远不知道该优化谁**。

```
Query
  ↓
Retrieval → 评估 Retrieval Quality
  ↓ context
Generation → 评估 Answer Quality
  ↓
Final Answer
```

## Retrieval Eval

### 必需指标

| 指标 | 含义 | 怎么算 |
|---|---|---|
| **Hit Rate @ K** | top-K 里是否包含至少 1 个 ground truth doc | 包含 = 1，不包含 = 0 |
| **MRR (Mean Reciprocal Rank)** | 第一个相关 doc 的倒数排名 | 第 1 位 = 1.0, 第 3 位 = 0.33 |
| **NDCG @ K** | 考虑排序的相关性累计 | 越前面的权重越高 |
| **Context Precision** | top-K 中有多少是真有用的 | 有用 / K |
| **Context Recall** | 回答所需信息有多少被检索到 | 检索到的关键信息 / 应有的关键信息 |

### 怎么建 ground truth

```
1. 从线上日志采样 200-500 个真实 query
2. 人工/LLM-as-judge 标注每个 query 对应的"理想检索片段"
3. 这就是 golden dataset
```

## Generation Eval

### 必需指标

| 指标 | 含义 | 怎么测 |
|---|---|---|
| **Faithfulness** | 答案是否基于检索片段（vs 凭空编） | LLM-as-judge：检查每句答案能否在 context 找到依据 |
| **Answer Relevancy** | 答案跟问题相关吗 | LLM-as-judge：从答案反推问题，跟原问题比相似度 |
| **Answer Correctness** | 答案对不对 | 跟 ground truth 答案对比（用 LLM judge）|
| **Answer Conciseness** | 没有废话 | 字数 / 信息密度 |

### Faithfulness 是 RAG 最重要的指标

幻觉的最常见形式：**模型基于训练数据回答**，而不是基于给的 context。
监控 faithfulness 能及时发现 "context 喂了但模型没用" 的问题。

## 工具选型

### Ragas（标杆）

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset=ds,
    metrics=[faithfulness, answer_relevancy, context_precision],
)
```

最全的 RAG eval 指标库，事实标准。

### LangSmith / Phoenix / Langfuse

提供完整 dataset + experiment + trace 平台，跟 Ragas 配合最好。

## 实战流程

```
1. 从生产采样真实 query（≥ 200）
2. 人工/LLM 标注 ground truth context + answer
3. 每次改 prompt / 模型 / 检索：跑全套指标
4. 看哪个指标退步：定位是 retrieval 还是 generation 的锅
5. 单独优化那一端
```

## 常见诊断

| 现象 | 可能原因 |
|---|---|
| Hit Rate 低 | embedding 烂 / chunking 烂 / query 太短 |
| Hit Rate 高但 Faithfulness 低 | LLM 没用 context，要改 prompt 强调 |
| Faithfulness 高但 Answer Correctness 低 | 检索到了但内容本身就错（数据质量问题）|
| Context Precision 低 | 召回多但不准，需要 reranker |
| 大多数指标都低 | 数据/任务本身有问题 |

## 一个朴素结论

> 不分开测的 RAG 优化都是瞎调参。
> Hit Rate + Faithfulness 这两个指标先建起来，**80% 问题都能定位**。
