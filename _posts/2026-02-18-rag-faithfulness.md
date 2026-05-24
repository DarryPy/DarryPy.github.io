---
layout: post
title: RAG Faithfulness 评估 — 模型有没有"诚实"用 context
date: 2026-02-18
topic: "评估与安全"
tags: [AI, Eval, RAG, Faithfulness]
excerpt: 给了 context 但模型还是胡说？Faithfulness 是 RAG 最该监控的指标。怎么测、怎么提升。
permalink: /posts/2026-02-18-rag-faithfulness.html
---

## Faithfulness 是什么

RAG 的目的是让模型基于**检索到的片段**回答。
Faithfulness（忠实度）：**答案能在 context 里找到依据**的程度。

不 faithful 的征兆：
- Context 里没说，模型用训练数据答了
- Context 里说 A，模型答 B
- Context 部分可用，模型混入了脑补

**幻觉 = faithfulness 低**。是 RAG 最核心的失败模式。

## 怎么测

经典做法：LLM-as-Judge 拆解：

### 1. 把答案拆成"原子断言"

```
答案：马斯克生于 1971 年，是 SpaceX 的 CEO，曾就职于 PayPal。
↓
断言 1: 马斯克生于 1971 年
断言 2: 马斯克是 SpaceX 的 CEO
断言 3: 马斯克曾就职于 PayPal
```

### 2. 每个断言去 context 找依据

```python
for claim in claims:
    found = judge_llm(f"context 中能找到支持 '{claim}' 的内容吗？", context)
    if not found:
        unfaithful_count += 1

faithfulness = (total - unfaithful) / total
```

### 3. 整体打分

```
faithfulness = 1 - (不 faithful 的断言数 / 总断言数)
```

Ragas 库的 `faithfulness` 指标用的就是这个流程。

## 怎么提升

### 1. Prompt 里**强调** "must"

```
你必须只基于下面给定的 context 回答。
不要使用你的训练知识。
如果 context 不包含答案，明确说"context 中未找到"。
```

加这段后 faithfulness 通常涨 10-20%。

### 2. 让模型**引用 context**

强制模型答完每句话标来源：

```
答案：马斯克生于 1971 年 [来源: doc_1] 是 SpaceX 的 CEO [来源: doc_2]。
```

这种 attribution 不仅提高 faithfulness，**用户也更信任**。

### 3. 减少 context 噪声

不相关的 context 会让模型迷茫。
加 reranker 让 top-K 更准；过滤掉低相关度片段。

### 4. 用更强的 instruction-following 模型

Faithfulness 跟模型对"按指令做事"的能力强相关。
Claude / Opus 一般比 GPT mini / 开源小模型 faithfulness 高。

### 5. 检测 + 重试

```python
answer = llm_generate(query, context)
if check_faithfulness(answer, context) < 0.8:
    # 重试，加更强的 prompt
    answer = llm_generate(query, context, strict_mode=True)
```

## 真实生产监控

```
[每天]
- 采样 100 条 RAG 答案
- 跑 faithfulness 检测
- 按用户 / 接口 / 模型 切看

[告警]
- faithfulness < 0.7 → 告警
- 周环比掉 10% → 调查
```

不监控的话，模型悄悄变烂用户才反馈。

## 跟 Answer Correctness 的区别

| | Faithfulness | Answer Correctness |
|---|---|---|
| 测什么 | 答案是否基于 context | 答案是否真的对 |
| 需要 ground truth | ❌ | ✅ |
| 测的是什么问题 | "模型有没有用 context" | "答案是不是真理" |

Context 本身就是错的，Faithfulness 高但 Correctness 低——这种 case 也存在，要分开测。

## 一个朴素结论

> RAG 最该上的指标不是 retrieval hit rate，而是 **Faithfulness**。
> 因为 hit rate 高但 faithfulness 低 = "材料对了，但模型瞎说"，用户体验最糟。
>
> 监控好它，能挡住 80% 的 RAG 投诉。
