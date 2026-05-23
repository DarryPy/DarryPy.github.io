---
layout: post
title: Temperature / Top-p / Top-k 调参 — 让模型该确定时确定、该发散时发散
date: 2026-02-04
topic: "Prompt 与推理"
tags: [AI, Sampling, Parameters]
excerpt: 一堆采样参数不知道怎么调？分类用 0.0，创意用 0.7-1.0，按任务给出明确推荐值。
permalink: /posts/2026-02-04-temperature-sampling.html
---

## 三个核心参数

LLM 输出每个 token 时，从概率分布里抽。3 个参数控制这个过程：

| 参数 | 作用 | 调高 = | 调低 = |
|---|---|---|---|
| **temperature** | 概率分布的"陡峭度" | 更随机 / 多样 | 更确定 / 趋同 |
| **top_p** | 从累计概率 top_p 内抽（nucleus） | 词表更大 | 词表更小 |
| **top_k** | 只从概率最高的 k 个抽 | 词表更大 | 词表更小 |

## 任务推荐值

| 任务 | temperature | top_p | top_k |
|---|---|---|---|
| 分类 / 抽取 / 路由 | 0.0 | — | — |
| 严肃问答 / 摘要 | 0.0-0.3 | 0.9 | 50 |
| 代码生成 | 0.0-0.2 | 0.95 | — |
| 推理 / 数学 | 0.0 | — | — |
| 通用聊天 | 0.5-0.7 | 0.9 | 50 |
| 创意写作 | 0.7-1.0 | 0.95 | — |
| 头脑风暴（要多样性）| 0.9-1.2 | 0.95 | — |
| Self-Consistency 多次采样 | 0.7 | — | — |

## 经典误用

### 1. 严肃任务也开 0.7

```python
# ❌ 信息抽取也开 temp=0.7
client.complete("从下面文本提取人名和日期：...", temperature=0.7)
```

每次结果不一样、字段顺序乱、偶尔多 / 少字段。
**抽取任务用 0.0**。

### 2. 创意任务用 0.0

```python
# ❌ 写小说也用 0.0
client.complete("写一个关于太空旅行的短篇小说", temperature=0.0)
```

每次都是同一个开头，干瘪没灵感。**创意用 0.7-1.0**。

### 3. 同时调高 temp 和 top_p

```python
# ⚠️ 容易产生 garbage
temperature=1.5, top_p=1.0
```

太宽松，输出可能跑飞。**调温度时把 top_p 收紧**（≤0.95）；调 top_p 时温度别太高。

## temperature 跟 top_p 联动

通常**只调一个**：

- 调 temperature：保持 top_p=1.0（默认）
- 调 top_p：保持 temperature=1.0

两个都调容易产生意外结果。

## seed 参数：复现实验

OpenAI / Anthropic 等支持 `seed` 参数：

```python
response = client.complete(prompt, temperature=0.7, seed=42)
```

**注意**：seed 只是"尽力而为"复现——同一参数同一 prompt 大概率得相同结果，但不保证 100%（底层硬件 / batch size 微小变化都可能让结果不同）。

调试用 seed 很有用。生产不要依赖严格复现。

## 怎么选

按这个简单决策树：

```
任务有标准答案？
├── 有 → temperature = 0.0
└── 没有
     └── 想要稳定输出？
          ├── 是 → temperature = 0.3-0.5
          └── 否 → temperature = 0.7-1.0
```

复杂场景再加 top_p / top_k 微调，**80% 任务只调 temperature 就够**。

## 实战常见配方

```python
# 信息抽取 / 分类
{"temperature": 0.0, "max_tokens": 200}

# 通用问答
{"temperature": 0.3, "top_p": 0.95}

# 代码生成
{"temperature": 0.0, "max_tokens": 2000}

# 创意写作
{"temperature": 0.8, "top_p": 0.95, "presence_penalty": 0.5}

# Self-consistency 多采样
{"temperature": 0.7, "n": 5}
```

## 一个朴素结论

> 参数调对，效果直接换一档；调错，模型表现得像换了个差模型。
> **大多数生产场景该用 0.0**——稳定可预测才是工程价值。
> 创意 / 多样性场景再开温度。
