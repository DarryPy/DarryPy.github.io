---
layout: post
title: Chain-of-Thought 与 Thinking 模式 — 让模型先想再答
date: 2026-05-13
tags: [AI, CoT, Reasoning, Prompt Engineering]
excerpt: 从最朴素的"Let's think step by step"，到 Claude Extended Thinking 和 o-series，推理模型的实战用法。
permalink: /posts/2026-05-13-chain-of-thought.html
---

## CoT 的起源很朴素

2022 年那篇出名的 CoT 论文核心发现只有一句：**在 prompt 末尾加"Let's think step by step"，复杂推理任务正确率显著上升**。

为什么？因为：

- LLM 是"自回归"生成的——它生成第一个 token 时还没"想"
- 直接让它说答案，等于让它**没想好就开口**
- 让它先输出推理过程，等于**给它一个"思考的纸面"**

到 2026 年，CoT 已经从一个 trick 演化成了多个分支。

## Zero-shot CoT

最简单的形式，prompt 里加一句让模型先推理：

```
问：一个农场有 12 只鸡和 5 头牛。它们一共有多少条腿？
答：让我们一步一步思考。
鸡有 2 条腿，12 只鸡共 24 条腿。
牛有 4 条腿，5 头牛共 20 条腿。
总共 24 + 20 = 44 条腿。
```

加这一句，复杂数学 / 多步逻辑任务的准确率往往能涨 10-30%。

## Few-shot CoT

更稳的做法是给 1-3 个示例，明确展示**你想要的思考方式**：

```
问：小明有 7 个苹果，给了小红 3 个，又买了 5 个，现在有几个？
思考：起点 7，减 3 等于 4，加 5 等于 9。
答：9

问：一个班 30 个学生，男生比女生多 4 人，问男女各几人？
思考：设女生 x 人，男生 x+4 人，总和 2x+4=30，所以 x=13。
答：女生 13 人，男生 17 人

问：{你的问题}
思考：
```

模型会模仿示例的推理风格，效果通常比 zero-shot 稳定。

## Extended Thinking / Reasoning Models

2024 年开始的新一代模型直接内置了"长思考"：

- **Claude 4.x extended thinking**：开启后模型在回答前先生成大段 `<thinking>` 内容，可以配置思考长度
- **OpenAI o-series（o1 / o3）**：内置 chain-of-thought，开发者看不到思考过程
- **Gemini 2.x thinking**：类似机制

调用方式（Claude 为例）：

```python
response = client.messages.create(
    model="claude-opus-4-7",
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "..."}]
)
```

实测：

- 复杂数学 / 代码 / 推理任务，开 thinking 能从"勉强能用"跳到"几乎全对"
- 简单任务上开 thinking 是浪费 token
- thinking 输出大概占总成本 30-60%（按 budget 算）

## 什么时候用 CoT / Thinking

**适合**：

- 多步推理（数学、逻辑、规划）
- 需要权衡多个因素的判断
- 代码生成 / 调试
- 复杂分析（"分析这份合同的风险点"）

**不适合**：

- 简单事实问答（"今天星期几"）
- 风格转换 / 改写
- 已经有 deterministic 流程的任务

简单任务上 CoT **反而可能让模型自我说服走偏**——这叫"过度推理"。

## Self-Consistency：多次采样投票

CoT 的升级版：

1. 让模型用 CoT 跑 5-10 次（temperature > 0）
2. 各次答案投票，多数获胜

适合**有标准答案**的任务（数学题、代码题）。
能再涨 5-15% 准确率，代价是成本 10x。
生产场景里一般只用在评估 / 关键决策点。

## 一些容易踩的坑

### 1. CoT 不是越长越好

研究多次表明，**推理过长会引入更多错误**。设个上限（budget_tokens 或 prompt 里写"用不超过 5 步推理"）。

### 2. CoT 输出别直接给用户

`<thinking>` 内容是给模型用的，不是给用户看的。给用户的应该是**最终结论 + 简短解释**，不是几百字的推理流水账。

### 3. Reasoning 模型不一定是最佳选择

o1 / o3 / extended thinking 模型贵且慢。
简单任务用 Sonnet 4.6 反而又快又准。
**让流量分级**：先用便宜模型，需要硬推理的 case 再升级。

### 4. CoT 不能弥补知识缺失

模型不知道某个事实，让它推理也推不出真相，反而会"看起来很有道理地编"。
这种 case 要走 RAG，不是堆 CoT。

## 一个朴素的判断

**CoT 解决的是"推理能力"问题，不是"知识"问题、不是"格式"问题、不是"风格"问题**。
诊断准了，再上工具。

实战推荐组合：

```
简单任务 → Sonnet 4.6 直接答（不用 CoT）
中等推理 → Sonnet 4.6 + zero-shot CoT
复杂推理 → Opus 4.7 + extended thinking（10k tokens budget）
关键决策 → 上面任一 + self-consistency（5 次投票）
```

按场景选，不要一刀切。
