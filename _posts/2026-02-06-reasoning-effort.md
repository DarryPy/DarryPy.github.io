---
layout: post
title: Reasoning Effort 控制 — 让模型按需"用力想"
date: 2026-02-06
topic: "Prompt 与推理"
tags: [AI, Reasoning, Thinking]
excerpt: o-series 和 Claude extended thinking 都支持调节"推理预算"。怎么按任务复杂度分级、用对量。
permalink: /posts/2026-02-06-reasoning-effort.html
---

## 推理预算控制

主流推理模型 2026 年都暴露了"用多少 token 思考"的参数：

- **OpenAI o-series**：`reasoning_effort: "low" | "medium" | "high"`
- **Claude 4 thinking**：`thinking.budget_tokens: 1024 ~ 64000`
- **Gemini 2.x thinking**：`thinking_config.budget_tokens`

更高预算 = 更准的推理，但**更贵 + 更慢**。

## 不同任务的甜区

| 任务类型 | 推荐 budget | 理由 |
|---|---|---|
| 简单问答 / 闲聊 | 0（不开 thinking） | 浪费 |
| 信息抽取 / 分类 | 1024-2048 | 中等推理 |
| 数学题 / 代码生成 | 4000-8000 | 多步推理刚好 |
| 复杂规划 / 多步推理 | 16000-32000 | 需要充分展开 |
| 极复杂研究 / 证明 | 64000+ | 投入产出比降低 |

## 实测涨幅

GSM8K（小学数学）：
- 无 thinking: ~78%
- 4k thinking: ~92%
- 16k thinking: ~95%
- 32k thinking: ~95.5%（边际递减明显）

简单任务上加 thinking **可能让模型过度推理反而走偏**。

## 路由分级

生产场景按复杂度路由不同 budget：

```python
def estimate_reasoning_budget(query):
    # 简单分类：直接答
    if is_simple_question(query):
        return 0
    # 中等：数学 / 代码 / 推理
    if has_math_or_code(query):
        return 4000
    # 复杂：多步分析
    if requires_planning(query):
        return 16000
    return 8000  # default
```

跑分类的小模型成本 < 1% 总成本，但能让贵的 thinking 模型只在该用的时候用。

## 用 reasoning 的几个工程要点

### 1. thinking 输出不要给用户

```python
response = client.messages.create(
    model="claude-opus-4-7",
    thinking={"type": "enabled", "budget_tokens": 8000},
    messages=[...]
)
# response.content 里同时有 thinking blocks 和 text blocks
# 只给用户看 text，thinking 仅用于内部
visible = [b for b in response.content if b.type == "text"]
```

### 2. thinking 不能跨轮共享

每轮 thinking 是独立生成的。多轮对话场景下 thinking 不会累计——
但每轮可以独立选预算。

### 3. budget 是上限不是目标

模型可能用 budget 的 30% 就够了。**bill 按实际用量算**，不是按上限。

### 4. thinking 跟 tool 联动

Claude 4 thinking 跟 tool use 配合很好——边想边规划工具调用。
比纯 ReAct 更高效（一次 thinking 决定多个 tool calls）。

## 一份分级建议

```python
TASK_BUDGETS = {
    "simple_qa": 0,
    "summarization": 2000,
    "classification": 1024,
    "extraction": 2048,
    "code_simple": 4000,
    "code_complex": 16000,
    "math_easy": 4000,
    "math_olympiad": 32000,
    "planning_simple": 8000,
    "planning_complex": 32000,
    "research": 64000,
}
```

按业务调，**不要一刀切**。

## 一个朴素结论

> 推理预算是新一代模型的核心调节钮。
> 用太低：复杂任务答错；用太高：简单任务浪费 + 慢。
>
> 按任务分级路由不同 budget，是生产 AI 应用的基本功。
