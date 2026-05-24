---
layout: post
title: Reflection / Reflexion Agent 实现 — 让 Agent 自我反思
date: 2026-03-18
topic: "Agent 与工具"
tags: [AI, Agent, Reflection]
excerpt: 单次生成往往有错，让 agent 把自己刚才的输出审一遍再改一遍，正确率能涨 15-25%。原理与实战。
permalink: /posts/2026-03-18-reflection-agent.html
---

## Reflection 的核心循环

```
Generate → Critique → Revise → ...
```

让同一个或另一个 LLM 把刚才的输出**挑刺**，然后基于挑刺**重写**。

跟人类工作流类似：写完先放一晚上，第二天读自己写的发现一堆错。Reflection 就是把这个"放一晚上"的反思压缩到几秒。

## 一个最小实现

```python
def reflection_loop(question, max_iters=3):
    answer = llm_generate(question)
    for i in range(max_iters):
        critique = llm_critique(question, answer)
        if "no issues" in critique.lower():
            return answer
        answer = llm_revise(question, answer, critique)
    return answer

def llm_critique(q, a):
    prompt = f"""
评估下面这个回答的问题。指出：
- 事实错误
- 逻辑漏洞
- 不完整的地方
- 格式问题

如果没问题，输出 "no issues"。

问题：{q}
回答：{a}
"""
    return llm.complete(prompt)
```

## 适合什么任务

| 任务 | Reflection 收益 |
|---|---|
| 代码生成 | +15-25%（编译错 / 边界 case 漏掉）|
| 数学题 | +10-20% |
| 复杂分析报告 | 写作质量明显提升 |
| 简单事实问答 | 几乎没收益 |
| 翻译 | 中等收益（修语法 / 风格）|

## Reflexion：Reflection + 记忆

Reflexion 在 Reflection 基础上加了**长期记忆**：
agent 把过去尝试中犯的错记下来，**下次遇到类似任务避免再犯**。

```
任务 → 尝试 → 评估 → 反思（记录 "我犯了 X 错"）
                                ↓
                          存到 memory bank
                                ↓
        下次类似任务，先取出相关 memory 作为额外 context
```

适合 agent 在同一任务族上反复跑（如代码自动修复任务连续多轮）。

## Critique 用单 agent 还是双 agent

| 模式 | 优势 | 劣势 |
|---|---|---|
| 单 agent self-critique | 简单 / 一个模型 | 模型容易给自己开绿灯 |
| 双 agent（generator + critic） | 角色清晰，critic 不留情 | 多一个模型调用 / 复杂 |

复杂任务推荐双 agent。critic 可以用同模型不同 system prompt，也可以用更强的模型评弱模型。

## 工程坑

1. **无限反思**：critic 永远能挑出小问题。**设 max_iters = 2-3**
2. **越改越糟**：critic 错了反而把对的改坏。**保留每轮版本，最后挑最好的**
3. **成本翻倍以上**：3 轮 reflection ≈ 4-7 倍 LLM 调用。不是无脑用
4. **critic 跟 generator 同模型不太够**：换个角度 / 加更具体 rubric

## 一个朴素结论

> Reflection 是"用 token 换质量"的典型工具。
> 输出质量要求高 + 错误代价大的场景再上；常规问答上是浪费。
