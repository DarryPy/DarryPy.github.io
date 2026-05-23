---
layout: post
title: ReAct 模式详细拆解 — 让 Agent "想一步、走一步"
date: 2026-04-11
topic: "Agent 与工具"
tags: [AI, Agent, ReAct]
excerpt: ReAct 是 2026 年最常用的 agent 模式。从最朴素的 Thought-Action-Observation 循环到生产可用的工程实现。
permalink: /posts/2026-04-11-react-pattern-deep.html
---

## ReAct 的核心结构

ReAct = Reasoning + Acting，2022 年提出，到 2026 年还是 agent 设计的基础范式。

它的循环只有 3 步：

```
Thought（想）：当前该做什么？
Action（做）：调用某个工具
Observation（看）：拿到工具结果
↓ 回到 Thought，再想下一步
```

终止条件：模型在某个 Thought 里说"我已经知道答案了"，输出 Final Answer。

## 一个真实的 trace

```
User: 上海今天比北京气温高几度？

Thought: 我需要先查上海今天的气温，再查北京的。
Action: get_weather(city="上海")
Observation: 上海今日温度 24°C

Thought: 拿到上海了。现在查北京。
Action: get_weather(city="北京")
Observation: 北京今日温度 18°C

Thought: 24 - 18 = 6。可以回答了。
Final Answer: 上海今天比北京高 6 度。
```

整个过程**每一步都被显式记录**，方便审计、debug、replay。

## 工程实现：最小可用版

```python
def run_react(query, tools, model, max_steps=10):
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    for step in range(max_steps):
        resp = model.complete(messages=messages, tools=tools)
        if resp.stop_reason == "end_turn":
            return resp.content
        # 处理 tool_use
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tool_use in resp.tool_uses:
            try:
                result = execute_tool(tool_use.name, tool_use.input)
            except Exception as e:
                result = {"error": str(e)}
            results.append({"tool_use_id": tool_use.id, "content": result})
        messages.append({"role": "user", "content": results})
    raise RuntimeError("max_steps exceeded")
```

## 让 ReAct 真正能用的几个工程点

### 1. max_steps 必须设

模型可能陷入循环（重复调同一个工具问同一个问题）。
**生产中 max_steps 一般 5-15**。超了直接报错或降级回答。

### 2. context 会爆炸

每一轮的 Thought/Action/Observation 都堆进 context。
跑 8 轮就可能 20k token。两个应对：

- **摘要压缩**：每 5 轮把老的 Thought 总结成一段
- **工具结果剪裁**：工具返回的 JSON 别全塞，只留 agent 需要的字段

### 3. 错误恢复

工具失败时，**不要扔异常**给 LLM，要返回结构化错误：

```json
{
  "error": "USER_NOT_FOUND",
  "message": "user_id 'abc' 不存在",
  "hint": "可以用 search_user_by_email 反查"
}
```

agent 看到 hint 就知道怎么调整下一步。

### 4. 并行 Tool Calls

模型支持时（Claude 4 / GPT-4.5），一轮可以返回多个 tool_use。
**客户端要并发执行**，全部完成后再喂回模型。能省一半延迟。

## 什么时候 ReAct 不合适

ReAct 不是万能。这些场景用别的：

| 场景 | 更好的模式 |
|---|---|
| 流程固定（每次都 step1→2→3）| Sequential Workflow |
| 任务可整体规划 | Plan-and-Execute |
| 输出质量要求极高 | Reflection / Self-critique |
| 需要并行探索多条路径 | Tree of Thoughts |
| 长任务（>20 步）| 拆成子 agent 委派 |

ReAct 的甜点：**步骤数不固定、需要根据反馈决定下一步**的场景。

## 一些容易忽略的细节

### 1. system prompt 里**强制**模型先 Thought

不写的话有些模型会跳过 Thought 直接 Action，破坏可审计性。

### 2. Final Answer 加显式标记

```
Final Answer: <answer>...</answer>
```

便于程序判断"输出完了，可以停"。

### 3. 工具结果给上下文，不是给用户

```
[工具结果]
get_weather 返回：{"temp": 24, "city": "上海"}
```

模型要基于这个用自然语言回答用户，**不要把 JSON 原文吐给用户**。
在 system prompt 里强调这一点。

## 一个朴素结论

> ReAct 是 agent 的"if/while"。
> 没了它，agent 就是个 prompt；有了它，agent 才能跟外部世界打交道。

生产用 ReAct，记得把 4 个工程点（max_steps / context 压缩 / 错误恢复 / 并行）都做了，
否则上线 3 天就会被各种边界 case 教做人。
