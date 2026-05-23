---
layout: post
title: Multi-Agent 框架对比 — CrewAI / AutoGen / Swarm / LangGraph
date: 2026-03-16
topic: "Agent 与工具"
tags: [AI, Multi-Agent, 框架]
excerpt: 多 agent 协作框架百花齐放。4 家主流框架横向对比，按场景选型。
permalink: /posts/2026-03-16-multi-agent-frameworks.html
---

## 4 家主流框架

| 框架 | 出品方 | 风格 | 学习曲线 |
|---|---|---|---|
| **CrewAI** | 第三方 | Role-based，类似剧组 | 低 |
| **AutoGen** | Microsoft | Conversation-based | 中 |
| **OpenAI Swarm** | OpenAI | 极简轻量，handoff 中心 | 低 |
| **LangGraph** | LangChain | State machine / graph | 中-高 |

## CrewAI

把 agent 想成"剧组角色"——researcher、writer、editor 各司其职。

```python
from crewai import Agent, Task, Crew

researcher = Agent(role="Researcher", goal="找最新 AI 新闻", llm="claude-opus-4-7")
writer = Agent(role="Writer", goal="写一份简报")

task1 = Task(description="搜本周 AI 大事", agent=researcher)
task2 = Task(description="基于搜索结果写 300 字简报", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
result = crew.kickoff()
```

**优势**：心智模型简单（角色 + 任务），上手 5 分钟。
**缺点**：复杂控制流（条件分支 / 循环）表达起来费劲。
**适合**：流程清晰的串行任务、研究 + 写作类。

## AutoGen

让多个 agent 在群聊里**对话讨论**直到达成共识。

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent(name="assistant", llm_config={"model": "gpt-4.5"})
user_proxy = UserProxyAgent(name="user", code_execution_config={"work_dir": "."})

user_proxy.initiate_chat(assistant, message="写个 fibonacci 函数并测试")
```

**优势**：群聊范式自然支持多 agent 协商、code execution 集成好。
**缺点**：对话能跑很长，控制不好成本爆。
**适合**：代码生成 + 自动执行、需要 agent 互相质疑的任务。

## OpenAI Swarm（极简轻量）

最朴素的 multi-agent：每个 agent 有自己的工具，通过**handoff 函数**互相传递。

```python
from swarm import Agent, Swarm

triage = Agent(name="Triage", instructions="判断用户问题属于哪类，转给对应 agent")
billing = Agent(name="Billing", instructions="处理账单问题")
tech = Agent(name="Tech", instructions="处理技术问题")

triage.functions = [lambda: billing, lambda: tech]  # handoff 函数

client = Swarm()
client.run(agent=triage, messages=[{"role":"user","content":"我账户被扣费了"}])
```

**优势**：超轻量，代码量小，可控性强。
**缺点**：复杂场景需要自己实现状态、记忆等。
**适合**：客服分流、tool routing、最小化抽象。

## LangGraph

把 agent 流程显式建模成**有向图**（状态机）。

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)
graph.add_node("research", researcher_node)
graph.add_node("write", writer_node)
graph.add_node("review", reviewer_node)
graph.set_entry_point("research")
graph.add_edge("research", "write")
graph.add_conditional_edges("write", check_quality, {"good": END, "needs_revision": "review"})
graph.add_edge("review", "write")
app = graph.compile()
```

**优势**：显式状态机 / 条件分支 / 并发 / checkpoint / human-in-the-loop，**控制力最强**。
**缺点**：学习曲线最陡。
**适合**：复杂生产 agent、需要完整 observability。

## 决策矩阵

| 场景 | 推荐 |
|---|---|
| 简单串行（搜索 → 写）| CrewAI |
| 代码生成 + 自动执行 | AutoGen |
| 客服分流 / tool routing | OpenAI Swarm |
| 复杂分支 / 并发 / 生产 | LangGraph |
| 想要最少抽象 | 原生 SDK + 自己写 |

## 一个朴素建议

> 先用原生 SDK 写一版能跑的，遇到痛点再加框架。
> 80% 的"multi-agent" 其实是 sequential workflow + tool calling，没必要上框架。

复杂度是负债，不是资产。
