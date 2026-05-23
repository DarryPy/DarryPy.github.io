---
layout: post
title: LangChain vs LangGraph vs 原生 SDK — 框架选型实战
date: 2026-04-29
tags: [AI, LangChain, LangGraph, 框架]
excerpt: LangChain 还能用吗？LangGraph 和原生 SDK 怎么选？什么时候必须上框架、什么时候反而是负担。
permalink: /posts/2026-04-29-langchain-vs-langgraph.html
---

## 一份直白的现状

2024 年大家都在骂 LangChain"抽象漏水、版本破坏多、调试难"。
2026 年的现实是：

- **LangChain** 依然是入门首选，但社区已经把核心抽象稳定下来了
- **LangGraph** 是 LangChain 团队主推的 agent 框架，地位类似 React 之于 jQuery
- **原生 SDK**（`anthropic` / `openai` / `mistralai`）越来越好用，简单场景反而比框架轻

下面分别说清楚什么时候用什么。

## LangChain — 当胶水用

LangChain 的最佳定位是**多模型 / 多工具的胶水层**。

适合用的场景：

- 你的应用要同时跑 Claude / GPT / Gemini / 本地模型，不想 if/else 一堆
- 用现成的 Retriever / VectorStore / DocumentLoader 等组件
- 写**chain**风格的简单流水线：`prompt → llm → output_parser`

LangChain 的核心抽象现在稳定下来后还是好用的：

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatAnthropic(model="claude-opus-4-7")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 SQL 专家。"),
    ("human", "请把这个需求转成 SQL：{requirement}"),
])
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"requirement": "找出 5 月销量 top 10 的商品"})
```

**优点**：写起来直观，组件能复用。
**不适合**：复杂 agent 流（多步、循环、条件分支、并发）——拿 chain 表达就费劲。

## LangGraph — 复杂 Agent 的事实标准

复杂 agent 系统**LangGraph 比 LangChain 强一个量级**。
它把 agent 流程显式建模成**有向图**：节点是逻辑步骤，边是状态转换。

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List[dict]
    next_step: str

def call_model(state):
    # 调 LLM
    return {"messages": [...]}

def call_tool(state):
    # 调工具
    return {"messages": [...]}

def should_continue(state):
    last = state["messages"][-1]
    return "call_tool" if last.get("tool_calls") else END

graph = StateGraph(AgentState)
graph.add_node("model", call_model)
graph.add_node("tool", call_tool)
graph.set_entry_point("model")
graph.add_conditional_edges("model", should_continue, {"call_tool": "tool", END: END})
graph.add_edge("tool", "model")
app = graph.compile()
```

**优点**：

- 显式状态机，每一步都可视化、可追踪、可重放
- 支持 checkpoint / 断点续传 / human-in-the-loop
- 并行节点、子图、流式输出
- 跟 LangSmith 集成的 observability 很完整

**缺点**：

- 学习曲线偏陡（要理解 reducer、state 合并、checkpoint）
- 简单任务上 overkill

## 原生 SDK — 简单场景最爽

如果你只用一个模型 + 一两个工具，**原生 SDK 比框架还顺**：

```python
from anthropic import Anthropic
client = Anthropic()

def chat(messages, tools=None):
    while True:
        resp = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=2048,
            messages=messages,
            tools=tools or [],
        )
        if resp.stop_reason == "end_turn":
            return resp
        # 处理 tool_use
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = [run_tool(tu) for tu in resp.content if tu.type == "tool_use"]
        messages.append({"role": "user", "content": tool_results})
```

**优点**：
- 零额外依赖
- 调试简单（没有黑盒）
- 性能最好（没有框架开销）

**缺点**：
- 多模型切换得自己写适配
- 复用 RAG / agent 组件得自己造

## 决策矩阵

| 场景 | 推荐 |
|---|---|
| 一个简单 chat / 一个分类 prompt | 原生 SDK |
| RAG，但流程线性 | LangChain（用现成 retriever） |
| 多模型 / 多供应商抽象 | LangChain 或 LiteLLM |
| 复杂 agent（多步、循环、人工审批） | LangGraph |
| 多 agent 协同（manager + workers） | LangGraph 或 CrewAI |
| 极致性能 / 生产关键路径 | 原生 SDK |
| 想要 observability + tracing | LangChain/LangGraph + LangSmith |

## 其他值得知道的框架

- **LiteLLM**：纯供应商抽象层，比 LangChain 轻。只想"一个客户端打所有模型"用它就够
- **CrewAI**：multi-agent 的另一种风格，role-based，比 LangGraph 更高层但灵活度低
- **Mastra**（TS 生态）：TypeScript 版的 LangGraph 思路
- **Pydantic AI**：Pydantic 团队出的，强类型 + 轻量
- **Inspect**（AI 安全/eval 框架）：评估专用

## 一个朴素的原则

> 复杂度只有不可避免时才引入。

太多团队一上来就 LangChain + LangGraph + Pinecone + LangSmith 全套堆上去，结果发现 80% 的代码在跟框架斗争。

正确顺序：

1. **原生 SDK 写一个能跑的版本**（验证场景）
2. **遇到痛点再加抽象**（多模型？加 LiteLLM；复杂流程？换 LangGraph）
3. **生产前补 observability**（LangSmith / Phoenix / Langfuse）

别让框架先于场景存在。
