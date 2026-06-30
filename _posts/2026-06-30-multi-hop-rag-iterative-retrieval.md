---
layout: post
title: Multi-hop RAG 实战 — 拆解复杂问题的迭代检索策略
date: 2026-06-30
topic: "RAG 与检索"
tags: [RAG, Multi-hop, LangGraph, 检索, 推理]
excerpt: 单次检索解不了跨跳问题，Multi-hop RAG 让每一次检索结果成为下一次查询的输入，通过迭代推理逐步逼近复杂问题的答案。本文拆解三种主流实现模式并给出 LangGraph 可跑代码。
permalink: /posts/2026-06-30-multi-hop-rag-iterative-retrieval.html
---

你打了一条查询进去，向量库吐出 Top-K 文档，LLM 综合回答——这条流水线在 80% 的场景里运转良好。直到你遇到这类问题：

"A 公司的 CEO 上个月在哪个城市签了合同，那个城市的主要竞争对手是谁？"

这里藏着两个跳跃：先找 CEO 和城市，再用城市找竞争对手。单次检索只能命中其中一跳，另一跳的上下文根本不在返回的文档里，LLM 要么胡编，要么回答残缺。这就是 Multi-hop RAG 要解决的核心问题：**让检索本身具备迭代推理能力，每一跳的结果成为下一跳的输入。**

## 什么是 Multi-hop RAG 🔁

Multi-hop RAG 不是单次 retrieve-then-read，而是一个循环：

1. 分解原始问题，提炼出第一跳子问题
2. 检索并阅读，得到中间答案或中间上下文
3. 用中间上下文补全下一跳查询
4. 重复直到能回答原始问题，或达到跳数上限

与普通 RAG 的本质差异：**查询是动态生成的，而不是静态传入的。** 这一点看起来微小，但意味着整个检索链路必须有状态，中间结果必须被维护和传递。

Multi-hop 并不是 LLM 时代的新概念——传统 QA 系统里的多步推理早已存在，但 LLM 让这套流程可以用自然语言驱动，不再需要手写规则链和实体抽取器。

## 为什么单次检索不够用

单次检索失败的根本原因是**查询语义与文档分布的错位**。当问题跨越两个以上独立知识点时，一次向量搜索很难同时命中所有相关文档——它会选择语义最近的那一堆，把另一跳的文档留在排名之外。

另一个隐藏问题是**上下文缺失导致的 LLM 幻觉**。模型在缺少关键中间事实时不会诚实地说"我不知道"，而是倾向于用训练数据里的参数知识填补空白，这正是 RAG 最常见的失控场景之一。

多跳检索从根本上规避这个问题：每一跳都是有据可查的子问题，LLM 只在有文档支撑的情况下推理，幻觉概率大幅降低。

## 三种主流实现模式

**模式 1：IRCoT（交替推理与检索）**

LLM 每输出一步 Chain-of-Thought，就检索一次相关文档，再把文档追加到上下文继续推理，直到给出最终答案。实现简单，与现有 RAG pipeline 兼容。但上下文随跳数线性膨胀，三跳以上容易超出窗口，适合跳数固定且较少的场景。

**模式 2：ReAct + 检索工具**

把检索变成一个 tool，LLM 通过 `Thought → Action(search) → Observation` 循环自主决定何时检索、检索什么。

```python
tools = [
    Tool(
        name="search",
        func=vector_store.similarity_search,
        description="检索知识库，输入为子问题字符串，返回相关段落列表",
    ),
]
agent = create_react_agent(llm, tools, prompt)
result = agent.invoke({"input": user_question})
```

LLM 自主决策跳数，不用硬编码几跳，灵活性最强。缺点是 token 消耗大，debug 困难，也容易陷入无效循环，必须加跳数上限硬截断。

**模式 3：Query Decomposition + 并行检索**

先用 LLM 把原始问题分解为 N 个独立子问题，并行检索各自答案，最后合并生成。延迟可控，适合子问题相互独立的场景。但当子问题之间存在顺序依赖时——"第二跳的查询需要第一跳的结果"——这个模式直接失效。

三种模式选哪个，先画子问题的依赖图：无依赖边用并行分解，有依赖边用 IRCoT 或 ReAct。

## LangGraph 实现：顺序依赖型多跳检索

当子问题存在顺序依赖，用 LangGraph 构建状态机是目前最清晰的方案，每跳的上下文自然流转：

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class HopState(TypedDict):
    question: str
    sub_questions: List[str]
    contexts: List[str]
    hop_count: int
    final_answer: str

def decompose_node(state: HopState):
    prompt = (
        "将以下问题拆解为需顺序回答的子问题列表（最多3个，JSON数组格式）：\n"
        + state["question"]
    )
    import json
    raw = llm.invoke(prompt).content
    sub_qs = json.loads(raw)
    return {"sub_questions": sub_qs, "hop_count": 0}

def retrieve_node(state: HopState):
    hop = state["hop_count"]
    query = state["sub_questions"][hop]

    # 将前序上下文摘要注入查询，增强语义连贯性
    if state["contexts"]:
        prefix = state["contexts"][-1][:200]
        query = f"{prefix} {query}"

    docs = vector_store.similarity_search(query, k=4)
    ctx = "\n".join(d.page_content for d in docs)
    return {
        "contexts": state["contexts"] + [ctx],
        "hop_count": hop + 1,
    }

def should_continue(state: HopState):
    if state["hop_count"] < len(state["sub_questions"]):
        return "retrieve"
    return "synthesize"

def synthesize_node(state: HopState):
    combined = "\n\n---\n\n".join(state["contexts"])
    answer = llm.invoke(
        f"基于以下多段检索结果，回答原始问题：{state['question']}\n\n{combined}"
    ).content
    return {"final_answer": answer}

graph = StateGraph(HopState)
graph.add_node("decompose", decompose_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("synthesize", synthesize_node)
graph.set_entry_point("decompose")
graph.add_edge("decompose", "retrieve")
graph.add_conditional_edges("retrieve", should_continue)
graph.add_edge("synthesize", END)

chain = graph.compile()
result = chain.invoke({"question": user_question, "sub_questions": [], "contexts": [], "hop_count": 0, "final_answer": ""})
```

`retrieve_node` 里把前一跳摘要的前 200 字前缀追加到当前查询，这是防止第二跳向量搜索跑偏最简单有效的手段。子问题分解用 JSON 数组而不是自由文本，省去解析歧义。

## 踩坑清单

- **跳数别超过 4**：每多一跳引入一次 LLM 调用加检索延迟，实测 3 跳以上 p95 延迟常破 15 秒，用户体验直线下滑。
- **必须加跳数硬截断**：ReAct 模式下 LLM 有时会死循环检索同一类文档，`max_hops=5` 是保险栓，不要省。
- **中间答案只取前 200 字**：不要把整段 context 全量塞入下一跳 prompt，否则上下文雪球越滚越大，三跳就撑破窗口。
- **子问题分解用 structured output**：让 LLM 返回 JSON 数组，配合 Pydantic 验证，解析失败直接抛异常而不是静默跳过。
- **评估要分跳打分**：整体答案对不代表每跳检索都对，用 RAGAS 的 `context_precision` 分别评每跳，才能定位到底哪一跳在拖后腿。
- **并行 vs 顺序看依赖图**：先画出子问题的 DAG，有依赖边的顺序执行，无依赖边的并行检索，不要一刀切地串行。

Multi-hop RAG 的核心洞察只有一句话：**检索不是查询的终点，而是推理链上的中间站。**
