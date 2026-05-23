---
layout: post
title: Plan-and-Execute Agent 实战 — 先想清楚再开始走
date: 2026-04-09
topic: "Agent 与工具"
tags: [AI, Agent, Planning]
excerpt: ReAct 是"边走边看"，Plan-and-Execute 是"先想后做"。复杂多步任务上后者更快、更便宜、更稳。
permalink: /posts/2026-04-09-plan-and-execute.html
---

## 跟 ReAct 的关键区别

ReAct：每一步都让 LLM 决定下一步——灵活但贵且慢。
Plan-and-Execute：**先用一次 LLM 调用规划整体计划**，然后按计划执行——快且省。

```
[规划阶段]
LLM 一次性输出：
  Step 1: 查用户 A 的当月订单
  Step 2: 查用户 B 的当月订单
  Step 3: 计算两人订单金额差
  Step 4: 给出结论

[执行阶段]
顺序执行 step 1-4，每一步可以用小模型 / 直接代码
最后一步可能再回到 LLM 总结
```

## 优势

按 2026 的研究数据：

- **完成率**：复杂多步任务 92%（ReAct 约 75%）
- **延迟**：相比 ReAct **快 3.6 倍**（少了多轮 LLM 调用）
- **成本**：低 50-70%（规划阶段一次 LLM 调用，执行阶段可用更便宜的模型/代码）

## 一个最小实现

```python
def plan(query):
    """规划阶段：一次性输出整体计划"""
    prompt = f"""
你是一个任务规划器。把下面的任务拆成 1-7 个可顺序执行的 step。
每个 step 用一句话描述要做什么，可以是工具调用或推理。
输出 JSON 数组：[{{"step": 1, "action": "..."}}, ...]

任务：{query}
"""
    return llm.complete(prompt, response_format="json")

def execute(plan, query):
    """执行阶段"""
    results = []
    for step in plan:
        # 把累计结果作为上下文喂给当前 step
        ctx = {"query": query, "results_so_far": results}
        result = execute_step(step, ctx)
        results.append(result)
    return results[-1]  # 最后一步通常是总结

def run(query):
    p = plan(query)
    return execute(p, query)
```

## Replan 机制 — Plan-Execute 的关键能力

简单的 Plan-Execute 有个致命缺点：**计划一旦错了，全错**。

解决：执行某步失败时，**回到规划阶段重新规划**：

```python
def execute_with_replan(plan, query, max_replans=2):
    for replan in range(max_replans + 1):
        try:
            return execute(plan, query)
        except StepFailure as e:
            if replan == max_replans:
                raise
            plan = replan_from_failure(plan, e, query)
```

这种"Plan-Execute-Replan"循环比纯 Plan-Execute 鲁棒得多，且仍比 ReAct 便宜。

## 什么时候选 Plan-Execute

| 场景 | 推荐 |
|---|---|
| 任务步骤结构清晰（数据分析 / 报告生成）| ✅ Plan-Execute |
| 任务依赖上一步的实时反馈 | ❌ 用 ReAct |
| 步骤可并行（多源数据汇总）| ✅ Plan-Execute（还能并行执行） |
| 实时交互 / 用户中途插话 | ❌ 用 ReAct |
| 成本敏感、规模大 | ✅ Plan-Execute |

## 进阶：并行执行

Plan 输出的步骤之间如果**没有依赖**，可以并行：

```
[Plan]
Step 1: 查 A 的订单（依赖：无）
Step 2: 查 B 的订单（依赖：无）
Step 3: 计算差额（依赖：Step 1, Step 2）

[Execute]
并行执行 Step 1 + Step 2
完成后执行 Step 3
```

这种 DAG 式执行是 LangGraph、Mastra 这类框架的强项。

## 工程上的几个坑

### 1. Plan 别让 LLM 写代码

Plan 应该是**自然语言或结构化 step**，不要让 LLM 直接生成 Python 代码（除非你有沙箱）。
Code 难审计、难调试、容易注入。

### 2. Step 描述要够具体

❌ "Step 2: 处理数据"
✅ "Step 2: 调 calculate_user_clv(user_id=X) 工具，输入是 Step 1 返回的 user_id"

模糊的 step 描述会让执行阶段又退化成小型 ReAct。

### 3. 累计结果不要全塞下游

如果 step 多，每步结果都塞给后面的 step，context 会爆。
**摘要 + 引用**：把详细结果存到一个 key-value store，下游 step 只看摘要 + 可按需查询。

### 4. 失败要传上下文

replan 时不仅要告诉 LLM"上一次失败了"，还要告诉**为什么失败、错在哪一步、还有多少预算**。

## 一个朴素结论

> ReAct 像探险，Plan-Execute 像项目管理。
> 任务可规划性高，用 Plan-Execute；变数大，用 ReAct；都不沾的话，先用 Sequential Workflow。

不要无脑用 ReAct——很多场景 Plan-Execute 又快又准又便宜。
