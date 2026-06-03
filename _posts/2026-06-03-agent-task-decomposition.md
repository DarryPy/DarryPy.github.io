---
layout: post
title: Agent 任务拆解的五种模式 🗂️
date: 2026-06-03
topic: "Agent 与工具"
tags: [Agent, 任务拆解, LLM, 工程实践, 多智能体]
excerpt: 一个复杂目标摆在 Agent 面前，如何拆、拆多细、谁来拆——这三个问题决定了你的 Agent 是聪明的流水线还是无头的循环机器。本文梳理五种任务拆解模式，带代码带对比表，帮你选对方案少走弯路。
permalink: /posts/2026-06-03-agent-task-decomposition.html
---

你让 Agent "帮我整理本周所有 GitHub PR 的评审意见，生成一份改进报告"。

这句话对人类来说清晰明了，但对 Agent 来说是一个没有边界的请求——它需要先知道有哪些 PR，再逐一拉取评审，再提炼关键意见，最后结构化输出。每一步都是独立的工具调用，彼此有依赖，部分可以并行，失败了还需要局部重试。这就是任务拆解要解决的核心问题：把一个模糊的大目标，变成有序、可执行、可恢复的子任务集合。

拆得好，Agent 是流水线工厂，每个子任务职责清晰、边界分明，失败了局部重试，成功了直接接力；拆得差，Agent 是没有调度器的多线程程序——乱、慢、容易在某个中间步骤死循环，而你完全不知道它卡在哪里。

任务拆解不是 LLM 会自动做好的事情，它是你作为工程师的架构决策。你选择了顺序还是并行，选择了几层分层，就决定了整个系统的可扩展性和可调试性。以下五种模式覆盖了绝大多数实际场景，理解它们的适用边界，才能在面对新需求时做出正确判断。

## 顺序拆解：最简单，也最容易被滥用

顺序拆解是最直观的模式：子任务 A 的输出作为 B 的输入，B 的输出作为 C 的输入，严格按顺序一步步执行。整个流程像一条流水管道，前后有明确的数据依赖。

```python
pipeline = [
    {"tool": "list_prs",      "input": {"repo": "myrepo", "since": "7d"}},
    {"tool": "fetch_reviews", "input": "$prev.pr_ids"},
    {"tool": "summarize",     "input": "$prev.reviews"},
]
context = {}
for step in pipeline:
    resolved = resolve_refs(step["input"], context)
    context = agent.run_tool(step["tool"], resolved)
```

适合的场景是步骤之间存在硬依赖、当前步骤必须等上一步完成才能知道输入参数的情况。比如拉 PR 列表之前根本不知道 PR ID，顺序在这里是必然选择，没有任何并行化的空间。

问题在于它天生是串行的——任何一步卡住，整条链路等待。很多初学者会把所有任务都写成顺序拆解，这是最常见的性能杀手。实际工程里，顺序拆解只用在确实存在数据流依赖的地方，其他情况优先考虑并行。

## 并行拆解：速度提升的关键，但有隐患

同一批次的子任务之间没有数据依赖时，完全可以同时跑，最后汇总结果再进行下一步。这是把时间复杂度从 O(n) 压缩到 O(1) 的核心手段。

```python
import asyncio

async def process_all_prs(pr_list: list[str]):
    semaphore = asyncio.Semaphore(5)  # 最多同时跑 5 个

    async def fetch_one(pr_id):
        async with semaphore:
            return await fetch_and_extract(pr_id)

    results = await asyncio.gather(
        *[fetch_one(pr_id) for pr_id in pr_list],
        return_exceptions=True
    )
    ok = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    return ok, failed
```

有两个地方必须注意。第一，`return_exceptions=True` 是关键，一个子任务失败绝不能取消其他正在运行的任务，否则一个偶发的超时会让整批工作全部白做。第二，并行并发数不能无限扩张——LLM API 有 rate limit，下游工具也有并发限制，`max_concurrent=5` 是比较安全的经验起点，视具体 API 配额调整。

并行拆解对吞吐量的提升是指数级的。10 个独立 PR 的评审，并行处理可以把总时间从 10 倍单次耗时压缩到接近 1 倍，延迟完全由最慢的那个子任务决定，而不是全部子任务叠加。

## 分层拆解：Multi-Agent 的标准拓扑

当任务足够复杂，单个 Agent 的 context 窗口装不下所有工具、所有中间状态时，就需要引入分层结构：主 Agent 负责规划与任务分发，子 Agent 负责具体执行，形成树状层级。

```
Orchestrator（规划层）
├── Sub-Agent A：拉取 PR 列表与元数据
├── Sub-Agent B：逐 PR 提炼评审意见
│   ├── Worker：PR #101 评审分析
│   ├── Worker：PR #102 评审分析
│   └── Worker：PR #103 评审分析
└── Sub-Agent C：汇总并生成最终改进报告
```

这种结构有几个显著优势。一是 context 隔离——主 Agent 只看到子任务的输入描述和输出摘要，不会被几百条评审原文撑爆上下文。二是模型成本可以按层级分配，规划层用推理能力强的 Opus 做任务分解，执行层用响应快、价格低的 Haiku 处理重复性工作，整体成本能降低 40%~60%。三是子 Agent 可以独立测试，不依赖整个链路联调。

分层拆解的架构约定很重要：主 Agent 不直接调用工具，只下达子任务指令并等待结果；子 Agent 不自己决定下一步该做什么，只执行分配到的明确任务。这个职责分离是整个系统可调试性的基础，一旦混用就很难排查是哪一层出了问题。

## 条件拆解：根据中间结果走不同分支

不是所有任务路径都是预先确定的。很多时候，下一步做什么取决于上一步的分类或判断结果，这就是条件拆解——在任务图中引入有向的条件边。

```python
# 先做分类
clf = agent.run_tool("classify_pr_type", {"pr_id": pr_id})

# 根据分类结果选择下一个工具
dispatch = {
    "bug_fix":    "analyze_regression_risk",
    "api_change": "check_breaking_compatibility",
    "feature":    "evaluate_scope_and_impact",
}
next_tool = dispatch.get(clf["type"], "generic_summarize")

# 带着分类上下文执行下一步
final = agent.run_tool(next_tool, {
    "pr_id": pr_id,
    "classification": clf,
    "confidence": clf["confidence"],
})
```

条件拆解让 Agent 具备了业务判断能力，而不只是机械执行固定步骤。但这里有一个高危点：LLM 作为分类器的误判率不可忽视，一旦走错分支，后续全部跑偏且往往不会报错，最终输出看起来合理但实际上基于错误的前提。

两个防护建议：在分类结果里要求 LLM 输出置信度字段，低于阈值时走保守的通用分支；在分叉节点记录完整日志，包括分类依据和置信度，方便事后审计。

## 递归拆解：最灵活，也最危险

当 Agent 在执行过程中发现某个子任务仍然复杂，无法直接用单一工具完成，就需要对它再次拆解，直到任务粒度足够细可以原子执行。代码研究类 Agent、网络爬取类 Agent 经常用到这个模式。

```python
MAX_DEPTH = 4

def decompose_and_run(task: dict, depth: int = 0) -> dict:
    if depth >= MAX_DEPTH:
        return agent.execute_directly(task)

    subtasks = agent.plan(task)      # LLM 生成子任务列表

    if len(subtasks) <= 1:           # 无法继续拆解，直接执行
        return agent.execute_directly(task)

    results = [decompose_and_run(t, depth + 1) for t in subtasks]
    return agent.merge_results(results)
```

递归拆解的灵活性来自它的自适应能力——同一框架可以处理简单任务（一层即终止）和极复杂任务（多层递归）。但危险也在这里：一个措辞模糊的开放性任务，在没有硬性约束的情况下会让 Agent 无限展开，烧掉远超预期的 token 和时间。

硬约束必须是两层的：深度上限防止垂直爆炸，总 token 预算或总工具调用次数上限防止横向爆炸。两个约束缺一不可，只设深度而不限总量，Agent 可以在每一层都产出大量并行子任务，最终总量依然失控。

## 颗粒度怎么选

| 维度 | 粒度太粗 | 粒度太细 |
|---|---|---|
| 单次失败代价 | 高，整块重跑 | 低，可局部重试 |
| Context 占用 | 每步上下文大 | 可按步骤截断 |
| 编排复杂度 | 低 | 高，依赖管理复杂 |
| 可调试性 | 差，中间状态不透明 | 好，每步有清晰日志 |
| Token 消耗 | 少 | 多，每步都有系统提示开销 |

经验法则：一个子任务对应一次工具调用，输入输出边界清晰。如果某个子任务需要两次以上的工具调用才能完成，它应该再拆一层；如果两个子任务总是同时成功或同时失败，合并成一个。

五种模式在实际项目里往往是混用的：顺序串联几个并行块，并行块内部有条件分支，某些深度复杂的分支触发有限递归。关键不是选哪一种，而是把拆解逻辑显式写出来、记录下来，而不是让 Agent 在运行时自由发挥，否则每次行为都不一样，根本没法调试。拆解结构是你设计的，不是 LLM 生成的，这个边界要始终清晰。

## 踩坑清单

- Planning prompt 要短：让 LLM 规划任务时，prompt 控制在 500 token 以内；过长的规划提示会让规划成本反超执行成本，得不偿失。
- 子任务必须带 ID 和父任务引用：任务量一上来，没有 ID 的日志根本无法回溯是哪条链路出问题，排查成本翻倍。
- 并行错误要隔离，不要级联取消：一个子任务超时不等于整批失败，先收集所有成功结果，再决定是否对失败项重试。
- Planner 和 Executor 必须分开：把规划逻辑和执行逻辑混在同一个 Agent 里是最难测试、最难迭代的设计，宁可多一层也要拆开。
- 递归必须双重限流：深度上限 + 总调用预算，缺任何一个都可能失控。
- 终止条件要面向结果，不要面向步骤：不是"跑完 N 步就停"，而是"当所有叶子任务都有了对应的执行工具时停止规划"——这样更健壮，也更容易验证。
