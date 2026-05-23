---
layout: post
title: Tree of Thoughts (ToT) 详解 — 让模型在多条路上探索
date: 2026-03-26
topic: "Prompt 与推理"
tags: [AI, ToT, Reasoning]
excerpt: CoT 是单条推理链，ToT 是同时探索多条分支再回溯择优。复杂规划 / 数学题 / 创意任务上比 CoT 强一截。
permalink: /posts/2026-03-26-tree-of-thoughts.html
---

## ToT 跟 CoT 的核心区别

CoT：一条线推到底。中间出错只能将错就错。
ToT：每一步**生成 N 个候选 thought**，评估后选最优分支继续。错了就回溯到上一层换条路。

```
        Question
        /  |  \
   thought_A thought_B thought_C
      |        |        |
   评估     评估       评估
      ↓
   选 B 继续 → 再分 3 个候选 → ...
```

## 4 个核心组件

1. **Thought generator**：每步生成 N 个候选 thought（通常 3-5）
2. **State evaluator**：给每个候选打分（用 LLM 判断 "promising / sure / impossible"）
3. **Search algorithm**：BFS（每层都展开）或 DFS（深一层失败再回溯）
4. **Termination**：到达目标 / 深度上限 / 评估器说"完成"

## 一个伪代码骨架

```python
def tot_solve(question, depth=4, breadth=3):
    states = [question]
    for d in range(depth):
        candidates = []
        for s in states:
            # 每个状态生成 breadth 个候选 thought
            candidates += llm_propose(s, n=breadth)
        # 评估打分
        scored = [(c, llm_evaluate(c)) for c in candidates]
        # 保留 top-K
        states = [c for c, score in top_k(scored, k=breadth)]
        if any(is_solved(s) for s in states):
            return best(states)
    return best(states)
```

## 适合什么任务

| 任务 | CoT | ToT |
|---|---|---|
| 简单事实问答 | ✅ 够用 | ❌ 浪费 |
| 多步数学（GSM8K）| OK | 更准 |
| 复杂规划 / 24 点游戏 | 较差 | **明显更好** |
| 创意写作 | 一般 | 多分支可挑最好 |
| 代码生成 | OK | 复杂任务上更稳 |

## 工程坑

- **贵**：每层 × breadth 倍 LLM 调用，整体可能 10-100x
- **评估器自己也可能错**：评估器选错分支等于全错
- **延迟高**：BFS 串行慢，要并行 propose

## 一个朴素结论

> ToT 是"暴力搜索 + LLM 当评估器"的组合拳。
> 复杂推理任务必备工具；简单任务上是过度工程。
