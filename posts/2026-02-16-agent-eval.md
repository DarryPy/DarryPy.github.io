---
layout: post
title: Agent Eval 方法论 — 多步骤任务怎么打分
date: 2026-02-16
topic: "评估与安全"
tags: [AI, Eval, Agent]
excerpt: 单轮答对答错好评，多步 agent 任务怎么测？任务完成率、效率、工具使用对错、成本，4 个维度撑起 agent eval。
permalink: /posts/2026-02-16-agent-eval.html
---

## Agent eval 跟 LLM eval 不同

LLM eval：input → output，看 output 对不对。
Agent eval：**有过程**——多步推理 + 工具调用 + 状态变化。每一步都可能出错。

要评的不只是"最终答案"：

- 走对路了吗（用对工具）？
- 走得快吗（步数）？
- 走得贵吗（成本）？
- 安全吗（没碰危险工具）？

## 4 个核心指标

### 1. Task Success Rate（任务完成率）

二元：任务做完了吗？
对于有明确目标的任务（"找出 X 公司的 CEO"），最直接。

```python
success = (final_answer matches expected) or (target_state reached)
```

复杂任务可以拆步骤：

```
成功率 = 完成步骤 / 总步骤
"找到公司" → ✓
"找到 CEO" → ✓  
"找到 CEO 邮箱" → ✗
得分: 2/3
```

### 2. Step Efficiency（步数效率）

完成同一任务，agent 用了几步？

```
理想步数：3（直接路径）
实际步数：8
效率分 = 3/8 = 0.375
```

低分意味着：要么 agent 在试错，要么 prompt 没让它直接想到正确路径。

### 3. Tool Usage Quality

每个工具调用单独评：

- **Tool Selection**：选对工具了吗
- **Argument Correctness**：参数填对了吗
- **Error Recovery**：失败时恢复对了吗

```python
for step in trace:
    if step.tool_used != ground_truth_tool[step.idx]:
        tool_selection_errors += 1
    if step.args != ground_truth_args[step.idx]:
        arg_errors += 1
```

### 4. Cost & Latency

成本和延迟也是 agent eval 的核心维度。
agent 完成任务的成本可能是简单 LLM 调用的 10-100 倍。

```
平均 LLM 调用：1
平均 task 调用：8
成本倍率：8x

平均 LLM 延迟：3s
平均 task 延迟：30s
延迟倍率：10x
```

## 怎么建 ground truth

Agent eval 的 ground truth 不只是"最终答案"，还要：

1. **理想步骤序列**（虽然 agent 可能用别的路径也行）
2. **必须用的工具集合**
3. **绝对不能用的工具**（如 delete / send_email）
4. **预算上限**（步数 / token / 时间）

由专家人工标注。难，但绕不开。

## 自动化测试集

经典 agent benchmark：
- **GAIA**：助手类任务（搜索 / 文件操作）
- **SWE-bench**：代码 agent（修真实 GitHub bug）
- **WebArena / VisualWebArena**：网页 agent
- **ToolBench**：工具调用准确性

跑这些 benchmark 比自建数据集快得多——对开发期评估好。
生产期还是要建自己业务的 golden dataset。

## Trace 才是 agent eval 的核心

跑完一个 task，把整个 trace 存下来：

```json
{
  "task_id": "...",
  "user_query": "...",
  "trace": [
    {"step": 1, "thought": "...", "tool": "search", "args": {...}, "result": "...", "duration_ms": 1200, "tokens": 543},
    {"step": 2, ...}
  ],
  "final_answer": "...",
  "total_duration_ms": 32000,
  "total_tokens": 12453,
  "cost_usd": 0.087
}
```

分析时按 trace 切：
- 哪一步耗时最久？
- 哪一步失败重试？
- 哪个工具调用率最高 / 最低？

## 调试工具

- **LangSmith / Langfuse**：trace 可视化 + 重放 + 评估
- **Phoenix**：开源 trace + eval
- **Inspect**：英国 AI 安全研究院的 agent eval 框架

不可视化 = 闭眼调 agent。

## 一个朴素结论

> Agent eval 不是单一分数，是**4 个维度的画像**：成功率 / 效率 / 工具质量 / 成本。
> 没监控这 4 个的 agent 上线 = 用户当 QA。
>
> 投产前跑 100 个 task 看分布，比上线后改 prompt 高效得多。
