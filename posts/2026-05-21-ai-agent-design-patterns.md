---
layout: post
title: AI Agent 七大设计模式 — ReAct / Plan-Execute / Reflection / Tool Use / Multi-Agent
date: 2026-05-21
category: "Agent 与工具"
tags: [AI, Agent, 架构, 设计模式]
excerpt: 2026 年最常见的 7 种 Agent 设计模式，分别适合什么场景、坑在哪、什么时候不要用。
permalink: /posts/2026-05-21-ai-agent-design-patterns.html
---

## 为什么要谈"模式"

LLM 自带的能力是**一问一答**，要让它做"自动化任务"——多步骤、调工具、有反馈循环——
就需要在它外面套一层框架。这一层就是**Agent**。

行业一年时间从"Agent 框架百花齐放"演化到现在，大家逐渐收敛到**几个稳定可复用的模式**。
下面这 7 个模式是 2026 年最常被提到的。理解它们，看到任何 agent 系统都能拆解出来。

## 1. ReAct（Reason + Act）

**核心循环**：

```
Thought → Action → Observation → Thought → Action → ...
```

模型先想"我下一步要做什么"，再执行（通常是调工具），观察结果，再想下一步。
一直循环到模型认为可以给最终答案。

**什么时候用**：
- 任务需要**调外部工具**（搜索、计算、查 API）
- 步骤数不固定，需要根据反馈动态决定下一步
- 需要**可审计**：每一步的 thought 都是显式记录，方便调试

**坑**：
- 模型容易陷入循环（重复调同一个工具）
- 长任务时 context 会爆炸——thought / observation 堆积
- 单步出错会污染后续推理

**修复**：
- 加最大循环次数兜底
- 在 prompt 里明确要求"如果你已经知道答案，立刻给最终答复"
- 长任务时引入 summarization 步骤压缩历史

## 2. Plan-and-Execute

**核心**：先让 LLM **整体规划**好所有步骤，再按计划执行。

```
Step 1: 规划阶段 - 输出 [step1, step2, step3, ...]
Step 2: 执行阶段 - 顺序执行每个 step
Step 3: 失败时 replan
```

**和 ReAct 的区别**：ReAct 是边走边看，Plan-and-Execute 是先想清楚再开始。

**什么时候用**：
- 任务**结构清晰**，可以提前规划
- 步骤间依赖明确
- 需要**省 token**——一次规划 + 多次小执行，比 ReAct 全程跑大模型便宜

**实测优势**：根据 2026 一份研究，Plan-and-Execute 在复杂多步任务上完成率 92%、相对 ReAct **快 3.6 倍**。

**坑**：
- 不擅长应对中途的意外变化
- 规划阶段如果错了，后面全错（所以 replan 机制很重要）

## 3. Reflection（反思）

**核心**：让 agent **检查自己刚才的输出**，发现问题再改。

```
Generate → Critique → Revise
```

可以是单 agent 自我反思，也可以是两个 agent（一个生成、一个审稿）。

**什么时候用**：
- 输出质量要求高（代码、文案、复杂分析）
- 单次生成容易出明显错误（语法、逻辑、风格）

**实际效果**：代码生成场景里加一次 reflection，正确率能涨 15-25%。

**坑**：
- 反思阶段也会出错——critique 错了反而把对的改坏了
- 成本翻倍（至少多一次 LLM 调用）
- 容易陷入"无限完善"——加个最大迭代次数

## 4. Tool Use（工具调用）

严格说不算单独的模式，更像是**所有 agent 模式的基础组件**。
但单独拿出来是因为**工具设计的好坏直接决定 agent 能力上限**。

工具设计要点：

- **职责单一**：每个工具只做一件事
- **输入有 schema**：用 JSON Schema 明确参数类型和必填项
- **错误信息要友好**：失败时 agent 要能看懂为什么失败
- **粒度合适**：太粗 agent 没法精细控制，太细 agent 需要调一堆工具才能干一件事

## 5. Multi-Agent Collaboration（多 agent 协作）

把一个复杂任务**拆给多个 agent**，每个 agent 角色不同。

常见的子模式：

- **Sequential**：研究员 → 写作 → 编辑（串行流水线）
- **Parallel**：4 个 agent 同时搜不同方向，然后汇总
- **Hierarchical**：一个 manager agent 协调下面的 worker agents
- **Debate**：两个 agent 互相挑刺，让答案更鲁棒

**什么时候用**：
- 任务太大单 agent 装不下
- 不同子任务需要**不同的 system prompt / 不同的工具**
- 想用便宜模型并行替代昂贵模型串行

**坑**：
- agent 之间通信容易丢信息
- 协调成本高，简单任务不要硬上 multi-agent

## 6. Sequential Workflows（顺序工作流）

最朴素的模式：**像 Unix pipe 一样把 LLM 调用串起来**。

```
[Input] → LLM-step1 → LLM-step2 → LLM-step3 → [Output]
```

每一步任务单一、prompt 短、可单独测试。

**和 Plan-and-Execute 的区别**：Sequential 是写死的流水线，没有"规划阶段"；用什么 step 是开发者预先决定的。

**什么时候用**：
- 流程稳定、不需要 agent 自己决定步骤
- 想要**最高的可控性和可预测性**
- 生产环境，准确率和稳定性优先于灵活性

**实战中**：很多被宣传成"agent"的产品，本质就是精心设计的 sequential workflow。这不是缺点——简单、稳定、易调试。

## 7. Human-in-the-Loop

**核心**：关键节点**让人拍板**。

适合：
- 高风险决策（执行交易、删除资料、发对外邮件）
- 第一阶段上线没把握、需要人审核
- 不确定时让人补充信息

工程上要做的事：
- 把 agent 停在等待点
- 把决策上下文清晰地呈现给人
- 接受人的反馈后无缝继续

**最容易低估的价值**：用 human-in-the-loop 跑 1-2 周收集真实交互数据，比闭门设计 prompt 强多了。

## 怎么挑

| 场景 | 推荐模式 |
|---|---|
| 客服问答、检索类 | 简单 Sequential + RAG，不一定要 agent |
| 数据分析、需要多次查询 | ReAct |
| 复杂报告生成 | Plan-and-Execute + Reflection |
| 代码自动修复 | ReAct + Tool Use + Reflection |
| 大型研究任务 | Hierarchical Multi-Agent |
| 任何涉及不可逆操作 | Human-in-the-Loop（加在关键节点） |

## 一个朴素的判断

**先用最简单的模式，跑不通再升级**。
80% 的"agent"任务其实用 Sequential Workflow 就能做好，没必要一上来就 ReAct + Multi-Agent。
能用一个 prompt 解决的，不要拆成 agent；能用 5 个 prompt 串起来的，不要套 ReAct；能用 ReAct 跑通的，不要硬上 multi-agent。

复杂度是负债，不是资产。
