---
layout: post
title: System Prompt 设计 — 让系统提示词真正起作用
date: 2026-04-19
topic: "Prompt 与推理"
tags: [AI, System Prompt, Prompt Engineering]
excerpt: 80% 的 prompt 工程其实是 system prompt 工程。一份让 system prompt 高效、稳定、可维护的实战清单。
permalink: /posts/2026-04-19-system-prompt-design.html
---

## 为什么 system prompt 是最大的杠杆

模型每次推理时都会读 system prompt。它定义"你是谁、能做什么、不能做什么"——
这意味着你在 system prompt 里写的每一句话**会乘以会话次数无限放大**。
改一个 user prompt 影响一条对话；改 system prompt 影响所有对话。

好的 system prompt 能让一个普通模型表现得像专家，差的 system prompt 能让 Opus 4.7 瞎说。

## 一份好的 system prompt 结构

按下面 6 段写，覆盖率最高：

```
<role>你是谁</role>
<task>你要做什么</task>
<context>你能用什么资源</context>
<constraints>不能做什么</constraints>
<style>怎么说话</style>
<failure>不确定时怎么办</failure>
```

具体例子：

```
<role>
你是一位资深的产品经理助手，专门帮 SaaS 团队梳理需求。
</role>

<task>
将用户描述的零散想法整理成结构化需求文档（PRD 大纲）。
</task>

<context>
你能调用工具：search_existing_requirements（搜索现有需求库），
analyze_user_feedback（分析用户反馈）。
</context>

<constraints>
- 不要发明用户没提到的功能
- 不要承诺时间表（开发资源不由你决定）
- 涉及法务/合规问题时显式标注"需法务确认"
</constraints>

<style>
- 简洁，每条需求一句话
- 用产品语言，不用纯技术术语
- 关键点加粗
</style>

<failure>
如果用户描述太模糊，主动问 1-2 个关键澄清问题；
不要硬编需求。
</failure>
```

## 关键原则

### 1. 用正面表达，少用否定

❌ "不要使用专业术语"
✅ "用普通用户能理解的日常语言"

模型对"不"字的注意力机制是有缺陷的——你越强调"不要 X"，它越容易想到 X。

### 2. 给硬约束的例子

不要只写"输出要简洁"。要写：

```
输出长度约束：
- 短回答场景：1-3 句话
- 列表场景：最多 5 个 bullet
- 详细解释：分小标题，每段 ≤ 100 字
```

可量化的约束 = 可验证。

### 3. 提供"判官标准"

```
你回答完后，自检一遍：
1. 有没有用到工具？没有的话理由是什么？
2. 回答是否基于事实？引用了哪个来源？
3. 用户是否会觉得回答有用？
```

让模型自己做内置 eval，正确率显著高。

### 4. System prompt 要稳定，user 部分塞变化的东西

```
[system] = 角色 + 工具 + 风格 + 通用规则（不变）
[user] = 当前问题 + 这次相关的上下文（每次变）
```

为什么？**prompt cache** 只缓存稳定前缀。system 越稳定，越省钱。

## 容易踩的坑

| 坑 | 修法 |
|---|---|
| 一段长描述，没结构 | 用 XML 标签或编号列表分块 |
| 写"请你尽力" | 改成可量化标准 |
| 把工具列表塞进 user prompt | 应该在 system 里 |
| 每个会话改 system | 让 system 稳定，变化放到第一条 user |
| system 里塞海量背景知识 | 改成 RAG，按需检索 |

## 一个朴素心智

> System prompt 是合同，不是建议。

写得越像一份带条款的合同——角色明确、约束可验证、失败兜底有规则——
模型表现越专业、越稳定。
