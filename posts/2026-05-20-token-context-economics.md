---
layout: post
title: Token 与上下文窗口经济学 — 让你的 AI 应用便宜又快
date: 2026-05-20
category: "Prompt 与推理"
tags: [AI, Token, 性能优化, 成本]
excerpt: 上下文窗口越来越大，但 token 越用越贵。怎么省、怎么缓存、怎么裁剪，一份工程视角的清单。
permalink: /posts/2026-05-20-token-context-economics.html
---

## 为什么要算 token

主流模型都是按 token 计费的：

- **Input token**：你发给模型的内容（system prompt + 历史 + 当前问题）
- **Output token**：模型生成的回答

很多人忽略一个事实——**input 通常远多于 output**。一个长 system prompt + 历史对话 + RAG 检索片段，能轻松塞进 20k+ tokens；而模型回答可能只有几百。
所以**真正吃成本的是 input**，不是回答长度。

## 一个能用的成本公式

```
单次请求成本 ≈ 
    input_tokens × input_price
  + output_tokens × output_price
  - cached_tokens × (input_price - cache_read_price)
```

举例（Claude Opus 4.7 的近似价位）：

| 维度 | 价格（每 1M tokens） |
|---|---|
| Input | $15 |
| Output | $75 |
| Cache write | $18.75（×1.25）|
| Cache read | $1.5（×0.1）|

如果一个 20k token 的 system prompt **被缓存命中**，input 部分成本从 $0.30 一下子降到 $0.03，相当于 **省 90%**。

## Prompt Caching：最大的杠杆

主流模型（Claude / GPT / Gemini）都支持 prompt cache，但策略不同：

- **Anthropic**：5 分钟（默认）或 1 小时（ephemeral_1h）TTL，命中能让 input 成本下降 90%
- **OpenAI**：自动缓存超过 1k token 的前缀，对开发者透明
- **Gemini**：显式 cache API，需要主动管理

工程建议：

- **稳定不变的部分放最前面**：system prompt → tools 定义 → few-shot examples → 动态内容
- 把缓存边界做成"前面 90% 不变，后面 10% 变化"
- 调用频率高时主动用 1h cache，避免 5 分钟空窗

## 上下文裁剪策略

context 越大不代表越好——3 个原因：

1. **贵**：input token 是钱
2. **慢**：每多 1k token 增加几十毫秒延迟
3. **lost in the middle**：研究多次表明模型对超长上下文中间部分的注意力会下降

裁剪几种思路：

### 1. 滑动窗口

只保留最近 N 轮对话历史，老的丢掉。简单粗暴，适合客服聊天。

### 2. 摘要压缩

把"老消息们"压缩成一段摘要替换。可以用同一个 LLM 异步压缩。

```text
[前 20 轮对话摘要] 用户在咨询信用卡换卡流程，已确认身份和原卡末四位...
[最近 3 轮原文]
用户：那我现在能直接换吗？
助手：可以的，需要你...
用户：好，下一步？
```

### 3. 检索式上下文（RAG-style memory）

不把所有历史塞进 context，而是 embed 后存进向量库，需要的时候**按当前问题检索 top-K 历史片段**塞回去。
长会话场景里这套比"全量塞"省 70%+ token。

### 4. 工具结果剪裁

工具返回的 JSON 别整段贴回——只保留 agent 真正需要的字段。
经常一个 SQL 查询返回 5MB JSON，agent 只需要 totalCount 这一个字段。

## 输出 token 的省法

输出贵且慢，能省就省：

- **结构化输出**：让模型直接返回 JSON，没废话
- **明确长度约束**：在 prompt 里写"用 50 字以内总结"
- **stop sequence**：用 `</answer>` 这种标签收尾
- **小模型扛冷流量**：判断当前请求复杂度，简单的扔给 Haiku，复杂的才给 Opus

## 一份监控清单

生产 AI 应用至少要监控：

- 平均 input / output token 数
- p50 / p99 延迟
- cache hit rate（特别是 prompt cache）
- 单请求成本分布
- 按用户 / 按 endpoint 的成本归因

**没有这些指标，"AI 太贵"就只是一句感叹**，没法做优化。

## 一句话总结

> Token 经济学的核心：把不变的东西做成可缓存的前缀，把变化的东西塞进尾巴；老历史压缩或者检索，不要全量贴；输出尽量短且结构化。

做好这几件事，AI 应用的成本可以**降到原来的 1/3 到 1/5**，体验还更快。
