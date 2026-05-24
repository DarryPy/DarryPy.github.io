---
layout: post
title: Prompt Management 平台对比 — LangSmith / Braintrust / Langfuse / Helicone
date: 2026-02-08
topic: "工程实战"
tags: [AI, Prompt Management, 工具]
excerpt: 把 prompt 当成代码管，需要版本、灰度、对比、回滚。4 家主流平台横向对比。
permalink: /posts/2026-02-08-prompt-management.html
---

## 为什么要专门平台

Prompt 散在代码里、文件里、Notion 里，最终会变成：
- 不知道线上跑的是哪版
- 多团队各自改一份，互相冲突
- 改了 prompt 不知道效果
- 出问题不能秒回旧版

专门平台解决这些。

## 4 家横评

### LangSmith

LangChain 团队出品。集 Prompt + Eval + Trace 一身。

**核心功能**：
- Prompt Hub：UI 编辑、版本、注释
- Playground：side-by-side 对比不同 prompt / 模型
- Experiment：跑 dataset 看指标对比
- Trace：每次调用的完整追踪
- Annotation：人工打标流程

**优势**：跟 LangChain / LangGraph 零接入；功能最全。
**劣势**：托管 SaaS 收费偏贵；自部署版本 enterprise 才有。
**适合**：用 LangChain 栈的中大团队。

### Braintrust

定位**现代化 prompt 工程 IDE**。

**核心功能**：
- Playground：浏览器里多 prompt / 多模型对比，diff view 强
- Datasets：版本化 dataset
- Evals：脚本式 eval + 历史对比
- CI 集成：每次 PR 自动跑 eval
- 简洁的 TypeScript / Python SDK

**优势**：UX 现代，prompt 迭代体验最舒服；CI 集成做得好。
**劣势**：社区比 LangSmith 小；收费偏贵。
**适合**：重视 prompt 工程迭代效率；TypeScript 项目。

### Langfuse

**开源版本的 LangSmith**。

**核心功能**：
- Trace / Dataset / Evaluator / Annotation 全有
- 完全开源，自部署友好
- 支持任意 SDK（不限于 LangChain）
- Prompt Management（版本 + 灰度 + 引用）
- Cost / Latency 监控

**优势**：开源、自部署、功能全、社区活跃。
**劣势**：UI 比商业产品略糙；自部署运维要自己来。
**适合**：数据合规要求自部署；预算紧。

### Helicone

主打**轻量 LLM proxy + observability**。

**核心功能**：
- 改 baseURL 到 `oai.helicone.ai` 就自动 trace
- 成本面板（最强）
- Caching（自动 semantic cache）
- 速率限制
- 简单的 Prompt Management

**优势**：接入零代码（改 URL 就行）；成本管理细。
**劣势**：eval 工作流相对弱；功能不如 LangSmith / Braintrust 深。
**适合**：只要看"花了多少钱、慢请求"；不要完整 eval pipeline。

## 决策矩阵

| 你的需求 | 推荐 |
|---|---|
| LangChain 栈 + 完整功能 | LangSmith |
| 重 prompt 迭代效率 | Braintrust |
| 开源 / 数据合规 / 自部署 | Langfuse |
| 极简接入 + 成本管理 | Helicone |
| 都不喜欢 | 用 SDK 自建（Mongo + S3 + 自家 UI）|

## 共同必备功能

不管选哪个，prompt 管理平台至少要有：

1. **版本号 + 历史**：每次改可追溯
2. **灰度发布**：按比例切流量
3. **快速回滚**：故障时秒切回旧版
4. **变量模板**：Jinja / 类似机制
5. **跟 Eval 联动**：改 prompt → 自动跑测试集 → 看指标
6. **多团队权限**：谁能改、谁只能看

## 实战配置

不管什么平台，建议套这个流程：

```
1. 在平台里建 prompt "intent_classifier"
2. v1 上线，灰度 100%
3. 设计 v2，提交到平台
4. CI 自动跑 eval 比 v2 vs v1，要求核心指标不退步
5. 通过 → v2 灰度 5% → 24h 监控
6. OK → v2 灰度 50% → 24h
7. OK → v2 灰度 100%，v1 保留 1 周
8. 出问题 → 一键回滚 v1
```

这套流程下 prompt 改动是**可控、可观测、可回退**的工程操作，不是"改个字符串"的随手活儿。

## 一个朴素结论

> 1 个 prompt：放代码里。
> 5 个 prompt：放文件。
> 20+ prompt 或多团队：**必须用平台**。
>
> 没平台 = AI 应用工程化做不起来。
