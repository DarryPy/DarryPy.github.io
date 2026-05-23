---
layout: post
title: AI 评估工具横评 — LangSmith / Phoenix / Ragas / Braintrust
date: 2026-04-23
topic: "评估与安全"
tags: [AI, Eval, LangSmith, Phoenix, Ragas]
excerpt: 不再自己 print 调试。主流 LLM 评估和 observability 工具横向对比，按场景选型。
permalink: /posts/2026-04-23-ai-eval-tools.html
---

## 为什么需要专门工具

前面写过 [AI Evals 怎么做](/posts/2026-05-17-ai-evals.html)，原则是"建数据集、跑指标、看趋势"。
但工程上要做好这件事，**手工跑很快就崩溃**：

- 每次改 prompt 要手动 run 一遍数据集
- 看历史指标变化得自己存 / 自己画
- LLM-as-judge 的代码每个项目重复写一遍
- trace 一次 agent 跑了 8 个 LLM call，调试满屏 print
- 团队协作时数据集 / 实验结果不好分享

主流的 LLM observability + eval 工具就是解决这些。

## 横向对比

### LangSmith（LangChain 团队出的）

**定位**：LangChain / LangGraph 生态首选 observability 平台。

**核心功能**：
- 完整的 trace 视图（每个 LLM call、tool call、retriever call 都能看）
- Dataset + Experiment：把数据集跑一遍生成 experiment，对比版本
- Evaluator：内置 + 自定义 evaluator
- Annotation queue：人工打标的工作流
- Production monitoring：上线后实时追踪

**优点**：跟 LangChain / LangGraph 零接入；UI 干净；上手快。
**缺点**：托管 SaaS（自部署版本贵）；强绑 LangChain 生态，纯原生 SDK 也能用但功能稍弱。

**适合**：用 LangChain / LangGraph 的团队，需要全套 trace + eval。

### Phoenix（Arize 团队开源）

**定位**：开源 LLM observability，主打"私有部署 + 强可视化"。

**核心功能**：
- OpenTelemetry-based tracing（不强绑某个框架）
- 内置常见 LLM 指标和 dashboard
- Embedding drift 检测（embedding 分布变化告警）
- RAG retrieval 评估
- 跟 LlamaIndex / LangChain / 原生 SDK 都能集成

**优点**：开源、自部署友好、跟数据科学 / ML 生态融合好。
**缺点**：UI 没 LangSmith 那么炫；功能在快速演进中，文档偶尔跟不上。

**适合**：数据敏感不能上 SaaS 的团队、已经在用 Arize ML 平台的团队。

### Ragas（专攻 RAG 评估）

**定位**：RAG 专用评估库。

**核心功能**：
- 一组成熟的 RAG 指标实现：
  - `faithfulness`（回答是否忠于检索片段）
  - `answer_relevancy`（回答跟问题相关吗）
  - `context_precision`（检索的片段都有用吗）
  - `context_recall`（关键信息都检索到了吗）
- LLM-as-judge 标准实现
- 跟 LangChain / LlamaIndex 集成

**优点**：RAG 评估的"事实标准"指标体系。
**缺点**：不是完整平台——只是个 Python 库，没自己的 UI。配合 LangSmith / Phoenix 用最好。

**适合**：做 RAG 的团队。**几乎所有 RAG 项目都该用**。

### Braintrust（新生代）

**定位**：现代化 prompt 工程 + eval IDE。

**核心功能**：
- Prompt playground：在浏览器里 side-by-side 对比不同 prompt / 不同模型的输出
- Evaluation：跑数据集 + 多 evaluator + 历史对比
- Trace + log
- TypeScript / Python SDK
- 强力的 diff view

**优点**：UX 现代，prompt 迭代体验最舒服；自动 eval CI 集成做得好。
**缺点**：SaaS 收费偏贵；社区比 LangSmith 小。

**适合**：重视 prompt 工程迭代效率的团队、TypeScript 项目。

### Langfuse（开源新秀）

**定位**：开源版本的 LangSmith。

**核心功能**：
- Trace / dataset / evaluator / annotation 都有
- 完全开源、自部署友好
- 支持任意 SDK 接入（不限于 LangChain）
- 内置 prompt management

**优点**：开源、功能全、社区活跃。
**缺点**：UI 比商业产品略糙；自部署运维要自己来。

**适合**：想要 LangSmith 全套但不想付钱 / 不能上 SaaS 的团队。

### Helicone（轻量 proxy）

**定位**：作为 LLM API proxy 自动收集 trace 和成本。

**核心功能**：
- 把 `api.openai.com` 改成 `oai.helicone.ai` 就自动 trace
- 成本面板、缓存、速率限制
- 简单的 eval 工作流

**优点**：接入零代码（改 baseURL 就行）；成本管理强。
**缺点**：evaluation 功能比专门工具薄。

**适合**：只想看"花了多少钱、有什么慢请求"，不要完整 eval pipeline 的团队。

## 决策矩阵

| 需求 | 推荐 |
|---|---|
| 入门 / 简单观察 | Helicone（代码改一行） |
| LangChain / LangGraph 用户 | LangSmith |
| 自部署 + 不绑框架 | Phoenix 或 Langfuse |
| RAG 评估专项 | Ragas（指标库） + 任一平台（UI） |
| Prompt 重度迭代 | Braintrust |
| 完全开源栈 | Langfuse + Ragas |
| 已经在 Arize ML 平台 | Phoenix |

## 实战配置组合

### "便宜 + 全功能"

```
Langfuse (自部署) + Ragas (RAG 指标)
```

零成本，覆盖 trace、dataset、experiment、RAG eval。

### "省事 + 强体验"

```
LangSmith (SaaS) + Ragas (RAG 指标)
```

不用运维，UI 顺手，跟 LangChain 生态集成最好。

### "重度 prompt 迭代"

```
Braintrust + Helicone (cost)
```

prompt 工程体验最现代化。

## 共通的最佳实践

不管选哪个工具：

1. **从第一天就把 trace 接上**，不要"等出问题再加"
2. **维护 golden dataset**：50-500 条人工标注的典型 case
3. **每次改 prompt 跑一次完整 dataset**，不要靠"感觉"
4. **CI 里设 eval 门禁**：核心指标不能跌
5. **存储所有线上请求 trace 至少 7 天**，方便事后排查

## 一个朴素观察

> AI 工程的成熟度 = eval 工具的使用深度

只看 print 调试的团队还停留在 2023 年；
建了 dataset 跑 CI eval 的团队已经摸到了 2025 年；
有 LLM-as-judge + retrieval eval + production trace 的团队，做出来的 AI 产品**质量碾压前两类**。

工具选错没关系，**不用工具问题大**。
