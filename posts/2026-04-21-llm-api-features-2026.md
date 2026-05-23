---
layout: post
title: 2026 年 AI API 新功能盘点 — Anthropic / OpenAI / Google
date: 2026-04-21
topic: "工程实战"
tags: [AI, API, Anthropic, OpenAI, Gemini]
excerpt: Claude Opus 4.7、GPT-5、Gemini 2.x 各家这一年都加了什么硬货？哪些是真有用，哪些是营销话术。
permalink: /posts/2026-04-21-llm-api-features-2026.html
---

## 一句话总结现状

2026 年的 AI API 已经从"模型能聊天"演化到**完整的 agent 平台**——每家都在卷工具调用、推理、长上下文、prompt 缓存、多模态、批处理、原生 agent 框架。

下面按"真有用"和"营销大于实质"两类盘一盘。

## Anthropic（Claude）

### Extended Thinking（真有用）

Claude 4 系列引入"显式推理预算"——调用时指定 `thinking.budget_tokens`，模型先内部推理再回答。

```python
client.messages.create(
    model="claude-opus-4-7",
    thinking={"type": "enabled", "budget_tokens": 10000},
    ...
)
```

实测复杂任务正确率提升显著。**编程、数学、合规推理**场景默认开。

### Prompt Caching（极有用）

5 分钟默认缓存，1 小时 ephemeral 模式。input token 命中后成本降到 1/10。

```python
messages=[{
    "role": "user",
    "content": [{
        "type": "text",
        "text": LONG_SYSTEM_CONTEXT,
        "cache_control": {"type": "ephemeral"}
    }, ...]
}]
```

高频调用场景几乎是免费 90% 折扣，必用。

### Tool Use 增强（真有用）

- 并行工具调用（一次返回多个 `tool_use`）
- 工具 schema 支持 enum / pattern / required
- 跟 thinking 联动：模型边想边规划工具调用

### Computer Use（极客向）

让 Claude 直接看屏幕、移鼠标、点击。**还不算生产可用**——慢、贵、有时点错。
但展示了未来 agent 的方向：从"调 API"到"用 GUI"。

### Files API（实用）

上传 PDF / 图片 / 长文本一次，后续直接引用，省得每次 base64 编码。

## OpenAI

### Responses API（重要）

旧的 Chat Completions API 之上的新一代接口，**整合了 chat + tool use + state**：

- 多轮 conversation 自动管理（不用每次塞历史）
- 内置 tool runner（hosted tools 自动跑）
- 流式 + 结构化输出原生支持
- 跟 Assistants API 替代关系

新项目用 Responses API，老项目可以慢慢迁移。

### Realtime API（语音场景必备）

WebSocket-based，**原生处理音频流**——
说话进去、声音出来，端到端延迟 < 500ms。
适合做：实时语音助手、客服 IVR、口语陪练。

### Structured Output strict 模式（必用）

`response_format` 设 strict 后，模型 100% 符合 JSON Schema。
信息抽取、路由、表单生成场景都该用。

### Batch API（省钱神器）

非实时任务用 Batch API，**成本对半砍**：

- 提交一个 jsonl 任务文件
- 24 小时内异步处理完
- 价格 50% off

适合：离线评估、批量内容生成、嵌入向量计算。

### Predicted Outputs（少见但有用）

当输出大部分可预测（比如改一段代码里的几行），可以提前传"预测内容"——
模型只生成差异部分，**延迟大幅降低**。
特别适合 IDE 里的 code edit 场景。

## Google（Gemini）

### 长上下文（真有用）

Gemini 1.5/2.x 系列：1M-2M token 上下文窗口。

实测：

- 100 页 PDF 一次塞进去 → 直接问问题
- 整个代码仓库（500k token）→ 全局重构建议
- 整段视频 → 时间戳级理解

需要长上下文的场景，Gemini 是首选。

### Context Caching（必用）

跟 Anthropic prompt cache 类似，缓存大上下文降成本。
长文档分析场景几乎必开。

### 原生多模态（真有用）

Gemini 是**真正的"原生多模态"**——
文本 / 图 / 音频 / 视频是同一个模型，不是拼接的。

- 直接传 mp4 视频
- 直接传 wav 音频
- 直接传 PDF（不用先转图）

很多需要多模态融合理解的场景，Gemini 比拼接式方案省心。

### Live API（实时多模态）

Realtime 风格，但**多模态实时**——
摄像头画面 + 麦克风 + 屏幕共享一起送进去，模型实时响应。
适合：现场翻译、AR / VR 助手、教育辅导。

## 跨家通用功能

主流家都有了：

- **Streaming**：SSE / WebSocket
- **Tool use / Function calling**
- **Structured output / JSON mode**
- **Vision input**
- **Embedding API**
- **Fine-tuning API**（OpenAI / Gemini；Anthropic 还没开）
- **Batch API**
- **File upload**

## 选哪家：场景导向

| 场景 | 首选 |
|---|---|
| 复杂推理 / 代码 / 合规 | Claude Opus + thinking |
| 实时语音 | OpenAI Realtime |
| 长文档 / 整库分析 | Gemini 2.x |
| 多模态融合（视频 / 音频） | Gemini |
| 性价比通用 | Claude Sonnet 或 GPT-4.5-mini |
| 批量非实时 | OpenAI Batch |
| 国产替代 | DeepSeek / Qwen / Kimi |

## 营销大于实质的部分

- "AGI Coming Soon" — 每家都在吹，没人交付
- "0 hallucination" — 物理上不可能
- "Beats GPT/Claude on X benchmark" — benchmark game，看自家挑的指标
- 大部分 "Agent platforms" — 还是早期，建议自己用 LangGraph + 原生 SDK 拼

## 一个朴素建议

**别忠诚于某一家**。

技术栈层面用 LiteLLM / OpenRouter 之类的适配器，
业务层面按场景灵活路由：

```
简单聊天 → Sonnet 4.6（便宜）
复杂推理 → Opus 4.7 + thinking
长文档 → Gemini 2.x
语音 → OpenAI Realtime
batch 任务 → OpenAI Batch（省钱）
国内合规 → DeepSeek
```

把每家最强的能力用在它擅长的场景，**总账单和效果都最优**。

模型这玩意儿迭代太快，别绑死。
