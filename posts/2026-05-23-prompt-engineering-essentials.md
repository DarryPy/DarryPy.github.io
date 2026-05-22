---
layout: post
title: Prompt Engineering 实战要点 — 写给开发者的入门到进阶
date: 2026-05-23
tags: [AI, Prompt Engineering, Claude, LLM]
excerpt: 一份从 0 到能跑生产的 Prompt 工程实战清单 — 清晰度、示例、XML 结构、role、thinking、链式提示。
permalink: /posts/2026-05-23-prompt-engineering-essentials.html
---

## 为什么 Prompt 还重要

很多人觉得"模型越来越强，Prompt 就不重要了"。
但实际上**模型越强，Prompt 的杠杆越大**。Claude 4.x、GPT-5 这一代模型在精心设计的 Prompt 下，能从 60 分跳到 95 分。
随便写的 Prompt，模型只能猜你想要什么——猜对了是运气，猜错了是必然。

下面这份清单是从 Anthropic、OpenAI 的官方指南和社区最佳实践里提炼出来的，可以当成 checklist 用。

## 核心原则：清晰、具体、有约束

最重要的一条原则只有一句话：**Prompt 写得像一份合同**——明确的角色、可验证的成功标准、明确的约束、不确定时的兜底。

对比一下：

❌ "帮我总结这篇文章"

✅ "你是一位资深的技术编辑。请用 3 句话总结下面这篇文章，每句不超过 25 字。如果文章不包含足够信息得出结论，请直接说'信息不足'。"

后者把**任务、风格、长度、失败兜底**全说清楚了。模型不用猜。

## 技巧 1：用具体示例（Few-shot）

直接告诉模型"你按这个格式输出"——给 2-3 个示例。
这比写一大段文字描述格式更有效。

```text
输入：今日天气如何？
输出：{"intent": "query_weather", "slots": {}}

输入：明天上海下雨吗？
输出：{"intent": "query_weather", "slots": {"date": "tomorrow", "location": "上海"}}

输入：{用户的真实问题}
输出：
```

## 技巧 2：用 XML 标签结构化输入

Claude 在训练时见过大量 XML 风格的结构化输入，对 `<tag>` 这种分隔的边界识别很好。
特别适合**多段拼接的复杂 Prompt**：

```xml
<task>提取下面文章里所有提到的产品名</task>

<rules>
- 只提取真实存在的产品名
- 不要包含公司名
- 输出为 JSON 数组
</rules>

<article>
{文章内容}
</article>
```

比起靠空行和 Markdown 分段，XML 标签让模型边界识别更准。

## 技巧 3：让模型"先想再答"

模型直接给最终答案，错误率会比让它**先输出推理过程**高很多。
两种用法：

- 让模型在答之前先写 `<thinking>...</thinking>`，再给最终结果
- 在支持 thinking 的模型（Claude 4 extended thinking、o-series）里直接开启 thinking 模式

实测复杂推理任务上，开 thinking 能让正确率上涨 20%+。

## 技巧 4：Role / System Prompt 是最强的杠杆

不要把 Role 写成"你是一个 helpful assistant"——太弱。

```text
你是一位专门处理金融合规问题的法律顾问。
你只用基于事实的语言回答，所有结论必须给出依据。
不确定时，明确说"我不确定"而不是猜测。
```

把 Role + 风格 + 边界 + 失败行为全写进 system prompt，对话里的回答会**整体抬一个层级**。

## 技巧 5：拆 Prompt（Prompt Chaining）

复杂任务不要硬塞到一个 Prompt 里。拆成多步：

1. **Prompt 1**：抽取原文里的关键事实，输出 JSON
2. **Prompt 2**：基于这份 JSON 做分析
3. **Prompt 3**：把分析结果写成给用户看的报告

每一步任务单一、上下文窄、可单独测试和重试。
生产系统里 chain prompts 的效果通常远好于"一个超长 Prompt 一把梭"。

## 常见反模式

| 反模式 | 为什么不好 |
|---|---|
| 一段没有结构的长描述 | 模型抓不住重点 |
| 用否定句："不要做 X" | 模型注意力反而被 X 吸引；改用正面表达 |
| 任务模糊："写得好一点" | "好"没有可验证标准；改成"读起来像 BBC 新闻稿" |
| 没有失败兜底 | 模型遇到边界条件会瞎编；显式给"信息不足时如何回应" |
| 把所有上下文都塞进 system | system 应该稳定；变化部分放 user 消息 |

## 一个可复用的模板

```text
<role>
你是 {领域专家身份}。
</role>

<task>
{一句话说清要做什么}
</task>

<rules>
- {可验证的约束 1}
- {可验证的约束 2}
- 如果信息不足，回答 "信息不足"
</rules>

<examples>
{1-3 个示例}
</examples>

<input>
{真实输入}
</input>

请先在 <thinking> 标签内分析，再给出最终结果。
```

把这个模板套上去，大部分新场景都能跑出能用的初版，再针对失败案例迭代就行。
