---
layout: post
title: Skeleton-of-Thought Prompting — 先提纲后并行扩写，生成速度翻倍
date: 2026-07-20
topic: "Prompt 与推理"
tags: [Prompt, 推理, 并行生成, 性能优化]
excerpt: Skeleton-of-Thought 把长文生成拆成两步：第一步让 LLM 只输出骨架，第二步对每个要点并行调用模型扩写，总耗时可降低 40%-60%，同时保持内容连贯性。
permalink: /posts/2026-07-20-skeleton-of-thought-prompting.html
---

你有没有遇过这种场景：让 LLM 写一篇 2000 字的技术文档，光等模型逐字吐完就需要 30 秒，而实际上大部分时间都花在"一个字一个字往下写"这件事本身——不是在思考，而是在打字。

Skeleton-of-Thought（SoT）就是要干掉这段浪费。

## 核心思路：把串行生成变成并行扩写

传统的长文生成是线性的：模型从第一个 token 写到最后一个 token，无论内容结构多清晰，都必须顺序完成。SoT 的思路则完全不同，分两步走：

**第一步（骨架 pass）**：给模型一个极短的 prompt，让它只输出要点列表，不展开，不写正文。这一步通常在 2-3 秒内完成。

**第二步（扩写 pass）**：对骨架里的每一个要点，单独发起一次 LLM 调用，并行运行，各自扩写成完整段落。N 个要点并发 N 个请求，总耗时等于"最慢那个要点的扩写时间"，而不是所有要点之和。

实测数据来自 SoT 原论文（Ning et al., 2023）：在 GPT-3.5 / GPT-4 上，对问答类和说明类任务，端到端延迟降低 40%-60%；在部分任务上，质量评分（GPT-4 作为 judge）还略有提升，原因是每个要点单独扩写时模型的"注意力"更集中。

## Prompt 设计：骨架 pass 怎么写

骨架 pass 的关键是约束输出格式，让后续解析不出错。下面是一个实战模板：

```text
你是一个技术写作助手。
用户问题：{question}

请仅输出回答的骨架，格式为编号列表，每条不超过 10 个字，不要展开，不要写正文。
示例输出格式：
1. 问题背景
2. 核心概念
3. 实战方案
4. 注意事项

现在输出骨架：
```

扩写 pass 的模板：

```text
你是一个技术写作助手，正在帮用户回答以下问题：
{question}

文章骨架如下：
{full_skeleton}

现在请展开第 {index} 点"**{point}**"，写成 150-200 字的完整段落。
只输出这一段的正文，不要重复标题，不要加其他内容。
```

注意把 `full_skeleton` 也传进扩写 prompt——这让模型知道整篇文章的结构，避免各段之间语义断裂。

## Python 实现示例

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

SKELETON_PROMPT = """
用户问题：{question}
请仅输出回答骨架，编号列表，每条 ≤ 10 字，不展开。
""".strip()

EXPAND_PROMPT = """
用户问题：{question}
文章骨架：
{skeleton}

请展开第 {idx} 点"{point}"，写 150-200 字正文段落，只输出段落本身。
""".strip()

async def expand_point(question, skeleton, idx, point):
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EXPAND_PROMPT.format(
            question=question, skeleton=skeleton, idx=idx, point=point
        )}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

async def sot_generate(question: str) -> str:
    # Step 1: 骨架
    skel_resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": SKELETON_PROMPT.format(question=question)}],
        temperature=0.3,
    )
    skeleton = skel_resp.choices[0].message.content.strip()
    points = [line.split(". ", 1)[-1] for line in skeleton.splitlines() if line.strip()]

    # Step 2: 并行扩写
    tasks = [expand_point(question, skeleton, i+1, p) for i, p in enumerate(points)]
    sections = await asyncio.gather(*tasks)

    return "\n\n".join(f"### {points[i]}\n\n{sections[i]}" for i in range(len(points)))

# 使用
result = asyncio.run(sot_generate("RAG 系统如何处理多跳问题？"))
print(result)
```

整个流程的核心是 `asyncio.gather`——N 个扩写请求同时飞出去，不等上一个回来才发下一个。如果你用的是同步客户端，可以改用 `concurrent.futures.ThreadPoolExecutor` 达到同样效果。

## 适用场景 vs 不适用场景

| 场景 | 适合 SoT？ | 原因 |
|---|---|---|
| 技术文档、说明文 | ✅ 强烈推荐 | 天然有结构，并行扩写质量好 |
| FAQ / 问答列表 | ✅ 推荐 | 每个 Q&A 完全独立 |
| 代码生成 | ⚠️ 视情况 | 函数间有依赖时骨架难切分 |
| 叙事/故事创作 | ❌ 不推荐 | 情节连贯性强，并行扩写会断层 |
| 单句/短回答 | ❌ 不划算 | 两轮调用反而更慢 |
| 需要跨段推理的分析 | ❌ 谨慎 | 各段孤立扩写会丢失全局逻辑链 |

决策规则很简单：**内容能被切成相对独立的块，就用 SoT；块之间强依赖，就老老实实串行。**

## 变体与进阶

**动态骨架深度**：先让模型估计问题的复杂度（1-5 分），再决定骨架要几个节点。简单问题 3 个节点，复杂问题 6-8 个节点，避免并发请求过多触发 rate limit。

**骨架验证层**：在扩写之前，用一个轻量 prompt 让模型检查骨架是否覆盖了问题的所有关键角度，有缺失就补上，再进入扩写 pass。这一步只需 0.5 秒，但能大幅提升结构完整性。

**混合模型**：骨架 pass 用便宜小模型（GPT-4o-mini / Haiku），扩写 pass 用强模型（GPT-4o / Sonnet）。骨架阶段不需要太强的推理，省下来的钱花在正文质量上更值。

## 踩坑清单

- **骨架 prompt 不加格式约束**：模型可能返回散文而非列表，导致解析崩。解决：在 prompt 里给出示例格式，并配合 JSON mode 或 regex 做兜底解析。
- **扩写 prompt 里不传完整骨架**：各段会"不知道自己在文章里的位置"，出现重复开头或逻辑跳跃。解决：每个扩写请求都附上完整的骨架字符串。
- **并发数不加限制**：骨架有 10 个节点就同时发 10 个请求，容易触发 OpenAI / Anthropic 的 rate limit（特别是 RPM 限制）。解决：用 `asyncio.Semaphore(5)` 限制最大并发数。
- **把 SoT 用在代码生成上**：函数 A 调用函数 B，两个人并行写，结果接口对不上。解决：代码生成先串行写接口定义，再并行写实现体。

SoT 不是银弹，但它提供了一个重要视角：**LLM 的推理和 LLM 的"打字"是两件事，能拆开的地方就拆开，能并行的地方就并行**。把这个思路用好，你的 AI 应用在用户感知延迟上能甩竞品一截。
