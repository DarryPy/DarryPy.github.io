---
layout: post
title: Prompt Chaining 实战 — 把复杂任务拆成多轮 LLM 调用
date: 2026-06-08
topic: "Prompt 与推理"
tags: [Prompt Engineering, LLM, Python, Workflow]
excerpt: 一个 Prompt 搞不定复杂任务？Prompt Chaining 把大任务拆成多个独立的 LLM 调用，每步只做一件事，输出作为下一步输入，可调试、可验证、可复用。
permalink: /posts/2026-06-08-prompt-chaining-workflow.html
---

你写了一个超长 Prompt，把所有需求全塞进去，结果模型输出像一锅乱炖——质量差、难调试、一旦出错满盘皆输。你改了三次 Prompt，每次都在猜到底是哪句话出了问题。这不是模型能力不够，是你把太多责任压在了一次调用上。

Prompt Chaining 的核心思路：把一个复杂任务分解成若干个相互依赖的子任务，每个子任务用独立的 LLM 调用完成，上一步的输出作为下一步的输入。就像工厂流水线，每个工位只做一件事，整条线才能稳定产出。这不是在增加复杂度，而是在把复杂度变得可控、可观测、可修复。出了问题，你知道去哪里找原因；想改某个环节，你不用担心影响其他步骤。

## 什么时候该切换成链式调用

单一 Prompt 能搞定的情况：逻辑简单、输出格式固定、不需要中间判断。遇到以下场景，就该认真考虑 Prompt Chaining 了。

**任务包含分支判断**：你需要根据中间结果走不同路径，而不是一条直路到底。比如先判断用户意图是投诉还是咨询，再调用对应的回复 Prompt。把判断和回复拆成两步，每个 Prompt 的职责收窄，各自的准确率都会提升。

**输出格式互相依赖**：第二步的 Prompt 依赖第一步的结构化输出（比如某个 JSON 字段的值），如果第一步直接输出自然语言，第二步解析就是薛定谔实验——模型不是你的解析器，结构化传递才可靠。

**需要独立校验**：用另一个 LLM 调用充当"审稿人"，检查上一步是否达标，不达标就重试或降级。这比在同一个 Prompt 里让模型"自我审查"要有效得多——自我审查容易陷入自我认可。

**上下文窗口不够**：长文档分块处理，每块独立调用，避免因为超窗口被截断导致关键信息丢失，同时每块的结果可以单独缓存。

**调试需要中间态**：可以单独检查每一步的输入和输出，快速定位是哪个环节出了问题，而不是面对一个黑盒结果无从下手。

核心原则只有一条：**单一职责**——每个 Prompt 只负责一件事，输出清晰可预期。

## 三种常见的链式模式

**顺序链（Sequential Chain）** 是最基础的形式，A 步完成后把输出传给 B，B 完成后传给 C，严格线性推进。适合文档撰写（提纲 → 初稿 → 润色校对）、数据处理（信息提取 → 格式转换 → 结构验证）等有明确先后顺序的任务。每一步的 Prompt 可以独立在沙箱里测试，上线后排查问题路径也极其清晰。你甚至可以把链中的某一步替换成规则引擎或其他服务，LLM 只负责它最擅长的部分。

**分支链（Branching Chain）** 在关键节点输出一个决策标签，后续根据标签走不同路径。比如客服系统先把用户输入分类为 `COMPLAINT / INQUIRY / PRAISE` 三类，再分别调用针对性的回复 Prompt。这样每类回复的措辞策略可以独立调整，投诉回复改得更诚恳，不会影响咨询回复变得更冗长。分支节点的分类 Prompt 要尽量简短、输出标签清晰，避免模糊类别让后续分支判断失效。

**验证链（Validation Chain）** 在每步输出后插入一个独立的质检调用，输出不达标则重试或走兜底。生产环境里格式校验失败比你想象的更频繁——模型在高并发下有概率输出不规范的结果，你无法假设它每次都乖乖按格式来。用验证链主动兜底，比上线后发现问题再修要省力得多。验证链通常和顺序链或分支链组合使用，不是单独存在的一种模式。

## 实战：用 Python 构建内容审核链 🔗

假设需求：对用户生成内容（UGC）做三步处理——先检测语言，再做合规检查，最后生成回复。用 Anthropic SDK 实现如下。

```python
import anthropic

client = anthropic.Anthropic()

def call(prompt: str, content: str, max_tokens: int = 256) -> str:
    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": f"{prompt}\n\n内容：{content}"}]
    )
    return msg.content[0].text.strip()

def review_chain(user_content: str) -> dict:
    # Step 1: 语言检测，输出极短，max_tokens 压小省成本
    lang = call(
        "判断以下内容的语言，只输出：zh / en / other，不要其他说明。",
        user_content,
        max_tokens=8
    )

    # Step 2: 分支 — 非中文走简化流程
    if lang != "zh":
        return {"lang": lang, "status": "skip", "reply": None}

    # Step 3: 合规检查，输出固定格式便于解析
    compliance = call(
        "判断以下中文内容是否合规。第一行只输出 PASS 或 FAIL，"
        "第二行给出不超过 20 字的原因。",
        user_content,
        max_tokens=64
    )
    status = "PASS" if compliance.startswith("PASS") else "FAIL"

    # Step 4: 仅合规内容才生成客服回复
    reply = None
    if status == "PASS":
        reply = call(
            "用温和、专业的语气对以下用户内容写一段客服回复，不超过 60 字。",
            user_content,
            max_tokens=128
        )

    return {"lang": lang, "status": status, "reply": reply}
```

整个链有四步、一个分支、清晰的中间态，每一步的 `max_tokens` 都精确控制。语言检测只需要输出 `zh` 这两个字，给它 256 token 就是在浪费钱。链式调用里，**资源应该按需分配，不是按不安全感分配**。

## 加上验证重试

生产环境里，合规检查输出的格式偶尔会飘。给质检步骤加简单的重试逻辑：

```python
def validated_call(
    prompt: str,
    content: str,
    validator_fn,
    max_retries: int = 2,
    max_tokens: int = 64
) -> str:
    for attempt in range(max_retries + 1):
        result = call(prompt, content, max_tokens)
        if validator_fn(result):
            return result
        # 最后一次还没过就走兜底
    return ""

# 确保合规检查输出以 PASS 或 FAIL 开头
compliance = validated_call(
    "判断内容合规性，第一行只输出 PASS 或 FAIL，第二行给原因（20 字内）。",
    user_content,
    validator_fn=lambda r: r.startswith("PASS") or r.startswith("FAIL"),
    max_retries=2
)
```

最多重试 2 次，超限返回空字符串，由调用方决定如何处理，不会死循环。

## 链式调用的成本与延迟

链式调用的最大代价是**延迟叠加**——每一步都是同步 HTTP 调用，四步串行意味着四倍 latency。有几个缓解方向：

- **能并行的步骤并行化**：如果两步之间没有依赖关系，用 `asyncio.gather` 同时发起两次调用
- **中间结果缓存**：如果同一文档会被多次处理，把语言检测结果缓存起来，不用每次都调一遍
- **用小模型做分类**：分支判断用 Haiku 这类轻量模型，主要生成任务再上 Opus，整体成本和延迟都会下来

还有一点常被忽视：链中每步的 token 成本是独立计费的。如果一个链有五步，而第一步的输出会原样传给后四步，考虑开启 prompt caching，把重复的上下文缓存住，能显著降低费用。Anthropic 的 prompt caching 在 cache hit 时输入 token 成本可降到原来的十分之一，链式系统里的收益远比单次调用明显，值得在架构设计阶段就规划进去。

## 踩坑清单

- **中间步骤不要传 raw 自然语言**：上一步输出 JSON 或明确的关键词，下一步才好解析；传自然语言给模型"自己理解"，等于给它出选择题让它随机选
- **每步单独设 max_tokens**：中间步骤输出短就压小，不要全链统一用一个大值；短输出用大 max_tokens 不省钱，模型还可能填充废话
- **验证链最多重试 2-3 次**：超限必须走兜底路径，绝对不能无限循环；无限重试是生产事故的常见来源，用户在那端等着
- **链越长越要记录中间态**：每步输入输出都打 log 或存到结构体里，线上出问题时这些中间态是唯一的救命稻草
- **每个节点的 Prompt 单独维护**：不要图省事把相邻两步合并到一个 Prompt 里——职责边界一模糊，你又回到了"一锅乱炖"的起点
- **测试每步的边界输入**：链式系统最容易出问题的地方是步骤之间的接缝——第一步的边界输出能不能被第二步正确处理，单独测，不要等集成测试才发现

Prompt Chaining 不是"更多 API 调用"，是**更清晰的职责划分**。每个 LLM 调用只做一件事，整个系统才有机会稳定做对——出错了知道去哪里修，想优化了知道动哪一步，不需要每次都推倒重来。
