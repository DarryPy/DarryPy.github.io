---
layout: post
title: ReAct Prompting — 让模型边推理边行动
date: 2026-05-25
topic: "Prompt 与推理"
tags: [Prompt, ReAct, 推理, Agent]
excerpt: ReAct 是"推理 + 行动"的缩写，核心在于让模型在每一步交替输出思考轨迹和可执行动作，把一个大问题拆解成可被外部工具验证的步骤链。本文从格式定义到 Python 落地，给你一套开箱可用的模板。
permalink: /posts/2026-05-25-react-prompting.html
---

你用 Chain-of-Thought 让模型把推理步骤写出来，但推理链是在模型参数内部完成的——训练数据截止日期之后的信息它不知道，中间结论的对错也无法被外部数据验证。当问题需要查实时行情、算准确数值、调用内部数据库的时候，CoT 就走到了边界。

这种边界在生产系统里很常见：用户问"今天人民币对美元汇率下，我的 1000 美元等值多少人民币"，CoT 会给你一个言之凿凿但数据不对的答案；用户问"查一下订单号 20240528001 的物流状态"，模型压根没有这条数据。

ReAct（Reasoning + Acting）由 Yao 等人在 2022 年提出，核心思路是让模型在每一步都先"想"再"做"，输出一条交替的 Thought / Action / Observation 序列，而不是一口气生成最终答案。你不需要用任何 Agent 框架才能实现 ReAct——它本质上是一种 prompt 格式加上一个简单的调用循环。

## 为什么 CoT 在这里不够

CoT 的推理步骤都在模型的参数空间里完成。模型写下"第一步……第二步……"，但这些步骤无法触发外部工具，也无法验证中间结论的真实性。一旦训练数据里没有相关信息，模型就会开始"合理推断"，也就是大家熟悉的幻觉现象。

更麻烦的是，CoT 里一旦某个中间步骤出错，后续的推理会在错误的基础上继续演绎，最终得出一个听起来完全合理、但事实上完全错误的结论。模型不会停下来质疑自己，因为它没有外部事实的参照物。

ReAct 引入了外部"行动"节点来打断这个问题。每次 Thought 之后，模型可以输出一个 Action（调用工具、查询 API、执行代码）；工具返回 Observation 之后，模型再继续 Thought，根据真实数据调整推理方向，直到输出 `Final Answer` 为止。中间任何一步出错，Observation 里的真实数据都可以把推理方向纠正回来。

| 能力 | CoT | ReAct |
|------|-----|-------|
| 拆分推理步骤 | ✅ | ✅ |
| 调用外部工具 | ❌ | ✅ |
| 基于真实数据自我纠错 | ❌ | ✅ |
| 适合纯逻辑推理 | ✅ | ✅（但杀鸡用牛刀）|
| 实现复杂度 | 低 | 中 |

另一个常被忽视的点是：ReAct 并不是 CoT 的升级替代品，两者可以共存。你可以在 Thought 步骤里使用 CoT 风格的多步推理，在 Action 步骤里触发工具，在 Observation 之后继续用 CoT 分析工具返回的数据。把两者结合起来，在纯推理段用 CoT 提高准确率，在需要外部数据时切换到 Action，是生产级 Agent 里最常见的模式。

需要说清楚的是：纯逻辑推理题用 CoT 就够了。强行加 ReAct 只会增加格式错误的概率，同时浪费工具调用的 latency。ReAct 的价值在于它能接上外部真实数据，没有这个需求就不要用。

## ReAct 的 Prompt 结构

ReAct prompt 分两个部分：system-level 的角色定义和工具说明，加上至少一个完整的 few-shot 示例。两个部分缺一不可。

工具说明需要写清楚三件事：工具名称、参数类型和期望的返回格式。模糊的工具描述是 ReAct 跑不起来的最常见原因之一——模型猜不清楚该传什么参数，就会随机生成一个"看起来对"的参数，然后工具调用失败。

```
你是一个能使用工具的助手。每一步必须严格按照以下格式输出：

Thought: <当前推理，说明为什么选择这个工具，1-3 句>
Action: <工具名>[<参数>]
Observation: <工具返回值，由系统填入，不要自己编造>
...（Thought / Action / Observation 可重复多轮）
Final Answer: <最终回答，直接给结论>

可用工具：
- search[自然语言查询]    → 返回网络搜索摘要，适合实时信息
- calculator[数学表达式]  → 返回计算结果，仅支持纯数学表达式
- lookup[精确关键词]      → 查询内部知识库，返回相关段落
```

few-shot 示例是格式能否被模型稳定学会的关键。一个只有 `Thought → Final Answer` 的示例会让模型学会"绕过工具直接回答"。有效的示例必须包含至少一次完整的 `Thought → Action → Observation → Thought → Final Answer` 循环，让模型看到什么时候应该用工具、用之后怎么继续推理。

few-shot 示例的质量直接决定 ReAct 的稳定性。一个好的示例需要展示：遇到不确定的信息时主动调用工具而不是猜测，Action 参数尽量简洁而非把整段自然语言直接塞进去，在 Observation 内容不完整时知道换一个工具或换一个参数再查一次，而不是硬把缺失信息脑补上去。把这几个行为模式通过示例展示给模型，比在 system prompt 里写十条规则要有效得多。

示例要贴近你的实际使用场景，别直接复制论文里的 Wikipedia 查询示例然后套到数据库场景上用——两个场景的工具返回格式差距很大，模型会被混淆。

## 实战：一个可落地的 Python 循环

ReAct 的工程实现比 CoT 多一个核心步骤：在模型输出 Action 后立即暂停，由代码注入真实的 Observation，再让模型继续。这个暂停靠 `stop_sequences` 实现。

```python
import re

TOOLS = {
    "calculator": lambda expr: str(eval(expr, {"__builtins__": {}})),
    "lookup": lambda term: knowledge_base.get(term, "未找到相关条目"),
    "search": lambda query: web_search(query),
}

def react_loop(client, user_query, max_steps=6):
    messages = [{"role": "user", "content": user_query}]

    for step in range(max_steps):
        resp = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
            stop_sequences=["Observation:"],  # Action 输出后立刻暂停
        )
        text = resp.content[0].text
        messages.append({"role": "assistant", "content": text})

        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()

        # 解析 Action，兼容全角冒号和中文括号
        match = re.search(r"Action[：:]\s*(\w+)[【\[](.+?)[】\]]", text)
        if not match:
            break  # 格式错误，终止而不是死循环

        tool, arg = match.group(1), match.group(2)
        obs = TOOLS.get(tool, lambda x: "工具不存在")(arg)

        # 截断 Observation，防止上下文膨胀
        obs = obs[:600] + "…（已截断）" if len(obs) > 600 else obs

        # 以 user 身份注入 Observation，让模型继续推理
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return "未能在规定步骤内得出结论"
```

`stop_sequences=["Observation:"]` 是整个实现里最重要的一行。没有它，模型会在 Action 之后继续生成，自己编出一个 Observation——这等于让模型在幻想工具的返回值，ReAct 的整个意义就消失了。有了 stop_sequences，模型输出 Action 后就停下来等待，代码填入真实数据后，模型才继续下一步推理。

Observation 以 `user` 身份注入而不是拼接到 `assistant` 消息里，原因是要让模型清楚地知道这是外部数据、不是自己生成的内容。把 Observation 混入 assistant message 是一个很容易犯的错误，会导致模型在后续推理里把"自己的话"和"工具数据"混淆。

## 三个让 ReAct 跑稳的配置细节

**格式要锁死，同时用宽松 regex 兜底。** 模型有时候会把 `Action: search[...]` 写成 `Action：search（...）`，把全角标点混进来。在 system prompt 里明确给出错误格式的反例，同时在代码里的 regex 里兼容常见的全角变体。两手都要准备——靠 prompt 减少错误率，靠代码容忍残余错误。

**Observation 必须截断，不是可选的。** 工具返回的内容可能非常长：一次数据库查询轻松返回几千 token，一个 API 响应可能包含大量无关字段。直接注入会导致上下文在第二三轮就膨胀到 context window 的限制附近，模型在后续步骤输出质量明显下降，而且每轮调用费用急剧上升。截断到 500-800 token 是经验起点，根据具体任务的信息密度调整。

**max_steps 不是可选的，超出要记录 trace。** 遇到模型认为无法解决的问题，它会进入反复尝试不同工具、循环推理的模式，永远不输出 Final Answer。`max_steps=6` 是起点，更复杂的任务可以放宽到 10，但必须有上限。超出后不要直接抛错，而是把到目前为止的完整 messages 序列存下来——这个 trace 是你后续优化 prompt 和工具描述时最有价值的调试素材。

## 踩坑清单

- **few-shot 示例不完整**：只给 Thought → Final Answer 的示例，模型就不会用工具，会直接从参数知识里编答案。
- **没有 stop_sequences**：模型自填 Observation，ReAct 变成换了个格式的幻觉生成。这是最常见的致命错误。
- **工具描述含糊**：`search[query]` 里的 query 是自然语言还是布尔关键词？不写清楚，模型只能猜，工具命中率会低到让你怀疑整套方案。
- **Observation 注入方式搞错**：拼进 assistant message 导致模型把外部数据当自己生成的内容，后续推理边界模糊。
- **在不需要工具的场景里用 ReAct**：数学逻辑推理、写作润色、代码解释这类任务不需要外部工具，强行套 ReAct 格式只会增加解析复杂度和出错概率。

理解了 ReAct，你会发现几乎所有 AI Agent 框架底层都在跑 Thought / Action / Observation 这个循环，只是用不同的抽象层封装了起来。剥掉框架的外壳，它就是这几百行代码。自己实现一遍，对调试框架层的问题会有质的帮助。

决定 ReAct 能否在生产里跑稳的，往往不是模型能力，而是 prompt 里工具描述的清晰度、few-shot 示例的完整度，以及代码层对异常格式的容错处理。这三件事做扎实了，ReAct 循环的稳定性会比你预期的高很多。
