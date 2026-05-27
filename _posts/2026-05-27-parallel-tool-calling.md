---
layout: post
title: 并行工具调用实战 🔀 — 一次请求，多把刀同时挥
date: 2026-05-27
topic: "Agent 与工具"
tags: [Agent, Tool Calling, Parallel, LLM, Anthropic]
excerpt: 大多数 Agent 教程默认顺序调用工具，但 Claude 和 GPT-4o 早就支持一次 response 返回多个 tool_use block。本文拆解并行工具调用的触发机制、Anthropic SDK 正确写法、串行 vs 并行决策树，以及三个能让你在生产环境踩坑的边界情况。
permalink: /posts/2026-05-27-parallel-tool-calling.html
---

你写了一个"早报助手"：查北京天气、抓今日 AI 新闻、读用户日历。默认实现大概是这样——先问天气，等结果，再问新闻，再等，最后问日历。三次 LLM round-trip，加上三次上游 API 延迟，全部串联。

三件毫无依赖关系的独立任务，被串成一条链，白白翻了两三倍的等待时间。

现代 LLM 的 tool use API 有一个鲜为人知的能力：一次 response 可以返回**多个** `tool_use` block，告诉你"这几件事同时去做"。这就是并行工具调用（Parallel Tool Calling）。用好它，某些场景下延迟能砍掉 60% 以上。

## 模型如何决定"并行还是串行"

模型并不总是并行。它有自己的判断逻辑，核心标准是**工具之间是否存在数据依赖**。

**会并行的情况**：工具的输入不依赖彼此的输出。"查北京天气"和"查上海天气"完全独立，"搜索论文摘要"和"查数据库用户信息"也互不干扰——模型会在同一轮返回两个 `tool_use` block。

**不会并行的情况**：后一个工具需要前一个的返回值。"先根据用户名搜索 user_id，再用 user_id 拉取订单列表"——这个顺序不能乱，模型会自动串行。

Claude 3 系列起支持多 `tool_use` block 并行，OpenAI GPT-4o 同样支持。判断权在模型端，你能影响它的方式是：**在 tool description 里清楚说明工具的独立性**。加上类似"此工具与其他工具无输入依赖，可与其他工具并发调用"的说明，确实会提升并行触发概率。

## Anthropic SDK 的正确写法

先看触发并行的请求姿势：

```python
import anthropic
import asyncio

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "获取指定城市的当前天气，与其他工具无依赖关系",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如北京"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_news_headlines",
        "description": "获取指定关键词的最新新闻头条，与其他工具无依赖关系",
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"}
            },
            "required": ["keyword"]
        }
    },
    {
        "name": "get_calendar_events",
        "description": "读取今日日历事项，与其他工具无依赖关系",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "格式 YYYY-MM-DD"}
            },
            "required": ["date"]
        }
    }
]

response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    tools=tools,
    messages=[{
        "role": "user",
        "content": "给我今天北京的天气、最新的 AI 新闻，以及今天的日程安排"
    }]
)

# 提取所有 tool_use block（而不是只取 [0]）
tool_calls = [b for b in response.content if b.type == "tool_use"]
print(f"模型返回了 {len(tool_calls)} 个并行工具调用")
# 输出：模型返回了 3 个并行工具调用
```

## 并发执行与结果回传

拿到多个 `tool_use` block 后，流程分两步：

**第一步**：用 `asyncio.gather` 并发执行所有工具调用，而不是逐个等待。

**第二步**：把所有 `tool_result` 打包进**同一条** `user` 消息，一次性回传模型。这是最关键的约束——同一轮的结果不能拆成多条消息分开发，否则 Claude 会报错或产生混乱输出。

```python
sem = asyncio.Semaphore(5)  # 控制对外部 API 的最大并发数

async def execute_tool(tool_call) -> dict:
    """派发并执行单个 tool_use，返回 tool_result 格式"""
    async with sem:
        name = tool_call.name
        inp = tool_call.input
        
        if name == "get_weather":
            result = await fetch_weather(inp["city"])
        elif name == "get_news_headlines":
            result = await fetch_news(inp["keyword"])
        elif name == "get_calendar_events":
            result = await fetch_calendar(inp["date"])
        else:
            result = f"未知工具: {name}"
        
        return {
            "type": "tool_result",
            "tool_use_id": tool_call.id,   # 必须与请求里的 id 对应
            "content": str(result)
        }

async def run_parallel_agent(user_message: str) -> str:
    # 第一轮：让模型决定调用哪些工具
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": user_message}]
    )
    
    tool_calls = [b for b in response.content if b.type == "tool_use"]
    if not tool_calls:
        return response.content[0].text
    
    # 并发执行，比串行快 N 倍（N = 工具数量）
    results = await asyncio.gather(*[execute_tool(tc) for tc in tool_calls])
    
    # 所有 tool_result 合并进同一条 user 消息
    followup = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2048,
        tools=tools,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": list(results)}  # 关键：list 格式，不是字符串
        ]
    )
    return followup.content[0].text
```

## 决策树：什么时候用并行

| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| 多个独立数据源（天气+新闻+日历） | 并行 | 无依赖，等待时间取最长者而非求和 |
| 批量同类查询（查 5 个城市气温） | 并行 | 天然独立 |
| 步骤有依赖（搜索→拿 ID→查详情） | 串行 | 后者的 input 来自前者的 output |
| 写操作后立刻读（INSERT→SELECT） | 串行 | 顺序敏感，并行会读到旧数据 |
| 有副作用且顺序敏感（扣款+发通知） | 串行 | 并行风险：扣款失败但通知已发出 |
| 上游 API 有严格 rate limit | 受控并行 | 加 Semaphore，并发数 ≤ rate limit |

## 三个生产级踩坑点

**坑一：只处理第一个 tool_use**

很多示例代码里有 `block = response.content[0]`，默认只有一个工具调用。并行场景下，第二、三个工具调用直接被丢弃，结果看起来"正常"但数据缺失——这个 bug 很难被单测覆盖到。改法：`[b for b in response.content if b.type == "tool_use"]`。

**坑二：tool_result 拆成多条 user 消息**

把三个结果分三条消息发回，Claude 的消息历史结构会不合法，要么报 API error，要么模型看到不完整上下文给出奇怪答案。正确做法：所有同轮结果必须打包进单条 user 消息的 `content` 数组。

**坑三：忽略外部 API 的并发限制**

`asyncio.gather` 默认无限并发。如果你同时触发了 10 个 tool_use，而上游 API 的 rate limit 是 5 QPS，你会在高负载下密集踩 429。一个 `asyncio.Semaphore(N)` 解决问题，N 按上游限制设定。

---

**踩坑清单**：
- [ ] 遍历所有 `tool_use` block，不要取 `[0]`
- [ ] 所有同轮 tool_result 合并进单条 user 消息
- [ ] 用 Semaphore 控制对外部 API 的最大并发数
- [ ] 有副作用或依赖顺序的工具链，不要强行并行
- [ ] 用至少 2 个独立意图的 prompt 验证模型是否真的触发了并行
- [ ] 检查 tool description 是否清楚表明工具间的独立性
