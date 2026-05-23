---
layout: post
title: MCP 协议入门 — 让 AI Agent 之间互通的开放标准
date: 2026-05-11
category: "Agent 与工具"
tags: [AI, MCP, Agent, 协议]
excerpt: Model Context Protocol 是 Anthropic 提的开放协议，正在变成 agent 工具集成的事实标准。看看它解决什么问题、怎么写一个 MCP server。
permalink: /posts/2026-05-11-mcp-protocol.html
---

## 为什么需要这个协议

LLM 应用越来越复杂，每家都在做"自己的工具集成方式"：

- OpenAI 的 function calling
- Anthropic 的 tools
- LangChain 的 Tool 抽象
- 各种 agent 框架的私有协议

结果是——**一个"查数据库"的工具，要给每个 agent 框架重写一遍**。
插件生态各自为战，工具的复用性几乎为 0。

MCP（Model Context Protocol）就是要解决这个：**统一 agent 和外部能力之间的接口**。
有点像"AI 时代的 USB"——一个口，插哪个 agent 都能用。

## MCP 的核心抽象

MCP 把"外部能力"抽象成三类：

### 1. Resources（资源）

只读的数据 / 文件 / 数据库表 / API 响应。
LLM 想看"我有哪些资源"时，MCP server 列出可用资源；LLM 想读某个时，MCP server 返回内容。

例子：本地文件、Notion 页面、GitHub PR 列表。

### 2. Tools（工具）

可执行的动作。LLM 调它会触发**有副作用**的操作。

例子：发邮件、写文件、执行 SQL、调外部 API。

### 3. Prompts（预制提示词模板）

server 端预定义好的 prompt 模板，client 可以拿来用。
适合给"特定场景"提供推荐用法。

例子：`/summarize-pr` 这种命令在 MCP server 里是一个 prompt 模板。

## 谁是 Client、谁是 Server

- **MCP Server**：暴露 resources / tools / prompts 的程序。比如一个 "GitHub MCP server" 暴露 PR / issue / commit 相关的工具
- **MCP Client**：使用这些能力的 AI 应用。Claude Desktop / Cursor / VSCode 都是 MCP client

通信走 JSON-RPC，传输可以是 stdio / SSE / WebSocket。

## 一个最小的 MCP server

用官方 Python SDK 写一个"会查天气"的 server：

```python
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent
import mcp.server.stdio
import httpx

server = Server("weather-mcp")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description="查询给定城市的当前天气",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"],
            },
        )
    ]

@server.call_tool()
async def call_tool(name: str, args: dict) -> list[TextContent]:
    if name == "get_weather":
        async with httpx.AsyncClient() as client:
            r = await client.get(f"https://wttr.in/{args['city']}?format=j1")
            data = r.json()
            return [TextContent(type="text", text=f"{args['city']} 当前天气：{data['current_condition'][0]['weatherDesc'][0]['value']}, 温度 {data['current_condition'][0]['temp_C']}°C")]
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, InitializationOptions(
            server_name="weather-mcp",
            server_version="0.1.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            ),
        ))
```

把这个 server 注册到 Claude Desktop 或 Cursor 的配置里，agent 就能调它了。

## MCP 生态正在爆发

到 2026 年中，社区已经有几百个开源 MCP server：

- **本地操作**：filesystem / git / shell / docker
- **数据**：Postgres / SQLite / Elasticsearch
- **SaaS**：GitHub / Slack / Notion / Linear / Stripe
- **搜索**：Brave Search / Tavily / Perplexity
- **自动化**：Puppeteer / Playwright

很多公司也在把内部工具 MCP 化——以后**给员工新装的 agent**直接连上公司的 MCP server，立刻能用所有内部能力。

## 跟 Function Calling 的区别

| | Function Calling | MCP |
|---|---|---|
| 定义在哪 | 每次 API 调用时传 | server 暴露 |
| 谁负责 | 应用代码 | 独立进程 |
| 复用性 | 应用内 | 跨应用 |
| 适用场景 | 应用绑定的工具 | 通用能力 |

理解上：**function calling 是 LLM 层面的能力，MCP 是工具层面的协议**。两者不冲突，互补。

## 实操建议

### 用现成的 server

90% 的需求都有现成 MCP server 实现。先去 [github.com/modelcontextprotocol](https://github.com/modelcontextprotocol) 翻翻有没有现成的，没必要自己造轮子。

### 自己写 server 的场景

- 暴露公司内部 API
- 包装某个特定业务流程
- 桥接老系统（SOAP / 私有协议）

### 给 server 做好 description

跟 tool 一样，**写得好的 description 决定 agent 用得对不对**。
"Get user data" 不够，要写清楚"在什么场景用、返回什么字段、有什么不能用的"。

## 一个朴素观察

MCP 现在的状态有点像 2015 年的 Docker——
"我能不能不用它"还有讨论空间，
但**到了 2027 年，再不用它的人会发现自己跟世界脱节**。

会写 MCP server 已经是 AI 工程师的基本功了。
