---
layout: post
title: LLM 请求链路追踪实战 🔍
date: 2026-07-18
topic: "工程实战"
tags: [tracing, observability, agent, OpenTelemetry, LLM]
excerpt: 当一条用户请求穿过多个 Agent、多次 LLM 调用、工具函数和外部 API，你怎么知道哪一步慢了、哪一步错了？本文带你用 OpenTelemetry 把整条链路打通，真正做到"看得见、查得快、改得准"。
permalink: /posts/2026-07-18-llm-distributed-tracing.html
---

你有没有遇到这种情况：用户反馈"回答很慢"，你打开日志，看到的是一堆割裂的 `tool_call`、`llm_response`、`api_error`——根本无法还原那条请求到底经历了什么，只能靠时间戳拼凑，猜了半小时还没定位到问题。

这就是 LLM 应用缺少链路追踪（Distributed Tracing）的代价。单体应用还好说，顺序执行，日志基本够用。但现在的 AI 应用动辄三五层：用户请求进来，先到 Orchestrator Agent 做意图理解，再分发给子 Agent 执行具体任务，子 Agent 调工具函数，工具函数打外部 API，结果回流给 LLM 做二次整合，最终才给用户。每一层都有延迟，每一层都可能出错，每一层的 context 都在变。没有 tracing，你的排障效率约等于瞎猜。

## 为什么日志和 Metrics 不够

很多团队在上了 Prometheus 和结构化日志之后，会误以为可观测性已经"做好了"。这是个认知误区，需要把三种工具的定位说清楚。

Metrics 是聚合数据，告诉你"整体成功率是 98%、平均延迟 3.2 秒"，但无法回答"这一次特定请求，哪个环节花了 2.8 秒"。它适合监控和告警，不适合排障。

日志是离散事件，每一条日志描述某一时刻发生了什么，但日志天然是割裂的。你有 Agent A 的日志，有 Agent B 的日志，但它们之间的调用关系、时序依赖、参数传递——这些在日志里要么没有，要么需要你人工关联一个 `request_id`，然后在几十个文件里 grep。

只有 Trace 能把一条请求的完整调用树还原出来：这条请求从哪来、经过了哪些节点、每个节点耗时多少、入参出参是什么、哪一步触发了重试。它是时序的、有层级的、端到端的。三者缺一不可，但在 LLM 应用里，Tracing 往往是最容易被忽视、也最值钱的那个。

有个具体的例子能说明问题：一个多轮对话 Agent，用户说"帮我分析一下这份合同"，系统先调文件解析工具，再把文本切片后做向量检索，检索结果喂给 LLM 做初步分析，LLM 输出不确定时会再调一次搜索引擎补充信息，最后综合两轮结果给出回答。整个过程涉及四五次外部调用，每次调用都有自己的超时和重试逻辑。如果你只有日志，出问题时大概率只能看到"最终响应超时"，根本看不出是哪一层拖慢的。

## 选型：OpenTelemetry + LangFuse

业界目前最实用的方案是以 OpenTelemetry 做标准埋点，再接一个支持 LLM 语义的后端。OpenTelemetry 是 CNCF 的开放标准，一次埋点，后端可以随时切换——今天用 Jaeger，明天换 Grafana Tempo，代码一行不动。

后端选型上，通用 APM 工具（Datadog、Jaeger）能用，但它们不理解 LLM 的概念，不知道 `prompt_tokens` 和 `completion_tokens` 有什么含义，也没有 prompt 版本对比这类功能。LangFuse 专门为 LLM trace 设计，天然支持 token 成本统计、prompt 模板追踪、评分回标，对 AI 应用更合适。两者可以共存：OpenTelemetry 同时导出到 LangFuse 做 AI 侧分析，导出到 Grafana Tempo 做基础设施侧监控。

整体架构很简单：

```
用户请求
  └─ 应用代码（OpenTelemetry 埋点）
       └─ OTLP Exporter（HTTP/gRPC）
            └─ OpenTelemetry Collector
                 ├─ LangFuse（LLM trace 分析）
                 └─ Grafana Tempo（全链路视图）
```

安装只需要几个包：

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp langfuse
```

## 三步把链路串起来

**第一步：初始化全局 Tracer**

这部分在应用启动时做一次，后续所有模块直接 `import tracer` 使用。

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# 生产环境采样 10%，避免把 Collector 打爆
sampler = TraceIdRatioBased(0.1)
provider = TracerProvider(sampler=sampler)
exporter = OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("my-llm-app", schema_url="https://opentelemetry.io/schemas/1.26.0")
```

**第二步：在每次 LLM 调用外包一个 Span**

核心思路是：每次调用 LLM API，用 `start_as_current_span` 包裹，把模型名、token 用量、延迟都记录进 span 属性。这样一条请求里有几次 LLM 调用，Trace 视图里就能看到几个并排或嵌套的 span，每个 span 上都挂着耗时和成本。

```python
def call_llm(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.prompt_preview", prompt[:200])  # 只存前 200 字

        response = anthropic_client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        usage = response.usage
        span.set_attribute("llm.input_tokens", usage.input_tokens)
        span.set_attribute("llm.output_tokens", usage.output_tokens)
        # 成本估算存进去，方便后续按 trace 统计花了多少钱
        cost = usage.input_tokens * 0.000003 + usage.output_tokens * 0.000015
        span.set_attribute("llm.cost_usd", round(cost, 6))
        return response.content[0].text
```

**第三步：跨 Agent 传递 Trace Context**

这是最容易漏掉的一步，也是链路断掉的最常见原因。子 Agent 在独立线程、独立进程或独立服务里运行时，OpenTelemetry 的 context 不会自动跟过去。必须在调用方把 context 注入 HTTP header，在接收方从 header 里提取出来继续。

调用方注入：

```python
from opentelemetry.propagate import inject

def call_sub_agent(payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    inject(headers)  # 注入 traceparent、tracestate 等字段
    resp = requests.post("http://sub-agent/run", json=payload, headers=headers)
    return resp.json()
```

子 Agent 接收方提取：

```python
from opentelemetry.propagate import extract

@app.post("/run")
def sub_agent_handler(request: Request, body: dict):
    ctx = extract(dict(request.headers))  # 从 header 恢复 context
    with tracer.start_as_current_span("sub_agent.execute", context=ctx) as span:
        span.set_attribute("agent.name", "sub_agent_v2")
        result = do_work(body)
        return result
```

做完这三步，打开 LangFuse 或 Jaeger 的 Trace 视图，你就能看到一棵完整的调用树：顶层是用户请求 span，往下是 Orchestrator span，再往下是各个子 Agent span，子 Agent 下面挂着它的 LLM 调用和工具调用，每个节点都有耗时和属性。那种感觉，就像第一次给黑盒系统装上了 X 光机。

## 用 Trace 数据做成本分析

链路追踪最容易被忽视的价值是成本分析。把每次 LLM 调用的 token 用量存进 span 属性，然后在 LangFuse 或 Grafana 里按 trace 聚合，你就能看到：哪类请求消耗 token 最多、哪个子 Agent 是成本大户、同一个用户一天花了多少 API 费用。

这些数据对于优化提示词和控制成本非常直接。比如你发现某类请求平均消耗 8000 input token，仔细一看是 system prompt 里塞了太多示例，把示例压缩到 3 条后，token 用量直接降到 3000，同样的功能，成本砍掉六成。没有 tracing，这种分析得靠手动统计日志，费时费力还容易出错。

另一个实用的用法是给高成本请求加告警。在 Grafana 里设置规则：单次 trace 的总 token 超过阈值就触发告警，帮你快速发现有没有用户在滥用系统或者有没有提示词设计失误导致的无限循环。

## 工具调用（Tool Call）的 Span 规范

工具调用是 Agent 系统里最常见的耗时来源。每次调用外部 API、执行搜索、读数据库，都应该包一个 span，并记录以下属性：

| 属性名 | 示例值 | 说明 |
|--------|--------|------|
| `tool.name` | `search_web` | 工具函数名，方便按工具聚合分析 |
| `tool.input_hash` | `a3f2c1...` | 入参的哈希值，不存原文防止泄露 |
| `tool.duration_ms` | `342` | 工具执行耗时，毫秒 |
| `tool.output_size_bytes` | `1240` | 返回数据字节数 |
| `tool.error_type` | `timeout` | 出错时填写错误类型 |
| `tool.retry_count` | `1` | 重试次数，大于 0 说明有不稳定性 |

把工具调用耗时聚合起来看，往往会发现 80% 的延迟集中在少数几个外部依赖上，这比你靠直觉猜要准确得多。

## 实战踩坑清单

- **进程退出前必须调 `provider.shutdown()`**：`BatchSpanProcessor` 是异步上报的，主进程如果直接退出，队列里还没发出去的 span 全部丢失。在 `atexit` 里注册 shutdown，或者在 FastAPI 的 `lifespan` 里处理。

- **asyncio 任务的 context 要手动传**：`asyncio.create_task()` 会复制当前 context，这点很多人不知道，其实它是安全的。但如果你用的是 `loop.run_in_executor()` 把任务丢到线程池，线程里就没有 context 了，必须用 `contextvars.copy_context().run(fn)` 手动带过去。

- **采样率别在开发环境和生产环境用同一个配置**：开发时全量采样（`ALWAYS_ON`）方便调试；生产高流量下全量采样会把 Collector 撑爆，建议按 `TraceIdRatioBased(0.05~0.1)` 随机采样，同时对出错的请求强制保留（`ParentBased` + 错误采样器）。

- **不要把完整 prompt 存进 span 属性**：LLM 的 system prompt 加上用户消息动辄几千 token，全存进 span 属性，一方面超过 span 属性的大小限制（通常是 4KB），另一方面让 trace 后端存储成本暴增。正确做法是存前 200 字的摘要预览，加上 token 总数和 prompt 模板版本号，足够排障了。

- **子 Agent 出错时不要吞异常**：捕获异常后先调 `span.record_exception(e)` 和 `span.set_status(StatusCode.ERROR, str(e))`，再 re-raise 或返回错误码。如果直接 catch 掉，父 span 只知道"子 Agent 返回了错误响应"，不知道错误发生在哪一行、是什么类型，调试效率会大打折扣。

- **Trace ID 要透传到用户侧**：在 HTTP 响应头里带上 `X-Trace-Id`，或者在前端 UI 的错误提示里展示这个 ID。当用户反馈问题时，直接让他们提供 Trace ID，你在后端一秒就能定位到那条请求的完整调用树，不用再问"你是几点操作的""你输入了什么""当时系统响应了什么"。这个小改动能把排障沟通成本减少一半以上。

---

分布式 tracing 是你给 LLM 应用装上的"飞行记录仪"——平时静静运转、无感知，出事了一键回放。越早装越值钱，等系统复杂了再补，要花三倍的工作量。
