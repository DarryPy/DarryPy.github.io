---
layout: post
title: "Agent 可观测性 — 多步骤任务的 Trace / Log / Metrics"
date: 2026-06-24
topic: "Agent 与工具"
tags: [AI, Agent, Observability]
excerpt: Agent 任务出了问题，你靠什么调试？打印 print 够吗？不够。你需要结构化的 trace、每一步的输入输出、工具调用记录、成本统计。
permalink: /posts/2026-06-24-agent-observability.html
---

## 为什么 Agent 的可观测性特别难

普通 HTTP 服务出问题，看日志，找对应请求的错误行，定位。

Agent 出问题：
- 任务可能跑了 20 步
- 每步都调了不同工具
- 中途 LLM 做了某个决策导致路径偏了
- 到第 15 步才暴露错误
- 中间有分叉、重试、状态变更

你需要的不是日志，是**整个执行树的快照**。

## 需要捕捉什么

### 最小可用集

每一步 LLM 调用记录：
```
{
  "span_id": "abc123",
  "parent_span_id": "root001",
  "timestamp": "2026-06-24T10:23:11Z",
  "model": "claude-sonnet-4-6",
  "input_tokens": 1250,
  "output_tokens": 387,
  "latency_ms": 2341,
  "cost_usd": 0.0034,
  "input_messages": [...],   // 完整的 messages 数组
  "output_message": {...},   // 模型回复
  "tool_calls": [            // 如果有工具调用
    {
      "tool_name": "web_search",
      "input": {"query": "..."},
      "output": "...",
      "latency_ms": 890
    }
  ]
}
```

### 完整 Trace 结构

```
Task: "Research and summarize recent AI papers"
├── Step 1: Planning [LLM call, 1.2s]
│   └── Output: "需要搜索 3 个方向"
├── Step 2: Tool call - web_search("latest transformer papers 2026") [0.8s]
│   └── Output: 10 results
├── Step 3: Tool call - web_search("diffusion model advances 2026") [0.7s]
│   └── Output: 8 results
├── Step 4: LLM call - analyze results [2.1s]
│   └── Output: 摘要草稿
├── Step 5: LLM call - refine and format [1.8s]
│   └── Output: 最终答案
│
Total: 5 steps, 6.6s, $0.012
```

## 工具选择

### LangSmith

LangChain 官方的可观测平台。如果你用 LangChain/LangGraph，几行代码就能接入：

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "my-agent"

# 之后正常用 LangChain，自动上报所有 trace
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

model = ChatAnthropic(model="claude-sonnet-4-6")
agent = create_react_agent(model, tools=[...])
# 所有调用自动被 trace
```

### Langfuse（开源，可自托管）

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"  # 或自托管地址
)

@observe()  # 自动追踪这个函数
def run_agent_step(step_name: str, messages: list) -> str:
    langfuse_context.update_current_observation(
        name=step_name,
        metadata={"step_type": "llm_call"}
    )
    # ... LLM 调用
```

### Phoenix（Arize，适合本地开发）

```python
import phoenix as px
from openinference.instrumentation.anthropic import AnthropicInstrumentor

# 启动本地 UI（localhost:6006）
px.launch_app()

# 一行注入所有 Anthropic 调用
AnthropicInstrumentor().instrument()

# 之后正常调用 Anthropic SDK，Phoenix 自动捕捉
import anthropic
client = anthropic.Anthropic()
# 所有 client.messages.create() 调用都会出现在 Phoenix UI
```

### 自己实现（不依赖第三方）

```python
import time
import uuid
import json
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import anthropic

@dataclass
class Span:
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    status: str = "ok"  # ok | error

    def finish(self):
        self.end_time = time.time()

    @property
    def duration_ms(self):
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

class Tracer:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.trace_id = str(uuid.uuid4())[:12]
        self.spans: list[Span] = []
        self._current_span_id: Optional[str] = None

    @contextmanager
    def span(self, name: str, **attributes):
        s = Span(
            name=name,
            parent_id=self._current_span_id,
            attributes=attributes
        )
        self.spans.append(s)
        prev = self._current_span_id
        self._current_span_id = s.span_id
        try:
            yield s
        except Exception as e:
            s.status = "error"
            s.attributes["error"] = str(e)
            raise
        finally:
            s.finish()
            self._current_span_id = prev

    def llm_call(self, client: anthropic.Anthropic, **kwargs):
        """包装 LLM 调用，自动记录 token 和延迟"""
        with self.span("llm_call", model=kwargs.get("model")) as s:
            response = client.messages.create(**kwargs)
            s.attributes.update({
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "stop_reason": response.stop_reason,
            })
            return response

    def tool_call(self, tool_name: str, tool_input: Any, tool_fn, **kwargs):
        """包装工具调用"""
        with self.span(f"tool:{tool_name}", input=str(tool_input)[:200]) as s:
            result = tool_fn(tool_input, **kwargs)
            s.attributes["output_preview"] = str(result)[:200]
            return result

    def report(self) -> dict:
        total_cost = sum(
            (s.attributes.get("input_tokens", 0) * 3 +
             s.attributes.get("output_tokens", 0) * 15) / 1_000_000
            for s in self.spans
            if "input_tokens" in s.attributes
        )
        return {
            "trace_id": self.trace_id,
            "task": self.task_name,
            "total_steps": len(self.spans),
            "total_duration_ms": sum(s.duration_ms or 0 for s in self.spans),
            "estimated_cost_usd": round(total_cost, 6),
            "spans": [asdict(s) for s in self.spans],
        }

# 使用示例
def run_research_agent(topic: str):
    tracer = Tracer(f"research:{topic}")
    client = anthropic.Anthropic()

    with tracer.span("planning", topic=topic):
        plan_response = tracer.llm_call(
            client,
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": f"给出研究 '{topic}' 的3个方向，简短列举"}]
        )
        plan = plan_response.content[0].text

    # 模拟工具调用
    def mock_search(query):
        return f"搜索结果：{query} 相关文章 5 篇"

    results = tracer.tool_call("web_search", topic, mock_search)

    with tracer.span("synthesis"):
        final = tracer.llm_call(
            client,
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[
                {"role": "user", "content": f"计划：{plan}\n检索结果：{results}\n请综合写出研究摘要"}
            ]
        )

    report = tracer.report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return final.content[0].text

run_research_agent("量子计算最新进展")
```

## 从 Trace 调试卡住的 Agent

典型场景：Agent 任务跑了 12 步后"卡死"，没有完成也没有报错。

**调试步骤：**

1. **找最后一个 span**：看最后成功完成的步骤是什么
2. **看那步的 LLM 输出**：模型输出了什么？是否发出了工具调用？
3. **看工具调用结果**：工具有没有返回？返回了什么？
4. **找"循环"模式**：如果同一个工具被调了 5+ 次，说明 agent 陷入了循环
5. **看 token 消耗曲线**：如果某步 output_tokens 突然变大，说明模型在生成废话

常见卡死原因：

```python
# 问题1：工具返回空，模型不知道怎么继续
def search_tool(query):
    results = db.search(query)
    return results  # 可能返回 []

# 修复：明确告诉 agent 没有结果时怎么办
def search_tool(query):
    results = db.search(query)
    if not results:
        return "没有找到相关结果。请尝试更换关键词或换一个方向。"
    return results

# 问题2：没有设置最大步数限制
# 修复：
MAX_STEPS = 15
for step in range(MAX_STEPS):
    ...
else:
    tracer.spans[-1].attributes["warning"] = "达到最大步数限制，强制终止"
```

## 关键 Metrics 该监控什么

```
生产 Agent 服务需要 alert 的指标：

1. 任务成功率 < 85% → 模型或工具有问题
2. 平均步数突然增加 > 20% → agent 可能在循环
3. p95 任务耗时 > SLA → 某工具变慢
4. 每任务成本 > 阈值 → 某次 context 爆了
5. 错误率 > 1% → 工具或 API 异常
```

## 一个朴素结论

> Agent 可观测性的核心只有一件事：**把每一步的输入、输出、耗时、费用都记下来**。
>
> 工具选哪个不重要，重要的是你出了问题能够回放整个执行过程。
> 没有 trace 的 agent 就是黑盒，出了 bug 只能重跑猜原因。
