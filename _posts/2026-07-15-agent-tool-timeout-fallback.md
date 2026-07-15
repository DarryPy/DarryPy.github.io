---
layout: post
title: Agent 工具超时与降级策略 — 工具挂了 Agent 别跟着挂
date: 2026-07-15
topic: "Agent 与工具"
tags: [Agent, 工具调用, 超时, 降级, 可靠性]
excerpt: 工具调用是 Agent 最脆弱的那一环。超时设错、重试不分场合、没有降级路径——任何一个缺失都能让整条 pipeline 僵死。本文从超时分档、指数退避、fallback 设计到一个可复用的 ToolRunner，把工具调用可靠性从"碰运气"变成可控的工程问题。
permalink: /posts/2026-07-15-agent-tool-timeout-fallback.html
---

你的 Agent 调用了一个外部 API，然后……没有然后。30 秒过去了，60 秒过去了，整条 pipeline 就这样僵在那里。这不是假设，是大多数生产 Agent 都踩过的坑。

工具调用的可靠性 = 工具本身的可靠性 × 调用策略的健壮性。前者你控制不了，后者你完全可以设计。

## 工具调用会以哪三种方式"挂"

外部工具挂掉的原因很多：网络抖动、下游过载、响应体过大、接口变更、速率限制。对 Agent 来说，这些问题有三种表现：

**超时（Timeout）**：工具一直没返回，Agent 傻等。最危险，因为 Agent 感知不到异常，只会越等越久。

**报错（Error）**：工具返回 4xx/5xx 或抛出异常。相对好处理，至少有明确的错误信号。

**静默失败（Silent Failure）**：工具返回 200，但结果是空的或乱码。最难排查，因为 Agent 会把错误数据当正常数据继续处理。

三种里超时最容易被忽视，也最容易把整个任务拖垮。先从这里入手。

## 超时：别让 Agent 无限等待

给每个工具调用设置硬超时是最基础也最容易被忽视的一步。Python 里用 `asyncio.wait_for` 最直接：

```python
import asyncio

async def call_tool_with_timeout(tool_fn, args: dict, timeout: float = 10.0):
    try:
        result = await asyncio.wait_for(tool_fn(**args), timeout=timeout)
        return result
    except asyncio.TimeoutError:
        return {"error": "timeout", "tool": tool_fn.__name__, "args": args}
```

关键是超时值不能全局统一，要按工具类型分档：

| 工具类型 | 建议超时 |
|---|---|
| 搜索 / 轻量 API | 5s |
| 数据库查询 | 10s |
| 代码执行 | 30s |
| 文件处理 / 大模型嵌套调用 | 60s+ |

不同工具的 SLA 差距可以是 10 倍以上，用统一超时要么太短（误杀慢工具）要么太长（保护不了快工具）。经验值：用该工具 P95 响应时间 × 1.5 来设超时，而不是 P50 × 3。

超时后务必记录结构化日志，不然线上抖动完全不可见。

## 重试：指数退避的正确姿势

工具报错后是否重试，要先判断错误类型，不能无脑重发：

| 错误类型 | 是否重试 | 原因 |
|---|---|---|
| 网络抖动（502/504） | 是 | 临时性，重试大概率成功 |
| 速率限制（429） | 是，带退避 | 等一等就好 |
| 认证失败（401/403） | 否 | 重试没用，先修鉴权 |
| 参数错误（400） | 否 | LLM 给的参数有问题，重发一样错 |
| 写操作超时 | 谨慎 | 先确认幂等性再决定 |

指数退避加抖动的基本写法：

```python
import random, asyncio

async def retry_with_backoff(
    tool_fn, args: dict,
    max_retries: int = 3,
    base_delay: float = 1.0
):
    last_error = None
    for attempt in range(max_retries):
        try:
            return await call_tool_with_timeout(tool_fn, args)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # 指数退避 + 随机抖动，避免多 Agent 同时打爆下游
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(delay)
    raise last_error
```

两个细节值得重点记：加 jitter 是为了防止多个 Agent 实例在同一时间集体重试把下游打崩；写操作（POST/PUT/DELETE）在重试前先确认幂等性，不然一个订单可能被重复创建——这类 bug 在测试环境几乎看不出来，生产上才会炸。

## 降级：备用路径不是可选项

重试耗尽还是失败了怎么办？这时候需要降级策略。几种常见模式：

**备用工具降级**：主工具是实时搜索 API，备用工具是本地知识库。速度慢一点，但不会彻底失败：

```python
async def search_with_fallback(query: str):
    try:
        return await web_search_tool(query, timeout=8)
    except Exception:
        return await local_rag_search(query)
```

**缓存降级**：如果这个查询最近 5 分钟执行过，直接用缓存结果。对天气、汇率、新闻这类实时性要求不高的工具特别有效，能同时降成本和提可靠性。

**跳过并标注**：当工具对当前任务不是关键路径时，允许 Agent 在结果里注明"工具暂不可用，以下内容基于已有知识生成"，继续完成剩余任务。不要因为一个边缘工具挂了就让整条链路停摆。

**人工介入**：对于高风险操作（发邮件、写数据库、触发支付），工具失败时发告警等人确认，而不是盲目重试或默默跳过。

## 实战：把三层策略封进一个 ToolRunner

把超时、重试、降级三件事合在一个可复用的类里，使用时按工具独立配参：

```python
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
import asyncio, random

@dataclass
class ToolConfig:
    timeout: float = 10.0
    max_retries: int = 2
    base_delay: float = 1.0
    fallback: Optional[Callable] = None
    retryable_exceptions: tuple = (asyncio.TimeoutError, ConnectionError)

class ToolRunner:
    def __init__(self, tool_fn: Callable, config: ToolConfig = ToolConfig()):
        self.tool_fn = tool_fn
        self.cfg = config

    async def run(self, **args) -> Any:
        last_error = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self.tool_fn(**args),
                    timeout=self.cfg.timeout
                )
            except self.cfg.retryable_exceptions as e:
                last_error = e
                if attempt < self.cfg.max_retries:
                    delay = self.cfg.base_delay * (2 ** attempt) + random.uniform(0, 0.3)
                    await asyncio.sleep(delay)
            except Exception as e:
                # 非可重试异常直接跳出重试循环
                last_error = e
                break

        if self.cfg.fallback:
            return await self.cfg.fallback(**args)

        raise last_error
```

按工具类型单独配置，而不是全局一刀切：

```python
# 搜索工具：快超时 + 本地 RAG 兜底
search_runner = ToolRunner(
    web_search,
    ToolConfig(timeout=6, max_retries=2, fallback=local_rag_search)
)

# 代码执行：长超时 + 不自动重试（避免重复执行副作用）
code_runner = ToolRunner(
    execute_code,
    ToolConfig(timeout=30, max_retries=0)
)

# 数据库写入：中等超时 + 不降级（宁可失败也不走备用写错数据）
db_writer = ToolRunner(
    write_to_db,
    ToolConfig(timeout=15, max_retries=1, fallback=None)
)
```

调用时只需要 `await search_runner.run(query="...")` 即可，超时、重试、降级全部在 runner 内部消化。

---

**踩坑清单，对照检查**

- 超时设太长 → Agent 假死，用户以为系统挂了，实际只是在等一个永远不会成功的 HTTP 请求
- 写操作不确认幂等直接重试 → 数据重复、双重扣款、消息重复发送
- 所有工具用同一套重试参数 → 一个失败任务重试链路可能拖出 20s 以上的等待
- fallback 从来没测过 → 主路径挂了，才发现备路径也挂了，而且没有告警
- 静默失败不校验返回值 → 空结果进入下一步工具调用，错误沿链路扩散，最后一步才爆
- 超时后不记日志 → 线上偶发抖动完全不可见，只能靠用户反馈发现问题
