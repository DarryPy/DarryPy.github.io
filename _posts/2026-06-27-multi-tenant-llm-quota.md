---
layout: post
title: 多租户 LLM 应用的配额与隔离 — 从一个用户扩到一千个用户
date: 2026-06-27
topic: "工程实战"
tags: [多租户, 配额管理, Rate Limiting, 工程实战, LLM]
excerpt: 单用户跑得很爽，用户一多，某个大客户把配额打光、别人全挂。这不是概率问题，是架构问题。本文拆解多租户 LLM 应用必须做的三层隔离、三类配额设计、优先级队列和用量告警，每个坑都踩过才知道疼。
permalink: /posts/2026-06-27-multi-tenant-llm-quota.html
---

你做了一个 LLM 应用，跑起来很顺——一个用户，一个数据库，一把 API key，一切美好。然后用户数长到 100、500、1000，突然发现：某个大客户把所有配额打光了，别人全挂。这不是概率问题，是架构问题。

## 隔离层：一个租户失控不能影响其他人

多租户的第一原则：任何一个租户的行为，不管是恶意的还是无意的，都不应该影响到其他人。这需要在三个维度同时隔离。

**API Key 隔离**。不要让所有租户共用一把 provider API key。按租户组（比如 enterprise / pro / free）分配不同的 key，超限只影响本组。这样一个 enterprise 客户突刺，不会把你所有 free 用户的配额也干掉。

**Rate limit 隔离**。在你自己的 gateway 层对每个 `tenant_id` 单独计数，不要依赖 provider 的全局限制：

```python
import redis, time

r = redis.Redis()

def check_rate_limit(tenant_id: str, limit_per_min: int) -> bool:
    key = f"ratelimit:{tenant_id}:{int(time.time() // 60)}"
    count = r.incr(key)
    if count == 1:
        r.expire(key, 120)
    return count <= limit_per_min
```

代码很简单，但这行 `tenant_id` 就是你隔离的边界。没有这条线，就没有隔离。

**数据隔离**。Prompt 历史、对话记录按 tenant 分区存储，row-level security 或 schema 隔离都行，但别把所有人的数据塞进一张没分区的大表——不只是性能问题，是合规问题。

## 配额设计：用量、预算、并发都要管

配额不只是"每分钟多少次"。生产里你需要同时控三类：

| 配额类型 | 示例 | 重置周期 |
|---------|------|---------|
| 请求次数 | 1000 次 / 天 | 日历日 |
| Token 用量 | 5M tokens / 月 | 自然月 |
| 并发席位 | 最多 5 个并发请求 | 实时 |

三者要同时检查，任何一个超限都返回 `429 Quota Exceeded`，响应头里带 `Retry-After` 和剩余量，让客户端知道等多久。

用量数据要异步写入，别让计费 IO 卡在主请求链路上：

```python
import asyncio

async def record_usage(tenant_id: str, tokens_used: int):
    pipe = r.pipeline()
    pipe.incrby(f"usage:{tenant_id}:tokens:month", tokens_used)
    pipe.incrby(f"usage:{tenant_id}:requests:day", 1)
    await asyncio.to_thread(pipe.execute)
```

LLM 响应完成之后，fire-and-forget 这个函数。主链路不等它，也不因为记账失败而报错给用户。记账失败另起告警处理。

## 优先级队列：让 VIP 合理插队

当 LLM API 有并发上限（比如 tier 1 账户只有 5 RPM），你需要一个排队机制，而且 VIP 要能插队，不然付费用户跟免费用户排同一条队，没有意义。

用 Redis Sorted Set 实现优先级队列，score 的计算公式：

```
score = tier_weight × 1e12 - enqueue_time_ms

tier_weight: enterprise=100 / pro=50 / free=10
```

score 越大越先处理。enterprise 天然排在 pro 前面，pro 天然排在 free 前面；同 tier 内按入队时间 FIFO。队列消费端用信号量控制并发上限，从队头取任务。

这个设计的好处：免费用户在低峰期完全能用，只是稍微等一下；付费用户不会被免费用户的流量峰值卡死。

## 计量与告警：让运营看得见

租户用量如果只有工程师看得见，运营会抓瞎。至少要提供三类可见性：

**实时计数**：Redis 存当前 window 的用量，给 dashboard 读；响应延迟要低，不超过 10ms。

**明细日志**：每条请求落一条记录，字段至少包含 `tenant_id / request_id / model / input_tokens / output_tokens / latency_ms / timestamp`。推荐用 ClickHouse 或 TimescaleDB 存，Redis 只做实时计数，不做历史查询。

**自动告警**：

- 租户用量超过当月配额 80% → 邮件 + 站内通知
- 5 分钟内用量超过日均的 50% → 触发人工审核（可能是爬虫或异常调用）
- 每小时输出 top 10 高消耗租户报表，给运营定价参考

## 踩坑清单

- **别用全局计数器做配额**。一个 Redis key 所有租户共用，等于没有隔离，第一个超限的租户把所有人都卡死。
- **配额重置时间要对齐租户时区**。你说"每天 1000 次"，北美用户的"每天"和你的 UTC+8 不是同一天。要么统一用 UTC 零点，要么存租户注册时区并转换。
- **并发配额必须用分布式锁**。多实例部署下，本地计数器各算各的，加起来会远超你设的上限。用 Redis SETNX 或 Lua 脚本做原子操作。
- **测试时要模拟配额耗尽路径**。大多数团队只跑正常流程，配额耗尽路径从没测过，上线才发现返回的是 500 而不是 429，客户端重试把问题放大三倍。
- **不要在超限时静默丢弃请求**。要么排队，要么明确拒绝并告知原因，让调用方知道发生了什么。

多租户不是一个功能点，是一套系统决策。从你签下第一个企业客户起，这套决策就该落地。
