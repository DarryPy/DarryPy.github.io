---
layout: post
title: AI 应用的成本监控与告警 — 别等账单出来才知道烧了多少钱
date: 2026-03-28
topic: "工程实战"
tags: [AI, 成本, 监控]
excerpt: LLM 应用最大的隐性风险是成本失控。一份从 metric 到告警再到归因的完整成本监控手册。
permalink: /posts/2026-03-28-ai-cost-monitoring.html
---

## 为什么成本监控特别重要

传统服务的成本可控：流量翻倍，服务器加一倍。
LLM 应用的成本是**用 token 算的**，一次复杂查询可能花 $0.50，一次 agent 调用可能花 $5。
两个原因让成本失控的风险特别大：

1. **单次成本不可预测**：用户问"帮我写报告"——可能 1k token 也可能 100k token
2. **滥用代价惨重**：一个被恶意调用的接口，一天能烧光半个月预算

**没有监控的话，等月底账单出来主人才发现已经晚了**。

## 必收的核心指标

```
total_cost / day
total_cost / user / day
total_cost / endpoint / day
total_tokens / day
avg_tokens_per_request
p99_tokens_per_request
cache_hit_rate
```

每个都按维度切：用户、接口、模型、错误类型。

## 数据怎么采

每次 LLM 调用打一条日志：

```json
{
  "ts": "2026-03-28T10:30:15Z",
  "request_id": "req_abc123",
  "user_id": "user_42",
  "endpoint": "/chat",
  "model": "claude-opus-4-7",
  "input_tokens": 5234,
  "output_tokens": 412,
  "cached_input_tokens": 4800,
  "input_price": 0.0000015,    # 单价 per token
  "output_price": 0.0000075,
  "cache_read_price": 0.00000015,
  "cost_usd": 0.00876,
  "latency_ms": 3245,
  "status": "success"
}
```

存到 ClickHouse / Datadog / OpenSearch，做聚合查询。

## 关键告警

### 1. 异常突增

```sql
今日 vs 7 日均值，单日成本涨 50%+ → 告警
```

可能是：滥用、bug 导致重试风暴、流量突增。

### 2. 单用户成本异常

```sql
top 5 用户 / 总成本 > 30% → 告警
```

可能是：单个用户被攻击、自动化脚本调用、定价漏洞。

### 3. 缓存命中率掉

```sql
cache_hit_rate 周环比掉 20%+ → 告警
```

可能是：prompt 改了破坏缓存、上游缓存挂了。

### 4. 单次请求 token 超阈值

```python
if response.total_tokens > 50000:
    alert("Single request used 50k+ tokens", request_id)
```

可能是：上下文塞太满、bug 导致历史无限累积。

### 5. 预算门槛

```
当月累计成本 > 80% 预算 → 严重告警
当月累计成本 > 100% 预算 → 自动限流降级
```

## 成本归因

知道"花了多少"还不够，要知道"为什么"。
按维度切看，找头部贡献者：

```
按 endpoint 排：哪个接口最贵？
按 user 排：哪个用户最贵？
按 model 排：是不是用错了昂贵的模型？
按时段排：是不是有非工作时段的异常调用？
```

经典反模式：

| 表现 | 可能原因 |
|---|---|
| 简单接口成本高 | 用了 Opus，其实 Haiku 就够 |
| 同一用户单日 1000+ 次调用 | 没限流，可能被自动化滥用 |
| input_tokens 持续上涨 | 对话历史不裁剪，越聊越贵 |
| cache_hit 接近 0% | prompt 结构没设计缓存 |

## 控制成本的几个杠杆

按优先级（容易实现 → 难）：

### 1. Prompt Caching（最大杠杆）

把 system prompt + 工具定义放最前面，标记 cache_control。命中后 input 成本降到 1/10。

### 2. 模型路由

简单任务 → 便宜模型，复杂任务 → 贵模型。
预先用一个分类器（Haiku / 关键词 / 长度）路由。

### 3. 上下文裁剪

老历史摘要压缩，工具结果只留必要字段。

### 4. 输出长度约束

prompt 里加 "用 100 字以内回答"，或设 `max_tokens` 硬上限。

### 5. Batch API

非实时任务用 Batch API，**成本砍半**。

### 6. 限流 + 配额

每用户每分钟 / 每天的请求数 + token 配额。
超额温和降级（用更便宜模型）或直接拒绝。

## 用户级配额示例

```python
class UserQuota:
    daily_token_limit = 100_000
    daily_request_limit = 200

    def check(self, user_id, est_tokens):
        used = redis.get(f"quota:{user_id}:{today()}")
        if used + est_tokens > self.daily_token_limit:
            raise QuotaExceeded("Daily limit reached")
        redis.incrby(f"quota:{user_id}:{today()}", est_tokens)
```

每个免费用户 100k token/天，付费用户 500k，企业级谈定。

## 一个朴素结论

> "AI 太贵" 是没有具体数据时的感叹。
> 有了 dashboard + 告警 + 归因 + 配额这 4 件套，AI 成本变成可优化的工程问题。

不要等账单震惊，提前监控。
