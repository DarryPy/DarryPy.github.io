---
layout: post
title: AI 应用的 Rate Limiting 设计 — 不只是"每秒 N 次"
date: 2026-03-06
topic: "工程实战"
tags: [AI, Rate Limiting, API]
excerpt: LLM 应用的限流跟传统 API 不一样——既要按请求数限，也要按 token 用量限，还要防止 burst 烧光预算。
permalink: /posts/2026-03-06-rate-limiting.html
---

## LLM 限流的特殊性

普通 API 限流通常按 RPS（requests per second）。
LLM API 不够——一次请求可能 100 token 也可能 100k token，**RPS 一样的话成本可以差千倍**。

LLM 限流的 2 个维度：

1. **RPM (Requests Per Minute)**：每分钟请求数
2. **TPM (Tokens Per Minute)**：每分钟 token 数

主流模型 API 都同时限这两个。自己的应用对外暴露也建议这么做。

## 三种主流算法

### 1. Fixed Window

```
窗口[10:00:00 - 10:01:00] 内 < 100 次
窗口[10:01:00 - 10:02:00] 内 < 100 次
```

简单但有边界问题：10:00:59 到 10:01:01 的 2 秒钟内可能允许 200 次。

### 2. Sliding Window

```
看过去 60 秒：用 Redis ZSET 记录每次请求时间戳
当前请求来时：清理 60 秒前的，count 剩下的
```

精确，但内存开销大（每请求一条记录）。

### 3. Token Bucket（推荐）

桶里有 100 个 token，每秒补 1.67 个（= 100/60）。
请求来时拿走一个 token；没 token 拒绝。

```python
class TokenBucket:
    def __init__(self, capacity, refill_per_sec):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_per_sec
        self.last_refill = time.time()

    def try_consume(self, n=1):
        now = time.time()
        # 补充 tokens
        self.tokens = min(self.capacity, self.tokens + (now - self.last_refill) * self.refill_rate)
        self.last_refill = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False
```

**支持 burst**（满桶时可以一次性 100 个），又能限平均速率。

## 双维度限流

LLM 应用同时做 RPM 和 TPM：

```python
def check_quota(user_id, est_tokens):
    # 检查请求数
    if not request_bucket[user_id].try_consume(1):
        raise RateLimitError("RPM exceeded")

    # 检查 token 数
    if not token_bucket[user_id].try_consume(est_tokens):
        raise RateLimitError("TPM exceeded")

    return True
```

`est_tokens` 来自 prompt 长度估计 + 历史平均输出 token。

## 分层配额

按用户级别设不同上限：

```
免费用户：100 RPM / 100k TPM / 日 1M token
付费用户：500 RPM / 500k TPM / 日 10M token
企业用户：按合同
```

Redis 存配额状态，每用户一个 key：

```
key: quota:user_42:2026-03-06
value: {"rpm": 35, "tpm": 12000, "daily_tokens": 234567}
TTL: 86400 秒
```

## Backpressure 设计

到达限制不要直接 503，要给客户端**优雅降级**：

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 45
X-RateLimit-Limit-Tokens: 100000
X-RateLimit-Remaining-Tokens: 0
X-RateLimit-Reset-Tokens: 45
```

客户端按 `Retry-After` 等待重试，**不要瞎重试**。

## 公平性：单用户不能挤死大家

一个滥用用户可能撑满整个系统的 LLM 容量。
解法：**全局配额 + 用户配额双约束**：

```
全局：1M token/min（你的总能力）
用户：10k token/min（单用户上限）

→ 任何用户最多吃掉总能力的 1%
```

加权公平队列也行——优先级高的用户先服务。

## 实战参考配额

中等规模 SaaS 的初始建议：

| 等级 | RPM | TPM | 日 token |
|---|---|---|---|
| 免费 | 20 | 20k | 200k |
| 付费 ($20/月) | 100 | 200k | 5M |
| Pro ($100/月) | 500 | 1M | 50M |
| 企业 | 自定 | 自定 | 自定 |

跑一周看分布，调整。

## 一个朴素结论

> AI 应用的限流必须**RPM + TPM 双维度**。
> Token Bucket 算法 + Redis 状态 + 分层配额是 90% 场景的标准答案。
>
> 没限流的 LLM 应用就是裸奔，**一次滥用能烧光你一个月的预算**。
