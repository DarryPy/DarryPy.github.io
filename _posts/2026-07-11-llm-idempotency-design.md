---
layout: post
title: LLM 请求幂等性设计 — 重试不重做，退款不漏做
date: 2026-07-11
topic: "工程实战"
tags: [幂等性, LLM, 工程实践, API设计, 分布式]
excerpt: LLM 调用不幂等会造成重复扣费和重复副作用，本文拆解幂等 Key 设计、去重缓存、客户端重试策略和服务端三层防护方案，帮你在生产系统里既安全重试又不多花一分钱。
permalink: /posts/2026-07-11-llm-idempotency-design.html
---

## 为什么 LLM 请求的幂等性比普通 API 更难

普通 REST API 写幂等很简单：PUT 覆盖、POST 加 idempotency-key，完事。LLM 调用难在三点。

第一，钱直接跟请求挂钩。每次调 `/chat/completions` 就是真实扣费，网络抖动触发的一次重试，可能在账单上变成两条 token 消费记录，调用量越大损失越可观。

第二，副作用不只在 LLM 侧。你的 agent 调完 LLM 还要写数据库、发消息队列、调第三方 API——LLM 返回成功但你没收到响应，重试时 LLM 又成功一次，数据库就双写了。

第三，LLM 的响应本身不可复现。同样的 prompt，两次调用大概率输出不同，没法像数据库那样用"查到就是成功"来判幂等。

所以 LLM 幂等设计要覆盖三层：**去重（不二次调 LLM）→ 防重（调了也别二次落库）→ 补偿（漏了能追回来）**。

---

## 幂等 Key 的设计：从调用出发，别从业务 ID 出发

很多人直接把 `user_id + session_id` 拼成 idempotency key，这是个常见错误。同一个用户同一个对话里可能发两条不同消息，key 一样就把第二条消息的请求当重试给挡掉了。

正确做法：**客户端在发请求前生成 UUID，作为本次调用的唯一标识，随请求一起传给后端**。这个 UUID 和"业务语义"完全解耦——它只标识"这一次 HTTP 调用"。

```typescript
// 客户端：每次用户点发送，生成新 UUID
const idempotencyKey = crypto.randomUUID();

const response = await fetch('/api/chat', {
  method: 'POST',
  headers: {
    'X-Idempotency-Key': idempotencyKey,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ messages, sessionId }),
});
```

后端收到请求后，先查 Redis：

```python
import redis, json

r = redis.Redis()

def handle_chat(idempotency_key: str, payload: dict):
    cache_key = f"idem:{idempotency_key}"

    # 命中缓存直接返回上次结果，不调 LLM
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)

    result = call_llm(payload)

    # 写缓存，TTL 24h（超过这个窗口的重试视为新请求）
    r.setex(cache_key, 86400, json.dumps(result))
    return result
```

TTL 设多少取决于你的重试策略。如果客户端最多重试 3 次、每次间隔 30s，24 小时完全够用。跨天有补偿场景可以拉长到 72 小时，但要评估 Redis 内存压力。

---

## 客户端重试的正确姿势：指数退避 + 抖动

网络超时、502、429——这三种错误都值得重试。但重试策略写错了，雪崩时你在帮倒忙。立即重试等于在服务压力最大的时候再补一刀，绝对禁止。

使用指数退避加随机抖动是标准解法：

```python
import time, random

def retry_with_backoff(fn, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (TimeoutError, RateLimitError) as e:
            if attempt == max_retries:
                raise
            # 指数退避 + ±20% 抖动
            delay = base_delay * (2 ** attempt)
            delay *= (0.8 + random.random() * 0.4)
            time.sleep(delay)
```

`base_delay=1.0` 时三次重试间隔约为 1s、2s、4s，加上抖动分散在 0.8–1.2s、1.6–2.4s、3.2–4.8s 之间。多个客户端同时失败时，抖动能有效错开重试洪峰，避免"重试风暴"打垮刚恢复的上游。

哪些错误绝对不要重试：400（参数错）、401（认证失败）、内容审核拒绝。这些重试一百次都没用，直接上报错误即可。

---

## 服务端三级防护：原子性是关键

光靠 Redis 查缓存还不够——高并发下两个请求可能同时通过缓存检查，然后都去调 LLM。需要加分布式锁：

```python
import time
from contextlib import contextmanager

@contextmanager
def idempotency_lock(r, key, timeout=30):
    lock_key = f"lock:{key}"
    # SET NX EX 原子操作，timeout 必须比 LLM 超时长 10s
    acquired = r.set(lock_key, "1", nx=True, ex=timeout)
    if not acquired:
        yield False
    else:
        try:
            yield True
        finally:
            r.delete(lock_key)

def handle_chat_safe(idempotency_key: str, payload: dict):
    cache_key = f"idem:{idempotency_key}"

    # 无锁快路径
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)

    with idempotency_lock(r, idempotency_key) as acquired:
        if not acquired:
            # 等上一个持锁请求完成后再查缓存
            time.sleep(0.5)
            cached = r.get(cache_key)
            return json.loads(cached) if cached else {"error": "concurrent_request"}

        # double-check：持锁后再查一次
        cached = r.get(cache_key)
        if cached:
            return json.loads(cached)

        result = call_llm(payload)
        r.setex(cache_key, 86400, json.dumps(result))
        return result
```

这个"查缓存 → 加锁 → 再查缓存 → 调 LLM"的四步模式是经典 double-checked locking，确保同一个 idempotency key 下 LLM 只被调用一次。

数据库层还要补一道：在业务结果表上对 `idempotency_key` 加唯一索引，让数据库的唯一约束兜底。即便应用层出 bug，数据库也不会双写。

| 防护层 | 手段 | 保护目标 |
|---|---|---|
| 客户端 | 带 UUID、指数退避 | 不触发无效重复调用 |
| 缓存层 | Redis + 分布式锁 | 不重复调 LLM |
| 数据库 | 唯一索引约束 | 不重复写业务数据 |

---

## 踩坑清单

- **客户端刷新页面就生成新 UUID**：这是对的，刷新等于"我想要一个新回答"，不应复用旧 key
- **UUID 要存 localStorage 而不是内存**：网络中断后重试才能带同一个 key，存内存的话页面一刷就丢了
- **锁 TTL 必须比 LLM 超时长 10s**：锁先过期会让并发请求以为没人持锁，又去调一次 LLM
- **不要把 idempotency key 混入 prompt**：放在 HTTP header 或请求元数据里，污染 prompt 会影响输出质量
- **只做 Redis 去重没做数据库唯一约束**：Redis 故障时应用层防线崩塌，数据库那道墙不能省
- **流式响应没有特殊处理**：streaming 场景建议先落一条"调用中"状态到 DB，完成后更新为"已完成"，重试时检测到"调用中"就等待，不要重新调 LLM

三层缺一不可——出事的时候往往就是以为"有一层就够了"。
