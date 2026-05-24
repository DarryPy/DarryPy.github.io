---
layout: post
title: LLM API 错误处理与重试机制设计
date: 2026-03-30
topic: "工程实战"
tags: [AI, API, 错误处理, 重试]
excerpt: 调 LLM API 不像调普通 REST。429 / 529 / context overflow / 内容审核拒绝，每种错误都有自己的最佳应对。
permalink: /posts/2026-03-30-llm-api-retry.html
---

## LLM API 错误的特殊性

跟普通 HTTP API 比，LLM API 的错误处理麻烦多了：

- **429 速率限制**：抖动大，重试策略要讲究
- **529 服务过载**：跟 429 不一样的限流，是供应商侧扛不住
- **400 context length exceeded**：不可重试，要降级处理
- **400 content policy violation**：不可重试，要 fallback 模型
- **stream 中途断开**：部分已生成的 token 怎么办？
- **timeout**：长生成超时，要 cancel 上游
- **模型超长不收敛**：max_tokens 触底了但模型还没说完

每种错误**应对策略完全不同**，不能一套 retry 走天下。

## 错误分类与策略

| HTTP / 错误码 | 含义 | 策略 |
|---|---|---|
| 200 | 成功 | — |
| 400 (context_length_exceeded) | 输入太长 | 不可重试，截断或换长上下文模型 |
| 400 (content_policy) | 安全审核拒绝 | 不可重试，换提示词或换模型 |
| 401 | 认证失败 | 不可重试，刷 token |
| 403 | 权限拒绝 | 不可重试 |
| 408 / Timeout | 超时 | 可重试，但缩短请求 |
| 429 | 速率限制 | 指数退避 + 抖动 |
| 500 | 供应商内部错 | 可重试 1-2 次 |
| 503 / 529 | 过载 | 长退避 |
| Stream interrupted | 流中断 | 可重试，要重发 + 续生成 |

## 退避策略

**指数退避 + 抖动（exponential backoff with jitter）** 是标配：

```python
import random, time

def retry_with_backoff(call_api, max_retries=5):
    for attempt in range(max_retries):
        try:
            return call_api()
        except RateLimitError as e:
            wait = e.retry_after or min(60, 2 ** attempt + random.uniform(0, 1))
            time.sleep(wait)
        except OverloadedError as e:
            wait = min(120, 4 ** attempt + random.uniform(0, 2))
            time.sleep(wait)
        except (TimeoutError, InternalServerError):
            time.sleep(2 ** attempt)
    raise MaxRetriesExceeded()
```

加抖动是防"惊群效应"——多个客户端同时重试把对方挤死。

## 429 的特殊处理

很多 API 返回 `Retry-After` header，告诉你**确切的等待时间**：

```
HTTP/1.1 429 Too Many Requests
Retry-After: 30
```

优先用这个值，不要自己 2 ** attempt 瞎算。

OpenAI / Anthropic 还返回更细的：

```
x-ratelimit-limit-tokens: 1000000
x-ratelimit-remaining-tokens: 234567
x-ratelimit-reset-tokens: 12.5s
```

监控这些 header，**接近上限时主动降速**，比触发 429 重试更优雅。

## Context Overflow 的降级

```
400 context_length_exceeded
```

这个错不能重试，重试还是错。三种降级：

### 1. 截断历史

```python
def truncate_messages(messages, max_tokens):
    # 保留 system + 最近 N 条
    system = messages[0] if messages[0]["role"] == "system" else None
    recent = messages[-max_tokens:]
    return [system] + recent if system else recent
```

### 2. 摘要压缩

把老的消息用 LLM 压缩成 200 字的摘要，替换原文。

### 3. 换长上下文模型

`Claude Sonnet 4.6` 上下文打不下？切到 `Gemini 2.x Pro`（1M+ 上下文）。
现代设计应该**多模型路由**，按需切换。

## Stream 中断的处理

流式响应中途断开是常见故障：

```python
async def stream_with_resume(messages):
    accumulated = ""
    attempt = 0
    while attempt < 3:
        try:
            async for chunk in client.stream(messages):
                accumulated += chunk
                yield chunk
            return  # 完成
        except StreamError:
            attempt += 1
            # 把已收到的部分作为 assistant prefix，继续生成
            messages.append({"role": "assistant", "content": accumulated, "partial": True})
```

部分 SDK 支持 `prefill` 让模型从已有 prefix 继续，比从头重发省 token。

## 模型 fallback 路由

主模型挂了或被审核拒了，要有 fallback：

```python
MODELS = [
    "claude-opus-4-7",       # 主
    "claude-sonnet-4-6",     # 性价比降级
    "gpt-4.5",               # 跨供应商兜底
    "deepseek-v3",           # 国内合规兜底
]

def call_with_fallback(prompt):
    last_error = None
    for model in MODELS:
        try:
            return call(model, prompt)
        except ContentPolicyError as e:
            last_error = e
            continue  # 这家拒了，换一家
        except RateLimitError as e:
            last_error = e
            continue
    raise last_error
```

**别把"主模型挂了"当成端到端失败**——多 vendor 路由能把可用性从 99% 拉到 99.9%。

## 监控必须做的几件事

- p50 / p99 latency
- 错误率按类型拆（429 / 529 / 400 / timeout）
- token 用量 vs 限额比例
- 重试次数分布
- fallback 触发频率（>5% 说明主模型出问题了）

没监控 = 半夜被叫醒不知道为啥。

## 实战 checklist

发布 LLM 应用前过一遍：

- [ ] 区分可重试错和不可重试错
- [ ] 用 Retry-After header，不要自己瞎算
- [ ] 指数退避 + 抖动
- [ ] context overflow 有降级
- [ ] content policy 拒绝有 fallback
- [ ] stream 中断有续传
- [ ] 至少 2 个 vendor 的 fallback 路由
- [ ] 关键指标有监控告警

跑通这套，**API 类的事故大幅减少**。

## 一句话总结

> LLM API 不是普通 REST API。
> 错误类型多、不可重试的多、stream 让事情更复杂。
> 一套通用 retry 不够用——必须按类型分别处理。
