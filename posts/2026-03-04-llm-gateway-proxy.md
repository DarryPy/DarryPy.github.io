---
layout: post
title: LLM Gateway / Proxy 设计 — 一个口对内、多个口对外
date: 2026-03-04
topic: "工程实战"
tags: [AI, Gateway, Architecture]
excerpt: 应用直连 LLM 供应商有 5 个隐性代价。中间加一层 gateway 后，多模型路由、统一鉴权、成本归因、降级容灾全都顺了。
permalink: /posts/2026-03-04-llm-gateway-proxy.html
---

## 直连供应商的痛点

应用代码直接调 OpenAI / Anthropic / Gemini API，会逐渐遇到这些问题：

1. **每家 SDK 不一样**——一个新模型要改 N 处代码
2. **没有统一的成本归因**——按用户/产品线/接口归账困难
3. **重试 / fallback / 降级**重复实现
4. **API key 散落**——撤销/轮换难
5. **观测分散**——日志在各家 dashboard，看不全

**LLM Gateway**（也叫 AI Proxy）就是统一这些。

```
┌─── App A ───┐
├─── App B ───┤      ┌──── Anthropic
│             │      │
├─── App C ───┼──→  GATEWAY ──┼──── OpenAI
│             │      │
└─── Worker ──┘      └──── Gemini / DeepSeek / 本地
```

## 应该做什么

### 1. 统一接口

对外暴露一份 OpenAI-compatible API：

```
POST /v1/chat/completions
{ "model": "any-string", "messages": [...] }
```

后端按 model 字段路由：

```python
def route(model):
    if model.startswith("claude-"):
        return anthropic_handler
    if model.startswith("gpt-"):
        return openai_handler
    if model.startswith("gemini-"):
        return gemini_handler
```

应用代码只对接一种 API，**换供应商不用改代码**。

### 2. 集中鉴权

应用拿自己的内部 token 调 gateway，不直接持有外部 API key。

- Gateway 后端持有真正的 OpenAI / Anthropic key
- 应用持有自家的 token（可按团队 / 项目细分）
- 撤销 / 轮换全在 gateway 一处做

### 3. 成本归因

每条请求记录：
- 内部 token 是谁（团队 / 用户）
- 调了哪家模型
- 用了多少 token
- 花了多少钱

汇总成 dashboard，**每个团队都能看自己的 LLM 账单**。

### 4. 重试 + Fallback

主模型失败时自动 fallback：

```yaml
routes:
  - model: claude-opus-4-7
    primary: anthropic/claude-opus-4-7
    fallback:
      - anthropic/claude-sonnet-4-6  # 同家降级
      - openai/gpt-4.5               # 跨家兜底
```

应用永远收到结果（除非全挂了），不用自己写 fallback 逻辑。

### 5. 限流 + 配额

按内部 token 限流（RPM + TPM），见前一篇文章。
按月配额，超额告警 / 自动降级。

### 6. 缓存

prompt 完全一样的请求直接返回缓存（semantic cache）。
高频场景能砍 30-50% 调用。

### 7. 日志 + Trace

所有请求 + 响应落日志（脱敏后），方便事后排查。
集成 LangSmith / Phoenix / 自家 ELK。

## 几个开源 / 商业方案

| 方案 | 类型 | 优势 |
|---|---|---|
| **LiteLLM Proxy** | 开源 | 100+ 模型适配，社区活跃 |
| **OpenRouter** | 商业 | 一个 key 用所有模型，按量付 |
| **Helicone** | 商业 | trace + 成本 dashboard 很强 |
| **Portkey** | 商业 | 全套 gateway + observability |
| **Kong AI Gateway** | 开源 | Kong 网关的 AI 扩展 |
| **自建（基于 LiteLLM SDK）** | 自定义 | 完全可控 |

## 自建关键代码

```python
# pseudo-code
from litellm import completion
from collections import defaultdict

usage_per_user = defaultdict(int)

@app.post("/v1/chat/completions")
def chat(req, user=Depends(auth)):
    # 限流
    check_quota(user.id, req.estimated_tokens)
    
    # 路由 + fallback
    for model in routing_chain(req.model):
        try:
            resp = completion(model=model, messages=req.messages)
            # 记账
            log_usage(user.id, model, resp.usage)
            return resp
        except RateLimitError:
            continue
    raise HTTPException(503, "all upstream failed")
```

## 性能注意

Gateway 不能成为瓶颈：
- 异步 IO（FastAPI + httpx async）
- 连接池复用
- 缓存层（Redis）
- 多实例水平扩展

LLM 调用本身就慢（数百 ms 到几秒），gateway 额外开销控制在 10ms 内即可。

## 一个朴素结论

> 1-2 个应用 + 1 个模型：直连。
> 3+ 应用 / 2+ 模型 / 想做成本归因 → 立刻上 gateway。
>
> 没有 gateway，AI 工程化做不到生产级。
