---
layout: post
title: AI 应用监控指标 — 16 个必收的指标清单
date: 2026-01-15
topic: "工程实战"
tags: [AI, Monitoring, Observability]
excerpt: AI 应用比传统服务多出一堆要监控的东西：token、成本、质量、安全。一份从零搭起 dashboard 的指标清单。
permalink: /posts/2026-01-15-ai-app-metrics.html
---

## 监控 = 工程化的最低门槛

没监控 = 出问题靠用户告诉你 = 上线就是裸奔。
AI 应用要监控的东西比普通服务多——既要看技术指标，也要看质量指标。

## 16 个必备指标

### 性能（4个）

1. **Latency p50 / p95 / p99**
   - 通常 p99 < 5s 算可用，> 10s 体验差
2. **Time to First Token (TTFT)**
   - 流式响应的关键 UX 指标
3. **Tokens per Second (TPS)**
   - 输出速度
4. **Concurrent requests**
   - 当前并发数，看接近不接近系统容量

### 成本（3个）

5. **Daily total cost**
6. **Cost per request**
7. **Cost per user**
   - 哪些用户最贵

### 流量（2个）

8. **RPM / RPS**：请求数
9. **TPM**：token 数

### 质量（3个）

10. **Faithfulness**（RAG 必看）：答案是否基于 context
11. **User feedback rate**：thumbs up / down 收集率
12. **Retry rate**：用户多次发同一问题（说明上次没答好）

### 错误（2个）

13. **Error rate by type**：429 / 503 / 400 / timeout 分别看
14. **Refusal rate**：模型拒答的比例（高了说明 prompt 太保守，低了可能有安全问题）

### 安全（2个）

15. **Jailbreak attempts**：检测到的攻击次数
16. **PII leak events**：输出包含敏感字段的次数

## 数据采集

每次 LLM 调用打一条结构化日志：

```json
{
  "ts": "2026-01-15T10:30:15Z",
  "request_id": "req_abc",
  "session_id": "sess_xyz",
  "user_id": "u_42",
  "endpoint": "/chat",
  "model": "claude-opus-4-7",
  "prompt_template_id": "v3",
  "input_tokens": 5234,
  "output_tokens": 412,
  "cached_input_tokens": 4800,
  "cost_usd": 0.00876,
  "latency_ms": 3245,
  "ttft_ms": 312,
  "status": "success",
  "error_type": null,
  "tool_calls": ["get_weather"],
  "feedback": null
}
```

存到时序数据库（ClickHouse / TimescaleDB / DataDog）做聚合查询。

## 维度切片

每个指标按维度切看：

```
- 按 endpoint：哪个接口慢
- 按 user_id：哪个用户成本异常
- 按 model：模型表现对比
- 按 prompt_template：哪版 prompt 慢 / 错
- 按 region：哪个区域用户体验差
- 按 时段：哪个时段流量高
```

不切维度，问题永远定位不到。

## 告警阈值

```yaml
alerts:
  - name: latency_p99_high
    condition: latency_p99 > 8000 ms for 5 min
    severity: high
  
  - name: cost_spike
    condition: cost_per_hour > 2 * 7d_avg
    severity: critical
  
  - name: error_rate_high
    condition: error_rate > 5% for 10 min
    severity: high
  
  - name: cache_hit_dropped
    condition: cache_hit_rate < 0.5 * 7d_avg for 30 min
    severity: medium
  
  - name: jailbreak_attempts_spike
    condition: jailbreak_count > 10 per hour
    severity: high
```

告警要 actionable——什么是异常、做什么响应、谁负责。

## Trace：不只看指标看上下文

指标告诉你"出问题了"。**trace 告诉你"为啥"**。

每次请求保留完整 trace：
- 哪个 prompt 模板
- LLM 调用的输入 / 输出
- 工具调用的步骤
- 每步耗时

工具：LangSmith / Phoenix / Langfuse 都支持。
没 trace 的 LLM 应用 = 闭眼 debug。

## Dashboard 分层

```
[CEO/Manager Dashboard]
- Total cost / 用户活跃 / 满意度
- 趋势图，无 trace

[Engineering Dashboard]
- Latency / Error rate / Token usage
- 按 endpoint / model 切

[On-call Dashboard]
- 告警列表 / 实时 trace
- 快速跳到具体请求
```

不同角色看不同 dashboard，**不要把所有人塞同一个面板**。

## 实战 stack

便宜组合：
- 日志：自家服务 → JSON to S3 / Postgres
- 时序：Prometheus / VictoriaMetrics
- Dashboard：Grafana
- Trace：Langfuse（自部署）
- 告警：AlertManager / PagerDuty

商业组合：
- DataDog / NewRelic / Honeycomb（统一）
- LangSmith / Helicone（LLM 专用）

## 一份发布前 checklist

- [ ] 每次 LLM 调用结构化日志已落地
- [ ] 16 个核心指标都有 dashboard
- [ ] 5+ 个核心告警配置好
- [ ] On-call 流程明确
- [ ] Trace 系统可查任意 request_id
- [ ] 监控数据保留至少 30 天

## 一个朴素结论

> 上线 AI 应用 = 把可观测性当成第一类公民。
> 没监控的 AI 应用就是定时炸弹——成本失控、质量退化、安全事件都是悄悄发生的。
>
> 第一周做监控不算延后上线，**算保命**。
