---
layout: post
title: Agent 错误恢复与重试机制 — 让 Agent 别在一个坑里跳
date: 2026-01-29
topic: "Agent 与工具"
tags: [AI, Agent, Error Recovery]
excerpt: Agent 工具失败时不能简单 retry，要让模型"看明白错在哪"再决定下一步。错误反馈、循环检测、降级策略。
permalink: /posts/2026-01-29-agent-error-recovery.html
---

## 普通重试不够

普通服务出错：retry 3 次失败 → 抛异常。
Agent 不能这么粗：工具失败可能是参数错、可能是数据不存在、可能是临时网络抖动——**每种原因下 agent 该做不同的事**。

## 4 类错误 + 应对

### 1. 临时性错误（Transient）

**例**：网络超时、429、503、连接重置。
**应对**：自动重试，指数退避，**不告诉 LLM**——它无能为力。

```python
@retry(max_attempts=3, backoff="exponential")
def call_tool(name, args):
    return tool_registry[name](**args)
```

### 2. 参数错误（Bad Input）

**例**：user_id 格式错、必填字段缺、enum 不在范围。
**应对**：**告诉 LLM 具体错在哪**，让它改参数重试。

```json
{
  "error": "INVALID_ARGUMENT",
  "field": "user_id",
  "got": "abc",
  "expected": "格式 ^[a-z0-9_]{8,32}$",
  "hint": "请检查 user_id 拼写或先用 search_user_by_email 反查"
}
```

LLM 看到 `hint` 就能下一轮自己修正。

### 3. 业务错误（Business Logic）

**例**：用户不存在、余额不足、权限拒绝。
**应对**：**告诉 LLM 这种情况**，让它换路径或告知用户。

```json
{
  "error": "USER_NOT_FOUND",
  "message": "user_id 'xyz' 在系统中不存在",
  "hint": "如果用户提供的是 email，请用 search_user_by_email"
}
```

### 4. 不可恢复错误（Fatal）

**例**：DB 挂了、依赖服务全宕机。
**应对**：**告诉 LLM 这条路走不通，结束任务并告知用户**。

```json
{
  "error": "SERVICE_UNAVAILABLE",
  "retry_recommended": false,
  "user_message": "查询服务暂时不可用，请稍后再试"
}
```

## 循环检测

Agent 最常见的失败模式：**死循环重试同一个失败工具**。

检测策略：

```python
def is_loop(history, window=3):
    if len(history) < window:
        return False
    recent = history[-window:]
    # 看最近 N 步是否调了同名同参的工具
    if all(s.tool == recent[0].tool and s.args == recent[0].args for s in recent):
        return True
    return False

def run_agent(query):
    history = []
    for step in range(MAX_STEPS):
        if is_loop(history):
            inject_message("你似乎在循环。请换个思路或结束任务。")
            history.append(...)
        result = run_step(history)
        history.append(result)
```

或者更简单：每个工具相同参数最多调 3 次，第 4 次直接拒绝（在工具层做硬约束）。

## 显式让 LLM 反思

错误发生 3 次后，强制让 LLM 反思：

```
[System inject]
你在过去 3 步都失败了。停下来，
1. 总结到目前为止你了解了什么
2. 分析失败原因
3. 提出新方案

然后继续。
```

这种"被迫反思"能挽救很多陷入死路的 agent。

## Budget 兜底

无论怎么处理错误，agent 总有失败可能。设兜底：

```python
agent_config = {
    "max_steps": 15,         # 步数上限
    "max_total_tokens": 50000,  # token 上限
    "max_duration_sec": 120,   # 时间上限
    "max_cost_usd": 0.50,     # 成本上限
}
```

任一超限 → agent 强制终止，告诉用户"任务复杂未能完成，请简化或人工处理"。

## 用户层降级

agent 失败时不要 500 给用户。要：

1. **检查是否有部分结果**：能给的先给
2. **解释失败原因**：人话，不要 stack trace
3. **建议下一步**：让用户能继续而不是卡死

```
"我尝试帮你查询订单状态，但 5 次尝试都没能拿到完整数据。
已知信息：订单号 #12345 在系统中存在，但状态查询服务异常。
建议：
- 稍后重试
- 或联系客服 hotline: ...
- 或我可以帮你查别的"
```

## 一份发布 checklist

- [ ] 工具失败返回结构化 error + hint
- [ ] 区分 transient / business / fatal
- [ ] max_steps / max_tokens / max_cost 兜底
- [ ] 循环检测
- [ ] 3 次失败强制反思
- [ ] 用户层友好降级
- [ ] 错误日志带 trace_id 便于排查

## 一个朴素结论

> Agent 的可用性 = (LLM 不犯错率) × (错误能恢复率)。
> 提升前者难（换强模型），**提升后者性价比高得多**。
>
> 把 error handling 做透，agent 的"成熟度感"立刻拉满。
