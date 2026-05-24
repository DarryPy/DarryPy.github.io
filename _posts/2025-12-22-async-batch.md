---
layout: post
title: LLM Async Batch 处理 — 离线任务省 50% 成本的杀招
date: 2025-12-22
topic: "工程实战"
tags: [AI, Batch, 性能]
excerpt: 不是所有 LLM 调用都需要实时。批处理 API 让离线任务成本砍半，吞吐量翻倍。怎么设计 batch 流水线。
permalink: /posts/2025-12-22-async-batch.html
---

## 为什么要 Batch

实时 LLM 调用：贵、有限流、要排队。
但很多场景**不需要实时**：
- 批量文档处理 / 翻译
- 离线 embedding 计算
- 周期性数据清洗 / 分类
- 大规模 eval / 蒸馏数据生成
- 报告生成 / 月度分析

这些任务**可以等几小时**，没必要走贵的实时 API。

## 主流 Batch API

| Provider | 折扣 | SLA |
|---|---|---|
| **OpenAI Batch** | 50% off | 24h 内完成 |
| **Anthropic Message Batches** | 50% off | 24h 内 |
| **Google Vertex AI Batch** | 50% off | 几小时 - 24h |
| **DeepSeek 批量** | 50% off | 24h |

**全部主流家都半价**。

## OpenAI Batch 实战

```python
from openai import OpenAI
client = OpenAI()

# 1. 准备 JSONL 任务文件
with open("tasks.jsonl", "w") as f:
    for i, task in enumerate(tasks):
        f.write(json.dumps({
            "custom_id": f"task-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": task}],
            }
        }) + "\n")

# 2. 上传文件
batch_input = client.files.create(
    file=open("tasks.jsonl", "rb"),
    purpose="batch"
)

# 3. 提交 batch
batch = client.batches.create(
    input_file_id=batch_input.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

# 4. 轮询 / 等回调
while True:
    status = client.batches.retrieve(batch.id).status
    if status == "completed":
        break
    time.sleep(60)

# 5. 取结果
result = client.files.content(batch.output_file_id)
```

24h 内你的 jsonl 处理完，**成本对半砍**。

## Anthropic Message Batches

类似，区别：

```python
batch = client.messages.batches.create(
    requests=[
        {"custom_id": "task-1", "params": {"model": "claude-3-5-sonnet-20241022", "messages": [...]}},
        ...
    ]
)
```

up to 10k requests / batch。

## 适合什么任务

| 任务 | 适合 Batch？ |
|---|---|
| 用户实时聊天 | ❌（需要即时）|
| 批量翻译文档 | ✅ 完美 |
| 标注 100k 条数据 | ✅ |
| 训练数据生成（distillation）| ✅ |
| 每日报告生成 | ✅ |
| 月度数据分析 | ✅ |
| 批量 embedding | ✅（但 embedding 本来就快）|

## 工程设计

### 1. 任务拆分

太大的 batch（> 50k tasks）容易遇到限制，拆成多个 batch 跑。

### 2. 失败重试

batch 内单条失败不影响整体，但要 collect 失败 ID 重试：

```python
failed = [r for r in results if r.error]
if failed:
    retry_batch = create_batch([t for t in tasks if t.id in [f.id for f in failed]])
```

### 3. 优先级队列

不是所有离线任务都同等紧急。设优先级：

```
P0: 用户提交的"等会儿回结果"任务（24h batch）
P1: 周期性内部任务（48h-72h 也行）
P2: 大规模 distillation / eval（不限时间）
```

P0 立刻提交 batch，P1/P2 攒一攒再提交。

### 4. 监控

```yaml
metrics:
  - batch_queued_count       # 排队中的 batch 数
  - batch_processing_count   # 处理中
  - batch_complete_count
  - batch_failed_count
  - avg_completion_time      # 平均完成时间
  - cost_saved_vs_realtime   # 跟实时调相比省了多少
```

## 跟实时调用的混合架构

```
[Realtime API]
- 用户互动
- 实时 chatbot
- 实时分类 / 路由

[Batch API]
- 离线数据处理
- 报告 / 分析
- Distillation
- Eval

[Smart Router]
按"是否需要实时"路由
```

中等紧急的任务可以 fallback：
- 5 分钟内能等？→ 实时 API（贵但快）
- 1 小时？→ Express batch（如果 provider 有）
- 24 小时？→ Standard batch

## 一个朴素结论

> 50% 的离线 LLM 调用走 batch，**总账单立刻砍 25%**。
> 不是要不要用 batch 的问题——是**你的应用里多少调用其实是离线**的问题。
>
> 上线前梳理一下哪些 endpoint 不需要实时，**白送你一半成本**。
