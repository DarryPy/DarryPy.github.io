---
layout: post
title: Constrained Decoding — 让 LLM 输出 100% 符合 schema
date: 2026-01-13
topic: "Prompt 与推理"
tags: [AI, Constrained Decoding, JSON]
excerpt: 不靠 prompt"乞求"模型，让引擎在生成时硬约束每个 token，输出 100% 符合 schema / 正则 / 语法。
permalink: /posts/2026-01-13-constrained-decoding.html
---

## 朴素方法的痛点

让模型吐 JSON 用 prompt + retry：

```
请用 JSON 回答：{"intent": "...", "city": "..."}
```

效果：90% 时候对，10% 时候少引号、多括号、加前缀"Sure, here is:"。
生产里这 10% 就是事故。

## Constrained Decoding 的思路

**生成每个 token 时，过一遍 schema validator，不合法的 token 直接 mask 掉概率为 0**。

```
模型想输出："Sure, {"
                   ↑
         engine 看到 schema 要求第一个非空字符必须是 "{"
         → 把所有非"{"的 token logit 设 -inf
         → 模型被迫从"{"开始

继续生成："intent":  
                ↑
       enum 限定 intent 必须是 "query"|"create"|"delete"  
       → 后续 token 只能在这 3 个 string 范围
```

**输出 100% 符合 schema**，**不需要重试**。

## 主流实现

| 工具 | 提供方 | 用法 |
|---|---|---|
| **OpenAI Structured Outputs** | OpenAI | `response_format: {strict: true, schema: ...}` |
| **Anthropic Tool Use** | Anthropic | `tools` + `tool_choice` 强制走 schema |
| **Outlines** | 开源 | 自部署模型，支持 JSON / 正则 / CFG |
| **LMQL** | 开源 | 类 SQL 的约束语言 |
| **JSONFormer** | 开源 | 早期方案 |
| **Guidance** | 微软开源 | 用 Python DSL 控制生成 |

## OpenAI strict mode

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.5",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_intent",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "enum": ["query", "create", "delete"]},
                    "entity": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["intent", "entity", "confidence"],
                "additionalProperties": False
            }
        }
    }
)
```

`strict: True` 时引擎做 constrained decoding，**输出 100% 合法**。

## 自部署 LLM 的 Outlines

```python
import outlines
from outlines import models, generate

model = models.transformers("meta-llama/Llama-3.1-8B-Instruct")

# 按 JSON schema 约束
schema = '{"type":"object","properties":{"city":{"type":"string"},"temp":{"type":"number"}}}'
generator = generate.json(model, schema)
result = generator("查询天气 ...")  # 返回 dict
```

不光支持 JSON，还能约束**正则、CFG**：

```python
# 只能匹配电话号码格式
generator = generate.regex(model, r"^\d{3}-\d{4}-\d{4}$")
```

## 适合什么任务

| 任务 | constrained decoding |
|---|---|
| 信息抽取 / 路由 / 分类 | ✅ 必上 |
| 表单填写 / 数据提取 | ✅ |
| SQL / 代码生成（约束语法）| ✅ |
| 自然对话 / 创意 | ❌ 没意义 |
| 复杂嵌套（>5 层） | ⚠️ 可能慢 |

## 工程注意

### 1. 性能开销

constrained decoding 比 free decoding 慢 5-15%（schema validate 开销）。
简单 schema 可忽略，复杂嵌套就明显。

### 2. Schema 复杂度上限

太深嵌套 / 太多 `oneOf` 会让 validator 状态爆炸。
**能扁平就扁平**，跟普通 schema 设计原则一致。

### 3. 跟 prompt 配合

constrained decoding 保证**格式**，不保证**内容对**。
prompt 还是要写好 description / 示例，让模型懂"该填什么"。

### 4. 错误处理仍然必要

理论上输出 100% 合法。但实际偶尔遇到引擎 bug / 超长截断等情况。
**生产代码仍要 try/parse/retry 兜底**。

## 一个朴素结论

> 信息抽取 / 路由场景：**constrained decoding 不上是裸奔**。
> 它把"输出格式对" 从 prompt 的责任变成引擎保证。
>
> 用了之后 JSON 解析失败基本绝迹，代码可以扔掉一半的兜底逻辑。
