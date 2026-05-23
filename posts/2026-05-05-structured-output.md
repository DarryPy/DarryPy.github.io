---
layout: post
title: 结构化输出 (Structured Output) 实战 — 让 LLM 稳定吐 JSON
date: 2026-05-05
category: "Prompt 与推理"
tags: [AI, JSON, Structured Output, 工程实践]
excerpt: 让 LLM 输出 JSON 不再"有时合法、有时不合法"。JSON mode、schema 约束、function calling 三种路径横向对比。
permalink: /posts/2026-05-05-structured-output.html
---

## 为什么"让 LLM 吐 JSON"是个老大难

很多年里大家都用：

> "请用 JSON 格式回答，例如 {\"answer\": \"...\"}"

效果：

- 大多数时候它真给你 JSON
- 偶尔多一个 ```markdown ``` 包裹
- 偶尔少一个引号 / 多一个逗号
- 偶尔在 JSON 前面加一句"Sure, here's the JSON:"
- 偶尔嵌入了无法转义的换行

生产里这种"偶尔"就是事故。

到 2026 年，主流模型都提供了**真正的结构化输出能力**——不再靠"乞求"，而是引擎层保证。

## 三种路径横向对比

### 1. JSON Mode（最早的方案）

OpenAI 早期的 `response_format={"type": "json_object"}`，
**只保证语法合法的 JSON**，但不保证内容符合你的 schema。

适合：能接受"是 JSON 但字段缺失"，自己做后处理。
不适合：严格 schema 校验。

### 2. JSON Schema 约束（主流推荐）

主流模型都支持把 JSON Schema 直接喂给模型，**引擎层保证输出严格符合 schema**。

OpenAI：

```python
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
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["intent", "entity", "confidence"],
                "additionalProperties": False
            }
        }
    }
)
```

`strict: True` 时引擎做 constrained decoding——每个 token 都过 schema validator，不合法直接禁止生成。
**输出 100% 符合 schema**。

Anthropic 通过 tool use 间接实现（见下面）。

### 3. Tool Use 当结构化输出（Anthropic 风格）

Claude 还没有独立的 JSON schema 参数，但**tool use 等价于结构化输出**：

```python
response = client.messages.create(
    model="claude-opus-4-7",
    tools=[{
        "name": "extract_user_intent",
        "description": "提取用户消息中的意图、实体、置信度",
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["query", "create", "delete"]},
                "entity": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["intent", "entity", "confidence"]
        }
    }],
    tool_choice={"type": "tool", "name": "extract_user_intent"},
    messages=[{"role": "user", "content": "我想看一下张三的订单"}]
)
# response.content[0].input 就是结构化结果
```

`tool_choice` 强制模型调这个工具，相当于强制走 schema。
输出在 `tool_use` block 的 `input` 字段，已经是合法 JSON。

## Pydantic / Zod 集成

实战中很少手写 JSON Schema，更多用 Pydantic（Python）或 Zod（TypeScript）反向生成：

```python
from pydantic import BaseModel
from typing import Literal

class UserIntent(BaseModel):
    intent: Literal["query", "create", "delete"]
    entity: str
    confidence: float

# OpenAI 直接支持
response = client.beta.chat.completions.parse(
    model="gpt-4.5",
    messages=[...],
    response_format=UserIntent,
)
parsed: UserIntent = response.choices[0].message.parsed
```

类型安全 + 自动校验，**比裸 JSON 工程上整洁一个数量级**。

## 选哪种？

| 场景 | 推荐 |
|---|---|
| 信息抽取 / 路由分类 | strict JSON Schema（OpenAI） 或 forced tool use（Claude） |
| 复杂任务里的最终输出 | tool use |
| 早期原型 | JSON mode（够用） |
| 不能改后端的 web 端 | 在 prompt 里加 schema + 后处理校验 |

## 工程上必须做的几件事

### 1. 验证 + 重试

即使模型说"我保证 schema"，生产代码也要：

```python
try:
    obj = MySchema.model_validate_json(raw)
except ValidationError as e:
    # 把 error 反馈给模型重试一次
    retry_response = ... # 加上"上一轮 schema 校验失败：xxx" 重跑
```

主流引擎的 strict 模式下，失败率非常低，但生产里**还是要兜底**。

### 2. 字段 description 写清楚

JSON Schema 的 description 对模型有强引导。
不要让字段名孤零零地立在那里：

```json
{
  "confidence": {
    "type": "number",
    "minimum": 0,
    "maximum": 1,
    "description": "你对该 intent 判定的置信度。0 = 完全不确定，1 = 非常确定。模糊提问应该 < 0.5"
  }
}
```

### 3. 用 enum 取代自由字符串

每当字段值有有限可能，**永远用 enum**：

```json
{ "intent": { "type": "string", "enum": ["query", "create", "delete", "unknown"] } }
```

加一个 `"unknown"`，比让模型乱起一个 intent 强。

### 4. 别忘了 additionalProperties: false

不加这个，模型可能多吐字段，下游处理可能出问题。
**默认全部 schema 都加上**。

### 5. 嵌套 schema 要克制

嵌套两三层还行，深嵌套（5+ 层）会让生成不稳定、token 暴涨。
能扁平就扁平。

## 一些常被低估的用法

### Routing

让 LLM 输出一个 enum：

```json
{ "route_to": "billing" | "tech_support" | "sales" | "spam" }
```

比 string match 强大得多——LLM 能理解隐喻、错别字、上下文。

### 数据清洗

输入一段非结构化文本，输出严格 schema。比手写正则 / NLP pipeline 灵活。

### 表单填写

用户上传 PDF，LLM 按 schema 提取出表单字段，然后拼成 JSON 给业务系统。

### 决策日志

agent 每一步决策都强制走一个 `{ action, reason, confidence }` schema。
事后审计、replay、debug 都方便。

## 一个朴素总结

> 结构化输出是从"LLM 试验品"走向"生产组件"最重要的一步。
> 它把 LLM 的随机性收进了一个可验证的笼子。

学会用 strict mode + Pydantic 之后，AI 应用的鲁棒性会跳一个台阶。
