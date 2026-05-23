---
layout: post
title: Function Calling / Tool Use 实战 — 让 LLM 真的"动手"做事
date: 2026-05-18
category: "Agent 与工具"
tags: [AI, Tool Use, Function Calling, Agent]
excerpt: 工具调用是 agent 的灵魂。schema 怎么设计、错误怎么传回、并行调用怎么协调，一份避坑指南。
permalink: /posts/2026-05-18-function-calling.html
---

## Tool Use 是 agent 的灵魂

没有 tool 的 LLM 只能聊天。给它工具，它才能查数据、调 API、操作文件、写代码、订机票。

主流模型都支持 tool calling：

- OpenAI：`tools` + `tool_choice`
- Anthropic：`tools`（消息里返回 `tool_use` block）
- Gemini：`function_declarations`
- 主流 LLM 框架（LangChain / LlamaIndex / Mastra）都做了统一封装

接口虽然名字不同，**核心模型是一样的**：

```
模型看到工具定义 → 决定调哪个工具、用什么参数 → 你执行工具 → 返回结果 → 模型继续推理或给最终答案
```

## 工具定义的最佳实践

工具的 schema 是 JSON Schema 格式。**写得好坏直接决定 agent 成功率**。

### 1. 名字直白、动词起头

❌ `get_user_data`
✅ `fetch_user_profile_by_id`

LLM 是按名字理解工具用途的。模糊的名字 = 错误调用。

### 2. description 像写给一个新员工

不要只写一句"获取用户数据"。要写：

```
用 user_id 查询用户的基础资料。
- 适用：知道 user_id，需要拿 name / email / created_at
- 不适用：按 email 反查用户（请用 search_user_by_email）
- 不适用：拿历史订单（请用 list_user_orders）
返回字段：name, email, created_at, is_premium
```

明确**用法 + 反例**能让 agent 不乱调。

### 3. 参数 schema 要严

```json
{
  "name": "fetch_user_profile_by_id",
  "input_schema": {
    "type": "object",
    "properties": {
      "user_id": {
        "type": "string",
        "pattern": "^[a-zA-Z0-9_-]{1,32}$",
        "description": "用户 ID，仅字母数字下划线短横线"
      }
    },
    "required": ["user_id"],
    "additionalProperties": false
  }
}
```

- `required` 必填
- `additionalProperties: false` 防止 agent 瞎加字段
- 关键字段加 `pattern` / `enum` / `minimum` / `maximum`

### 4. 工具粒度要刚好

- 太粗：`do_user_thing(action, ...args)` — agent 不会用
- 太细：拆成 20 个工具 — token 暴涨、决策空间太大
- 刚好：每个工具职责单一，但能独立完成一件**用户能描述的事**

## 错误返回要友好

工具失败时 **不要扔异常**给 LLM，要返回 LLM 看得懂的 JSON：

```json
{
  "error": "USER_NOT_FOUND",
  "message": "user_id 'abc123' 在系统中不存在",
  "hint": "请确认 user_id 是否拼写正确，或先用 search_user_by_email 反查"
}
```

LLM 看到 `hint` 就知道下一步该怎么走。直接抛 stack trace 它只会重试同样的错误。

## 并行 Tool Calls

主流模型现在都支持一次返回**多个 tool_use**。
比如 agent 看到"对比 vid=A 和 vid=B 的播放数据"，可以一次性调用：

```
- fetch_video_stats(vid="A")
- fetch_video_stats(vid="B")
```

并行执行能省一轮 LLM 调用 + 一半延迟。

工程上要点：

- 客户端要**并发执行**所有 tool_use（不是顺序）
- 全部完成后再把结果一起喂回模型
- 任何一个工具失败也要返回完整的"错误结果对象"，不要 short-circuit

## 限制 agent 的访问范围

有些工具危险（删数据、转账、发邮件）。两种保护：

1. **白名单工具**：危险操作根本不暴露给 agent
2. **审批中间层**：agent 调危险工具时，先弹给人确认，人点 yes 才真执行

不要指望 prompt 里写"请谨慎使用"。LLM 没有"谨慎"的能力。

## 常见坑

| 坑 | 修法 |
|---|---|
| Agent 不调工具，直接编答案 | 在 system prompt 强调"必须用工具查证，不能基于训练数据回答" |
| Agent 死循环调同一个工具 | 加 max_tool_calls；给工具结果加"已查询过"提示 |
| Agent 用了错误参数（id 写错） | 工具 description 写明输入格式；返回 error 时给 hint |
| Agent 把工具结果原文当答案输出 | 在 system prompt 强调"基于工具结果用自然语言总结" |
| Tool schema 太大，token 浪费 | 按用户角色/场景只暴露相关工具子集 |

## 一个朴素观察

工具用的好不好，**80% 取决于工具定义**，**20% 才是 LLM 能力**。
精心写好 5 个工具的 description，比换一个更强的模型省事得多。
