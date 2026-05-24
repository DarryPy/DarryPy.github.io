---
layout: post
title: Tool JSON Schema 设计 — Agent 用得对不对，60% 取决于 schema
date: 2026-02-22
topic: "Agent 与工具"
tags: [AI, Agent, Tool, JSON Schema]
excerpt: LLM 调工具的成功率，跟工具 schema 写得好坏强相关。一份生产级 JSON Schema 设计清单。
permalink: /posts/2026-02-22-tool-schema-design.html
---

## Schema 决定上限

写 agent 的工具时，schema 是模型理解工具的唯一渠道。
**模糊的 schema = 模型瞎调；精准的 schema = 模型用得准**。

实测：同一个工具，schema 改三次描述能让调用准确率从 65% 涨到 92%。

## 必须有的字段

```json
{
  "name": "fetch_user_profile_by_id",
  "description": "...",
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

5 个关键点：

1. **name 动词起头 + 具体**：`get_user` 不够，`fetch_user_profile_by_id` 才精确
2. **description 写清楚用法 + 反例**
3. **每个字段都加 description**
4. **`required` 标必填**
5. **`additionalProperties: false`** 防 agent 自由发挥

## description 模板

工具描述按这个结构：

```
[用途] 简述工具做什么。
[何时用] 适合的场景。
[何时不用] 不适合的场景（防止 agent 误用）。
[返回] 字段列表。
[示例] 1-2 个调用例子。
```

具体：

```
按 user_id 查询用户基础资料。

适用：
- 知道 user_id，想取 name / email / created_at

不适用：
- 按 email 反查（用 search_user_by_email）
- 拿订单历史（用 list_user_orders）

返回字段：name, email, created_at, is_premium

示例：
fetch_user_profile_by_id(user_id="u_12345")
→ {"name":"张三","email":"...","created_at":"...","is_premium":true}
```

清晰的反例能挡住 60% 的误用。

## 用强约束代替 prompt 提醒

不要在 prompt 里写 "请只发给内部邮箱"。
在 schema 里硬约束：

```json
{
  "to": {
    "type": "string",
    "pattern": "^[a-z0-9._-]+@yourcompany\\.com$",
    "description": "收件人邮箱，必须是公司内部 @yourcompany.com"
  }
}
```

模型生成不符合 pattern 的值会被引擎拒绝（strict mode），**比 prompt 约束硬核得多**。

## 枚举优先于自由字符串

每当字段值有有限可能：

```json
{
  "status": {
    "type": "string",
    "enum": ["pending", "approved", "rejected"]
  }
}
```

避免 agent 创造 `"approve"` 或 `"已批准"` 这种偏差。

## 嵌套要克制

```json
// ❌ 太深
{
  "filter": {
    "conditions": {
      "and": [...]
    }
  }
}
```

```json
// ✅ 扁平
{
  "filter_field": "...",
  "filter_value": "...",
  "filter_op": "eq"
}
```

嵌套 3 层以上 agent 容易写错。能扁平就扁平。

## 工具粒度

太粗：

```python
do_something(action="get_user_or_post_or_comment", ...)
```

agent 根本不知道怎么用。

太细：

```python
get_user_name(id), get_user_email(id), get_user_avatar(id)...
```

agent 要调 10 个工具完成一件事，token 暴涨。

刚好：每个工具职责单一但能完成一个"用户能描述的事"。
经验值：每个工具 1-5 个参数最舒服。

## 错误返回也要 schema

工具失败时不要扔异常：

```json
{
  "error": "USER_NOT_FOUND",
  "message": "user_id 'abc' 不存在",
  "hint": "确认 user_id 拼写，或用 search_user_by_email 反查"
}
```

`hint` 让 agent 知道下一步怎么走，**不会原地循环**。

## 一份发布前 checklist

- [ ] name 是动词 + 具体
- [ ] description 包含用途 + 何时用 + 何时不用 + 返回 + 示例
- [ ] 每个参数有 description
- [ ] required 标注
- [ ] additionalProperties: false
- [ ] 关键字段用 pattern / enum / minimum / maximum 硬约束
- [ ] 嵌套 ≤ 2 层
- [ ] 工具数量 ≤ 30（再多分组暴露）
- [ ] 错误返回结构化

## 一个朴素结论

> 工具 schema 是 agent 跟世界的接口。
> Schema 设计 = API 设计 + LLM 提示工程。
>
> 在 schema 上花 1 小时，比换更强模型省事得多。
