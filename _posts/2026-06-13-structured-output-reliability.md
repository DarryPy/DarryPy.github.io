---
layout: post
title: Structured Output 可靠性实战 — 让 LLM 稳定输出 JSON 的 7 个技巧
date: 2026-06-13
topic: "工程实战"
tags: [结构化输出, JSON, LLM, 工程实战, 可靠性]
excerpt: LLM 输出 JSON 看起来简单，生产里却是重灾区：多余的 markdown fence、字段缺失、类型错、偶发乱码。这篇拆解 7 个可落地的手段，从 prompt 设计到解析层防护，帮你把 JSON 解析失败率压到 0.1% 以下。
permalink: /posts/2026-06-13-structured-output-reliability.html
---

你有没有遇到这种情况：prompt 明明写了"返回 JSON"，但模型偶尔会在 JSON 前面加一句"当然，以下是结果："，或者把数字字段包在引号里，或者直接漏掉某个 key？测试环境跑十次都正常，上了生产就炸一发。

结构化输出是大多数 AI 应用的核心链路，可靠性要求极高。这篇文章总结 7 个可落地的手段，把 JSON 解析失败率压到 0.1% 以下。

## 为什么 JSON 输出比你想象的脆

LLM 是语言模型，不是 JSON 序列化器。它生成 token 的方式是自回归预测，任何时候都可能"创作"出格式噪音：

- 在 JSON 前后输出说明文字（` ```json ... ``` ` 或解释性语句）
- 数字值用字符串表示（`"count": "5"` 而非 `"count": 5`）
- 布尔值用 `"yes"/"no"` 代替 `true/false`
- 嵌套结构在长上下文中被截断
- 偶发的 Unicode 转义或非法字符

这些问题的发生率通常在 1%–5%，但在高并发下会快速放大，且很难被集成测试覆盖到。

## 技巧 1–3：从模型侧控制

**技巧 1：使用原生 JSON mode 或 structured output API**

如果模型支持，优先用原生约束而非 prompt 约定。

| 提供商 | 方式 | 保证程度 |
|--------|------|---------|
| OpenAI GPT-4o | `response_format: {type: "json_schema"}` | 100%（schema 级别） |
| Anthropic Claude | `tool_use` 强制调用返回 | 接近 100% |
| Gemini 1.5 | `response_mime_type: application/json` | 高，偶有截断 |
| 开源模型（vLLM） | `guided_json` + grammar sampling | 100%（有延迟代价） |

Claude 没有原生 JSON mode，但用 tool_use 可以达到同等效果：定义一个 dummy tool，schema 就是你期望的输出结构，强制模型"调用"它，tool_input 就是结构化结果。

**技巧 2：在 system prompt 里写精确的 schema 约定**

不要只写"返回 JSON"，把 schema 直接贴进去：

```text
你必须返回且仅返回一个 JSON 对象，格式如下，不要包含任何其他文字：
{
  "label": string,   // 值域：["正面","负面","中性"]
  "score": number,   // 置信度，0.0–1.0
  "reason": string   // 50 字以内的理由
}
```

显式列出字段名、类型和值域，比自然语言描述失败率低 60%–80%。

**技巧 3：用 few-shot 示例锚定格式**

在 prompt 里放 2–3 个输入/输出对，尤其是边界案例（空字符串输入、特殊字符、长文本），模型会模仿格式而非自由发挥。注意示例要和真实分布匹配，否则会引入偏差。

## 技巧 4–5：解析层的防护

**技巧 4：分层解析 + 自动 repair**

第一层：先尝试 `JSON.parse`（或 Python 的 `json.loads`）。

第二层：失败后用宽松解析器提取，比如 `json5`（允许末尾逗号、注释）或正则提取 JSON 块：

```python
import re, json

def extract_json(text: str) -> dict:
    # 剥掉 markdown fence
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取第一个 {...} 块
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise
```

第三层：repair 仍失败，触发一次重试，prompt 里追加："上一次你的返回无法解析为 JSON，请严格按格式重新输出。"

**技巧 5：schema 校验，不只是解析**

解析成功不等于字段合规。用 Pydantic（Python）或 Zod（TypeScript）做 schema 校验，捕获类型错误和缺失字段：

```python
from pydantic import BaseModel, field_validator
from typing import Literal

class SentimentResult(BaseModel):
    label: Literal["正面", "负面", "中性"]
    score: float
    reason: str

    @field_validator("score")
    def check_range(cls, v):
        assert 0.0 <= v <= 1.0
        return v
```

`ValidationError` 和 `JSONDecodeError` 分开统计：前者说明模型理解了格式但填错了值，需要调整 prompt；后者说明格式本身就烂，先上 JSON mode 再说。

## 技巧 6–7：监控与 CI 守卫

**技巧 6：上线前的 schema 一致性测试**

在 CI 里跑 50–100 个真实样本，统计解析成功率和字段覆盖率。设阈值——成功率低于 98% 就 block merge。不要用纯 mock，必须打真实 API（用 prompt cache 控制成本）。这一关能拦住 80% 的格式回归。

**技巧 7：生产侧的解析失败监控**

每次解析异常打 structured log，至少记录：

```json
{
  "event": "json_parse_error",
  "model": "claude-sonnet-4-6",
  "prompt_version": "v1.3",
  "raw_output_len": 312,
  "error_type": "JSONDecodeError",
  "ts": "2026-06-13T12:00:00Z"
}
```

按 `prompt_version + model` 聚合失败率，一旦某个版本的失败率超过基线 2× 就告警。这能快速定位是 prompt 改动引入的，还是模型版本升级悄悄带来的。

## 踩坑清单

- 只在 prompt 里写"返回 JSON"没给 schema → 失败率高 5 倍
- 用 `response.text` 直接 `JSON.parse` 没处理 markdown fence → 线上必炸
- 解析成功就认为数据可用，没做 schema 校验 → 下游字段取 undefined
- 只测 happy path 的 10 条样本 → 上线后 1% 失败率被放大成几千次每天
- 换了模型版本没重新跑基准测试 → 模型升级悄悄改了输出风格

**一句犀利总结：LLM 的 JSON 输出像个不靠谱的外包——给清单、设验收标准、留返工机制，缺一不可。**
