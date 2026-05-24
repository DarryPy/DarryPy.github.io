---
layout: post
title: 拒答 Prompt — 让模型敢说"我不知道"
date: 2026-01-11
topic: "Prompt 与推理"
tags: [AI, Prompt, Refusal]
excerpt: LLM 默认啥都答——不知道也编。让它在不确定时显式拒答，是减少幻觉最便宜的手段。
permalink: /posts/2026-01-11-refusal-prompts.html
---

## 默认行为：什么都答

LLM 训练时被鼓励"helpful"，所以**不知道也尽力答**。
问"我家狗叫什么名字"——它会编一个。

生产场景里这种"瞎答"成本极高：
- 客服场景给用户错误信息 → 投诉
- RAG 场景 context 没答案 → 模型脑补 → 信任崩盘
- Agent 场景模型猜参数 → 工具调用失败链式反应

## 让它学会"不知道"

在 system prompt 里**明确指示**：

```
你的回答必须基于事实和给定的 context。

如果遇到以下情况，明确说"我不确定" 或 "我没有这方面的信息"：
- context 不包含问题相关信息
- 问题超出你的训练知识
- 问题需要实时数据但你没有

不要：
- 编造信息
- 用"可能"、"大概"等模糊词掩饰不确定
- 给一个不完整答案而不指出局限

如果不确定，宁可少说也不要错说。
```

这段加上后，**RAG faithfulness 平均涨 15-25%**。

## 不同场景的拒答策略

### RAG 场景

```
你只能基于下面提供的 context 回答。

如果 context 不包含问题的答案：
- 明确说 "提供的资料中没有这个信息"
- 不要使用你的训练知识补充
- 可以建议用户尝试其他相关查询
```

### Agent 场景

```
如果用户的请求需要的信息你没法通过工具获得：
- 告诉用户具体缺什么
- 建议怎么提供（"请提供 user_id" / "请上传文件"）
- 不要假设填值
```

### 高风险场景（医学 / 法律 / 金融）

```
你不是专业医生 / 律师 / 金融顾问。

任何具体诊断 / 法律建议 / 投资建议：
- 不要给出具体方案
- 强调"请咨询专业人士"
- 可以提供通用科普信息，但加免责声明
```

## 校准置信度

更进阶：让模型**自报置信度**，前端按置信度分级处理。

```
回答末尾加 JSON：
{
  "confidence": "high" | "medium" | "low",
  "reasoning": "..."
}
```

```python
if response.confidence == "low":
    show_warning("AI 对这个回答不太确定，建议人工核实")
elif response.confidence == "high":
    show_normal(response.text)
```

实测置信度跟实际正确率有相关性——**不完美但有用**。

## 反模式

### 1. 全 prompt 强调"必须自信回答"

某些场景写 "always provide a complete answer" 加重幻觉问题。
正确做法相反：**鼓励诚实**。

### 2. 用否定句太多

```
❌ "不要瞎编"
❌ "禁止猜测"
❌ "不许编造"
```

否定句让模型注意力偏向"瞎编"这个词。改成正向：

```
✅ "诚实回答，不知道时明确告知"
```

### 3. 拒答不给替代方案

```
❌ "我不知道。"
```

用户不满意。

```
✅ "我没有这方面的信息。你可以试试：1) 问 X 2) 查 Y 3) 联系客服"
```

给出路。

## 评估"拒答得当"

加 eval：
- **该拒未拒（hallucination）**：本应拒答时模型瞎答
- **过度拒答（over-refusal）**：明明能答却拒答

两个指标要平衡——**拒答率太高反而是问题**（10-20% 算正常，>30% 模型变得没用）。

## 工程实践

```python
def is_refusal(response):
    refusal_markers = ["我不确定", "我没有", "建议咨询", "无法回答"]
    return any(m in response for m in refusal_markers)

# 监控
if is_refusal(response):
    metrics.inc("refusal", labels={"endpoint": endpoint})
```

监控拒答率，**异常突增说明 prompt / context 出问题了**。

## 一个朴素结论

> "助手怕没用" → 学会拒答 → "助手值得信任"。
>
> 用户能接受"AI 不知道"，**接受不了 AI 编瞎话**。
> 这一行 prompt，是 trust 的护栏。
