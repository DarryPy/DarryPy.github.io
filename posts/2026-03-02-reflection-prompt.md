---
layout: post
title: Reflection Prompt — 让模型在回答前先挑自己的错
date: 2026-03-02
topic: "Prompt 与推理"
tags: [AI, Prompt Engineering, Reflection]
excerpt: 不需要做 agent 框架，靠一个 prompt 就能让 LLM "先想再答 + 自检 + 改写"。投入 10 行 prompt 换 15% 准确率。
permalink: /posts/2026-03-02-reflection-prompt.html
---

## 朴素的想法

让模型一次性答完往往出错。
**让它在 prompt 里就完成"先想 → 答 → 自检 → 改"** 的循环，效果显著好。

跟 Reflection Agent 区别：不是多次 LLM 调用，**就在一个 prompt 里完成**。简单、便宜、即插即用。

## 模板

```text
你将按以下步骤工作，分别用对应标签包裹：

<thinking>
- 我先理解问题
- 列出关键事实
- 推理过程
</thinking>

<draft>
[初步回答]
</draft>

<critique>
检查 draft：
- 事实有错吗？
- 逻辑通顺吗？
- 有没有遗漏？
- 风格匹配要求吗？
</critique>

<final>
[修正后的最终答案]
</final>

问题：{question}
```

模型按这个结构输出 4 段。**最后只把 <final> 给用户看**。

## 实战效果

GSM8K 数学题实测：
- 直接答：~78%
- CoT：~85%
- Reflection prompt：~92%

提升幅度跟 CoT 累加。代价：输出 token 多 3-5 倍。

## 关键设计点

### 1. critique 阶段要"换角度"

如果 critique 用同样的 prompt 风格，模型可能给自己开绿灯。
要明确："**假装你是另一个挑刺的专家**，找 draft 的所有可能错误"。

### 2. 给具体的检查项

模糊的"检查一下"不够。给清单：

```
<critique>
逐项检查：
[ ] 数字计算对吗？
[ ] 单位换算对吗？
[ ] 假设合理吗？
[ ] 答案回应了问题吗？
</critique>
```

可量化的 critique 远比模糊"自查"有用。

### 3. final 要修正 critique 发现的问题

明确告诉模型："如果 critique 发现问题，<final> 必须改正；如果没问题，<final> 跟 draft 一致即可。"

不写的话，有些模型会无视 critique 直接 copy draft。

## 什么时候用

适合：
- 复杂推理（数学 / 逻辑 / 代码生成）
- 严谨要求的输出（合规文档 / 法律分析）
- 错误代价高的场景

不适合：
- 简单对话 / 问候
- 实时低延迟场景（多 3-5x token 输出）
- 创意写作（critique 容易把妙处改没）

## 一个朴素结论

> 不需要 agent 框架，**一个会自检的 prompt 就能涨 10-15% 准确率**。
> 输出多花的 token 远便宜过错答案的代价。
