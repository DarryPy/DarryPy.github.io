---
layout: post
title: LLM-as-Judge 实战 — 让 AI 评 AI 的正确姿势
date: 2026-04-07
topic: "评估与安全"
tags: [AI, Eval, LLM-as-Judge]
excerpt: 没人评 AI 答案是工程化 eval 的最大瓶颈。让一个强 LLM 当评委是答案，但要避免位置偏差、过度宽容、评分漂移等坑。
permalink: /posts/2026-04-07-llm-as-judge.html
---

## 为什么需要 LLM-as-Judge

传统 eval 用人评：质量高但**贵且慢**。
1000 条样本人评要花一天，跑 10 次实验就 10 天。

LLM-as-Judge：让一个强模型当评委。代价：每条几分钱 / 几秒。**速度 100 倍、成本 1/50**。
当然它没人准，但配上规则 + 校准，**80% 的 case 上可用**。

## 两种基础范式

### 1. 评分式（Pointwise）

```
[Judge Prompt]
评估下面这个回答的质量，打分 1-5：
- 5：完美回答，准确、完整、清晰
- 4：很好，小瑕疵
- 3：能用，有明显不足
- 2：明显错误
- 1：完全错误或答非所问

[Question]
{question}

[Answer]
{answer}

输出 JSON: {"score": 1-5, "reason": "..."}
```

适合：单一指标、绝对评价。
缺点：分数漂移大，跨实验难比较。

### 2. 对比式（Pairwise）

```
[Judge Prompt]
对比 A 和 B 两个回答，哪个更好？
- A 好
- B 好
- 一样好

[Question]
{question}

[Answer A]
{answer_a}

[Answer B]
{answer_b}

输出 JSON: {"winner": "A" | "B" | "tie", "reason": "..."}
```

适合：A/B 测试、模型对比。
**比 pointwise 更稳定**，因为相对比较比绝对评分容易。

## 5 个必踩的坑

### 1. 位置偏差（Position Bias）

实验显示：**A 选项放在前面，胜率比放在后面高 5-15%**。
模型对"先看到的"有偏好。

**修法**：每个 case 跑两次，A/B 顺序对调，取平均。
或者用更稳定的判官（GPT-4.5 / Opus 比小模型轻得多）。

### 2. 长度偏差（Length Bias）

模型倾向于"看起来更详细"的答案。
**让回答字数差不多**，或在 judge prompt 里明确："不要根据长度判断质量"。

### 3. 自家偏好（Self-preference）

GPT 当判官评 GPT 的回答，会给自家打高分。
Claude 评 Claude 同理。
**用第三方模型当判官**——比如评 Claude 输出用 GPT，评 GPT 输出用 Claude。

### 4. 评分粒度过细

让模型打 1-100 分，相邻分数（如 78 vs 82）几乎是噪声。
**用 1-5 整数**。需要更细就用排序而不是分数。

### 5. Judge prompt 太模糊

❌ "评价质量"
✅ 给具体维度 + 每个分数的标准 + 反例

清晰的 rubric 让 judge 稳定性翻倍。

## 校准方法：判官靠不靠谱怎么知道

跑 50-100 条真实样本，人评 + LLM-judge **都跑一遍**，算一致性：

| 指标 | 含义 |
|---|---|
| Cohen's Kappa | 跟随机比，0.6+ 算可用 |
| Pearson 相关 | 适合 pointwise scoring，0.7+ 算稳 |
| Agreement % | Pairwise 直接看一致率，70%+ 算可用 |

不到阈值就调整 judge prompt 重做，**不要直接用没校准的 judge**。

## 进阶配方

### 多 judge 投票

让 3 个不同模型当 judge，多数决：

```python
votes = [gpt_judge(...), claude_judge(...), gemini_judge(...)]
final = majority(votes)
```

更稳定，代价是成本 3x。关键决策可以用。

### 带推理的 judge

Pairwise 让 judge **先写理由再下结论**：

```
[Judge Prompt]
对比 A 和 B 后：
1. 先列出 A 的优势
2. 再列出 B 的优势
3. 最后给结论

输出 JSON: {"a_pros": [...], "b_pros": [...], "winner": "..."}
```

研究显示这种 chain-of-thought judge **比直接打分准 10-20%**。

## 多维度评估

不要只用一个分数。常见 RAG 评估的 4 维：

| 维度 | 检查啥 |
|---|---|
| Faithfulness | 回答是否基于检索片段（vs 凭空编） |
| Answer Relevancy | 回答跟问题相关吗 |
| Context Precision | 检索的片段有用吗 |
| Context Recall | 关键信息都检索到了吗 |

每个维度单独让 judge 打分，比单一总分有用得多。

## 一个朴素 checklist

发布 judge 之前过一遍：

- [ ] judge 用比被测更强的模型？
- [ ] judge prompt 有具体 rubric？
- [ ] 跑过 50 条样本人评一致性？Kappa > 0.6？
- [ ] 位置/长度/自家偏差有兜底？
- [ ] 多维度评估，不是单一总分？

5 个都满足，你的 judge 可以拿出去用了。

## 一句话总结

> LLM-as-Judge 不能完全替代人评，但**能让 eval 跑得动**。
> 跑得动 > 完美但不跑。
