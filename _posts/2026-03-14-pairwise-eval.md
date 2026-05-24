---
layout: post
title: Pairwise Comparison Eval — A/B 评分比绝对打分更稳定
date: 2026-03-14
topic: "评估与安全"
tags: [AI, Eval, Pairwise]
excerpt: 让 judge 给绝对分数（4.7 / 3.2）噪声大；让 judge 在 A 和 B 之间挑一个，结论稳定得多。
permalink: /posts/2026-03-14-pairwise-eval.html
---

## 为什么 Pairwise 比 Pointwise 稳

绝对评分（pointwise）让 judge 给 1-5 分，问题：

- 不同样本之间分数没对齐
- 不同 judge 的"标准"不一样
- judge 自己每次心情不一样（漂移）

Pairwise 直接问 "A 和 B 哪个好"——
**相对比较是模型/人都更擅长的判断**，结果稳定得多。

## Chatbot Arena 就是这么做的

```
[问题]
今天上海天气怎么样？

[答案 A]
今天上海多云转晴，气温 22-28°C，建议带件薄外套。

[答案 B]
你可以查天气 app。
```

让 judge 选一个，记录胜率。
跑成千上万对，统计每个模型的 ELO 分数。

## 怎么实现

```python
def pairwise_eval(question, answer_a, answer_b, judge_llm):
    prompt = f"""
对比 A 和 B 两个回答，哪个更好？

判断标准：
- 准确性（事实对不对）
- 完整性（关键信息全不全）
- 相关性（是否切题）
- 表述（是否清晰）

不要根据回答长度判断质量。

问题：{question}

[答案 A]
{answer_a}

[答案 B]
{answer_b}

输出 JSON: {{"winner": "A" | "B" | "tie", "reason": "..."}}
"""
    return judge_llm.complete(prompt, response_format="json")
```

## 必踩的坑：位置偏差

实验显示：**A 选项放在前面，胜率天然比放后面高 5-15%**——模型对"先看到的"有偏好。

修法：**每对样本评两次，A/B 顺序对调**：

```python
def fair_pairwise(q, a, b, judge):
    r1 = pairwise_eval(q, a, b, judge)
    r2 = pairwise_eval(q, b, a, judge)  # 调换顺序

    # 投票决定最终结果
    a_wins = (r1.winner=='A') + (r2.winner=='B')
    b_wins = (r1.winner=='B') + (r2.winner=='A')
    if a_wins > b_wins: return 'A'
    if b_wins > a_wins: return 'B'
    return 'tie'
```

## 计算总体胜率

跑 100+ 个测试 query，统计：

```
A vs B: A 胜 62 次，B 胜 33 次，5 平
A 胜率 = 62 / (62 + 33) = 65%  (排除 tie)
```

要做显著性检验的话用 binomial test，看 p < 0.05。

## ELO 排名（多模型对比）

多个模型互相 pairwise 后用 ELO 算分：

```python
# 简化版
def update_elo(rating_a, rating_b, result_a, k=32):
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    return rating_a + k * (result_a - expected_a)
```

跑足够多对比后，每个模型有一个 ELO 分数，可以直接排名。
Chatbot Arena 用的就是这套。

## 适合 / 不适合

适合：
- 模型对比、prompt 版本对比
- 没有标准答案的开放任务（写作、对话）
- 需要稳定 ranking 的场景

不适合：
- 任务有 ground truth（数学题）→ 用 accuracy
- 需要绝对质量分数（"打 80 分以上才发布"）

## 一个朴素结论

> Pointwise 评分有"分数漂移"问题。
> Pairwise 强迫 judge 做相对判断，结果稳定 + 跨实验可比。
> **没有 ground truth 的任务上，pairwise 是 eval 的金标准**。
