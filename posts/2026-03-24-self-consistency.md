---
layout: post
title: Self-Consistency — 多次采样投票的稳定杀招
date: 2026-03-24
topic: "Prompt 与推理"
tags: [AI, Self-Consistency, CoT]
excerpt: 同一个问题让模型跑 N 次取多数答案，复杂推理任务正确率能再涨 10-20%。原理、代价、什么时候不值得。
permalink: /posts/2026-03-24-self-consistency.html
---

## 核心想法

CoT 让模型先推理再答。但单次推理可能走偏。
Self-Consistency：**用 temperature > 0 跑 N 次（通常 5-10 次），各次答案投票**，多数获胜。

直觉：错的推理路径是多样的、各自不同；对的推理路径相对一致。**多数票 ≈ 真理**。

## 一个最小实现

```python
def self_consistency(question, n=5, temperature=0.7):
    answers = []
    for _ in range(n):
        cot_resp = llm.complete(
            f"{question}\n请一步一步推理。",
            temperature=temperature,
        )
        ans = extract_final_answer(cot_resp)
        answers.append(ans)
    return Counter(answers).most_common(1)[0][0]
```

注意 `temperature > 0`，否则每次结果一样，投票无意义。

## 效果

GSM8K（小学数学题）实测：
- 单次 CoT：~85%
- Self-Consistency × 5：~92%
- Self-Consistency × 40：~95%（边际收益递减）

复杂推理任务上是**最朴素有效的涨点方式**。

## 什么时候用

适合：
- 任务**有标准答案**（数学题、代码题、SQL 生成）
- 错误代价高，愿意多花 5-10x 成本换准确率
- 答案能被"标准化"提取（数字、JSON 字段、最终结论）

不适合：
- 开放生成（写文章）—— 没有"多数"概念
- 答案差异本身就是价值（创意类）
- 成本敏感、要求快

## 实战注意

### 1. 答案归一化

```python
# 原始答案: "答：6 度" / "6°C" / "六度"
def normalize(s):
    nums = re.findall(r'\d+\.?\d*', s)
    return nums[0] if nums else None
```

不归一化的话投票会失败。

### 2. 加权投票

不是每次输出都同等可信。可以让 LLM 自评：

```
[在每次推理末尾]
你对这个答案的把握 0-1：0.85
```

按把握加权投票，效果比平投票好。

### 3. 早停

```python
# 前 3 次答案完全一致 → 提前结束
if len(set(answers[:3])) == 1:
    return answers[0]
```

省 token。

## 一个朴素结论

> 答案有"对错"判定的任务：**Self-Consistency 是最便宜的涨点手段**。
> 没 ground truth 的开放任务：用别的方法。
