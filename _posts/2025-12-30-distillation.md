---
layout: post
title: Distillation 实战 — 用大模型教小模型
date: 2025-12-30
topic: "模型与训练"
tags: [AI, Distillation, 训练]
excerpt: 大模型贵慢，小模型笨。Knowledge Distillation 让小模型在特定任务上达到 90% 大模型的水平，但成本 1/20。
permalink: /posts/2025-12-30-distillation.html
---

## 蒸馏的核心想法

让小模型（student）模仿大模型（teacher）的行为。
不是端到端训新模型，是**用 teacher 生成的数据 + 输出训 student**。

3 个层级的蒸馏：

| 层级 | 方法 | 复杂度 |
|---|---|---|
| **Output Distillation** | teacher 给输入产生输出，student 学这些 (input → output) 对 | 低 |
| **Logit Distillation** | student 学 teacher 的概率分布（不只是最终答案）| 中 |
| **Feature Distillation** | student 学 teacher 内部 hidden states | 高 |

最常用是 Output Distillation——简单且效果不错。

## Output Distillation 流程

```
1. 准备一批 prompt（5k-50k）
2. teacher（如 Claude Opus）对每个 prompt 生成回答
3. 把 (prompt, teacher_answer) 当 SFT 数据训 student（如 Llama 3 8B）
4. student fine-tune 后能在特定任务上接近 teacher
```

```python
# 1. 用 teacher 生成数据
teacher_data = []
for prompt in prompts:
    answer = teacher_model.complete(prompt)
    teacher_data.append({"input": prompt, "output": answer})

# 2. 用这份数据 SFT 训 student
fine_tune(student_model, teacher_data)
```

简单粗暴但有效。

## 为什么有效

Teacher 不只输出最终答案，还**示范了"怎么思考"**：
- 推理风格
- 错答的处理方式
- 拒答的姿态
- 输出格式

Student 模仿这些"高级行为"，**不需要从零学**。

## 适合什么场景

| 场景 | 蒸馏 |
|---|---|
| 大模型贵 / 慢，想找便宜替代 | ✅ |
| 任务专一（路由 / 抽取 / 分类）| ✅ 效果接近 teacher |
| 通用对话 | ⚠️ 效果折损大 |
| 复杂推理 / 数学 | ⚠️ 蒸馏后小模型 reasoning 跟不上 |
| 多模态 | ⚠️ 难，要 vision encoder 兼容 |

## 实测效果

7B student + Opus 4.7 teacher，在客服 FAQ 场景：
- Zero-shot 7B: 70% 准确
- 蒸馏后 7B: 88% 准确
- Opus 4.7: 92% 准确

**蒸馏后 7B 接近 Opus，但成本 1/30**。

## 工程要点

### 1. 选合适的 teacher

- teacher 不一定要最贵的——能在你任务上给 95% 准确就行
- teacher 跟 student 同家族更易蒸（架构相近）

### 2. 数据质量 > 数量

5k 高质量蒸馏数据 > 50k 凑数。
用 LLM-as-judge 过滤掉 teacher 的低质量输出再训。

### 3. 加 diversity

太单一的 prompt 让 student 学偏。
要覆盖：
- 不同 phrasing 同一意图
- 不同长度输入
- 简单 + 困难 case
- 各种边界

### 4. 跟人工数据混合

```
70% teacher 生成 + 30% 人工标注
```

人工数据矫正 teacher 的偏差（teacher 也会错）。

## Reasoning Distillation：教小模型推理

特殊场景：让 small student 学 reasoning model 的推理能力。

```
teacher: o1 / DeepSeek R1（强推理）
  prompt → 详细推理过程 + 最终答案

student: 7B 模型
  SFT on (prompt, full_reasoning + answer)
```

DeepSeek R1 团队就是这样把推理能力下放到 1.5B / 7B 小模型。

## 蒸馏的法律风险

Teacher 是商业模型（GPT / Claude）的话注意：

- OpenAI ToS：禁止用 GPT 输出训竞品模型
- Anthropic ToS：类似限制

实际操作：
- 内部使用 / 不开源 → 风险小
- 公开发布蒸馏模型 → 法务先看 ToS
- 用 Llama / DeepSeek 这种 permissive license 模型做 teacher 最安全

## 一个朴素结论

> 不要每个场景都用最大最贵的模型。
> 对于任务专一的场景，**蒸馏出来的小模型性价比无敌**。
>
> 关键：选对 task + 准备好数据 + 蒸馏后用 eval 验证。
> 做对了，单请求成本能砍到 1/20-1/30。
