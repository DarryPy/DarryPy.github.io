---
layout: post
title: RLHF 详解 — ChatGPT 之所以像人的关键技术
date: 2026-03-10
topic: "模型与训练"
tags: [AI, RLHF, 对齐]
excerpt: 从 SFT 到 RLHF：用人类偏好把模型"调教"成对话有礼、回答有用、拒答得体。原理、实现、坑全梳理。
permalink: /posts/2026-03-10-rlhf-explained.html
---

## RLHF 是什么

**Reinforcement Learning from Human Feedback** — 用人类偏好作为信号训练 LLM。

经典 pipeline：

```
1. SFT 微调 (Supervised Fine-Tuning)
   → 让模型会做任务

2. 收集偏好对 (chosen vs rejected)
   → 比如：同一个问题 2 个回答，人选哪个好

3. 训练 Reward Model
   → 学会给"好回答"打高分

4. PPO 优化策略模型
   → 让模型最大化 Reward Model 的分数
```

## 为什么是它让 ChatGPT 起飞

GPT-3 在 2020 年就出来了，但很笨拙。
GPT-3.5 / ChatGPT 让大众惊艳，**关键就是 RLHF 让它变得"懂事"**：
- 礼貌、清晰、有用
- 该拒答的拒答
- 不胡乱给危险建议

没 RLHF 的模型像没受过训练的实习生，**强但不实用**。

## 4 个阶段详解

### 1. SFT

把"指令 + 期望回答"配对数据微调进模型。**让模型会"按指令做事"**。

```
[Input] 帮我写一封辞职信
[Output] 尊敬的领导：经过...
```

这一步在前面 [SFT 完全指南](/posts/2026-04-03-sft-guide.html) 详细讲过。

### 2. 偏好数据采集

同一个 prompt 让 SFT 模型生成 2 个回答（温度高）：

```
prompt: 怎么减肥？
response_A: 节食 + 跑步是关键，建议每天慢跑 30 分钟...
response_B: 减肥就靠少吃。
```

让人类标注员选哪个更好：A vs B。
**收集几万到几十万对**，组成偏好数据集。

### 3. 训练 Reward Model

Reward Model (RM) 是一个分类/回归模型：
- 输入：prompt + response
- 输出：标量分数

训练目标：让 chosen 得分 > rejected 得分。

```python
loss = -log(sigmoid(reward(prompt, chosen) - reward(prompt, rejected)))
```

RM 通常用比 policy 小的模型（节省成本），从 SFT checkpoint 初始化。

### 4. PPO 优化 Policy

最复杂的一步。用 PPO（Proximal Policy Optimization）让 policy 模型最大化 RM 分数：

```
loss = -RM(prompt, policy_response) + β · KL(policy || ref)
```

- 最大化 RM 给的奖励
- 但加一个 KL 项防止偏离 ref（初始 SFT 模型）太远，避免"为了刷分疯狂胡说"

## 工程复杂度

PPO 阶段要同时维护 **4 个模型**：
- Policy（被训练的）
- Ref Policy（初始 SFT，冻结）
- Reward Model（冻结）
- Value Model（critic，跟 policy 一起训）

显存暴涨、训练不稳定、超参敏感——这就是为什么很多团队转 DPO。

## RLHF vs DPO

| | RLHF (PPO) | DPO |
|---|---|---|
| 步骤 | SFT → RM → PPO | SFT → DPO（一步）|
| 模型数 | 4 个 | 2 个（policy + ref）|
| 稳定性 | 难调 | 稳定 |
| 效果 | 上限略高 | 接近 |
| 工程复杂度 | 高 | 低 |

2024 年起开源圈基本切到 DPO；大厂还在用 PPO。

## RLHF 的常见问题

### 1. Reward Hacking

模型学会"刷 RM 分数"而不是真的变好。例子：
- 加套话："让我详细解释..."（看起来用心）
- 列表 over 段落（看起来结构化）
- 长 over 短（看起来认真）

防御：RM 训练时加入"hacking 反例"。

### 2. Mode Collapse

PPO 优化太狠，模型只生成"安全的 mainstream 回答"，多样性大降。

防御：β（KL 系数）调高，惩罚偏离 ref。

### 3. 标注质量

RM 上限被人类标注质量限制。
标注员有偏好 + 累 + 不一致，**dirty RM 训出来 PPO 也救不了**。

## 一个朴素结论

> RLHF 是 GPT-3.5 时代的对齐里程碑技术，但工程复杂度高。
> 2026 年想做对齐：**先用 DPO**；研究 / 极致效果场景才上 RLHF。

理解 RLHF 仍然重要——它是后续所有对齐方法的概念基础。
