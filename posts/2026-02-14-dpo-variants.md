---
layout: post
title: DPO 变种 — ORPO / KTO / SimPO 怎么选
date: 2026-02-14
topic: "模型与训练"
tags: [AI, DPO, ORPO, KTO, SimPO]
excerpt: DPO 火了之后衍生出十几个变种。ORPO 合并 SFT + DPO，KTO 不要 pair，SimPO 不要 ref model。各有适用场景。
permalink: /posts/2026-02-14-dpo-variants.html
---

## DPO 之后的家族

2023 年 DPO 简化了 RLHF。2024-2026 又有一系列变种解决 DPO 自身的问题：

| 算法 | 核心创新 | 适合场景 |
|---|---|---|
| **DPO** | 用偏好对直接优化，不要 reward model | 标准对齐 |
| **IPO** | 修复 DPO 的长度偏差 | DPO 跑飞时换它 |
| **KTO** | 不要 pair，只要"好 / 坏"标签 | 标注成本敏感 |
| **ORPO** | 把 SFT + DPO 合成一步 | 节省训练成本 |
| **SimPO** | 不要 ref model | 显存紧张 |
| **DNO** | 用对手模型替代偏好 | 不要人类偏好数据 |

## ORPO（Odds Ratio Preference Optimization）

**最大卖点**：跳过 SFT 直接对齐——一步搞定。

经典 pipeline：

```
Base → SFT → DPO
```

ORPO：

```
Base → ORPO（一步）
```

loss 同时包含 SFT loss（让模型会做任务）+ 偏好 loss（让模型偏好 chosen）：

```
L_ORPO = L_SFT + λ · L_OR(chosen, rejected)
```

**优势**：训练成本 -50%（少跑一遍 SFT）。
**适合**：从头训练新模型 / 数据量大 / 算力紧。

## KTO（Kahneman-Tversky Optimization）

DPO 要 (prompt, chosen, rejected) 三元组——**采集 pair 数据贵**。
KTO 只要 (prompt, response, label)——label 是 "好" 或 "坏" 二元标签。

```
DPO 数据: 一份回答 A 一份回答 B, A 好
KTO 数据: 一份回答, 标"好"或"坏"
```

**优势**：能用已有的"用户反馈点赞 / 踩"数据直接训。
**劣势**：理论上效果略弱于 DPO（损失了相对信息）。
**适合**：有大量单点 feedback 数据的产品（如客服满意度）。

## SimPO（Simple Preference Optimization）

DPO 训练时要维护 ref model（冻结的 SFT 版本）做 KL 参照——**多吃显存 + 多一次 forward**。
SimPO 完全去掉 ref model：

```
L_SimPO = -log σ(β · (log π(yw) / |yw| - log π(yl) / |yl|) - γ)
```

用 length normalization 替代 KL 项。**显存省一半**。
**适合**：显存紧张、追求训练效率。

## 这么多怎么选

| 你的处境 | 推荐 |
|---|---|
| 想稳，照标准走 | DPO |
| 长度偏差明显（chosen 总比 rejected 长） | IPO |
| 只有 thumbs up/down 数据 | KTO |
| 想 SFT + DPO 一步搞定 | ORPO |
| 显存紧 / GPU 少 | SimPO |
| 业界主流推荐 2026 | ORPO（性价比最好）|

## 实战代码（TRL 都支持）

```python
from trl import ORPOTrainer, ORPOConfig

config = ORPOConfig(
    output_dir="./orpo-out",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    learning_rate=5e-6,
    beta=0.1,  # ORPO 的 lambda
)

trainer = ORPOTrainer(
    model=model,
    args=config,
    train_dataset=ds,
    tokenizer=tokenizer,
)
trainer.train()
```

TRL（HuggingFace）现在已经支持 DPOTrainer / IPOTrainer / KTOTrainer / ORPOTrainer / SimPOTrainer 全家桶。

## 一个朴素结论

> DPO 是基线，但**2026 年新项目优先考虑 ORPO**——一步到位 + 训练成本低。
> 数据形态特殊时（KTO 用单点 feedback、IPO 处理长度偏差）选对应变种。
>
> SimPO 是显存救星，能用一张消费级 GPU 训 7B。
