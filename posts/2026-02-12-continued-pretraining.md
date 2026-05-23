---
layout: post
title: Continued Pretraining — 给模型注入领域知识
date: 2026-02-12
topic: "模型与训练"
tags: [AI, Pretraining, 领域微调]
excerpt: SFT 教模型"会做任务"，但教不会它"知道 X 领域"。Continued Pretraining 是把领域语料喂回模型的关键步骤。
permalink: /posts/2026-02-12-continued-pretraining.html
---

## 什么时候需要 CP

SFT 微调能教模型遵循指令，但**不能往模型脑子里塞新知识**。

如果你的场景是：
- 医学专业领域（罕见病、药物名称）
- 法律 / 合规（行业法规、判例）
- 公司内部知识（产品 SKU、内部黑话）
- 一种小语种（模型训练时见得少）

→ 单纯 SFT 不够，要 **Continued Pretraining (CP)**：用领域语料**继续做 next-token prediction**。

```
Base Model 
  → CP（在领域语料上 LM 训练）
  → 领域 base
  → SFT（教指令）
  → DPO/RLHF（对齐）
  → 部署
```

## 数据准备

不像 SFT 要 (instruction, response) 对，CP 要的是**纯文本**——领域内的书、论文、文档、对话记录。

数据量参考：

| 目标 | 数据量 |
|---|---|
| 微弱注入领域风格 | 50M-500M token |
| 显著提升领域知识 | 1B-10B token |
| 训练领域强 base | 50B+ token |

100B+ 量级就是"小公司接近不可能"——
**真正的领域 CP 一般在 1-10B token 级别做**。

## 数据质量是关键

CP 跟 pretraining 一样，垃圾数据训出来模型变笨：

- 去重（MinHash + LSH）—— 重复数据让训练浪费
- 质量过滤（语言识别 / 困惑度阈值）
- 数据混合：领域数据 + 通用数据（防止 catastrophic forgetting）

经典混合比例：
- 70% 领域语料
- 30% 通用语料（C4 / RedPajama 等）

太纯领域会让模型忘掉通用能力。

## 训练配置

```python
args = TrainingArguments(
    learning_rate=5e-5,            # 比 SFT 低
    lr_scheduler_type="cosine",
    warmup_steps=1000,
    num_train_epochs=1,            # 通常 1 epoch
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    bf16=True,
    save_steps=2000,
    logging_steps=50,
)
```

关键超参：

- **lr**：5e-5 是经验值；7B 模型可调到 3e-5；70B 更低
- **epochs**：1 epoch 通常够；多了过拟合 + 灾难遗忘加剧
- **mask loss**：可选；通用 LM 训练不 mask（全 token 算 loss）

## 灾难性遗忘（Catastrophic Forgetting）

最大风险：训完后模型在通用 benchmark（MMLU / HellaSwag）上跌 5-15%。

缓解：

1. **数据混合**：领域 + 通用（前面说的）
2. **学习率别太大**：避免权重剧烈变动
3. **regularization**：加 KL 项约束跟原模型
4. **LoRA**：只更新 adapter，不动 base
5. **EWC（Elastic Weight Consolidation）**：保护对原任务重要的权重

实战：上 LoRA + 数据混合，够 80% 场景。

## CP 还是 RAG

很多人混淆 CP 和 RAG 的角色：

| 需求 | CP | RAG |
|---|---|---|
| 模型理解领域**语言风格** | ✅ | ❌ |
| 模型知道**具体事实**（可频繁更新）| ❌（训练数据是 snapshot）| ✅ |
| 模型流畅讨论领域话题 | ✅ | 一般 |
| 准确引用最新信息 | ❌ | ✅ |
| 处理小语种 | ✅ | ❌ |

**经典组合 = CP + RAG**：CP 让模型懂领域，RAG 让模型回答时拿到最新事实。

## 一份发布 checklist

- [ ] 数据 ≥ 1B token，且去重 + 质量过滤过
- [ ] 70/30 混合通用语料
- [ ] lr 不超过 5e-5
- [ ] 1 epoch 够，看 eval loss
- [ ] 训练后在通用 benchmark 上不掉 > 5%
- [ ] 领域专门 benchmark 上明显涨

## 一个朴素结论

> 没必要从零训模型——**Continued Pretraining 已经能让你拥有领域专家级 base**。
>
> 100M-1B token 的精挑领域语料 + 一周 GPU 时间，效果远超大量"通用 SFT 数据"。
