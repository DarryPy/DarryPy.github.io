---
layout: post
title: DPO 详解 — 不需要 reward model 的 RLHF 替代品
date: 2026-04-01
topic: "模型与训练"
tags: [AI, DPO, RLHF, 对齐]
excerpt: DPO 把 RLHF 的复杂 pipeline 砍到剩"一对偏好数据 + 一个 loss 函数"。2024 起几乎所有开源模型对齐都在用。
permalink: /posts/2026-04-01-dpo-explained.html
---

## RLHF 的痛点

RLHF (Reinforcement Learning from Human Feedback) 是 GPT-3.5 之后大火的对齐方法。
但它的 pipeline 极其复杂：

1. SFT 微调
2. 收集人类偏好对（chosen vs rejected）
3. 训练 reward model
4. 用 PPO 优化策略模型 against reward model
5. 不断防止 reward hacking

PPO 阶段尤其难——多模型同时跑、超参敏感、不稳定。

## DPO 的核心想法

**不要 reward model，直接用偏好数据更新模型。**

DPO (Direct Preference Optimization) 的数学洞见：
"最大化 reward model 期望" 等价于 "增大 chosen 相对 rejected 的对数似然差"。

跳过中间的 reward model 训练 + PPO 优化，**直接一个 loss 解决**：

```
L_DPO = -log σ(β · (log π(yw|x) - log π_ref(yw|x))
                - β · (log π(yl|x) - log π_ref(yl|x)))
```

其中：
- `x`：prompt
- `yw`：chosen response
- `yl`：rejected response
- `π`：当前策略模型
- `π_ref`：参考模型（通常是 SFT 后的初始模型）
- `β`：温度参数，控制偏离 ref 的程度

**实战上不用看懂数学**，知道它在做什么即可：让 chosen 概率上升、rejected 概率下降，同时不偏离 ref 太多。

## DPO 数据集长什么样

每条样本一个三元组：

```json
{
  "prompt": "怎么写一封专业的辞职信？",
  "chosen": "尊敬的领导：\n经过深思熟虑...",
  "rejected": "Hi 老板，老子不干了"
}
```

数据规模：**通常 5k-50k 对**。比 SFT 少，比 PPO 简单很多。

## 实战代码

用 TRL 库，10 行起步：

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载 SFT 后的模型作为 policy 和 ref
model = AutoModelForCausalLM.from_pretrained("./sft-output")
ref_model = AutoModelForCausalLM.from_pretrained("./sft-output")
tokenizer = AutoTokenizer.from_pretrained("./sft-output")

# 2. 配置
args = DPOConfig(
    output_dir="./dpo-output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,           # DPO 用很低的 lr
    beta=0.1,                     # 关键超参
    max_length=2048,
    max_prompt_length=512,
)

# 3. 训练
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=args,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

## 关键参数 β

β 控制"偏离 ref 的代价"：

- β 小（0.01-0.1）：模型可以大幅偏离 ref，学得猛但容易跑偏
- β 大（0.3-1.0）：模型保守，行为接近 ref

经验值：**0.1-0.3 是甜区**。从 0.1 起调。

## 数据从哪来

1. **人工标注**：最贵最准
2. **现有模型对比**：让 GPT 评 Claude 输出 vs Llama 输出，分出 chosen/rejected
3. **同模型多采样**：同 prompt 让模型温度高跑 N 次，用 reward model 或 LLM-as-judge 挑出 chosen/rejected
4. **公开数据集**：UltraFeedback、HH-RLHF、Anthropic HH

实战常用 **3** —— 不依赖外部标注，可大规模扩展。

## DPO 的几个变种

| 变种 | 差异 |
|---|---|
| **IPO** | 改进 DPO 在长 response 上的偏差 |
| **KTO** | 不要 pair，只要"好" / "坏" 标签即可 |
| **ORPO** | 把 SFT 和 DPO 合并成一步 |
| **SimPO** | 不需要 ref model，更省显存 |

2026 年大多数生产用 DPO 或 ORPO；前沿研究偏 SimPO。

## SFT vs DPO vs RLHF：什么时候用啥

| 阶段 | 目的 |
|---|---|
| **SFT** | 让模型会做这个任务 |
| **DPO/RLHF** | 让模型在多个会做的选项里挑"用户喜欢的那个" |

经典 pipeline 是先 SFT 再 DPO：

```
Base Model → SFT (学任务) → DPO (学偏好) → 部署
```

跳过 SFT 直接 DPO 也行，但效果通常差一截——
**DPO 假设模型已经会做这件事**，只是要校准偏好。

## 工程踩坑

1. **lr 比 SFT 低一个数量级**：DPO 用 1e-6 到 5e-7，SFT 用 2e-4
2. **chosen/rejected 长度差太大**：长的那个会被天然偏好。建议长度差异在 30% 以内
3. **数据噪声敏感**：错的标签会让模型学反；标注质量比 SFT 更重要
4. **β 调错就翻车**：太小模型疯，太大跟没训一样
5. **ref model 不能丢**：训练时 ref 必须存活，省显存可以用 QLoRA 量化 ref

## 一句话总结

> 想做模型对齐？
> 简单任务：SFT 就够。
> 要做偏好对齐：直接上 DPO，省 80% 复杂度，效果跟 PPO 接近。

PPO 留给前沿研究和有 ML 团队的公司。
