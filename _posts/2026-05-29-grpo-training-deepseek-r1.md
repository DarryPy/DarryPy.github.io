---
layout: post
title: GRPO 训练 — DeepSeek-R1 背后的强化学习新武器
date: 2026-05-29
topic: "模型与训练"
tags: [GRPO, RLHF, DeepSeek, 强化学习, LLM训练]
excerpt: GRPO 是 DeepSeek-R1 采用的强化学习算法，用"同组相对比较"取代了 PPO 的 value model，显存占用近乎减半，训练稳定性更强。本文拆解它的核心逻辑、TRL 实现方式与奖励函数设计，帮你把它用到自己的训练管线里。
permalink: /posts/2026-05-29-grpo-training-deepseek-r1.html
---

你可能注意到 DeepSeek-R1 论文里用了一个陌生的缩写：GRPO。它不是 PPO 的笔误，而是一套专门为大语言模型强化学习阶段设计的新算法。DeepSeek 团队用它训出了在数学推理和代码生成上接近 o1 水平的开源模型，同时把训练成本控制在了一个相对可接受的范围。这篇文章拆解它的核心逻辑，告诉你它为什么能替代 PPO，以及在自己的训练管线里把它用起来需要注意什么。

## PPO 在 LLM 上的症结

RLHF 的标准流程是 PPO（Proximal Policy Optimization）。在 OpenAI 最初发布 InstructGPT 的论文之后，PPO 成了大家默认的选择，原因很简单：它经过了几年 RL 领域的验证，有足够多的工程经验积累。

但是把 PPO 搬到大语言模型训练上，你很快会遇到几个难以回避的问题。首先是显存压力。PPO 标准流程需要四个模型同时驻留在显存里：policy model（正在被训练的目标模型）、reference model（用来计算 KL 散度惩罚的冻结副本）、reward model（用来打分的偏好模型）、以及 value model（用来估计当前状态价值的 critic 网络）。前三个在 RLHF 框架里本来就有，value model 是额外引入的负担。麻烦在于，value model 的规模通常和 policy model 相当——如果你的策略模型是 70B，那么 value model 也是 70B，整体显存占用直接翻倍。

其次是 value model 本身的训练不稳定性。Value model 的目标是预测"从当前 token 开始，最终能拿到多少奖励"，这在 Atari 游戏这种低维状态空间里比较好学，但对于自然语言这种高维、离散、稀疏奖励的场景，value 估计误差会非常大。在训练早期，value loss 飙升、梯度爆炸是家常便饭，很多团队在调 PPO 超参数上花费的时间远超他们的预期。

第三个问题是采样效率。PPO 是 on-policy 算法，每次更新之后已经采集的数据就过期了。这意味着你需要反复做生成——打分——更新的循环，而生成本身就是 LLM 训练里最慢的那一步。

## GRPO 的核心思路 🔑

GRPO（Group Relative Policy Optimization）的想法可以用一句话概括：**用同组样本之间的相对比较来估计优势函数，完全不需要 value model**。

具体做法是这样的。对同一个 prompt，用当前 policy 采样 G 个不同的输出（论文里 G 通常取 8 到 16）。然后用 reward model 分别给这 G 个输出打分，得到 G 个奖励值 r₁, r₂, …, rG。接下来用这组奖励的均值和标准差对每个奖励做标准化，得到每个输出的"相对优势"：

```python
import torch

# rewards: [batch_size, G] 每个 prompt 的 G 个输出奖励
mean_r = rewards.mean(dim=-1, keepdim=True)
std_r  = rewards.std(dim=-1, keepdim=True)
advantages = (rewards - mean_r) / (std_r + 1e-8)
```

advantage 为正意味着"这个输出比同组平均水平好"，为负意味着"低于平均"。然后按照标准 PPO 的 clipped surrogate objective 进行参数更新，只是完全去掉了原来 value function 的那一项。

这套设计的几个好处值得单独说清楚。第一，baseline 是数据自动给出的，不需要任何额外的网络结构，实现简洁，显存占用少了将近一个模型大小。第二，同组比较天然适合语言模型的生成场景——相同 prompt、不同输出、统一打分，这正是你在采样时就会做的事，不需要额外设计。第三，当同一 prompt 的 G 个输出质量差异不大时，advantages 趋近于 0，更新幅度自动收缩，训练本身就有内建的稳定机制。

需要注意的是，GRPO 仍然保留了 reference model 和 KL 散度惩罚项，这是防止 policy 无限漂离 reference 分布、产生 reward hacking 的关键保护。

## 用 TRL 跑 GRPO

Hugging Face TRL 库从 0.9 版本开始内置了 `GRPOTrainer`，接口设计和 `PPOTrainer` 高度相似，如果你之前跑过 PPO，迁移的学习成本非常低。

```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model     = AutoModelForCausalLM.from_pretrained("your-sft-model")
ref_model = AutoModelForCausalLM.from_pretrained("your-sft-model")  # 冻结，不更新

config = GRPOConfig(
    num_generations=8,            # G，每个 prompt 采样几个输出
    max_new_tokens=512,
    temperature=0.8,              # 控制输出多样性，太低则 G 个输出趋同
    learning_rate=1e-6,
    kl_coef=0.05,                 # KL 惩罚系数
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
)

trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    train_dataset=dataset,
    config=config,
)
trainer.train()
```

几个关键超参数的取值参考：

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `num_generations` (G) | 8–16 | 太小则 advantage 估计方差高；太大则显存爆；8 是通常的起点 |
| `kl_coef` | 0.01–0.1 | 防止 policy 偏离 reference 太远；任务越开放，这个值越要谨慎 |
| `temperature` | 0.7–1.0 | 低于 0.5 时 G 个输出几乎一样，advantage 全部趋近于 0 |
| `clip_range` | 0.2 | PPO clipping 默认值通常不需要调 |

如果显存依然紧张，`ref_model` 可以用 `peft` 以共享基础权重的方式加载，或者启用梯度检查点（`gradient_checkpointing=True`）来换空间。

除 TRL 之外，字节跳动开源的 veRL 框架对 GRPO 的支持也很完整，并且在大规模多机训练场景下做了更深度的调度优化。如果你的训练规模超过单机八卡，veRL 的异步生成和参数同步机制值得评估。两个框架的 GRPO 核心逻辑是一致的，换用的迁移成本主要在数据加载和分布式配置上，算法层面不需要改动。

## 奖励函数设计是真正的杠杆

算法本身只是框架，reward function 的设计才决定训练的上限。DeepSeek-R1 在这块做了两个非常聪明的选择，值得认真借鉴。

第一个是**规则奖励（rule-based reward）**。对于数学题和代码题，可以直接通过执行器验证答案的对错，给出 0/1 的二值奖励。数学题的答案是一个确定的数，对就是对，错就是错；代码题可以跑单元测试，通过了给分，失败了不给。这类奖励信号准确、无噪声，不需要另外训一个 reward model，是 DeepSeek-R1 在推理任务上表现强劲的核心原因。如果你的任务能写出客观的验证逻辑，优先选择这种方式，胜过任何 learned reward model。

第二个是**格式奖励（format reward）**。DeepSeek-R1 要求模型把推理过程写在 `<think>...</think>` 标签里，把最终答案写在 `<answer>...</answer>` 里，格式满足要求才给额外奖励分。这个设计强迫模型显式地习得"先思考再回答"的行为模式，本质上是在用奖励信号引导推理链的结构化程度。你在自己的任务里也可以用类似的格式约束，比如强制输出 JSON、强制按步骤列清单，都能通过格式奖励来引导。

对于开放式问答、对话质量评估等无法用规则验证的任务，还是需要 reward model 打分。这时 reward model 本身的质量直接决定 GRPO 的天花板。不要在这块的数据和训练上节省成本，reward model 训不好，GRPO 再怎么跑都是在优化错误的目标。

## 踩坑清单

- **G 太小让训练基本无效**：G 小于 4 时，advantage 的估计噪声极大，几乎等于随机梯度；显存不够的情况下用梯度检查点或 offload 换空间，G 从 8 起步
- **reward hacking 来得比预期更快**：GRPO 的信号虽然比 PPO 稳定，但模型学会刷长度分、刷格式分的速度同样快；明确加入长度惩罚项或多维度奖励组合来防止这一点
- **reference model 必须是 SFT 后的检查点**：用 pretrained 原始权重做 reference 会让 KL 惩罚从第一步就极大，训练全程都在与 KL 约束搏斗；SFT 后的权重才是合理的起点
- **采样温度要够高**：temperature 低于 0.5 时，G 个输出在 token 层面几乎相同，所有 advantage 都趋近于 0，等同于梯度消失，训练没有进展
- **不要跳过 SFT 直接上 GRPO**：DeepSeek-R1-Zero 证明了从预训练权重直接做 GRPO 在理论上可行，但那是千卡量级的预算支撑的实验；你的模型如果没有 SFT 热身，reward 信号会极度稀疏，GRPO 很难收到有效反馈

相比 PPO，GRPO 去掉了最贵的一块，代价是对采样多样性和 reward 设计更敏感。理解这个权衡，选对任务，设计好奖励函数，它会比 PPO 好用得多。
