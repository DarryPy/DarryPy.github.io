---
layout: post
title: PPO 实现要点 — 让 RLHF 训得动的工程细节
date: 2026-03-08
topic: "模型与训练"
tags: [AI, PPO, RLHF]
excerpt: PPO 是 RLHF 里最难训的一环。Reference model、Value model、KL 控制、显存优化的实战细节。
permalink: /posts/2026-03-08-ppo-implementation.html
---

## PPO 的核心思想

PPO (Proximal Policy Optimization) 的目标：**最大化 reward，但不要偏离 ref 太多**。

```
objective = E[R(s,a)] - β · KL(π_new || π_old)
```

第一项让模型刷分，第二项防止它"为了刷分疯狂胡说八道"。

## 4 个必备模型

跑 PPO 你要同时维护：

| 模型 | 作用 | 是否更新 |
|---|---|---|
| **Policy** | 被训练的对话模型 | ✅ 更新 |
| **Reference** | 初始 SFT，作 KL 参照 | ❌ 冻结 |
| **Reward Model** | 给 (prompt, response) 打分 | ❌ 冻结 |
| **Value Model** | 估计未来累计 reward（critic）| ✅ 更新 |

7B policy 全参微调 PPO 需要 100GB+ 显存。

## 训练循环

```python
for epoch in epochs:
    for prompts in batch:
        # 1. Rollout: 用 policy 生成 response
        responses = policy.generate(prompts)

        # 2. 计算 reward
        rewards = reward_model(prompts, responses)

        # 3. 计算 value（每个 token 的预期累计 reward）
        values = value_model(prompts, responses)

        # 4. 计算 advantage = reward - value
        advantages = rewards - values

        # 5. 计算 KL divergence 跟 ref
        log_p_new = policy(responses)
        log_p_ref = ref_model(responses)
        kl = log_p_new - log_p_ref

        # 6. PPO loss
        ratio = exp(log_p_new - log_p_old)
        policy_loss = -min(ratio * advantages, clip(ratio, 1-ε, 1+ε) * advantages).mean()
        value_loss = (rewards - values).pow(2).mean()
        kl_penalty = β * kl.mean()

        loss = policy_loss + 0.5 * value_loss + kl_penalty
        loss.backward()
```

## 关键超参

| 参数 | 推荐 | 调整方向 |
|---|---|---|
| `learning_rate` | 1e-6 到 1e-5 | 大模型用低端 |
| `kl_coef (β)` | 0.01-0.1 | 偏离太多→升高，刷不动 reward→降低 |
| `clip_range (ε)` | 0.1-0.2 | 防 policy 更新过大 |
| `gamma` (折扣) | 0.99-1.0 | 短 response 用 1.0 |
| `gae_lambda` | 0.95 | GAE 的衰减 |
| `mini_batch` | 32-128 | 看显存 |
| `epochs_per_rollout` | 2-4 | 每次 rollout 用几遍 |

## 工程坑

### 1. 显存爆炸

4 个模型同时跑，7B × 4 ≈ 60GB FP16。
解法：
- Reference 和 RM 量化到 8-bit（INT8）
- Value head 跟 Policy 共享 backbone
- 用 LoRA 只训 policy 的 adapter

### 2. KL 失控

β 太低，policy 会迅速跑偏，输出垃圾。β 太高，模型学不动。
**实战**：用自适应 KL—— 监控 KL 值，超过目标范围就动态调 β。

### 3. Reward Hacking

policy 学会刷 RM 分数而不真变好。征兆：
- Reward 飙升但人评分变差
- 输出趋同（mode collapse）
- 套话 / 列表 / 长度 突然变多

防御：
- 多个 RM 投票
- RM 训练时加 hacking 反例
- 定期人评校准

### 4. Reward Sparsity

RM 只在 response 末尾给一个标量分，每个 token 拿到的 advantage 信号很弱。
解法：用 GAE (Generalized Advantage Estimation) 把末尾 reward 反向传播给前面 token。

## TRL 是事实标准实现

HuggingFace 的 [TRL](https://github.com/huggingface/trl) 库把所有这些封装好了：

```python
from trl import PPOTrainer, PPOConfig
config = PPOConfig(
    learning_rate=1.41e-5,
    kl_coef=0.05,
    cliprange=0.2,
    mini_batch_size=4,
)
trainer = PPOTrainer(config, model, ref_model, reward_model, tokenizer)
trainer.train(dataset)
```

10 行起步。但要调出好结果，还得理解上面那些超参。

## 一个朴素结论

> PPO 是 RLHF 的"重武器"。
> 工程复杂度极高，2024 年起开源圈大量切到 DPO。
> 真要训 PPO，**用 LoRA + 量化 + TRL**，否则 GPU 烧不起。
