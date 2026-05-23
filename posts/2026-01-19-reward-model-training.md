---
layout: post
title: Reward Model 训练 — RLHF 的隐藏命门
date: 2026-01-19
topic: "模型与训练"
tags: [AI, Reward Model, RLHF]
excerpt: RLHF 上限 = Reward Model 上限。RM 训不好，PPO 怎么调都白搭。数据、架构、训练、评估的实战。
permalink: /posts/2026-01-19-reward-model-training.html
---

## Reward Model 是 RLHF 的瓶颈

RLHF 让 PPO 优化策略模型最大化 Reward Model (RM) 的分数。
**如果 RM 烂，PPO 就在优化错的目标——模型变烂还以为自己变好**。

很多 RLHF 失败案例的根因都在 RM。

## 数据：偏好对

RM 训练数据是 `(prompt, chosen, rejected)` 三元组：

```json
{
  "prompt": "怎么减肥？",
  "chosen": "减肥的核心是热量赤字。建议：1) 控制饮食，每天比基础代谢少 300-500 卡 2) 增加运动量...",
  "rejected": "节食 + 运动呗。"
}
```

收集方法：
- 让同一个 prompt 在 SFT 模型上**用高温度生成多个回答**
- 人工标注员从中选 chosen / rejected
- 每个 prompt 收集 2-8 个回答两两对比

数据量：**至少 10k pairs，最好 50k+**。

## 模型架构

RM 通常**比 policy 模型小**（节省 RLHF 训练成本）：

- Policy: 70B → RM: 7B 或 13B
- Policy: 8B → RM: 1B-3B

架构：
- 用同家族的 base 模型
- 在最后一层加一个 **scalar regression head**（线性层映射到 1 维分数）

```python
class RewardModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base  # LM
        self.head = nn.Linear(base.config.hidden_size, 1)
    
    def forward(self, input_ids):
        last_hidden = self.base(input_ids).last_hidden_state[:, -1]  # 末尾 token
        return self.head(last_hidden).squeeze(-1)
```

## Loss 函数

经典 Bradley-Terry loss：

```python
def rm_loss(reward_chosen, reward_rejected):
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()
```

最大化 chosen 跟 rejected 的分差。

## 训练超参

```python
from transformers import TrainingArguments

args = TrainingArguments(
    learning_rate=1e-5,           # 比 SFT 低
    num_train_epochs=1,           # 1 epoch 通常足够，2-3 epoch 后过拟合
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    bf16=True,
    eval_strategy="steps",
    eval_steps=500,
)
```

关键观察：**RM 比 SFT 更容易过拟合**——常常 1 epoch 内 eval accuracy 拐点就出来。

## RM 评估

最重要指标：**pairwise accuracy**——给定 (chosen, rejected)，RM 给 chosen 的分数比 rejected 高的比例。

```python
def evaluate(rm, dataset):
    correct = 0
    for prompt, chosen, rejected in dataset:
        r_c = rm(prompt + chosen)
        r_r = rm(prompt + rejected)
        if r_c > r_r:
            correct += 1
    return correct / len(dataset)
```

经验值：
- < 60% 准确率：RM 没学到东西，重做
- 65-75%：能用，PPO 应该有效果
- > 80%：好 RM，PPO 收益最大化
- > 95%：过拟合，反而效果差（PPO 容易 reward hacking）

## 5 个常见坑

### 1. 数据标注不一致

不同标注员标准不一样 → RM 学到的是混乱信号。
解法：写**清晰的标注指南** + **多人交叉验证**。

### 2. RM 过拟合

太多 epoch / lr 太高 / 模型太大 → RM 把训练集背了，泛化烂。
监控 eval accuracy 拐点，及时早停。

### 3. Length Bias

长 response 自然得分高（人写偏好对时也偏好详细回答）。
PPO 用这种 RM 训出来的模型会越来越啰嗦。
解法：**训练数据控制长度差不超过 20%**；或者 loss 加 length 正则化。

### 4. Distribution Shift

RM 在 SFT 输出上训，但 PPO 把 policy 训跑了——
policy 的输出分布跟 RM 见过的不一样，RM 给的分数不再可靠。
解法：**周期性重新采样 + 重训 RM**。

### 5. 单 RM 投票更稳

训 3-5 个 RM（不同种子 / 数据子集），PPO 时用多 RM 平均分数。
能减少单 RM 的偏差。

## 实战代码

TRL 一行起步：

```python
from trl import RewardTrainer, RewardConfig

config = RewardConfig(
    output_dir="./rm-out",
    learning_rate=1e-5,
    num_train_epochs=1,
    per_device_train_batch_size=4,
)

trainer = RewardTrainer(
    model=base_model,
    args=config,
    train_dataset=pairs_dataset,  # 含 chosen / rejected
    tokenizer=tokenizer,
)
trainer.train()
```

## 一个朴素结论

> 想 RLHF 成功？把 80% 时间花在 RM 数据 + 训练上。
> 算法层 PPO 调参是表面文章，**RM 才是真正的天花板**。
>
> 不行的话试 DPO——完全跳过 RM，对很多场景够用。
