---
layout: post
title: "模型合并 — SLERP / TIES / DARE 混出更强的模型"
date: 2026-06-28
topic: "模型与训练"
tags: [AI, Model Merging, Fine-tuning]
excerpt: 把两个在不同任务上 fine-tune 的模型权重直接合并，往往能得到一个比两者都强的新模型。不需要 GPU 训练，只需要做矩阵运算。
permalink: /posts/2026-06-28-model-merging.html
---

## 为什么权重可以合并

一个基础模型（如 LLaMA-3-8B）经过 fine-tune 后，权重变化量（delta）通常是**稀疏的**——大部分参数几乎没动，只有一小部分参数发生了显著变化。

对于在不同任务上 fine-tune 的两个模型，它们的 delta 往往集中在不同的参数上。这意味着合并时的冲突比你想象的少得多。

直觉上：模型 A 学了代码能力，模型 B 学了数学能力，它们改动的参数基本不重叠，直接平均权重就能让新模型同时拥有两种能力。

这不是理论，是实践中被大量 Hugging Face 社区实验证明的。

## 三种核心方法

### SLERP（球面线性插值）

最简单的合并方式，适合**两个同源模型之间的平滑插值**。

普通线性插值（LERP）：
```
merged = (1 - t) * model_A + t * model_B
```

问题：在高维空间里，权重向量的模长会在插值中途缩小，导致合并后的模型"能力萎缩"。

SLERP 沿着球面插值，保持模长不变：

```python
import torch
import numpy as np

def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    t: 插值系数，0 = 纯 v0，1 = 纯 v1
    v0, v1: 要插值的权重张量（会 flatten 成向量处理）
    """
    orig_shape = v0.shape
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()

    # 计算两个向量之间的角度
    v0_norm = v0_flat / (torch.norm(v0_flat) + eps)
    v1_norm = v1_flat / (torch.norm(v1_flat) + eps)
    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    omega = torch.acos(dot)

    # 如果两个向量几乎平行，退化为线性插值
    if torch.abs(omega) < eps:
        return ((1 - t) * v0_flat + t * v1_flat).reshape(orig_shape)

    sin_omega = torch.sin(omega)
    result = (torch.sin((1 - t) * omega) / sin_omega) * v0_flat + \
             (torch.sin(t * omega) / sin_omega) * v1_flat

    return result.reshape(orig_shape).to(v0.dtype)

def merge_models_slerp(model_a_path: str, model_b_path: str, t: float = 0.5) -> dict:
    """
    合并两个模型的 state_dict
    t=0.5 表示各取一半
    """
    state_a = torch.load(model_a_path, map_location="cpu")
    state_b = torch.load(model_b_path, map_location="cpu")

    merged = {}
    for key in state_a:
        if key not in state_b:
            merged[key] = state_a[key]
            continue

        a, b = state_a[key], state_b[key]

        # 只对浮点参数做 SLERP，其他（如 embed token 表）直接取 A
        if a.dtype in (torch.float32, torch.float16, torch.bfloat16):
            merged[key] = slerp(t, a, b)
        else:
            merged[key] = a

    return merged
```

**适用场景**：两个非常接近的模型（如同一个 base，不同轮次训练），你想取中间某个点。

### TIES（Trim + Elect + Merge）

SLERP 的问题：当两个模型有**符号冲突**时（同一个参数，A 让它变大，B 让它变小），简单平均会互相抵消。

TIES 的解法：

**步骤 1：Trim（剪枝）** — 只保留每个模型里变化最大的 top-k% 参数

**步骤 2：Elect（选举）** — 对于有冲突的参数，用符号投票决定用哪个方向

**步骤 3：Merge（合并）** — 只合并方向一致的参数，方向冲突的丢弃

```python
def ties_merge(
    base_state: dict,
    finetuned_states: list[dict],
    density: float = 0.5,  # 保留 top 50% 的变化
) -> dict:
    """
    base_state: 基础模型权重
    finetuned_states: 多个 fine-tuned 模型权重
    density: trim 后保留的参数比例
    """
    # 计算每个模型相对 base 的 delta
    deltas = []
    for ft_state in finetuned_states:
        delta = {}
        for key in base_state:
            if key in ft_state:
                delta[key] = ft_state[key].float() - base_state[key].float()
        deltas.append(delta)

    merged_delta = {}
    for key in base_state:
        if not all(key in d for d in deltas):
            merged_delta[key] = torch.zeros_like(base_state[key])
            continue

        # Step 1: Trim — 每个模型只保留变化最大的参数
        trimmed = []
        for delta in deltas:
            d = delta[key].clone()
            # 计算 top-k% 的阈值
            threshold = torch.quantile(torch.abs(d).flatten(), 1 - density)
            d[torch.abs(d) < threshold] = 0.0
            trimmed.append(d)

        # Step 2: Elect — 统计符号投票
        signs = torch.stack([torch.sign(t) for t in trimmed], dim=0)
        # 对于每个参数：正数票 vs 负数票 vs 零票
        sign_sum = signs.sum(dim=0)
        elected_sign = torch.sign(sign_sum)  # 多数派的符号

        # Step 3: Merge — 只保留与 elected_sign 一致的参数
        valid_deltas = []
        for t in trimmed:
            # 符号一致的保留，冲突的置零
            mask = (torch.sign(t) == elected_sign) | (t == 0)
            valid_deltas.append(t * mask)

        if valid_deltas:
            merged_delta[key] = torch.stack(valid_deltas).mean(dim=0)
        else:
            merged_delta[key] = torch.zeros_like(base_state[key])

    # 加回 base
    result = {}
    for key in base_state:
        result[key] = (base_state[key].float() + merged_delta.get(key, 0)).to(base_state[key].dtype)

    return result
```

**适用场景**：合并 3 个以上来自不同任务的模型，任务差异大。

### DARE（Drop And REscale）

DARE 的思路更激进：**随机丢弃大部分 delta，然后重新缩放**。

```python
def dare_merge(
    base_state: dict,
    finetuned_state: dict,
    drop_rate: float = 0.9,  # 丢弃 90% 的 delta 参数
    merge_coeff: float = 1.0,
) -> dict:
    """
    DARE: 随机稀疏化 delta，然后用 1/(1-drop_rate) 重新缩放
    """
    result = {}
    for key in base_state:
        if key not in finetuned_state:
            result[key] = base_state[key]
            continue

        base_w = base_state[key].float()
        ft_w = finetuned_state[key].float()
        delta = ft_w - base_w

        # 随机 mask：drop_rate 的参数置零
        mask = (torch.rand_like(delta) > drop_rate).float()
        # 重新缩放，保持期望值不变
        sparse_delta = delta * mask / (1 - drop_rate)

        result[key] = (base_w + merge_coeff * sparse_delta).to(base_state[key].dtype)

    return result
```

DARE 通常与 TIES 配合使用（先 DARE 稀疏化，再 TIES 合并），称为 **DARE-TIES**。

## 用 mergekit 直接操作

```bash
pip install mergekit
```

```yaml
# merge_config.yaml
merge_method: ties           # slerp / ties / dare_ties / linear
base_model: meta-llama/Llama-3-8B

models:
  - model: my-code-llama
    parameters:
      weight: 0.5
      density: 0.7          # TIES trim 保留 70%

  - model: my-math-llama
    parameters:
      weight: 0.5
      density: 0.7

parameters:
  normalize: true
  int8_mask: true            # 节省内存

tokenizer_source: base       # 用 base model 的 tokenizer
dtype: bfloat16
```

```bash
mergekit-yaml merge_config.yaml ./merged-model \
  --cuda \                   # 用 GPU 加速
  --low-cpu-memory           # 内存受限时分批处理
```

## 何时用合并 vs 重新 Fine-tune

| 场景 | 推荐方案 |
|---|---|
| 两个模型互补，想合并能力 | 模型合并（TIES/DARE-TIES）|
| 想在特定任务上最大化性能 | 专门 Fine-tune |
| 手头没有训练数据 | 模型合并（0 训练数据）|
| 手头没有 GPU | 模型合并（只需 CPU）|
| 两个模型来自不同 base | 不适合合并 |
| 需要特定输出格式 | Fine-tune |

## 合并后如何验证

```python
from lm_eval import evaluator

# 在标准 benchmark 上评估合并前后的变化
results = evaluator.simple_evaluate(
    model="hf",
    model_args={"pretrained": "./merged-model"},
    tasks=["hellaswag", "mmlu", "gsm8k", "humaneval"],
    num_fewshot=0,
    batch_size=8,
)
print(results["results"])
```

检查合并后每个能力域的分数，确认：
- 目标任务有提升
- 基础能力没有大幅退化（通常不超过 2%）

## 一个朴素结论

> 模型合并是一个被严重低估的技术：不需要训练数据，不需要 GPU 时间，
> 只需要几分钟矩阵运算，就能造出能力更全面的模型。
>
> 开始一个新项目时，先逛 Hugging Face 找同类 fine-tuned 模型，
> 试试合并，很可能比自己从头 fine-tune 得到更好的起点。
