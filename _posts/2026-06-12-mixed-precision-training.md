---
layout: post
title: 混合精度训练 — BF16 / FP16 / FP8 背后的省显存账
date: 2026-06-12
topic: "模型与训练"
tags: [混合精度, BF16, FP16, 训练优化, LLM]
excerpt: 从 FP32 到 FP8，混合精度训练是 LLM 省显存最直接的手段。本文拆解三种精度的数值范围差异、AMP 训练流程、梯度缩放原理，以及 FP8 在 Hopper GPU 上的实战注意事项，帮你在不牺牲收敛的前提下把显存占用砍掉一半。
permalink: /posts/2026-06-12-mixed-precision-training.html
---

你想微调一个 7B 参数的模型，用全精度 FP32 跑——光是参数本身就要占 28 GB 显存，再加上梯度同等量级、Adam 的 optimizer state 又是参数的两倍、activation 还没算进来。一张 A100 80G 直接装不下一次完整的前向加反向。买更多卡？大多数团队没这个预算。

混合精度训练（Mixed Precision Training，AMP）是破局最直接的手段：前向和反向计算用低精度，关键状态保留高精度，显存减半、吞吐翻倍，收敛几乎不受影响。这套方案已经成为 LLM 训练的默认配置，但真正理解它的细节和背后的数值原理，才能在出问题的时候不抓瞎。

## 三种精度的本质差异 📊

浮点数格式由三部分组成：符号位、指数位、尾数位。指数位决定能表示多大/多小的数（数值范围），尾数位决定同一量级下有多少精细刻度（相对精度）。

| 格式 | 总位数 | 指数位 | 尾数位 | 数值范围 | 备注 |
|------|--------|--------|--------|----------|------|
| FP32 | 32 | 8 | 23 | ±3.4×10³⁸ | 训练标准格式 |
| FP16 | 16 | 5 | 10 | ±65504 | 容易溢出 |
| BF16 | 16 | 8 | 7 | ±3.4×10³⁸ | Google Brain Float |
| FP8 E4M3 | 8 | 4 | 3 | ±448 | 前向激活用 |
| FP8 E5M2 | 8 | 5 | 2 | ±57344 | 反向梯度用 |

表里最关键的是「指数位」这一列。BF16 的指数位和 FP32 一样宽，都是 8 位，两者能表示的数值范围完全相同，从极小的梯度到极大的激活值都不会越界。FP16 的指数只有 5 位，最大只能表示 65504——LLM 训练里某些层的激活值或梯度一旦超过这个上限，直接变成 inf，然后 loss 变 NaN，训练崩掉。

BF16 牺牲的是尾数位——从 10 位降到 7 位，相对精度大约从 0.01% 降到 0.8%。但实践证明，LLM 训练对「范围」的需求远超对「精度」的需求：你需要在同一次训练里同时表示量级相差十几个数量级的梯度，却不需要每个数都精确到小数点后好几位。这正是 BF16 能作为 A100 时代主流训练格式的根本原因。

从算力角度看，A100 上 BF16 矩阵乘法的吞吐量是 FP32 的两倍。H100 上 FP8 又是 BF16 的两倍。每向下走一级精度，理论显存减半、算力翻倍。代价是精度损失越来越难控制。

## AMP 训练的完整流程

AMP 的核心思路是「主副权重分离」：参数在内存中有两份，一份是 BF16/FP16 的工作副本用于计算，另一份是 FP32 的 master weights 用于更新。整个流程如下：

```
1. 前向传播（BF16）→ 得到 loss（BF16）
2. 反向传播（BF16）→ 得到 BF16 梯度
3. 梯度转换为 FP32
4. optimizer 用 FP32 梯度更新 FP32 master weights
5. 将 master weights 的值 cast 回 BF16，覆盖工作副本
6. 下一步迭代
```

值得注意的是，切换 AMP 之后你的显存占用反而可能短暂上升——因为模型参数现在有两份：BF16 工作副本 + FP32 master weights。7B 模型的参数在 FP32 下是 28 GB，BF16 副本是 14 GB，两者加起来 42 GB。这是"短暂"的，因为 activation 和中间计算的显存减少是更大的收益。训练过程中 activation 占比往往超过参数本身，AMP 整体来看还是显著降低了峰值显存。

这里有一个经常被忽视的细节：为什么 optimizer state 必须保留 FP32？

以 Adam 为例，它维护着每个参数的一阶矩（momentum）和二阶矩（variance）。参数更新量的公式大致是：`delta = lr * m / (sqrt(v) + eps)`。当学习率是 `1e-4`、梯度很小时，更新量可能只有 `1e-7` 量级。参数本身可能是 0.85 这样接近 1 的值。

在 BF16 下，0.85 + 1e-7 的加法会直接舍入——BF16 在 1.0 附近的精度大约是 `7.8e-3`，比 `1e-7` 大几千倍，小更新量完全被吃掉。长时间积累下去，参数陷入「伪稳定」，模型实际上已经停止有效学习，但 loss 曲线可能看起来还在缓慢下降，非常难发现。FP32 master weights 的存在，就是为了在这个关键环节保住数值精度。

PyTorch AMP 的用法：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # BF16 不需要；FP16 必须

for inputs, labels in dataloader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
    
    # FP16 路径：梯度缩放 + 裁剪 + 更新
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)          # 先还原缩放，裁剪才有意义
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    
    # BF16 路径（注释掉上面 5 行，改为）：
    # loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # optimizer.step()
```

## FP16 梯度缩放的底层逻辑

用 FP16 训练面临两个对立的危险：

- **上溢（overflow）**：数值太大超过 65504，变成 inf，通过反向传播污染整个梯度
- **下溢（underflow）**：数值太小低于 FP16 最小正数（约 6e-8），舍入到零，这部分参数从此不再更新

梯度缩放（Gradient Scaling）专门对付下溢问题，思路是：训练前把 loss 乘以一个大数 S（默认 65536），这样反向传播计算出的梯度也被放大了 S 倍，原本会下溢到零的小梯度现在有机会活下来。在用梯度更新参数之前，再把梯度除以 S 还原。

上溢由动态调整来处理：GradScaler 在每步检查是否出现 inf/nan，如果有就把 S 减半（缩放因子降低），跳过本步参数更新；连续若干步没有溢出就把 S 翻倍（扩大检测敏感度）。这套自适应机制让缩放因子在整个训练过程中自动调整到合适区间。

所以你有时会看到某一步 loss 没有更新，但训练没有崩溃——那是 GradScaler 检测到溢出主动跳步，属于正常现象。真正需要担心的是跳步频率很高，比如连续几十步都在跳，通常说明 loss landscape 太陡或者模型架构有问题，需要降学习率或者检查初始化。

BF16 不需要 GradScaler，因为它的指数范围和 FP32 一样大，几乎不可能发生上溢或下溢。在 Ampere（A100）及以后的 GPU 上，BF16 是优先选择。只有在使用 V100、T4 等不支持 BF16 的老卡时，才需要忍受 FP16 + GradScaler 的复杂度。

## FP8：H100 才开的新门

FP8 是 NVIDIA 在 Hopper 架构（H100）引入的格式，搭配 Transformer Engine 库使用。两种子格式有明确分工：

E4M3（4 位指数 + 3 位尾数）精度相对高，适合前向传播中的激活值——激活值分布通常比较集中，精度比范围更重要。E5M2（5 位指数 + 2 位尾数）范围更大，适合反向传播的梯度——梯度的动态范围变化大，范围比精度更重要。

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

fp8_recipe = DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=Format.HYBRID,   # 前向 E4M3，反向 E5M2
    amax_history_len=16,        # 历史最大值窗口，用于计算缩放因子
    amax_compute_algo="max",
)

model = te.Linear(in_features, out_features)

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    output = model(input_tensor)
```

FP8 最大的挑战是每个张量需要独立的缩放因子。E4M3 最大只能表示 448——激活值稍微大一点就溢出。Transformer Engine 通过「delayed scaling」解决这个问题：追踪最近 N 步里每个张量的历史最大值（amax），据此计算缩放因子，把数值压进 FP8 可表示的范围。这套机制引入了额外的内存开销和同步操作，实际端到端加速比通常比理论值低不少，大约在 1.3-1.6 倍之间，而非理论上的 2 倍。

另一个现实约束是框架支持：目前 Megatron-LM、nanotron、DeepSpeed 对 FP8 支持较好，HuggingFace Trainer 和 PyTorch FSDP 的 FP8 路径还在持续完善中。除非你在 H100 集群上跑几十亿参数的预训练，FP8 的工程投入很难在普通微调任务上得到回报，BF16 依然是性价比最高的选择。

## 实战决策：怎么选精度

三个问题可以快速定位你的选择：

**问题一：你用什么 GPU？**
V100 / T4 / K80 没有 BF16 硬件支持，只能用 FP16 + GradScaler。A100 / A10G / RTX 30 系以上，优先选 BF16。H100 / H200，大规模训练可以考虑 FP8。

**问题二：任务对精度有多敏感？**
代码生成、数学推理等任务对细微数值变化更敏感，有时 BF16 会比 FP16 稍差（因为尾数位更少）。大部分自然语言任务感知不到差别。如果发现 BF16 收敛比预期差，可以尝试仅对某些层保留 FP32。

**问题三：显存还是速度更紧张？**
两者在混合精度下都有提升，但如果主要目标是显存（装得下更大 batch），优先关注 optimizer state 的节省，可以搭配 8-bit Adam（bitsandbytes）进一步减小 optimizer state 的显存占用。

## 踩坑清单

- loss 某步不更新但没崩 → GradScaler 跳步，偶发正常；高频跳步才需要排查
- loss 变 NaN 无法恢复 → 检查 GradScaler 缩放因子是否降到 1 以下，或改用 BF16
- 多卡 DDP 下梯度裁剪不生效 → 裁剪必须在 `scaler.unscale_()` 之后，顺序错了裁的是放大后的梯度
- 换 BF16 后 loss 仍然 NaN → 检查 attention 的 softmax 输入，极长序列下 logit 可能超出范围，需要手动加 `/ sqrt(head_dim)` 缩放
- optimizer 吃显存太多 → 试 bitsandbytes 的 8-bit Adam，在 BF16 精度下存 optimizer state，Adam 的显存占用减少 75%
- A100 上想用 FP8 → A100 没有原生 FP8 矩阵运算指令，Transformer Engine 会自动 fallback 到 BF16，基本没有加速收益
- 忘记 `optimizer.zero_grad()` 位置 → 在 `scaler.step()` 之后、下一步 `autocast` 之前调用，不影响正确性，但位置不对有时会引发意外的显存峰值

精度不是越低越好，而是找到「不影响收敛的最低精度」。BF16 是目前这条线上最省心的驻留点——换了它你能省掉 GradScaler 的心智负担，还能享受一倍的吞吐提升，几乎没有理由在 A100 上还跑 FP32。
