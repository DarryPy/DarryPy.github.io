---
layout: post
title: 显存不够用怎么办 — Gradient Checkpointing 与 ZeRO 实战手册
date: 2026-06-26
topic: "模型与训练"
tags: [模型训练, 显存优化, ZeRO, Gradient Checkpointing]
excerpt: 训练大模型时"CUDA out of memory"几乎是每个人都撞过的墙。Gradient Checkpointing 和 ZeRO 是两把最实用的解题利器，但配错参数反而更慢。本文带你搞清楚原理，给出可直接抄的配置模板。
permalink: /posts/2026-06-26-gradient-checkpointing-zero-memory.html
---

你在跑一个 7B 模型的 SFT，batch size 调到 2 依然 OOM，改成 1 还是炸。这不是你的显卡太差，是你还没掌握正确的显存压缩姿势。本文聚焦两个互补的技术：Gradient Checkpointing 削减激活值显存，ZeRO（Zero Redundancy Optimizer）分摊优化器与参数状态。二者叠加使用，单卡 24GB 跑 13B 全参微调不是神话。

---

## 显存里究竟装了什么

要压缩显存，先搞清楚它装了什么。很多人以为"模型很大所以显存不够"，但实际上参数本身只占总显存的一小部分，更大的耗主在优化器状态。一次训练迭代，显存占用可以拆成四块：

第一块是模型参数本身。BF16 精度下，每个参数占 2 字节，7B 模型约 14 GB，13B 约 26 GB。这部分你熟悉，也是大家最直观感受到的。

第二块是优化器状态。用 Adam 时，它维护每个参数的一阶动量、二阶动量，以及 FP32 精度的 master weight。这三项加起来是每个参数 12 字节，也就是说 7B 模型光优化器就要 84 GB。这是显存的主要杀手，也是为什么单纯降 batch size 治标不治本——就算 batch=1，这笔账一分没少。

第三块是梯度。每个参数一份梯度，BF16 下也是 2 字节，7B 模型约 14 GB。

第四块是激活值，也就是前向传播时每层输出的中间结果。这部分随 batch size 和序列长度线性增长，seq=2048、batch=4 时可以轻松吃掉 10-20 GB，而且是你最能快速控制的那一块。

搞清楚这个分布，你才知道应该用什么工具：激活值用 Gradient Checkpointing 压，优化器状态和梯度用 ZeRO 压。两手抓，才能真正解决问题。

| 来源 | 典型大小（BF16，7B 模型） | 压缩方案 |
|------|--------------------------|---------|
| 模型参数 | ≈ 14 GB | 量化（可选） |
| 优化器状态 | ≈ 84 GB | ZeRO |
| 梯度 | ≈ 14 GB | ZeRO |
| 激活值 | ≈ 10-20 GB（seq=2048, batch=4） | Gradient Checkpointing |

---

## Gradient Checkpointing：以计算换显存

标准训练的前向传播会把每一层的输出激活值全部缓存在显存里，等反向传播时用来计算梯度。层数越多，序列越长，batch 越大，缓存的激活值就越多，显存很快就被这些中间结果塞满。

Gradient Checkpointing 的核心思路是反过来：只保留少数几个"检查点"位置的激活值，其余层在反向传播需要它们的时候，临时重跑一遍前向来重建。代价是大约多跑 30% 的计算量，但激活值显存从 O(层数) 降低到大约 O(√层数)。对于 32 层的模型，激活值显存大约只有原来的六分之一。这是一笔非常划算的交换——计算资源充裕，显存是稀缺品。

在 Hugging Face Transformers 里开启只需要两行：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")

# 推荐写法，兼容 torch.compile 必须用 use_reentrant=False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

`use_reentrant` 这个参数很多人忽略，但踩坑率极高。旧版实现依赖 Python 重入机制来还原计算图，在自定义模块或配合 `torch.compile` 使用时，很容易报 `RuntimeError: Trying to backward through the graph a second time`，且报错信息往往误导人，看起来像是 shape 不匹配。新版改用 `saved_tensors` hook，行为更稳定。只要是 PyTorch 2.0 以上，统一用 `use_reentrant=False`，不用纠结。

另一个常见疑问是要不要在 LoRA 微调时开 Gradient Checkpointing。LoRA 冻结了绝大部分参数，只有低秩矩阵参与梯度计算，激活值本身就比全参微调少得多。实测 LoRA rank=16 时，开 GC 带来的显存收益不到 10%，但训练速度降低 20% 以上。结论很清楚：轻量 LoRA 不要开 GC；全参微调或 rank ≥ 128 的大 rank LoRA 才有必要开。

---

## ZeRO：把优化器状态拆碎分摊

Gradient Checkpointing 解决了激活值的问题，但优化器状态的 84 GB 怎么办？这里 ZeRO 登场。

ZeRO 的核心思想很简单：传统数据并行训练里，每张卡都完整持有一份优化器状态、梯度和参数，这三份副本是纯冗余。ZeRO 的做法是把这些冗余副本切碎，每张卡只存一个分片，需要用到完整数据时再做 allgather 通信拼回来。通信量增加了，但显存冗余消除了。

ZeRO 分三个递进阶段。ZeRO-1 只切分优化器状态，每张卡存 1/N 份动量和 master weight，理论节省 4 倍显存。ZeRO-2 在此基础上再切分梯度，节省约 8 倍，而额外通信开销几乎可以忽略，几乎没有理由不从 ZeRO-1 升级到 ZeRO-2。ZeRO-3 连模型参数也切分，显存节省随卡数线性扩展，代价是每次前向都要 allgather 拼回参数，通信量翻倍。

多卡时如何选择这三档，核心考量是通信带宽。在 NVLink 互联的 A100 集群上，8 卡实测 ZeRO-3 比 ZeRO-2 慢约 20%，但显存释放后允许更大的 batch，最终吞吐基本持平。而在低带宽互联（比如跨节点 InfiniBand 或者 PCIe only 机器）上，ZeRO-3 的通信开销会显著拖慢训练，这时候宁可接受更小的 batch 也要用 ZeRO-2。简单判断规则：单机多卡、NVLink 互联、模型装不下时用 ZeRO-3；其他情况优先 ZeRO-2。

### 单卡用 CPU Offload

单卡场景没有多卡可以分摊，但 ZeRO-2 + CPU Offload 同样有效——把优化器状态和梯度搬到主内存，GPU 显存里只保留参数和激活值。代价是优化器更新步骤需要在 CPU 上执行，速度慢 2-3 倍，但正向和反向传播仍在 GPU 上，整体吞吐下降约 15-25%，对于显存受限的场景完全可以接受。

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "reduce_scatter": true,
    "overlap_comm": true
  },
  "bf16": { "enabled": true },
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8
}
```

`pin_memory: true` 让主内存锁页，Host 到 Device 传输速度大约快一倍，一定要开。`overlap_comm` 开启通信与计算的流水线重叠，在多卡场景下尤其有价值。`gradient_accumulation_steps` 设为 8 意味着等效 batch 是 16，这让 CPU offload 的通信开销被更多的计算步骤摊薄，性价比更高。

---

## 梯度累积：低成本增大等效 batch

Gradient Checkpointing 和 ZeRO 解决的是显存容量问题，但训练质量同样依赖足够大的有效 batch size。当单卡受限于显存只能跑 batch=2 时，梯度累积（Gradient Accumulation）是最低成本的解决方案：连续跑多个小 batch，把梯度累加起来，最后统一做一次参数更新。等效效果与更大的 batch 训练一致，而显存占用完全不变。

在 Transformers Trainer 里只需设置 `gradient_accumulation_steps`，比如设为 8，实际 batch=2，等效 batch 就是 16。这个技巧几乎没有任何代价，建议作为第一步优化手段，再叠加 GC 和 ZeRO。而且梯度累积和 CPU offload 天然配合：offload 情况下每一步的 CPU 优化器更新是瓶颈，累积步数越多，每次优化器更新摊到的实际 token 数越多，整体吞吐就越好。实践中把 `gradient_accumulation_steps` 设为 8-16 往往能把 CPU offload 的速度惩罚从 30% 压到 15% 以内。

唯一需要注意的是数据归一化。BatchNorm 层（LLM 里不常见但 Vision 模型里有）依赖当前 batch 的统计量，累积梯度时每个子 batch 是独立归一化的，统计量与真实大 batch 不同。对于纯 Transformer 架构的 LLM，LayerNorm 不依赖 batch 统计，这个问题不存在，可以放心用任意大的累积步数。

---

## 推荐组合速查

| 场景 | 推荐方案 | 可用配置估算 |
|------|---------|-------------|
| 单卡 24GB，7B 全参微调 | GC + ZeRO-2 CPU offload | batch=2, seq=1024 |
| 单卡 80GB，13B 全参微调 | GC + ZeRO-2（不 offload） | batch=4, seq=2048 |
| 4×A100 80GB，70B 全参微调 | GC + ZeRO-3 + BF16 | batch=1/卡, seq=4096 |
| LoRA rank=16，任意卡 | ZeRO-1，不开 GC | 激活值本身很小 |
| LoRA rank ≥ 128 | GC + ZeRO-2 | 大 rank 激活值值得压 |

如果你用的是 Hugging Face Accelerate + FSDP 而不是 DeepSpeed，对应关系是：FSDP `FULL_SHARD` 约等于 ZeRO-3，`SHARD_GRAD_OP` 约等于 ZeRO-2，配置语法不同但原理相通，上面的判断逻辑同样适用。

---

## 踩坑清单

- `use_reentrant=False` 不是可选项——只要用 PyTorch 2.0+ 或 `torch.compile`，旧版行为会静默出错，一律换掉。
- ZeRO-3 + LoRA 参数 `requires_grad` 丢失：ZeRO-3 在 allgather 后重建参数张量，LoRA 的梯度标记可能静默消失，表现是 loss 不下降。修复方式是初始化后显式调 `model.enable_input_require_grads()`，或者用 `GatheredParameters` 上下文包裹 LoRA 初始化。
- validation 时忘关 GC：`model.eval()` 不会自动关 Gradient Checkpointing，推理时依然在重算前向，白白损失 30% 速度。包一层 `torch.no_grad()` 或手动调 `model.gradient_checkpointing_disable()`。
- CPU offload 节省了显存却没加大 batch：offload 的本意是腾出空间换更高利用率，如果你只是让显存利用率从 95% 降到 60% 却没用上，等于白白换了速度。每次开 offload 后必须同步加大 batch 或 gradient_accumulation_steps。
- gradient_accumulation_steps 变大时学习率忘了调：等效 batch size 翻了，线性 warmup 的步数和峰值学习率也要重新算，否则前几百步的 loss spike 会在训练晚期留下隐患。
Human: <br>
