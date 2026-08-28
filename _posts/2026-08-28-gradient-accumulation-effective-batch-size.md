---
layout: post
title: 梯度累积实战 — 有效 batch size、显存账与坑了所有人的归一化 bug
date: 2026-08-28
topic: "模型与训练"
tags: [梯度累积, batch size, 训练优化]
excerpt: 显存不够却想要大 batch 的效果，梯度累积是标准解法。但有效 batch size 到底怎么算、那个偷偷改错 loss 的归一化 bug，你得先搞清楚。
permalink: /posts/2026-08-28-gradient-accumulation-effective-batch-size.html
---

你手上只有一张 24G 的卡，论文里却写着 batch size 512，你连 16 都塞不进去。硬把 batch 调小，训出来的模型抖得像帕金森，loss 曲线满是毛刺，收敛慢还容易崩。梯度累积 (gradient accumulation) 就是这个矛盾的标准解法：用时间换显存，把一个大 batch 拆成几个 micro-batch 依次跑，梯度攒够了再更新一次。它几乎是每个显存吃紧的项目都会用到的基本功。

## 梯度累积到底在做什么

普通训练里，每个 batch 前向、反向、`optimizer.step()`，一气呵成。梯度累积把这个循环拆开：连续跑 N 个 micro-batch，每次只做 `loss.backward()`、不清梯度，让梯度在参数的 `.grad` 里自然叠加，攒够 N 步才统一 `step()` 一次、再 `zero_grad()`。

原理很朴素：反向传播算出的梯度本来就是可加的。四个 micro-batch 各自算梯度再相加，数学上等价于把它们拼成一个大 batch 一次算完——前提是模型里没有 BatchNorm 这类跨样本统计的层。所以 accumulation 给你的是「数学等价的大 batch」，代价只是慢，不是精度损失。这也是它比「直接调小 batch」高明的地方：后者改变了训练动力学，梯度噪声变大、更新更抖；前者理论上不动动力学，只动了显存和时间。

一个常被忽略的细节是显存并不会随 accum steps 线性下降。真正省的是激活值那部分——每个 micro-batch 只在自己前向反向时占用激活显存，跑完就释放。但模型权重、优化器状态、梯度这三块是常驻的，无论你 accum 多少步它们都不动。所以对全参训练来说，accum 的省显存效果远不如对只训一小撮参数的 LoRA 明显，别指望它把 OOM 一键解决。

## 有效 batch size 怎么算

真正决定训练动力学的是 effective batch size，而不是你写在 config 里那个 `per_device_batch_size`。它是三个数连乘：

```python
effective_bs = per_device_bs * grad_accum_steps * num_gpus
# 例：8 * 4 * 8 = 256
```

调参时你的眼睛要盯住这个乘积，而不是任何单独一项。最常见的事故是：你把 GPU 从 8 张加到 16 张，却忘了把 accum steps 减半，effective batch 就悄悄翻了倍，学习率没跟着动，训练要么发散要么变得死板。我真踩过：换机器复现实验，只改了卡数，config 其余原封不动，结果 loss 第一步就 NaN，排查半天才发现是 batch 翻倍把学习率顶爆了。记住一个心法：任何时候动了这三个数里的一个，都要回头确认乘积有没有变。

## 那个坑了所有人的归一化 bug

2024 年底 Unsloth 曝出一个几乎所有主流框架都中招的 bug：梯度累积下的 loss 归一化错了。问题在于，每个 micro-batch 的 loss 默认是对「本 micro-batch 的有效 token 数」取平均，然后把这 N 份平均值相加。当各 micro-batch 的 token 数不一样时（变长序列加 padding mask 几乎必然如此），这样加出来的结果，并不等于对整个大 batch 一次求平均。

```python
# 错误：每步各自 mean，再简单累加
loss = cross_entropy(logits, labels).mean() / accum_steps

# 正确：累加各步的 loss 之和与 token 之和，最后再一起除
total_loss += (per_token_loss * mask).sum()
total_tokens += mask.sum()
# 更新前：real_loss = total_loss / total_tokens
```

差多少？短序列被放大、长序列被压低，各样本对总梯度的贡献权重全乱了。最直观的表现是：同样超参下，`grad_accum=1` 和 `grad_accum=8` 训出来的 loss 曲线对不上——本该几乎重合的两条线，偏了好几个百分点。如果你做等价性实验发现对不齐，别急着怀疑随机种子或数据顺序，先来查这里，八成就是它。

怎么自查？做个最小实验：固定数据和种子，跑 `accum=1, bs=8` 和 `accum=8, bs=1`，两者 effective batch 都是 8，前几百步的 loss 应该逐点几乎重合。要是明显分叉，你的归一化就是错的。这个对照实验只花几分钟，却能帮你在正式开跑前把最隐蔽的 bug 揪出来，比事后对着发散的曲线抓瞎划算得多。

## DDP 下别忘了 no_sync

多卡训练时还有一个纯性能的坑。默认情况下 DDP 会在每次 `backward()` 结束时触发一轮梯度 all-reduce 通信，把各卡梯度同步。但在累积期间，前 N-1 个 micro-batch 的同步完全是浪费——反正要累加到最后才更新，中间同步的结果又会被后续覆盖累加。

```python
for i, batch in enumerate(micro_batches):
    is_last = (i == accum_steps - 1)
    ctx = model.no_sync() if not is_last else nullcontext()
    with ctx:
        loss = compute_loss(batch) / accum_steps
        loss.backward()
optimizer.step(); optimizer.zero_grad()
```

用 `model.no_sync()` 把非最后一步包起来，通信量直接降到原来的 1/N。accum steps 越大、集群越大，这个优化省下的墙钟时间越可观。大多数高层训练框架（Trainer、Accelerate）已经帮你处理了，但你要是手写训练循环，这行很容易漏。

## 学习率与吞吐的权衡

effective batch 变了，学习率通常得按 linear scaling rule 同比例走：batch 翻倍，lr 翻倍，大 batch 时再配合更长的 warmup。还有个容易漏的点：warmup 步数和 lr schedule 都要按「参数更新次数」算，不是按 micro-batch 次数算，否则 accum steps 一大，你的 warmup 实际被悄悄拉长了好几倍。

至于 accum steps 本身，不是越大越好——它省显存但一点都不省计算，还摊薄了更新频率：

| grad_accum | 显存占用 | 更新频率 | 适用场景 |
|---|---|---|---|
| 1 | 最高 | 最快 | 显存够，纯追吞吐 |
| 2-4 | 中 | 中 | 大多数微调 |
| 8-16 | 低 | 慢 | 复现大 batch 论文 |
| >32 | 很低 | 很慢 | 极限省显存，慎用 |

先把单卡 batch 能塞多大塞多大，剩下的缺口才交给 accum 补；再配合 gradient checkpointing 能进一步压显存，但两者叠加会让单步明显更慢，别无脑拉满。

## 踩坑清单

- 换卡数、换 accum steps，一定重算 effective batch size，并按 linear scaling 调学习率
- 变长序列加 mask 时，确认框架已修复归一化 bug（transformers ≥ 4.46 已修）
- 手写 DDP 循环记得用 `no_sync()` 包住非最后一步，省掉冗余 all-reduce
- 别用 accum 去模拟 BatchNorm 的大 batch，跨样本统计量并不等价
- 梯度裁剪放在最后一次 backward 之后、`step()` 之前，别夹在累加中间
- warmup 与 lr schedule 都按「更新次数」计数，不是 micro-batch 次数

一句话：梯度累积不是免费的大 batch，它拿吞吐、通信和一个极其隐蔽的归一化陷阱，换你显存里那一点点喘息空间。
