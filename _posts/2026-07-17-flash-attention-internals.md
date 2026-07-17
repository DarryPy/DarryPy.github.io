---
layout: post
title: Flash Attention 工程解析 ⚡ — 重写注意力机制如何省掉 10x 显存
date: 2026-07-17
topic: "模型与训练"
tags: [Flash Attention, LLM, 训练优化, 显存]
excerpt: 标准 Attention 的显存是序列长度的平方级，训练长文本模型时往往第一个 OOM 的就是它。Flash Attention 没有改数学，只是换了计算顺序，却让显存从 O(N²) 降到 O(N)，同时还更快。
permalink: /posts/2026-07-17-flash-attention-internals.html
---

你第一次跑长上下文微调，几乎一定是被 Attention 的 OOM 绊倒的。序列长 4096，批次稍大一点，显存直接打满——但 GPU 的计算单元还空着大半。这不是算法问题，是内存搬运问题。Flash Attention 正是盯着这个痛点重写的。

## 标准 Attention 慢在哪

先看清楚敌人。标准 Scaled Dot-Product Attention 的公式是：

```
Attention(Q, K, V) = softmax(QKᵀ / √d) · V
```

看起来简洁，但实现时要把 `QKᵀ` 这张 `[N, N]` 的矩阵完整写回 HBM（显卡高带宽内存）再做 softmax，然后再读一次乘 V。序列长度 N=4096，这张矩阵就是 16M 个 float，每个训练步要被来回搬好几次。

问题的本质是：GPU 的计算速度（FLOP/s）远超显存带宽（GB/s）。计算一次乘法只需几纳秒，但把结果写回 HBM 再读回来要几微秒。标准 Attention 的 **算术强度**（FLOP / 访存字节比）太低，大量时间耗在等数据。

## Flash Attention 的核心思路：Tiling + Online Softmax

Flash Attention（Dao et al., 2022）没有改数学结果，只是换了执行顺序。

关键洞察：**不需要先把整个 `QKᵀ` 写全，再统一做 softmax**。softmax 可以"在线"边算边归一化。

具体做法：

1. 把 Q / K / V 切成小块（tile），一块一块装进 SRAM（片上高速缓存，比 HBM 快约 10x）
2. 对每个 Q-tile，遍历所有 K/V-tile，在 SRAM 里完成局部点积和局部 softmax
3. 利用数值稳定的 Online Softmax 公式合并每块结果，最后写回 HBM 的只有最终的 `[N, d]` 输出

```python
# 伪代码示意（非真实实现）
output = zeros(N, d)
for q_tile in split(Q, block_size):
    m_i, l_i = -inf, 0          # 运行最大值、归一化分母
    acc = zeros(block_size, d)
    for k_tile, v_tile in zip(split(K), split(V)):
        s = q_tile @ k_tile.T / sqrt(d)          # 局部得分
        m_new = max(m_i, s.max(dim=-1))          # 更新全局最大值
        exp_s = exp(s - m_new)
        l_i = exp(m_i - m_new) * l_i + exp_s.sum(-1)  # 更新分母
        acc = exp(m_i - m_new) * acc + exp_s @ v_tile
        m_i = m_new
    output[q_tile] = acc / l_i                   # 一次写回
```

整个过程 HBM 访问次数从 `O(N²)` 降到 `O(N)`，中间的大矩阵从未出现在显卡内存里。

## 实测对比

| 序列长度 | 标准 Attention 显存 | Flash Attention 显存 | 速度提升 |
|----------|---------------------|----------------------|----------|
| 1K       | ~1 GB               | ~0.1 GB              | 1.2x     |
| 4K       | ~16 GB              | ~0.4 GB              | 2–3x     |
| 8K       | OOM（A100 80G）     | ~1.5 GB              | 3–5x     |
| 32K      | OOM                 | ~6 GB                | 5–8x     |

数字来自 FlashAttention-2 论文，实际值因精度、头数、批次不同有出入，但量级可参考。

## Flash Attention 2 和 3 改了什么

**FA2**（2023）的主要改进：
- 减少非矩阵乘的运算比例（softmax 归一化操作移到尾部攒批做）
- Q-tile 外循环、K/V 内循环，更友好于 GPU warp 调度
- 吞吐量比 FA1 再提 2x

**FA3**（2024）针对 Hopper 架构（H100）的特殊硬件特性：
- 利用 TMA（Tensor Memory Accelerator）异步搬数据
- 把 WGMMA（Warpgroup Matrix Multiply Accumulate）指令流水线化
- FP8 支持，理论 FLOP 翻倍

使用时你通常不需要关心版本细节，`torch>=2.0` 里 `F.scaled_dot_product_attention` 会自动选 Flash Attention 后端：

```python
# PyTorch 2.x 自动使用 Flash Attention
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

或者直接装 `flash-attn` 包，用 `flash_attn_func`，对序列长度的控制更精细。

## 训练时三个容易踩的坑

**1. 数据类型必须是 FP16 或 BF16**
Flash Attention 的 CUDA kernel 不支持 FP32。如果你的训练脚本用 FP32，调用会静默回退到标准 Attention，显存炸了却不知道为什么。用 `BF16` 是最保险的选择，稳定性比 FP16 好。

**2. Head Dimension 有限制**
FA2 支持 head_dim ≤ 256，FA3 放宽到更大值，但不是任意的。用非标准头维度（比如 head_dim=96）要提前查版本文档，否则会报 `RuntimeError: head dim not supported`。

**3. 变长序列 padding 要用 `varlen` 接口**
批次里序列长度不一致时，硬 padding 到最长序列会让 Flash Attention 做无效计算。`flash_attn_varlen_func` 接收 cu_seqlens（累积序列长度），真正跳过 padding 部分，长文本微调场景下这个差距能到 30%+。

---

**踩坑清单**

- [ ] 开 Flash Attention 前确认模型权重是 BF16/FP16，不是 FP32
- [ ] 检查 head_dim 是否在当前 FA 版本支持范围内
- [ ] 变长批次用 `varlen` 接口，别无脑 padding
- [ ] H100 上优先装 FA3，A100 用 FA2，消费卡（RTX 系列）确认 CUDA 版本 ≥ 11.8
- [ ] 推理侧同样适用，不只是训练专属优化

Flash Attention 改变的不是注意力的数学，而是计算机组织数据的方式。理解它，你才能真正看懂为什么现在的 LLM 能跑 128K 上下文而不把显卡炸掉。
