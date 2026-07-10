---
layout: post
title: 分布式训练四板斧 — DP / TP / PP / SP 怎么选
date: 2026-07-10
topic: "模型与训练"
tags: [分布式训练, 张量并行, 流水线并行, LLM]
excerpt: 训练 70B 模型光装参数就要 560 GB 显存，数据并行 / 张量并行 / 流水线并行 / 序列并行各司其职，本文给你一张可落地的决策地图，帮你在不踩坑的前提下把 GPU 利用率真正打上去。
permalink: /posts/2026-07-10-distributed-training-parallelism.html
---

你想训一个 70B 参数的模型，光是参数本身就要 140 GB 显存——BF16 下每个参数占 2 字节，70 亿乘以 2 等于 140 GB。再加上 AdamW 的 optimizer state：动量和方差各存一份 FP32，共 560 GB。算上 backward 时的梯度、中间激活值，现实上要超过 700 GB 才能稳定跑一个 step。H100 SXM 的显存是 80 GB，单纯装参数就要 9 张卡起步。这还是理想状态下的估算，实际跑起来因为激活检查点策略、序列长度和 batch size，真实显存需求往往比理论值高 20–40%。

"装下"只是第一关，怎么切、怎么通信、怎么不让 GPU 空转才是真正的战场。分布式训练有四种主流并行方式：数据并行（DP）、张量并行（TP）、流水线并行（PP）、序列并行（SP）。它们不互斥，现代大模型训练几乎都是三者乃至四者叠加的 3D / 4D 并行，但搭配错了只会让你花大价钱租来的 GPU 利用率跌到 30%，还找不到原因。

## 数据并行：能用就别换

数据并行的思路最直接：每张 GPU 存一份完整的模型，各自拿到不同的 mini-batch 跑 forward 和 backward，最后对梯度做 all-reduce 同步，参数更新完再继续下一步。

PyTorch DDP（DistributedDataParallel）是最常见的实现。它在 backward 阶段就把梯度按 bucket 分批发出去，通信与计算高度重叠，只要网络带宽够，你几乎感觉不到额外开销。只要模型能塞进单卡，DDP 几乎是零代价的理想选项——先跑起来，再讨论别的。

当模型放不进单卡，进入 ZeRO（Zero Redundancy Optimizer）。它的核心洞察是：标准 DP 下，每张 GPU 都完整存着三份数据——模型参数、梯度、optimizer state——其中有大量冗余。ZeRO 把这三份冗余拆开分配给不同 GPU，需要时再广播或 all-gather 拿完整版本：

- **ZeRO-1**：只切分 optimizer state。Adam 的动量矩阵和方差矩阵每人只存自己那一份，向后更新时广播。相比 baseline，显存可降到约 1/3，通信量几乎不变，最容易上手，几乎没有额外调参工作。
- **ZeRO-2**：再切分梯度。backward 后梯度直接 reduce-scatter 给负责该分片的 GPU，不再每张卡存完整梯度。显存继续下降，forward 阶段每张卡仍然持有完整参数，性能损失很小。
- **ZeRO-3**：连参数也切。每次 forward 前先 all-gather 拿完整参数，用完立刻丢弃；backward 时同样先 gather、再 reduce-scatter 梯度然后丢参数。理论上把显存压到 1/N（N 为 GPU 数），代价是 all-gather 非常频繁，节点间带宽成为直接瓶颈。

Microsoft DeepSpeed 是 ZeRO 最成熟的实现，Hugging Face Trainer 也内置了 ZeRO 支持，`--deepspeed ds_config.json` 一行搞定。选哪个 stage 的原则很简单：能用 ZeRO-1 就别上 ZeRO-3，通信量越小，带宽利用率越高，整体吞吐越好。

## 张量并行：把单层切开

张量并行（Tensor Parallelism，TP）把每一层的权重矩阵在"宽度"方向切开，不同 GPU 各持一块，矩阵乘法结束后用 all-reduce 汇聚结果。比如把 8192×32768 的 FFN 权重按列切成 8 份，每张 GPU 只存 8192×4096，各自完成自己那部分的计算，最后一次 all-reduce 加和。

经典实现来自 Megatron-LM，关键洞察是让切分方式恰好使 forward 中只需一次 all-reduce 就能拿到正确输出，backward 同样只需一次，不需要额外的通信轮次。对 Transformer 层来说，Attention 的 Q/K/V 权重按"头"切，output projection 按行切，合计两次 all-reduce 完成一个 attention block；FFN 用 Megatron 标准切法同样两次。每层多出两次 all-reduce，但因为它们可以和非通信计算 overlap，整体带宽压力比你想的小。

TP 的硬性要求是**高带宽低延迟的互联**。节点内 NVLink（H100 NVLink 4.0 双向 900 GB/s）和节点间 RoCE 网卡（通常 400 Gbps）相差 20 倍以上，延迟也高出一个数量级。每层都有 all-reduce 的 TP 放到跨节点场景，等待时间会把计算优势全部抵消，整体吞吐通常下降 30–50%。所以业界几乎有一个不成文的铁律：**TP 只做节点内，TP=8 是单机 8 卡时的天花板，跨节点做 TP 是最常见的性能踩坑**。

推理侧用 vLLM 或 SGLang，一行 `tensor_parallel_size=4` 即可；训练侧建议直接用 Megatron-LM 或 NeMo，自己手写 TP 很容易在 backward 时出现梯度不同步而静默错误，等你发现时模型已经收敛到一个奇怪的方向，很难回溯。

## 流水线并行：按层切模型

流水线并行（Pipeline Parallelism，PP）把 transformer 层按"深度"分配到不同 GPU：GPU 0 跑第 0–11 层，GPU 1 跑第 12–23 层，以此类推。层边界只传 hidden state 激活（通常 batch × seq × hidden_dim），通信量比 TP 的 all-reduce 小得多，而且可以用带宽较低的 InfiniBand 或 RoCE，天然适合跨节点切分。

最朴素的 PP 存在严重的 GPU 空闲（pipeline bubble）：GPU 0 跑完把激活传给 GPU 1，然后就得等 backward 阶段梯度传回来，中间什么都干不了。空闲占比高达 `(p-1)/p`，4 个 stage 就有 75% 的时间在摸鱼，完全抵消了并行的收益。

GPipe 的解法是把 batch 切成 m 个 micro-batch：GPU 0 处理完 micro-batch 1 就立刻处理 micro-batch 2，不等 GPU 1 的返回，把 bubble 压到 `(p-1)/(p+m-1)`。Megatron 的 1F1B 调度在此基础上让 forward 和 backward 交替执行，不需要存全部 micro-batch 的激活，显存开销大幅降低：

| 调度策略 | Bubble 占比 | 激活显存 | 工程复杂度 |
|----------|------------|---------|-----------|
| 朴素 PP | (p-1)/p | 低 | 低 |
| GPipe | (p-1)/(p+m-1) | 高 | 中 |
| 1F1B | (p-1)/(p+m-1) | 低 | 高 |
| Interleaved 1F1B | 进一步降低 | 低 | 很高 |

*p = stage 数，m = micro-batch 数*

在实际工程里，PP 和 TP 几乎总是组合出现：**节点内做 TP，跨节点做 PP**。比如 8 机 8 卡共 64 张 GPU，每节点 TP=8，节点间 PP=8，两者乘积等于 64，这就是 Megatron 大规模训练的标准骨架。

## 序列并行：处理超长上下文

前三种并行方式都没解决"序列太长"的问题。128K 上下文下，即使模型参数被切得再碎，每个 token 对应的中间激活图本身就很大——batch × 128K × hidden_dim，这部分是张量并行切不到的维度，单卡还是放不下。

序列并行（Sequence Parallelism，SP）把序列维度也切开：不同 GPU 各自负责一段 tokens，在 Attention 层用 Ring Attention 机制完成跨 GPU 的因果注意力——每张 GPU 保存自己那段的 Q，轮流接收其他 GPU 的 K 和 V 做局部 attention，再把结果拼合回完整的 output，全程不破坏 causal mask。

这个机制让你可以把超长序列平摊到多个 GPU 上，每张卡只存 seq_len/SP 段的激活，显存随 SP 度线性下降。DeepSeek-V2、DeepSeek-V3 以及 LLaMA 3 的 128K 上下文训练阶段都依赖 SP 才得以实现。实现难度较高，通常和 TP 共用同一进程组（Megatron 里叫 SP+TP 联合模式），两者绑在同一组 GPU 上协同通信。建议直接用框架内置支持，Ring Attention 的通信顺序设计错了会让 causal mask 失效，产生的 loss 曲线看起来正常，模型却学歪了。

## 实战决策：怎么落地

四种并行度满足这个约束：`DP × TP × PP × SP = N_GPUs`。给你一个落地顺序：

```text
1. 模型能放进单卡？
   → 是：直接 DDP，先跑起来再优化
   → 否：继续

2. 节点内 GPU ≥ 4 且 NVLink 互联？
   → 是：TP = 4 或 8（严格限节点内）
   → 否：跳过 TP，直接看 ZeRO

3. 模型跨节点还装不下，或需要更多并行吞吐？
   → 单节点内放不下：ZeRO-3
   → 跨节点扩展：PP（按节点数切 stage）

4. 序列长度超过 32K？
   → 加 SP，和 TP 共用进程组

5. 剩余维度归 DP：
   DP = N_GPUs / (TP × PP × SP)
```

踩坑清单：

- **TP 绝对不跨节点**：跨节点 TP 因延迟高，整体吞吐通常降 30–50%，新手最常犯的错
- **PP micro-batch 数要够大**：m 小于 4 时 bubble 超过 20%，换 interleaved 调度或加大 global batch size
- **ZeRO-3 + PP 混用要观察带宽争用**：ZeRO 的 all-gather 和 PP 的激活传输会在同一网卡上抢带宽，先用 profiler 看通信利用率再决定是否叠加
- **Checkpoint 必须按 PP stage 分片保存**：恢复训练时 PP degree 不能随意修改，否则层权重无法正确 remap
- **每加一种并行就先验一次 loss**：TP bug 会让梯度静默不同步，大规模下几乎无法定位，单步对齐单机 loss 是最便宜的排查手段
- **MFU 才是真实指标，不是 GPU 显存利用率**：MFU（Model FLOPS Utilization）低于 40% 说明并行配置还有优化空间，光看显存满了会产生"跑得很充分"的错觉

分布式训练没有银弹。最常见的错误是一上来就把四种并行全堆上，然后花三天排查 GPU 利用率为什么只有 28%。先跑通单卡，再一层层叠加，每加一种并行就跑一次 profiler——这才是让钱花在刀刃上的节奏。记住：并行配置是超参数，它不会随着你训练进度自动调优，错误的并行策略会以一种非常隐蔽的方式消耗你的计算预算，直到账单结算那天才显现出来。
