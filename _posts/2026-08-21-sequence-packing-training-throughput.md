---
layout: post
title: 样本打包 Sequence Packing — 把训练吞吐榨到最后一滴
date: 2026-08-21
topic: "模型与训练"
tags: [Sequence Packing, 训练吞吐, FlashAttention]
excerpt: SFT 数据长度天差地别,padding 常吃掉一半算力。样本打包把短样本拼满定长序列,配合 varlen 注意力隔离,训练吞吐轻松翻倍。
permalink: /posts/2026-08-21-sequence-packing-training-throughput.html
---

## 你的显卡有一半在算废物

先看一个扎心的事实:SFT 数据里样本长度天差地别,有的只有 30 个 token,有的长到 2000。你按 batch 训练时,框架会把整个 batch 补齐到最长那条的长度。于是一条 30 token 的短样本后面跟着 1970 个 padding,GPU 老老实实地为这些空气做了一整遍前向和反向传播。

这不是小浪费。真实指令数据集的长度分布往往极度右偏:少数长样本拉高了 max_len,大量短样本被迫陪跑。padding 占比冲到 40%~60% 很常见,意味着你花了双倍的钱,一半算力喂给了 `<pad>`。样本打包(Sequence Packing)就是来堵这个窟窿的——把多条短样本拼进同一条序列,填满到 max_len,让每个位置都在干活。

## 拼接不难,难的是别让样本互相偷看

打包的第一直觉很简单:把 `[样本A][样本B][样本C]` 首尾相接,凑够长度就算一条,吞吐立刻上去。但这里藏着一个致命 bug——如果你什么都不改,注意力会让样本 B 的 token 回头去看样本 A 的内容。模型读到了根本不该同框的文本,等于训练时被悄悄投毒。

有两处必须跟着改。一是 attention mask,要做成块对角(block-diagonal),每条子样本只能看见自己内部的 token;二是 position_ids,每遇到一条新样本就从 0 重新计数。否则位置编码会把三条样本当成一整段超长文本,RoPE 的相对位置直接错乱,序列越长偏得越离谱。

## 用 FlashAttention 的 varlen 接口做隔离

手写块对角 mask 要实体化一个 seqlen×seqlen 的矩阵,会把打包省下的显存又吃回去,得不偿失。正确做法是走 FlashAttention 的变长(varlen)接口:只传一个 `cu_seqlens` 累积长度数组,内核在算子内部按样本边界做隔离,完全不实体化 mask。

```python
# 三条样本长度 [30, 500, 1494],拼成一条 2024
lengths = [30, 500, 1494]
cu_seqlens = torch.tensor([0, 30, 530, 2024], dtype=torch.int32)

# position_ids 每条从 0 重启,不能顺着往下加
position_ids = torch.cat([torch.arange(l) for l in lengths])

out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max(lengths),
    max_seqlen_k=max(lengths),
    causal=True,
)
```

记住两点:`cu_seqlens` 是长度的前缀和,数组长度等于"样本数 + 1";position_ids 必须逐样本重置。后者最容易漏,漏了不会报错,只是 loss 曲线悄悄变差,排查起来极其难受。

## 怎么拼:装箱算法与 loss 掩码

把样本塞进定长序列,本质是一道装箱问题(bin packing)。三种常见策略各有取舍:

| 策略 | 做法 | 优点 | 代价 |
|------|------|------|------|
| 贪心顺序拼 | 按原顺序累加到满 | 实现最简单 | 尾部碎片多 |
| 首次适应递减 | 长样本先放,短的填缝 | 填充率高、碎片少 | 需先排序 |
| 定长硬切打包 | 拼满就截断 | 几乎零 padding | 跨样本截断丢信息 |

生产里首次适应递减(FFD)是性价比最高的一档,填充率普遍能做到 95% 以上,又不会像硬切那样把问答对劈开。另外别忘了 loss masking:SFT 只在回答段算 loss,打包之后 prompt 段、padding 段、以及任何被截断的残样本,都要在 label 里置成 -100,否则模型会去拟合这些不该学的 token,白白稀释梯度。

## 吞吐到底能提多少

按 padding 占比 50% 粗估,打包后同样的 GPU 时间能多喂近一倍的真实 token,训练一个 epoch 的 wall-clock 时间常能砍掉 30%~45%。数据越碎、长尾越极端,收益越大。代价只是一次性的打包预处理和几十行工程代码,是训练侧少有的"纯赚"优化。

## 踩坑清单

- position_ids 忘了逐样本重置——最隐蔽,不报错只是悄悄掉点
- 用了普通 attention 而非 varlen,块对角 mask 把省下的显存又吃回去
- loss mask 没盖住 padding 和 prompt,梯度被垃圾 token 稀释
- 把一条样本从中间截断,问答对被劈开,数据质量当场塌方
- 评估或推理时忘了关打包,混进跨样本注意力,指标虚高虚低都可能

一句话:打包省的是真金白银,但每一处隔离都不能省。cu_seqlens、position_ids、loss mask 三件套,漏一个就是白干。
