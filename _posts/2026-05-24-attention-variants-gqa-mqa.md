---
layout: post
title: GQA / MQA / MHA — KV 头共享背后的速度与精度账
date: 2026-05-24
topic: "模型与训练"
tags: [Attention, GQA, MQA, KV Cache, 推理优化]
excerpt: KV cache 是大模型推理的内存大头，砍 K/V 头数就能省一大笔。GQA / MQA 怎么省、省多少、什么时候不能省，一次说清。
permalink: /posts/2026-05-24-attention-variants-gqa-mqa.html
---

你跑 LLaMA-2-70B 做长 context 推理时大概率撞过墙：batch 开不大、序列开不长、GPU 显存动辄 OOM。不是模型权重重，是 **KV cache 撑爆了**。每一层、每个 head、每个 token 都要存 K 和 V — 70B 模型 8K context、batch=4 的 KV cache 能轻松吃掉 40+ GB 显存。

GQA 和 MQA 就是为这件事生的。

## MHA 的推理瓶颈

标准 Multi-Head Attention 给每个 query head 独立配一组 K/V 矩阵。8 个 Q head 就有 8 个 K head 和 8 个 V head。KV cache 大小公式：

```
kv_cache = 2 × num_layers × num_kv_heads × seq_len × head_dim × dtype_bytes
```

70B 模型典型 64 层、64 个 head、head_dim 128。fp16 推理时，单 batch 单 token 的 KV cache 是 `2 × 64 × 64 × 128 × 2 ≈ 2 MB`。8K context、batch=4 → 64 GB。**KV cache 比模型权重还大**。推理时每生成一个 token 都要读全部 KV cache，显存带宽吃满之前就先被容量爆掉。

## MQA / GQA 怎么省

MQA（Multi-Query Attention）走极致路线：**所有 Q head 共用 1 个 K head 和 1 个 V head**。KV cache 直接缩 N 倍。GQA（Grouped-Query Attention）是折中：把 Q heads 分组，每组共享一组 K/V。8 个 Q + 4 个 KV = `GQA-4`，KV cache 缩 2 倍。

![三种注意力变体：Q 头数不变，K/V 头数递减](/assets/images/2026-05-24-attention-variants.svg)

只动 KV head 数，Q head 数不变 — 这是关键设计。Q 决定模型表达能力，KV 决定显存成本。

## 实测 trade-off

GQA 论文（Ainslie 等 2023）里给过对比数据，标准 transformer 8B 量级：

| 变体 | KV cache | 推理速度 | 困惑度 |
|---|---|---|---|
| MHA (32 KV) | 100% | 1.0× | 8.05 |
| GQA-8 | 25% | 1.6× | 8.08 |
| GQA-4 | 12.5% | 2.0× | 8.12 |
| MQA (1 KV) | 3.1% | 2.4× | 8.31 |

GQA-8 几乎零质量损失，速度翻 1.6 倍 — 这是当前的甜点。MQA 速度最快但 PPL 涨 3%，对长生成场景明显能感觉到退化。

## 主流模型怎么选

LLaMA 3 / Mistral 7B / Qwen2 都用 **GQA-8**。PaLM 2 用 MQA，Google 认为质量损失可接受。DeepSeek V3 走了 **MLA**（Multi-head Latent Attention），把 K/V 进一步压缩到 latent 空间，理论上比 GQA 再省 5-10 倍 cache，但训练复杂度更高。

如果你自己 fine-tune，**别动 attention 结构** — GQA 是预训练阶段决定的，SFT/LoRA 阶段改不了。模型选型时直接看 `config.json` 里的 `num_key_value_heads`。

## 踩坑清单

- **坑 1**：以为 MQA 能"零成本"换速度。MQA 适合 throughput 优先（搜索、代码补全），不适合长生成
- **坑 2**：自己写 attention kernel 没考虑 KV 复制语义。GQA 推理时 K/V 要 broadcast 到对应 Q group，FlashAttention 2 之后才原生支持
- **坑 3**：把 GQA 模型用 MHA 的代码加载，Q 和 KV head 数对不上直接 shape 报错
- **坑 4**：vLLM / TGI 这类推理框架对 GQA 有专门优化，自己拿 PyTorch 裸跑会丢一半性能
- **坑 5**：MLA 不是 MQA 的进化版，是另一条路 — 看到 DeepSeek 用 MLA 就盲目跟，没那本事就老老实实 GQA

KV cache 这一刀决定了你的 LLM 服务能不能上规模。GQA 是 2026 年的事实标准。
