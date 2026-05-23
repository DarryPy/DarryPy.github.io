---
layout: post
title: MoE 架构详解 — 为什么 GPT-5 和 DeepSeek 都用 Mixture of Experts
date: 2025-12-28
topic: "模型与训练"
tags: [AI, MoE, 架构]
excerpt: MoE 让模型"参数规模上千亿，每次推理只激活一小部分"。原理、工程实现、为什么 2025 起大模型都在切 MoE。
permalink: /posts/2025-12-28-moe-architecture.html
---

## MoE 解决什么问题

普通 Transformer：参数越多，推理越贵——线性关系。
**MoE (Mixture of Experts)**：**总参数大，但每次推理只激活一部分专家**。

```
普通模型：100B 参数 → 推理用 100B
MoE：     671B 总参数 → 每次激活 ~37B
```

效果接近 100B+ 模型，但推理成本约等于 37B 模型。**鱼和熊掌兼得**。

## 怎么工作

每个 Transformer 层的 FFN（feed-forward network）被替换成多个并行的"专家"：

```
input
  ↓
Router（路由器，小神经网络）
  ↓ 选 top-K 专家（如 top-2）
Expert 1 / Expert 2 / ... / Expert N
  ↓ 只激活选中的
combine outputs
  ↓
output
```

每个 token 由不同专家处理，每层只激活 2-8 个专家（共 8-128 个候选）。

## 为什么有效

直觉：专家化分工。
- 专家 1 擅长代码
- 专家 2 擅长中文
- 专家 3 擅长数学
- ...

模型自动学习"路由"——见到代码 token 就给 Expert 1，数学 token 给 Expert 3。
**比 dense 模型的"一个网络通吃"更高效**。

## 主流 MoE 模型

| 模型 | 总参数 | 激活参数 | 出品 |
|---|---|---|---|
| **Mixtral 8x7B** | 47B | 13B | Mistral（早期开源） |
| **Mixtral 8x22B** | 141B | 39B | Mistral |
| **DeepSeek-V3** | 671B | 37B | DeepSeek（2024 年末爆红）|
| **DeepSeek-R1** | 671B | 37B | DeepSeek（推理强）|
| **Qwen2.5-Max** | ~700B | ~70B | 阿里 |
| **GPT-4** | 推测 1.8T | ~280B | OpenAI（未公开）|
| **Claude 4** | 未公开 | — | Anthropic（推测也是 MoE） |

2024-2026 大模型趋势：**几乎全切 MoE**。

## 工程挑战

### 1. 路由不均衡

理想：每个 token 均匀分到各专家。
现实：模型容易"全去 Expert 1"，其他专家闲置。

解法：
- **Auxiliary loss**：训练时加均衡惩罚
- **Token dropping**：每个专家有容量上限，超出的 token 跳过
- **Expert parallelism**：不同 GPU 跑不同专家，路由跨 GPU

### 2. All-to-all 通信开销

Routing 后 token 要跨 GPU 送给对应专家，**通信成本大**。
DeepSpeed-MoE / Tutel 等框架专门优化这块。

### 3. 训练不稳定

MoE 训练更难调，router 容易"塌缩"到几个专家。
解法：warm-up 阶段慢慢启用 routing、加 noise 防止过度专家化。

### 4. 推理部署

MoE 显存占用按总参数算（要全装下），但**FLOPs 按激活参数算**。
所以 MoE 模型：**显存大 + 推理快**。

671B 模型至少需要 8x H100（80GB），但每次推理速度接近 37B dense 模型。

## 推理优化

### Expert Parallelism

不同专家放不同 GPU：

```
GPU 0: Expert 1, 2, ..., 16
GPU 1: Expert 17, ..., 32
...
GPU 7: Expert 113, ..., 128

每个 token 经路由后被 dispatch 到对应 GPU
```

### Selective Loading

不是所有专家都活跃。可以根据流量统计**只加载常用专家**到显存，冷专家放磁盘。
代价是冷启动慢。

## MoE vs Dense 选型

| | MoE | Dense |
|---|---|---|
| 大规模能力 | 上限高 | 受单 GPU 显存限 |
| 推理成本 | 低（按激活算）| 高 |
| 推理延迟 | 一般（all-to-all 开销）| 低 |
| 训练复杂度 | 高 | 低 |
| 部署门槛 | 高（多 GPU） | 低 |
| 适合 | 大模型 / API 服务 | 小模型 / 边缘 |

**< 30B 用 Dense，> 70B 优先 MoE**。

## 一个朴素结论

> MoE 是大模型时代的"扩展定律"——让参数规模再上一个量级，而推理成本不爆炸。
> 2024-2026 几乎所有前沿大模型都是 MoE。
>
> 工程师角度：用 MoE 模型时主要关注**部署复杂度**（多 GPU all-to-all）和**冷启动**（专家加载）。
