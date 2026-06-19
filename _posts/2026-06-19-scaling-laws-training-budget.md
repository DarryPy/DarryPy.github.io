---
layout: post
title: Scaling Laws 实战手册 🧮 — 训练前必须做完的预算数学
date: 2026-06-19
topic: "模型与训练"
tags: [Scaling Laws, LLM, 预训练, Chinchilla, 模型训练]
excerpt: 不懂 Scaling Laws，训练预算大概率打了水漂。从 Kaplan 到 Chinchilla，用具体数字告诉你算力、数据和参数三者怎么分配才不亏——附实战配比公式和五条踩坑清单，开 GPU 前先把数学题做完。
permalink: /posts/2026-06-19-scaling-laws-training-budget.html
---

你手上有 100 GPU-days，准备训一个 7B 参数的语言模型。数据备了 400B tokens，batch size 调到 4M tokens，学习率做了 warmup，训练曲线看起来平稳下降。跑完，loss 收敛了，但下游任务表现只有同规模开源模型的 70% 水准。

你怀疑是架构问题，换了 hidden dim，又跑了两轮超参搜索，结果差不多。你开始怀疑数据质量，清洗了一遍，还是差。

问题其实从规划那一刻就定了：**参数量和训练 token 数的比例，根本没算过**。Scaling Laws 解决的正是这件事——给定固定算力，怎么在参数与数据之间分配，才能把模型性能推到最高点。

## 两代定律：Kaplan 与 Chinchilla

2020 年，OpenAI 的 Kaplan 等人发表了首篇系统性研究，发现模型 loss 与三个变量呈幂律关系：参数量 N、训练 token 数 D、以及总计算量 C（单位 FLOPs）。他们的结论是：**固定算力下，应优先增大参数量，数据可以相对少**。这套逻辑直接催生了 GPT-3——175B 参数，喂了约 300B tokens 就收摊了。

2022 年，DeepMind 的 Chinchilla 论文（Hoffmann 等人）用更严格的控制实验推翻了这个结论。Kaplan 实验里每个配置的训练步数太少，严重低估了数据的价值。Chinchilla 团队用超过 400 个不同规模的模型、从 70M 到 16B 参数横跨数十个数据量配置，拟合出新的最优比例：

**最优 token 数 ≈ 20 × 参数量**

这意味着 7B 模型要训到最优点，至少需要 140B tokens。GPT-3 的 175B 配 300B tokens 严重欠训。Chinchilla 本身用 70B 参数配 1.4T tokens，性能超过了 Gopher 280B——参数量只有它四分之一，赢的是训练充分度，不是模型规模。

这不是细节差异，是整整一代训练策略的纠偏。

## 从算力推最优配置

训练一次 Transformer 所需浮点运算量的常用近似：

```
C ≈ 6 × N × D
```

C 是总 FLOPs，N 是参数量，D 是训练 tokens。前向约 2ND，反向约 4ND，合计 6ND，精度足够用于预算估算。

从这个公式出发，可以直接写出 Chinchilla 最优分配的估算器：

```python
import math

def chinchilla_optimal(gpu_days, mfu=0.4):
    # A100 理论 312 TFLOPS，实际按 MFU 打折
    flops_per_day = 312e12 * mfu * 86400   # ≈ 1.08e19 FLOPs/天
    C = gpu_days * flops_per_day
    # Hoffmann et al. 2022 拟合系数
    N_opt = math.sqrt(C / 120)   # 最优参数量
    D_opt = 20 * N_opt            # 最优 token 数
    return N_opt, D_opt

N, D = chinchilla_optimal(100)
print(f"100 GPU-days → {N/1e9:.1f}B 参数，{D/1e9:.0f}B tokens")
# 输出：约 3.0B 参数，60B tokens
```

100 GPU-days 的 A100，Chinchilla 最优点是 **3B 参数配 60B tokens**，而不是直觉里的"7B 模型随便跑"。想训 7B 并达到最优，算力需要翻到 500 GPU-days 以上，否则参数量买了，但 token 数不够喂满，性能留在桌上了。

## 什么时候可以主动偏离最优点

Chinchilla 给的是"给定算力下 loss 最低"的配方，但实际工程里有几个场景会让你主动偏离：

| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| 推理成本优先 | 小参数 + 超量训练 | 同性能下小模型推理便宜；LLaMA 的做法 |
| 高质量数据稀缺 | 增参数、减 epoch | 重复超 4 次会损性能，宁可参数大点 |
| 追下游任务 SOTA | 不能只盯 loss 指标 | 部分能力有涌现阈值，loss 层看不见 |
| 领域继续预训练 | 重新做小规模标定 | 通用最优比例在垂直数据上会失效 |

LLaMA 系列是"推理成本优先"的典范——故意在远超 Chinchilla 最优的 tokens 上训小模型，目标不是 loss 最低，而是在固定推理预算下性能最高。这套逻辑完全成立，只是目标函数换了。

Meta 在 LLaMA 2 技术报告里明确说明：Llama 2 70B 用了 2T tokens，远超 Chinchilla 最优的约 1.4T。多出来的训练成本换来的是推理阶段每次调用省下的算力——规模化服务时这笔账值得。

## 全量训练前先做比例预实验

这一步最容易被跳过，代价也最惨。任何超过 100 GPU-days 的训练，都值得先拿 1%–2% 的算力做一轮缩放验证。

具体做法：取目标参数量的 0.1x、0.3x、1x 三个档，分别配上各自的 Chinchilla 最优 token 数，各训到收敛，记录最终 loss。如果三个点在对数坐标上拟合出平滑的幂律曲线，说明你的数据管道、训练代码和超参设置没有隐蔽 bug，放大是安全的。如果点偏离幂律，必须先找原因——可能是数据有重复、tokenizer 有问题、或者 batch size 和 learning rate 没有随规模一起缩放。

发现问题花 2 GPU-days，不发现问题花 200 GPU-days 重跑，这笔账不需要算两遍。

## 踩坑清单

1. **用理论算力排预算**：A100 理论 312 TFLOPS，实际训练 MFU 通常 35%–50%，保守用 40% 估算不夸张。按 100% 排计划，实际用时直接乘 2.5，进度表准时烂掉。

2. **数据重复超 4 次**：Chinchilla 实验明确验证过，相同数据重复超过 4 epoch，loss 不降反升。数据量不够时，宁可缩参数，不要把同一批语料反复喂。

3. **把 loss 最优等同于下游任务最优**：推理、代码生成等能力有涌现阈值，在 loss 曲线上完全不可预见。Scaling Laws 告诉你 loss，不告诉你 MMLU 或 HumanEval 的分数。

4. **跨语言 / 跨数据源忽略 tokenizer 效率**：中文、数学公式、代码的 token 压缩率差异显著。同样"100B tokens"，中文稠密语料和英文稀疏爬虫数据传递的语义信息量不同，跨模型比较训练量时要折算。

5. **跳过缩放预实验直接全量**：这是最贵的坑。规模一上去，每次重跑的机会成本不只是算力，还有时间窗口。小实验是保险，不是可选项。

---

开 GPU 之前先把数学题做完。Scaling Laws 不是玄学，是你在签算力订单前本该查的公式。
