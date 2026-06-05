---
layout: post
title: 上下文长度扩展 🔭 — RoPE 缩放、YaRN 与长文本训练的实战陷阱
date: 2026-06-05
topic: "模型与训练"
tags: [LLM, RoPE, 长文本, YaRN, 位置编码]
excerpt: 模型训练时只见过 4K token，你却想让它处理 128K 长文档——这不是幻想，是工程问题。本文拆解 RoPE 缩放的三种主流方案、YaRN 的插值逻辑，以及踩坑最多的"假长文"陷阱，帮你搞清楚扩上下文到底在改什么。
permalink: /posts/2026-06-05-context-length-extension-rope.html
---

你训了一个基于 LLaMA 架构的模型，上下文窗口 4096 token，效果不错。但产品要处理合同、长报告、多轮对话历史，4K 根本不够用。直接截断？丢失关键信息。重新从头预训练一个长上下文版本？成本是原来的几十倍，而且还要等好几周。

上下文长度扩展（Context Length Extension）就是在这个夹缝里诞生的技术方向：用最小的代价，把已有模型的有效上下文撑长。它不是魔法，背后是对位置编码机制的精准手术。理解它需要先摸清 RoPE 的工作原理。

## RoPE 是什么，为什么它决定了上下文天花板

RoPE（Rotary Position Embedding）是目前主流 LLM 的标准位置编码方案，LLaMA、Mistral、Qwen、Gemma、DeepSeek 全都依赖它。它的核心思路与早期绝对位置编码不同：不把位置信息加到 token embedding 里，而是在计算 attention 时，把 query 和 key 按位置旋转一个角度，两者的内积自然包含了相对位置信息。

旋转角度由频率基数 `base`（默认 10000）和 head 的维度共同决定，公式如下：

```python
# RoPE 旋转频率
theta_i = base ** (-2i / d)   # i = 维度索引，d = head_dim

# 在位置 m 处，对 query/key 施加的旋转
q_rotated = apply_rotary(q, cos(m * theta_i), sin(m * theta_i))
k_rotated = apply_rotary(k, cos(m * theta_i), sin(m * theta_i))
```

这套设计的好处是相对位置信息通过旋转差角自然传递，外推性比绝对编码强。但问题在于：模型训练时只见过 0 到 `max_seq_len` 范围内的位置角度值。超出这个范围的 token，其旋转角度在训练阶段从未出现过，模型对这些角度值毫无经验，注意力分数崩塌，输出退化成重复乱码或截断式幻觉。这就是所谓的"位置外推失效"问题。换句话说，4K 上下文的模型不是"不想"处理 8K，而是它从没见过 4096 以外的位置，根本不知道那些 token 该放在哪个旋转坐标系里。

## 三种主流 RoPE 缩放方案对比

**Linear Scaling（线性插值）**

最简单直接：把所有位置索引按比例缩小，将 0~32K 的位置压进 0~4K 的已知训练范围。

```python
# scale_factor = target_len / train_len = 32768 / 4096 = 8
# 推理时位置 id 除以缩放系数
position_ids = original_position_ids / scale_factor
```

优点是零微调成本，推理时动态替换即可。缺点很明显：近距 token 的位置分辨率等比下降。原本位置 1 和位置 2 的旋转角差距显著，线性压缩后变成 0.125 和 0.25，模型对短距句法依赖和共指关系的感知能力变差，整体 perplexity 上升，短文本任务性能也会回退。这个方案适合"临时救急"，不适合生产部署。

**NTK-Aware Scaling**

这个方案源于 2023 年 Reddit 上的一篇讨论帖子（`/r/LocalLLaMA`），后来被大量工程实践验证效果显著优于线性插值。核心思路是：修改 `base` 参数，使不同频率维度的缩放比例不同——高频维度（短波长，负责短距细节）少压缩，低频维度（长波长，负责长距位置）多压缩。

```python
# NTK-Aware 修改后的 base
new_base = base * (scale_factor ** (d / (d - 2)))

# 示例：4K -> 32K，scale_factor = 8，d = 128
# new_base ≈ 10000 * 8^(128/126) ≈ 82694
```

这样短距依赖（句法、词义）对应的高频维度旋转角变化幅度基本不变，长距位置关系通过低频维度的合理压缩得到覆盖，两端精度都得到了保留。实测比线性插值好 2-3 个 PPL 点，且同样无需重训——推理时直接修改 config 即可生效。

**YaRN（Yet another RoPE extensioN）**

YaRN 是目前工程效果最稳定的方案，Qwen2.5、Mistral-Long 等主流模型都采用了这套思路。它把 head 的维度分成三类分别处理：

| 维度类型 | 处理方式 | 适用原因 |
|---|---|---|
| 高频维度（短波长） | 完全不插值，保持原始旋转角 | 短距依赖不存在位置超界问题 |
| 中频维度 | 线性插值（NTK 风格） | 中等距离关系需要平滑过渡 |
| 低频维度（长波长） | 按更大比例插值（外推侧重） | 长距位置关系由低频维度主要承载 |

此外 YaRN 还引入了 attention temperature 缩放系数 `t`，在 softmax 前对注意力分数做校正。这是因为插值后低频维度的旋转角密度降低，attention 熵会增大，导致 attention 分布过于平坦，token 之间区分度下降。温度系数把这个偏移补回来：

```python
# YaRN attention temperature scaling
attn_scores = attn_scores / (t * math.sqrt(head_dim))
# t ≈ 0.1 * ln(scale_factor) + 1
# scale_factor=8 时，t ≈ 1.207
```

一次完整的 YaRN 扩展只需在原始权重上做不超过 1000 步的轻量微调，数据量几百 MB 即可，成本远低于重新预训练。这是它被广泛采用的核心原因。

## 光有缩放不够 — 长文本微调的两个硬前提

上面三种方案解决了"位置不越界"，但真正能稳定处理长文档还需要两个配套条件。

**前提一：注入真实长文本训练数据**

位置编码修复只是让模型"不报错"，并不代表它真的学会了在 32K 范围内做有效的注意力分配。模型必须见过足够多的真实长序列样本，才能习得"在 token 间距拉大时如何权衡远近依赖"的策略。通常的做法是准备 5%~15% 的超长文档（长篇小说、代码库全文、多轮对话历史、法律合同）混入微调数据集，确保训练时有足够多的 "position > train_len/2" 的样本。

来源多样性同样关键：只喂书的模型处理长代码审查时注意力会漂移，只喂代码的模型对长叙事文本的语义连贯性建模能力差。建议代码：文档：对话 = 3:4:3 的比例起步，根据下游任务调整。

**前提二：Flash Attention 2 或等价实现**

标准 attention 的显存复杂度是 O(n²)。在单个注意力层，32K token 的 Q/K/V 矩阵乘法中间结果需要存储约 32K × 32K 的 float16 矩阵，单层即超过 4GB 显存，16 层 Transformer 直接 OOM。Flash Attention 通过分块计算（tiling）和 IO 感知调度，把中间矩阵的显存占用降到 O(n)，同时不损失数值精度。

```bash
pip install flash-attn --no-build-isolation
```

启用方式：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
```

没有 Flash Attention 的情况下，32K 上下文微调批大小只能到 1，单步耗时极长；启用后批大小可以到 4~8，整体速度差距在 4~6 倍之间。不启用就做长上下文训练，是在浪费算力。

## "假长文"的四个典型陷阱

很多团队扩完上下文后发现 PPL 数字好看，但实际任务表现很差。根本原因通常是这四个：

**陷阱一：只测 PPL，不测检索精度**

PPL 下降不等于模型真的利用了长距依赖。正确的评测方法是 Needle-in-a-Haystack（大海捞针）：在 8K / 16K / 32K 不同位置各埋一条关键信息，用问题逼模型找回。很多"扩展成功"的模型在 16K 位置之后检索准确率断崖下跌，而 PPL 完全看不出来这个问题。

**陷阱二：Packing 没设置文档边界掩码**

为了凑满长序列，大多数训练框架会把多条短文档拼接成一条长样本（sequence packing）。如果 attention mask 没有正确标注文档边界，模型会跨文档做 attention，学到错误的跨文档"长距依赖"，推理时遇到真正的单文档长序列反而困惑。修复方法是使用带 document boundary mask 的 packing 实现，HuggingFace TRL 的 DataCollatorForCompletionOnlyLM 在最新版本已支持。

**陷阱三：RoPE 配置没有在推理时同步**

用 LoRA 做长文本微调时，base 模型的 RoPE config 在 adapter 里并不会自动更新。如果推理时只加载 LoRA adapter 而不显式覆盖 rope_scaling，模型仍然用原始的 4K 位置配置跑推理，YaRN 或 NTK 的修改等于白做。

```python
# 正确姿势：显式传入 rope_scaling
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    rope_scaling={
        "type": "yarn",
        "factor": 8.0,
        "original_max_position_embeddings": 4096
    },
)
model = PeftModel.from_pretrained(model, lora_adapter_path)
```

**陷阱四：扩展系数过激，分阶段训练缺失**

一次性从 4K 跳到 128K（32×）成功率极低，即便 PPL 过得去，生成质量也会明显退化。正确做法是分阶段：4K → 16K → 32K，每个阶段做数百步微调让模型适应新的位置范围。Qwen 和 LLaMA3 的官方长上下文版本都采用了这种渐进式策略。

---

**踩坑清单**

- [ ] 扩展前先跑 Needle-in-a-Haystack 建立基线，扩展后对比不同位置的检索准确率
- [ ] 确认 Flash Attention 2 已正确安装，且 `attn_implementation="flash_attention_2"` 被模型加载时接受
- [ ] Packing 数据集检查 attention mask 是否正确标注了文档边界，防止跨文档污染
- [ ] LoRA 推理时显式覆盖 `rope_scaling` 配置，不要依赖 adapter 自动继承
- [ ] 长文本微调数据来源多样化：代码 + 文档 + 对话，比例 3:4:3 为起点
- [ ] 扩展系数分阶段进行，每次 2~4 倍，不要一步到位跳 32 倍

上下文扩展本质上是让模型的位置感知系统"相信"自己见过更长的序列——位置编码是基础，长文本数据是灵魂，Flash Attention 是门票，缺一样，你得到的只是 PPL 好看的花架子。
