---
layout: post
title: LoRA 变种选型 — rsLoRA / LoRA+ / DoRA / PiSSA 到底该不该换
date: 2026-11-20
topic: "模型与训练"
tags: [LoRA, DoRA, PiSSA, PEFT, 微调]
excerpt: 原版 LoRA 不是终点。rsLoRA 修缩放、LoRA+ 拆学习率、DoRA 拆方向与幅度、PiSSA 改初始化，各解决一个真问题，也各带一个代价。
permalink: /posts/2026-11-20-lora-variants-dora-pissa.html
---

你用 LoRA 微调了一个 7B 模型，rank 从 8 调到 64，指望容量翻倍效果也跟着涨，结果 eval 几乎纹丝不动，loss 曲线甚至比 rank 8 还慢。你又去翻论文，发现 DoRA、PiSSA、LoRA+、rsLoRA 一个比一个说自己更好，peft 里也都只是一行参数的事。

问题是：这些变种各自修的是 LoRA 的哪个毛病？哪些是白捡的收益，哪些要付显存和速度的账？这篇不讲 LoRA 的基础原理，只讲你已经在用 LoRA 之后，下一步该不该换、换哪个。

先给一个理解框架：所有变种都没有改变 LoRA“冻结主干、只训低秩增量”的本质，它们动的只是缩放系数、优化器分组、参数化方式和初始化这四个位置。弄清每个变种动的是哪个位置，你就知道它能解决什么，以及为什么它解决不了你的另一个问题。

## 原版 LoRA 的三个隐藏毛病

先把原版公式摆出来：更新量等于 (alpha / r) 乘以 B 乘以 A，其中 A 随机初始化，B 初始化为零，所以训练第一步模型输出和原模型完全一致。这个设计很优雅，部署也简单，但它藏着三个毛病，分别对应后面的几个变种。

第一是缩放。alpha / r 意味着 rank 越大，每个方向分到的缩放越小，rank 64 的更新幅度被压得很低，相当于偷偷降了学习率，这正是你开头遇到的现象。很多人的应对是把 alpha 跟着 rank 一起翻倍，这能缓解，但本质是在手动补一个本不该存在的偏差。

第二是学习率。A 和 B 共用一个学习率，但 B 从零起步、A 从随机起步，两者的最优步长理论上就不一样，模型越宽差得越多。结果就是同一个学习率对 A 偏大、对 B 偏小，你扫出来的最优值其实是两者之间的折中，谁都没吃饱。

第三是起点。B 为零意味着训练初期 adapter 在一片平坦区域里摸索，前几百步的梯度信号很弱，收敛偏慢。DoRA 则从另一个角度指出：LoRA 更新时幅度和方向是绑在一起变的，而全参微调里两者往往反向变化，这是 LoRA 和全参之间那道差距的来源之一。

## 四个变种各修一个问题

**rsLoRA** 只改一处：把缩放从 alpha / r 换成 alpha 除以 r 的平方根。推导结论是只有这样，rank 增大时更新量的尺度才保持稳定，梯度也不会随 rank 塌缩。实际效果是高 rank 终于能用了，rank 128 不再比 rank 16 学得慢。零额外开销，换上基本没有风险。

**LoRA+** 给 B 一个比 A 大得多的学习率，论文推荐比值约 16。它不改模型结构，只改优化器的参数分组，收敛速度通常能快一截，在宽度大的模型上收益更明显。代价是多了一个超参，比值太大时训练前期容易抖，而且它和 warmup 长度会互相影响，需要一起看。

**DoRA** 把权重拆成幅度向量和方向矩阵两部分，方向用 LoRA 更新，幅度单独作为可训练参数。它在低 rank 下更接近全参微调的学习行为，rank 8 的 DoRA 常常能追平 rank 32 的 LoRA。代价是训练时每步要算列范数，速度慢 20% 到 40%，显存也略涨；好在训练完可以合并回原权重，推理零开销。

**PiSSA** 改初始化：对原权重做 SVD，用前 r 个奇异值和奇异向量初始化 A 和 B，剩下的残差作为冻结的主干。等于让 adapter 一上来就握住最重要的那几个方向，收敛明显更快。配合 QLoRA 时还有额外好处：主成分被拿出来保持高精度，量化误差只落在残差上。

| 变种 | 改了什么 | 主要收益 | 训练开销 | 推理开销 | 坑点 |
|------|----------|----------|----------|----------|------|
| LoRA | 基线 | 简单稳定 | 基线 | 合并后为零 | 高 rank 不涨 |
| rsLoRA | 缩放系数 | 高 rank 可用 | 无 | 无 | alpha 要重调 |
| LoRA+ | A/B 学习率 | 收敛更快 | 无 | 无 | 比值需扫 |
| DoRA | 幅度/方向分解 | 低 rank 逼近全参 | 慢 20%-40% | 合并后为零 | 部分量化 kernel 不兼容 |
| PiSSA | SVD 初始化 | 前期收敛快 | 初始化要做 SVD | 合并后为零 | 保存时要转换格式 |

我自己的经验是：rsLoRA 默认开，几乎没有理由不开；数据量小、rank 被迫压低时上 DoRA；算力紧、想少跑几个 epoch 时试 PiSSA；LoRA+ 适合已经稳定的流水线做加速，不适合作为第一次跑通的配置。

## 在 peft 里怎么开

下面这段把几种写法放在一起。它们大部分可以叠加，比如 rsLoRA 加 DoRA，但不建议一次全开，否则出了问题你分不清是谁的锅。

```python
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
import torch

base_targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

# 方案 A：rsLoRA，高 rank 也能稳定学习
cfg_rs = LoraConfig(r=64, lora_alpha=16, use_rslora=True,
                    target_modules=base_targets, task_type="CAUSAL_LM")

# 方案 B：DoRA，低 rank 逼近全参
cfg_dora = LoraConfig(r=8, lora_alpha=16, use_dora=True,
                      target_modules=base_targets, task_type="CAUSAL_LM")

# 方案 C：PiSSA 初始化，用快速 SVD 节省初始化时间
cfg_pissa = LoraConfig(r=16, lora_alpha=16,
                       init_lora_weights="pissa_niter_4",
                       target_modules=base_targets, task_type="CAUSAL_LM")

model = get_peft_model(base_model, cfg_rs)

# LoRA+：只换优化器，B 的学习率是 A 的 16 倍
optimizer = create_loraplus_optimizer(
    model=model,
    optimizer_cls=torch.optim.AdamW,
    lr=2e-4,
    loraplus_lr_ratio=16,
)
```

PiSSA 有一个容易忽略的细节：因为它改动了主干权重（主干变成了残差），直接保存的 adapter 不能挂到原始 base model 上用。保存时要传入初始 adapter 做转换，把它变回标准 LoRA 格式，否则线上加载后输出全是乱的，而且不会报任何错。

DoRA 也有类似的部署细节：如果你的推理框架只支持加载标准 LoRA 而不支持幅度向量，就必须先合并成完整权重再部署，这会让多 adapter 热切换的方案失效。上线前先确认推理侧支持哪种格式，别等训练完才发现部署不了。

## 怎么做公平对比

换变种之前，先确保你的对比是公平的，否则结论毫无意义。最常见的错误是固定 alpha 和学习率不变，直接切换变种：rsLoRA 把有效缩放放大了好几倍，你等于同时调高了学习率，好坏都不能归功于变种本身。

我的做法是每个变种单独扫三档学习率，取各自最优再比；评测用下游任务指标而不是训练 loss，因为 DoRA 和 PiSSA 的 loss 曲线形状本来就不同；同时记录每步耗时和峰值显存，把墙钟时间也算进账里。很多时候 DoRA 的指标优势，拿同样的时间预算给 LoRA 多跑半个 epoch 就追平了。

还有一点经常被忽略：随机种子。LoRA 类方法在小数据集上的方差不小，两次相同配置的结果相差一两个点很正常。如果变种只比基线高零点几个点，至少跑三个种子取均值，否则你看到的可能只是噪声。另外别忘了测通用能力回退，有些变种在目标任务上涨了，在通用 benchmark 上掉得也更多。

如果你不想逐个实验，我建议的默认路径是：先用 rsLoRA 加一个中等 rank 跑通基线；效果不够且数据量有限时换 DoRA；训练预算紧时试 PiSSA 初始化；流程稳定后再用 LoRA+ 压缩训练时间。每一步只改一个变量，保留上一步的结果作为对照。

最后提醒一句：如果你的数据只有几百条，或者标注质量参差不齐，换哪个变种都救不回来。变种优化的是同一份数据下的学习效率，不是数据本身的上限，先把数据清洗做扎实，收益往往比折腾变种大得多。

## 踩坑清单

- rank 调大不涨效果，先开 rsLoRA，再怀疑数据。
- 开了 rsLoRA 记得重调 alpha，沿用旧值等于暗中放大学习率。
- DoRA 训练慢是正常的，别当成 bug 去排查，算账时把墙钟时间算进去。
- DoRA 和某些 4-bit 量化 kernel 组合会退回慢路径，上线前实测吞吐。
- PiSSA 保存必须做格式转换，否则线上加载静默出错。
- LoRA+ 比值别直接拉到 16 以上，先从 4 开始观察前 200 步的 loss。
- 一次只换一个变种，叠加实验留到确定单个收益之后。
- 对比结论要带上种子数和墙钟时间，否则下次复现时没人说得清当初为什么选它。

一句话：变种不是升级包，是针对具体症状的药，先确认你的问题是缩放、学习率、起点还是容量，再对症下药。
