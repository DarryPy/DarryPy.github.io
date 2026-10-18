---
layout: post
title: Checkpoint 平均与 EMA — 训练末期白捡的几个点
date: 2026-10-16
topic: "模型与训练"
tags: [训练, EMA, Checkpoint]
excerpt: 训练已经收敛，评测分却还在抖。别再反复重训，把末期的几个 checkpoint 平均起来，常能白捡一两个点的稳定提升。
permalink: /posts/2026-10-16-checkpoint-averaging-ema.html
---

你训练到最后几千步，loss 早就平了，可每存一个 checkpoint 去评测，分数就上下跳一两个点。你挑哪个发布？挑最高的那个，大概率是运气，换个测试集就打回原形。这里有个几乎零成本的办法：别挑，把末期的多个权重平均起来。这就是 checkpoint averaging，和它的在线表亲 EMA。

## 为什么平均权重会更好

训练后期模型在 loss 平面的一个宽谷里来回震荡，每个 checkpoint 落在谷壁不同位置，带着各自的噪声。单独看谁都不是谷底，但把权重逐元素平均，相当于把这些偏移抵消掉，落点更接近谷的中心。而宽平的极小值对应更好的泛化，平均正是把你推向那个更宽的中心。

关键前提是这些 checkpoint 要足够接近、处在同一盆地里。隔得太远（比如学习率还大时的早期点）平均出来是垃圾，因为根本不在同一个谷。所以这招只在训练末期、学习率退得很低时才稳定生效。

## 两种做法：离线平均 vs 在线 EMA

离线 checkpoint averaging 最简单：训练正常进行，末期多存几个权重，事后把它们逐参数取平均。代码几行就够：

```python
import torch, glob

paths = sorted(glob.glob("ckpt/step_*.pt"))[-5:]  # 取末期 5 个
avg = None
for p in paths:
    sd = torch.load(p, map_location="cpu")["model"]
    if avg is None:
        avg = {k: v.clone().float() for k, v in sd.items()}
    else:
        for k in avg:
            avg[k] += sd[k].float()
for k in avg:
    avg[k] /= len(paths)
torch.save({"model": avg}, "ckpt/averaged.pt")
```

在线 EMA 则是训练过程中维护一份影子权重，每步按 `ema = decay*ema + (1-decay)*model` 更新。它不占额外训练算力，只多一份权重的显存，发布时直接用影子权重。两者的权衡如下。

| 维度 | 离线平均 | 在线 EMA |
|------|---------|---------|
| 实现成本 | 训练后几行脚本 | 要改训练循环 |
| 额外显存 | 0（事后做） | 1 份权重 |
| 调参 | 选哪几个 ckpt | decay（如 0.999） |
| 灵活性 | 可事后试不同组合 | 训练时就定死 |

小团队、已经训完的模型，用离线平均补救；长期流水线、追求稳定，直接上 EMA。

## 实战中怎么选 checkpoint

离线平均的效果取决于你平均哪几个点。经验法则是只取学习率退火尾段、loss 已平稳的 checkpoint，个数 3 到 8 个，间隔别太密（太密几乎是同一个点）也别太疏。我习惯按验证集扫一遍末期所有点，把明显异常尖峰的那个剔掉再平均，避免一个坏点拖累整组。

还要注意 BatchNorm 之类带运行统计量的层——纯语言模型一般没有，但多模态的视觉编码器可能有。对这些层直接平均 running stats 会出问题，正确做法是平均完权重后再用少量数据重新校准一遍统计量。

## 踩坑清单

- 平均了训练早期的点：学习率还大时的 checkpoint 不在同一盆地，平均出来更差。
- float16 直接累加：逐个累加会丢精度，务必先转 float32 再平均。
- EMA decay 拍脑袋：0.999 是常见起点，训练步数短就调小，别照搬别人的数字。
- 忘了重算 BN 统计：含 running stats 的层平均后要重新校准，否则推理崩。
- 只信单点最高分：末期单点高分多半是噪声，平均后的稳定分才是真实水平。

一句话：训练收敛后别急着挑最高分发布，那是在赌运气；把末期几个权重平均起来，你赌的是统计规律。
