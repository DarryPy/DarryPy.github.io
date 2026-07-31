---
layout: post
title: Loss Spike 排查 — 大模型训练半夜炸了怎么救
date: 2026-07-31
topic: "模型与训练"
tags: [训练稳定性, loss-spike, 预训练]
excerpt: 大模型训练半夜炸出 loss spike 和 NaN,先判型再急救:跳步闸门、梯度裁剪、checkpoint 回滚,一套流程把两周进度从崩溃边缘救回来。
permalink: /posts/2026-07-31-loss-spike-training-crash-debug.html
---

你盯着 wandb 上那条 loss 曲线,本来稳稳往下走,凌晨三点突然一根尖刺冲上天,紧接着 NaN 铺满整块显存。八张卡跑了两周的进度,就悬在崩溃边缘。这一刻真正该做的不是手忙脚乱重启,而是先看清它到底属于哪一类 spike。

## 先分清:抖动、尖刺还是雪崩

不是所有 loss 波动都要救。正常训练里 loss 会有小幅抖动,幅度在均值百分之几以内,这是 mini-batch 采样噪声,别去管它,一管反而容易过拟合曲线本身。

真正危险的是尖刺:loss 在几步内跳高两三倍。它能不能自己回落,决定了是良性还是恶性——十几步后自动收敛说明模型还有自愈力,一路冲到 NaN 就是雪崩。

雪崩意味着梯度已经污染了全部参数,再往下训都是白烧算力。所以第一反应不该是重启,而是判型:先看曲线形状,再定位是哪张卡、哪个 step 最先炸开。

这里有个容易忽略的细节:分布式训练下,只有一张卡的 loss 异常,多半是那张卡的数据分片或硬件出了问题;所有卡同步飙升,才是模型层面的全局失稳,两者的处理路径完全不同。

判型时也别只盯着总 loss。把 grad norm、learning rate、每张卡的 loss 分别拉出来对齐时间轴,炸点往往在某一路指标上先露马脚,这比事后翻日志高效得多。落到具体排查,按固定顺序走最省时间,别东一榔头西一棒子:

- 看 loss:是 inf 还是 NaN?inf 多半是溢出,NaN 常是除零或 log 负数
- 看 grad norm:炸之前是否已经缓慢抬头?抬头就是可预警的爆炸
- 看 step 编号:是不是每轮固定位置复现?固定就锁定数据
- 看 GPU 分布:单卡还是全卡?单卡先查分片与硬件

## 五个最常见的元凶

先用一张表对号入座,再逐个说怎么快速证伪:

| 元凶 | 典型信号 | 快速验证 |
| --- | --- | --- |
| 学习率过高 | 训练早期规律性尖刺 | 砍一半 LR 复现 |
| 坏数据样本 | 固定 step 必炸 | 反推 batch 捞脏样本 |
| FP16 溢出 | loss 先 inf 后 NaN | 换 BF16 或调 loss scale |
| 梯度爆炸 | grad norm 冲到几百 | 打印 per-layer norm |
| Adam 二阶矩失稳 | 长期平稳后突炸 | 调大 eps 或降 beta2 |

学习率过高是最普遍的原因,信号是训练早期就出现规律性尖刺。把 LR 砍一半重跑一小段,如果 spike 明显收敛,基本就是它,配合更长的 warmup 一般能压住。

坏数据是最阴的一类:同一个畸形样本每个 epoch 都会经过,于是 spike 在固定 global step 精准复现。记下崩溃的 step 反推 batch 索引,十有八九能捞出一条乱码或超长拼接的脏样本。

这类脏样本的杀伤力在于它绕过了平均。哪怕百万条里只有一条全是重复符号,它带来的梯度也可能比正常样本大上几个数量级,一次前向就足以把参数推到发散区间。

FP16 的动态范围只到六万多,注意力分数一大就溢出成 inf,反传立刻 NaN。这就是为什么现在大模型预训练几乎都默认 BF16——牺牲尾数精度换更大指数位,稳得多。

梯度爆炸和 Adam 失稳偏隐蔽。前者 grad norm 会先冲到几百,打印 per-layer norm 能揪出是哪层失控,通常是某个 LayerNorm 或输出投影;后者在长期平稳后突炸,多半是二阶矩 eps 太小,调大 eps 或降 beta2 就能缓解。

## 现场急救:三板斧

急救的核心思想只有一句:别让一次 spike 毁掉整轮训练。给训练循环加一道跳步闸门,梯度一异常就跳过这步更新,保住已经练好的参数。

```python
prev_loss = None
for step, batch in enumerate(loader):
    loss = model(batch).loss
    # 闸门一:loss 非有限或突然翻倍,直接跳过这步
    if not torch.isfinite(loss) or (prev_loss and loss > 3 * prev_loss):
        optimizer.zero_grad(set_to_none=True)
        print(f"[skip] step={step} loss={loss.item():.3f}")
        continue
    loss.backward()
    # 闸门二:全局梯度裁剪,压住偶发爆炸
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if gnorm > 100:                      # 闸门三:norm 异常大就只记录不更新
        optimizer.zero_grad(set_to_none=True)
        continue
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    prev_loss = loss.item()
```

三道闸门层层兜底:跳过坏 loss、裁剪梯度、拦截异常 norm。跳步在预训练里是安全的,单步样本占比极小,漏掉几条不影响收敛,却能让你从崩溃边缘全身而退。

如果已经出了 NaN,别犹豫,直接回滚到最近一个健康 checkpoint,把学习率临时下调两三成再续训。硬着头皮从 NaN 状态往下跑,只会浪费更多卡时,还可能把坏状态又存进新档。

要提醒的是,跳步不能无限用。如果同一段数据反复触发跳过,说明问题出在数据或超参本身,而不是偶发噪声,这时候要停下来查根因,而不是靠闸门硬扛过去。

## 从源头把风险摁下去

急救是止血,真正省心还得靠预防。下面这份清单,是预训练开跑前就该配齐的:

- Warmup 拉长:前几千步慢慢升 LR,别让冷启动的大梯度直接冲垮模型
- 默认 BF16:除非硬件不支持,否则别碰 FP16 的溢出雷区
- grad clip 常开:norm 阈值设 1.0,几乎零成本的保险
- 高频存档:大模型每几百步存一次,回滚代价才扛得住
- 数据预清洗:超长、重复、乱码样本在入库前就滤掉
- 监控 grad norm:它比 loss 更早预警,曲线抬头往往就是前兆

这几条里,存档频率和梯度监控最容易被省。省下的那点存储和日志开销,真出事时会让你追悔莫及——差一个健康 checkpoint,可能就是几万块卡时的差距。

## 写在最后

Loss spike 不是玄学,是信号。曲线炸给你看的那一刻,模型其实已经在喊救命。记住三句话:BF16 保命、grad norm 先知、checkpoint 常备。半夜被告警叫醒时,先判型再动手,别一上来就 kill 掉两周的进度。
