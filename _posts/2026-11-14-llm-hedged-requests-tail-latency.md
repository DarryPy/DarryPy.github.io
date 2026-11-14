---
layout: post
title: LLM 请求对冲实战 — 多花一点冗余，砍掉 P99 长尾
date: 2026-11-14
topic: "工程实战"
tags: [Hedged Requests, 长尾延迟, LLM 工程]
excerpt: P50 很好看、P99 却慢得离谱？对冲请求用少量冗余调用换掉长尾，讲清触发时机、取消、成本上限和那些会让它适得其反的坑。
permalink: /posts/2026-11-14-llm-hedged-requests-tail-latency.html
---

你大概见过这样的监控面板：首 token 延迟的 P50 只有 600 毫秒，P99 却飙到 9 秒。用户不看中位数，他们只记得那一次转圈转到想关页面。超时重试救不了这种情况，因为你得先等满超时才动手，长尾早就发生了。今天聊一个在 Google 内部用了很多年的老办法：对冲请求（hedged requests），把它搬到 LLM 调用上。

## 长尾延迟到底从哪来

LLM API 的慢，很少是因为你的 prompt 本身。更多是排到了一台负载高的推理节点、碰上 KV cache 被挤出、跨区路由绕远，或者供应商侧短暂的调度抖动。这些因素是随机的、彼此独立的，同一个请求立刻再发一次，大概率会落到另一台机器上，表现完全正常。

这就是对冲成立的前提：慢是偶发的，而不是系统性的。如果你的慢来自超长上下文、输出本身就要生成三千 token，或者整个供应商在降级，那再发一份只会让两份一起慢，还白付一倍的钱。所以动手之前，先拿 trace 数据确认，长尾请求和正常请求的输入特征基本一样。

具体怎么确认？我的做法是把最近一周的请求按首 token 延迟分桶，对比最慢百分之一和中间那一段的输入 token 数、输出 token 数、模型名、region 和时间段。如果几个维度的分布几乎重叠，说明慢是运气问题，对冲有效；如果慢请求明显集中在长上下文或某个时段，那该做的是拆分上下文或错峰，而不是对冲。

还有一个容易漏掉的来源是你自己的客户端。连接池耗尽、DNS 解析偶发变慢、TLS 握手重建，都会在 trace 里表现成“上游慢”。先把客户端侧的耗时单独打点，排除掉自家问题，再决定是否值得为上游的随机抖动付冗余的钱，否则对冲只会掩盖真正的毛病。

## 对冲的基本做法

思路很朴素：先发主请求，等一个阈值时间；如果到点还没有收到首 token，就再发一个备份请求；谁先返回第一个 token 就用谁，另一个立刻取消。阈值通常取历史 P90 或 P95 的首 token 延迟，这样只有一成左右的请求会触发对冲，额外开销可控。

我在一个客服场景里试过，阈值设在 P95 的 1.8 秒，备份请求发到同一模型的另一个 region。上线一周，P99 首 token 从 7.6 秒降到 2.9 秒，调用量只多了 6%。关键在于“首 token”这个判定点：流式场景下拿到首 token 就锁定胜者，不需要等整段生成完再比较。

为什么不干脆一开始就同时发两份？因为那样成本直接翻倍，而大部分请求根本不需要帮助。延迟启动的意义就在这里：九成以上的请求在阈值前已经拿到首 token，备份永远不会发出；只有真正落进长尾的那一小撮，才值得多花一次钱。这是一种按需购买的保险，而不是全量冗余。

阈值的选取本质上是一道算术题。设得越低，触发越频繁，长尾砍得越干净，但成本越高；设得越高，越省钱，但对 P99 的改善有限。我一般从 P95 起步，观察一周的触发率和收益，再在 P90 到 P97 之间微调，别指望第一天就拿到最优值。

## 一个可取消的实现

下面是基于 asyncio 的最小实现。要点有三个：备份请求延迟启动、胜者确定后取消败者、全局有一个对冲预算，防止故障期间对冲把流量翻倍。

```python
import asyncio
import time

class HedgeBudget:
    """滑动窗口内，对冲请求数不超过主请求数的 ratio。"""
    def __init__(self, ratio=0.1, window=60):
        self.ratio, self.window = ratio, window
        self.primary, self.hedged = [], []

    def _trim(self, now):
        cut = now - self.window
        self.primary = [t for t in self.primary if t > cut]
        self.hedged = [t for t in self.hedged if t > cut]

    def record_primary(self):
        self.primary.append(time.monotonic())

    def try_acquire(self):
        now = time.monotonic()
        self._trim(now)
        if len(self.hedged) < len(self.primary) * self.ratio:
            self.hedged.append(now)
            return True
        return False

async def first_token(stream):
    it = stream.__aiter__()
    tok = await it.__anext__()
    return tok, it

async def hedged_stream(call_primary, call_backup, delay, budget):
    budget.record_primary()
    tasks = [asyncio.create_task(first_token(call_primary()))]
    done, _ = await asyncio.wait(tasks, timeout=delay)
    if not done and budget.try_acquire():
        tasks.append(asyncio.create_task(first_token(call_backup())))
    done, pending = await asyncio.wait(
        tasks, return_when=asyncio.FIRST_COMPLETED)
    for p in pending:
        p.cancel()  # 败者必须取消，否则 token 照样计费
    winner = done.pop()
    tok, rest = winner.result()
    yield tok
    async for t in rest:
        yield t
```

真实代码里还要处理胜者抛异常的情况：先完成的那个如果是报错，应该继续等另一个，而不是直接把错误抛给用户。另外 cancel 只是关掉你这边的连接，供应商是否停止计费要看文档，有的会把已生成的部分照算。

预算这一层建议放在进程外，比如用 Redis 做全局计数。单进程内的预算在多实例部署下会失真，十个实例各自允许一成对冲，加起来没问题；但实例数随流量扩容时，大家的窗口统计都不完整，故障期间仍可能集体放量。全局预算多一次网络往返，但换来的是可预期的上限。

## 什么时候该用，什么时候别用

对冲不是万能药，它和重试、多模型路由解决的是不同问题。下面这张表是我做选型时的对照：

| 手段 | 解决的问题 | 额外成本 | 典型触发点 |
| --- | --- | --- | --- |
| 超时重试 | 请求彻底失败或卡死 | 失败时才多一次 | 等满超时 |
| 对冲请求 | 偶发的长尾延迟 | 约 5%-10% 调用量 | P90-P95 首 token 延迟 |
| 多模型路由 | 质量或成本差异 | 视路由策略而定 | 请求进入时就决定 |
| 熔断降级 | 上游系统性故障 | 牺牲部分质量 | 错误率超过阈值 |

经验上，交互式、对首 token 敏感、输出较短的请求最适合对冲，比如聊天首屏、补全、意图分类。离线批处理、长文生成、带副作用的 tool call 都不要对冲。尤其是后者，两份请求都可能触发真实动作，除非你已经做好了幂等键。

还有一点常被忽略：备份请求发到哪。发到同一个 endpoint，可能又排进同一个拥塞队列；发到另一个 region 或另一个供应商的同款模型，独立性更好。但跨供应商时输出风格会有差异，对一致性要求高的场景要慎重，最好先在影子流量里对比一轮。

对冲和重试也可以叠加，但要算清楚放大倍数。主请求加备份请求，各自再带两次重试，最坏情况下一个用户请求会变成六次上游调用。我的建议是：对冲的备份请求不再重试，重试只留给主请求，并且所有重试和对冲共用同一个 deadline，超出就直接降级返回。

## 上线踩坑清单

- 阈值不能写死。延迟分布每天都在变，用最近十分钟的滑动 P95 动态计算，并设上下限，避免冷启动时阈值为零导致每个请求都对冲。
- 一定要有对冲预算。供应商整体变慢时，所有请求都会超过阈值，没有预算就等于在故障期间把流量翻倍，自己把自己打进限流。
- 败者必须取消，并核对计费口径。只关连接不等于不计费，在账单里单独统计对冲请求的 token，才能算清这笔冗余到底值不值。
- Prompt Caching 会打折扣。备份请求落到另一台机器或另一个 region，缓存大概率没命中，首 token 反而可能更慢，阈值要据此调高。
- 监控要拆开看：对冲触发率、备份胜出率、对冲带来的 P99 改善。备份胜出率长期低于两成，说明阈值太低，你在花冤枉钱。
- 带副作用的请求禁止对冲，除非全链路已有幂等键，否则就是在用延迟换重复扣款。

如果你只打算今天做一件事，那就先把首 token 延迟的分布按分钟打出来，看看长尾是否随机、占比多少。数据支持再上对冲，从一个低风险的只读接口开始灰度，观察一周的触发率、胜出率和账单变化，确认收益大于成本，再逐步推广到其他交互式接口。

一句话总结：对冲是用确定的小成本，去买不确定的长尾，前提是你得先证明长尾真的是随机的。
