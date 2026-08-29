---
layout: post
title: LLM 请求的超时预算实战 — deadline 怎么在调用链里层层传递
date: 2026-08-29
topic: "工程实战"
tags: [超时预算, deadline, 调用链, 工程实战]
excerpt: 只在最外层套一个 timeout 保护不了整体延迟。这篇讲怎么用 deadline 让超时预算在检索、rerank、生成之间层层传递,超时后还能把取消穿到最底层。
permalink: /posts/2026-08-29-llm-timeout-budget-deadline.html
---

你有没有遇到过这种情况:给 LLM 请求设了 30 秒超时,线上却总有请求跑到 80 秒才断。排查半天发现超时压根没生效——最外层裹了一个 timeout,底下的检索、rerank、模型调用各跑各的,谁也不知道整条链路还剩多少时间。这不是 bug,是超时预算(timeout budget)没设计好的必然结果。今天讲怎么让 deadline 在调用链里层层传下去。

## 一个全局 timeout 会骗你

大多数人的超时是这么写的:在 handler 最外层套一个 30 秒,里面顺序调用 embedding、向量检索、rerank、LLM 生成。问题是这四步串行执行,前三步慢了,留给 LLM 的时间就被悄悄吃掉,但 LLM 那层自己可能还傻乎乎地设了 25 秒超时。

结果就很离谱:检索花了 20 秒,LLM 又等满 25 秒,总耗时 45 秒,你那个 30 秒"全局超时"形同虚设。根因在于单一 timeout 只能保证"某一层不超时",保证不了"整体不超时"。层与层之间没有共享同一个时间账本,每一层都以为自己拥有全部时间,加起来自然爆表。

更隐蔽的是,这种设计在测试环境几乎暴露不出来。本地检索毫秒级返回,你看不到时间被吃掉的过程,一切正常。等到生产环境向量库偶尔抖动一下,或者上游模型排队,累积效应才会让尾部请求集体击穿超时线。这也是为什么超时问题往往在压测或大促当天才第一次现形。

## deadline 比 timeout 更靠谱

换个思路:别传"还能跑多久",传"必须在几点几分之前结束"。这就是 deadline。请求进来时算出一个绝对时间点 `deadline = now + 30s`,然后把这个时间戳一路带下去。

每一层要做的不再是"我设个超时",而是"看看离 deadline 还有多少,拿这个剩余量当我的超时"。这样无论前面消耗了多少,后面的层永远知道真实的剩余预算。timeout 是相对值,跨一层就失真;deadline 是绝对值,传多少层都不会算错。gRPC、Go 的 context 都是这个模型,不是巧合。

```python
import time

class Deadline:
    def __init__(self, budget_s: float):
        self.at = time.monotonic() + budget_s

    def remaining(self) -> float:
        return self.at - time.monotonic()

    def expired(self) -> bool:
        return self.remaining() <= 0

async def handle(req, budget_s=30):
    dl = Deadline(budget_s)
    docs = await retrieve(req.query, dl)   # 把 deadline 传下去
    if dl.expired():
        raise TimeoutError("检索后已超预算")
    return await generate(req, docs, dl)
```

注意用 `time.monotonic()` 而不是 `time.time()`。前者是单调时钟,不受系统对时、NTP 校正影响;后者一旦被运维改了系统时间,你的 deadline 可能瞬间过期或永远不过期,这种 bug 半夜排查能让人崩溃。

## 每一跳都要留买路钱

光传 deadline 还不够,你得给每一跳留缓冲。假设总预算 30 秒,不能让检索把 29 秒用光,只剩 1 秒给生成——那生成必然失败。所以每层拿到 deadline 后,先算 `min(自己的最大值, 剩余预算)`,再用这个值当超时,永远不要让某一层独吞全部剩余时间。

具体怎么分,看你各阶段耗时稳不稳定:

| 分配方式 | 做法 | 适合场景 |
|---|---|---|
| 固定配额 | 检索最多 5s、生成最多 20s、留 5s 缓冲 | 各阶段耗时稳定 |
| 按比例切 | 剩余预算的 40% 给检索,60% 给生成 | 阶段耗时波动大 |
| 硬下限 | 剩余不足 3s 直接放弃生成,返回降级答案 | 保护尾部延迟 |

固定配额最好懂但不够弹性,检索快的时候省下的时间没人接盘;按比例切能自适应,但要防止某一步异常拖慢把比例算歪。多数生产系统是两者混用:先按比例切,再用硬下限兜底。硬下限那一行尤其关键——与其让生成在只剩 2 秒时启动然后必然超时,不如提前认输,直接返回一个基于检索结果的降级答案,把这 2 秒还给用户。

## 重试别偷偷加倍预算

超时和重试是一对冤家。最常见的翻车姿势是:主请求设 20 秒超时,失败后重试又给了一个全新的 20 秒。你以为总耗时上限是 20 秒,实际用户等了 40 秒还多。重试必须在同一个 deadline 之内,而不是重启时钟。

```python
async def call_with_retry(req, dl: Deadline, tries=3):
    for i in range(tries):
        if dl.remaining() < 2:        # 剩余不够跑一次,别硬试
            break
        try:
            budget = min(8, dl.remaining())
            async with asyncio.timeout(budget):
                return await llm_call(req)
        except (TimeoutError, UpstreamError):
            continue
    return degrade(req)
```

规则很简单:每次重试前先问 deadline 还剩多少,不够就直接放弃,别为了"再试一次"把整体预算拖爆。指数退避的 sleep 时间也要算进预算里——退避 4 秒结果只剩 3 秒预算,那这次退避本身就已经注定超时,不如不退。

## 流式响应下超时要拆成两个

LLM 一旦走 streaming,单一超时就更不够用了。一个生成 800 token 的回答,整体耗时可能到 40 秒,但用户真正在意的是第一个 token 什么时候出来。这时候要把超时拆成两段:首 token 超时(TTFT)卡 5 秒,判断上游是不是卡死;token 间隔超时卡 10 秒,判断流是不是中途断掉。整体 deadline 依然存在,但它管的是上限,不是用户体验的主要抓手。

实现上就是每收到一个 token 就重置"间隔计时器",只要流还在往外吐字就不判超时;一旦某两个 token 之间隔太久,说明上游多半挂了,立刻断开走降级。别再用一个"30 秒没结束就杀"的粗暴逻辑——那会把一个正常但偏长的回答误杀,用户体验反而更差。

## 超时之后,取消要穿到底

设了 deadline 只是"知道该停",真停下来还得靠取消(cancellation)。Python 里用 `asyncio.timeout`,Go 里用 `context.WithDeadline`,核心都是把取消信号顺着调用链传到最底层的 HTTP 请求,连 LLM 那条 socket 一起断掉。

最容易漏的一点:很多人 timeout 触发后只是 `return`,底层那个 HTTP 连接还在后台跑,白白占着连接池和上游配额。一定要确认取消能穿透到最底层的 client,让 socket 真的关掉。否则你省下的只是响应时间,烧掉的还是那份 token 和连接数,并发一高连接池就被这些僵尸请求拖垮,雪崩往往就是这么开始的。

顺手提醒一句:取消要能被中断的地方接住。如果你在某一层写了不带超时的同步阻塞调用,比如一个没设 timeout 的 `requests.get`,取消信号到这里就断了,上面 deadline 算得再准也没用。设计调用链时,凡是会发起 I/O 的地方都要能感知取消,这是 deadline 传播能真正落地的前提。

## 别让超时变成黑盒

超时预算做得再漂亮,不埋点也是白搭。你至少要能回答三个问题:请求最后死在哪一层、每一层实际吃了多少预算、有多少请求踩着硬下限走了降级。把每一跳的 `remaining()` 记进 trace,超时时打上"哪一步耗尽预算"的标签,这些数据才是你后面调配额的依据。

有了观测你会发现很多反直觉的事:P99 慢往往不是 LLM 慢,而是检索偶发抖动把预算提前榨干;某类 query 总在 rerank 阶段超时,那就该给它单独设更短的检索配额。没有埋点,你只能靠猜,而猜出来的超时值不是太松就是太紧。更进一步,把"降级率"单独拉一条曲线盯着——它悄悄爬升时,往往是上游在变慢的第一个信号,比等报警响起来早得多。

## 踩坑清单

- 只在最外层设 timeout,内层各设各的——整体超时必然失守
- 传 timeout 数值而非绝对 deadline,跨层后剩余时间算不准
- 不给尾部阶段留缓冲,检索吃光预算,生成永远超时
- 重试给全新超时而非共享 deadline,总耗时悄悄翻倍
- timeout 后只 return 不 cancel,底层请求变僵尸,连接池被拖垮
- deadline 用 `time.time()` 而非 `monotonic`,系统一改时间就翻车
- 没准备降级路径,预算耗尽时只能返回 500 而不是一个次优答案

一句话:超时不是"设个数字",是"让整条链路共享同一个截止时间"。谁消耗、谁记账,剩多少大家都看得见,才不会有请求偷偷跑到天荒地老。
