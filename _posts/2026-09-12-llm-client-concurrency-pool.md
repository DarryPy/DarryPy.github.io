---
layout: post
title: LLM 客户端并发控制与连接池 — 高并发下别被自己先拖垮
date: 2026-09-12
topic: "工程实战"
tags: [并发控制, 连接池, 高并发]
excerpt: 生产 LLM 高并发挂掉，一半不是 provider 限速，是你自己的 client 没管好。讲 semaphore、连接池调参、背压和重试放大怎么把吞吐稳住。
permalink: /posts/2026-09-12-llm-client-concurrency-pool.html
---

你把 LLM 调用从 demo 搬进生产，压力一上来就发现：provider 疯狂返回 429，本地句柄爆满，内存曲线一路向上，延迟从 2 秒飙到 30 秒。你第一反应是模型限速，但把日志翻一遍会发现，一大半问题出在你自己的客户端——没复用连接、没管并发、没有背压。这篇讲怎么用 semaphore、连接池和背压把吞吐压稳，顺带说说重试放大、同步场景和监控。

## 第一个坑：每次请求都新建 client

每来一个请求就 new 一个 httpx.Client 或 OpenAI() 实例，是最常见也最隐蔽的性能杀手。每个新 client 都要重做 TCP 握手和 TLS 协商，连接用完即弃，keep-alive 完全失效，白白把几十毫秒耗在建连上。

高并发下你会看到大量 TIME_WAIT 连接堆在系统里，端口很快耗尽，握手开销甚至超过请求本身。更糟的是每个 client 各自维护一套连接池，实例越多连接越乱，你根本无法在一个地方统一限流。

正确做法是进程启动时建一个全局 client，整个生命周期复用它。httpx 和大多数官方 SDK 的 client 都是线程安全、协程安全的，一个实例就能扛住成百上千并发，底层由连接池自动调度复用。

一个反直觉的点：client 少不等于连接少。全局 client 底层就是一个连接池，能开多少连接完全由 limits 参数控制，跟你 new 了几个 client 无关。所以复用 client 不会限制并发，反而让所有连接归一处管理，并发更可控、更好观测。

```python
# 错误:每次请求新建,握手 + TLS 全部重来
def call(prompt):
    client = httpx.Client()      # 连接用完即弃
    return client.post(URL, json={"prompt": prompt})

# 正确:全局复用 + 显式连接池
client = httpx.Client(
    limits=httpx.Limits(max_connections=100,
                        max_keepalive_connections=20),
    timeout=httpx.Timeout(60.0, connect=5.0),
    http2=True,
)
```

## 用 semaphore 给并发上闸

复用 client 之后，下一个问题是并发数失控。一次涌进 5000 个协程全去调 LLM，provider 立刻用 429 招呼你，客户端再自动重试，叠加出更多请求，直接雪崩。你需要一个 semaphore，把同时在飞的请求数卡在可控上限。

上限怎么定？看你的 provider 配额：假设账号每分钟 500 请求、平均每请求 3 秒，那同时在飞约 500/60×3≈25 个就能跑满，再多只会撞限速。宁可略小，把余量留给重试和突发抖动。

注意 semaphore 要跟 client 一样做成全局单例，别在每个请求里新建，那等于没限。多进程部署时每个进程各有一个 semaphore，真实并发是「进程数 × 单进程上限」，配额要按这个总量倒推，否则单看一个进程觉得很稳，全局早就爆了。

```python
import asyncio
sem = asyncio.Semaphore(25)      # 同时在飞的上限,全局单例

async def call(prompt):
    async with sem:              # 超过 25 的协程在这里排队
        return await aclient.post(URL, json={"prompt": prompt})
```

别只信估算的 25，真实上限得压出来。写个阶梯压测脚本，从并发 5 开始、每隔 30 秒加 5，同时盯 p99 延迟和 429 比例两条曲线，一路加到明显撑不住。你会看到清晰拐点：再往上加吞吐不再涨，p99 却陡然抬头、429 开始冒头，这个拐点前一档就是安全上限，写死进配置并注明是压出来的。

压测有个容易漏的前提：必须用生产同规格的 prompt 长度和输出长度。短请求和长请求耗时能差好几倍，拿一句话的短请求压出来的上限，套到几千 token 输出的长请求上，照样一压就撞墙。

## 连接池调参：max_connections 与 keepalive

semaphore 管的是应用层并发，连接池管的是传输层连接，两者必须对齐。max_connections 至少不小于 semaphore 上限，否则协程拿到了信号量却卡在等连接，白白多排一层队，吞吐上不去还查不出原因。

max_keepalive_connections 决定空闲时保留多少热连接。设太小，一波流量峰过去连接就被回收，下一波又要重新握手；设太大，低峰期一堆空闲连接占着句柄。经验值是设成 max_connections 的四分之一到一半。

| 并发量级 | max_connections | max_keepalive | http2 |
| --- | --- | --- | --- |
| 低 (<10) | 20 | 10 | 可选 |
| 中 (10-50) | 100 | 20 | 建议开 |
| 高 (>50) | 200+ | 50 | 建议开 |

开 http2 的好处是多个请求能在一条连接上多路复用，句柄占用大幅下降，尤其适合大量小请求。但要确认 provider 和中间网关都支持，否则会静默回落到 http1.1，你以为生效了其实没有，得抓包或看连接数才发现真相。

## 背压：队列满了要拒绝，别无限堆

semaphore 会让超额请求排队，但队列本身也要有上限。如果上游流量持续高于处理能力，无限排队只会让延迟越拖越长、内存越堆越高，客户端还没等到响应就先超时白跑一趟，最后 OOM。

正确姿势是给等待设超时或队列长度上限，满了直接快速失败返回 503，把压力反推给调用方。这就是 backpressure：与其让所有请求一起慢死，不如让一部分快速失败并重试，保住大部分请求的延迟不崩。

背压的等待超时要留够安全边界，别设得跟请求超时一样长。否则请求光排队就耗掉大半时间预算，真正执行时所剩无几，超时率反而更高。经验是把排队超时设成请求超时的十分之一到五分之一，宁可早点拒绝，也别让它在队里慢慢烂掉。

```python
async def call_with_backpressure(prompt):
    try:
        await asyncio.wait_for(sem.acquire(), timeout=2.0)
    except asyncio.TimeoutError:
        raise HTTPException(503, "overloaded, retry later")
    try:
        return await aclient.post(URL, json={"prompt": prompt})
    finally:
        sem.release()
```

## 重试、同步与监控：三个容易漏的点

很多人给 LLM 调用加了指数退避重试，却忘了重试本身会放大并发。上游一抖动，成百上千请求同时失败、同时重试，瞬间把并发翻倍，本来几秒能自愈的小故障，被你的重试硬生生打成了雪崩。

两个动作必做：一是重试也要走同一个 semaphore，别让重试请求绕过限流从侧门涌入；二是退避加随机抖动 jitter，别让所有请求卡在同一个时间点齐刷刷重发。再配合熔断，上游挂了就快速失败一段时间，别一根筋硬重试。

重试还有个隐患：叠加并发容易造成重复计费和重复副作用。给每个逻辑请求带上幂等键，重试时复用同一个键，既能让下游去重、省下重复的 token 钱，也避免同一个操作被执行两次。

不用 asyncio 的同步服务，并发控制换成线程池加有界队列：ThreadPoolExecutor 的 max_workers 就是你的并发上限，队列满了直接拒绝任务。千万别在同步代码里硬套 asyncio.Semaphore，两套并发模型混用只会互相打架，还会引出诡异的死锁。

无论同步异步，都要把三个指标暴露出来：信号量当前占用、连接池活跃连接数、请求排队等待时长。占用长期顶格说明该扩容或调大上限；排队时长飙升就是背压该触发的信号。没有这几个指标，你的调参全靠拍脑袋。

## 踩坑清单

- client 全局复用，别在函数里 new；临时 client 忘了 close 会持续漏连接
- semaphore 做成全局单例并对齐 provider 配额，不是越大越好，撞限速反而更慢
- max_connections 必须 >= semaphore 上限，否则信号量形同虚设
- 多进程部署，真实并发是「进程数 × 单进程上限」，配额按总量倒推
- 重试走同一个 semaphore 且加 jitter，否则重试会把小故障放大成雪崩
- 给逻辑请求带幂等键，重试复用同一个键，去重省钱、避免副作用做两遍
- 同步代码用线程池 + 有界队列，别硬套 asyncio.Semaphore
- 排队等待超时设成请求超时的十分之一到五分之一，别让请求在队里烂掉
- 并发上限用生产同规格 prompt 阶梯压测，看 p99 与 429 拐点，别拍脑袋
- 背压优先「快速失败 + 重试」，无限队列等于把 OOM 留给未来的自己

一句话：LLM 高并发的稳定性，一半靠 provider，一半靠你有没有认真管好手里这一个 client。
