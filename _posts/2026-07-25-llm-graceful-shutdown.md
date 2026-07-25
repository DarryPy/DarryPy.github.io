---
layout: post
title: LLM 服务优雅停机 — 部署时别掐断正在跑的长请求
date: 2026-07-25
topic: "工程实战"
tags: [优雅停机, 部署, 可用性]
excerpt: 一次 LLM 请求动辄几十秒,滚动更新时老 Pod 硬退会把半句话掐断。讲清优雅停机的正确顺序、代码骨架和 K8s 配置。
permalink: /posts/2026-07-25-llm-graceful-shutdown.html
---

你有没有遇到过这种情况:一次例行发布,用户那边突然收到一批 502,客服群里炸开锅。你翻日志发现新版本已经起来了,老 Pod 也退了,可偏偏有几十个正在生成的长回答被硬生生掐断。对普通 Web 服务,请求几十毫秒就结束,滚动更新掐断几个无所谓;但 LLM 服务的一次请求动辄几十秒,优雅停机就成了绕不开的硬骨头。这篇就把这块硬骨头拆开:停机该按什么顺序做、代码骨架长什么样、K8s 和负载均衡两层怎么配合,最后给一份能直接照着抄的踩坑清单。

## 为什么 LLM 服务的停机格外难

传统微服务的请求生命周期以毫秒计,滚动更新时老实例只要等几百毫秒,in-flight 请求基本都能自然收尾。LLM 服务完全是另一回事:一次 chat completion 从首 token 到结束常在 20 到 90 秒之间,流式场景里连接会一直挂着。你按老经验设一个 30 秒的 grace period,照样会拦腰截断一大批还没写完的回答,用户看到的就是半句话加一个红叉。

更麻烦的是重试放大。客户端收到断连往往会自动重试,而重试请求又打到刚起来、连接池还在预热的新实例上,瞬间叠加一波流量尖峰。新实例这时候最脆弱:模型 client 没热、缓存是空的、上游连接还在握手,一波重试打过来很容易把它也拖垮,故障就从一个 Pod 蔓延成一片。

所以对 LLM 服务,停机不是"多等一会儿"的小事,而是一条必须显式设计的路径。你要同时回答三个问题:流量什么时候停止进来、正在跑的请求给多久排空、排不完的怎么体面收尾。这三点任何一个含糊,一次发布就可能变成一次自制的小型故障。

## 停机的正确顺序:先摘流量,再等排空

很多人把优雅停机理解成"收到信号就睡一会儿再退",其实顺序才是关键。正确的动作是四步:第一,把 readiness probe 翻成失败,让负载均衡把这个实例从转发列表里摘出去;第二,停止接收新连接,但已建立的请求继续服务;第三,等所有 in-flight 请求跑完或触达超时;第四,清理资源后退出。

顺序错了就全盘皆输。如果你先关服务再摘流量,新请求还会源源不断打进来,队列永远排不空;如果摘流量和关连接之间不留缓冲,负载均衡还没感知到探活失败,照样往这台已经在收尾的实例上转发。记住一句话:摘流量是异步生效的,你必须主动睡一小段时间去等它,而不是假设它瞬间完成。

如果你的 LLM 任务跑在异步 worker 上(比如队列消费者做批量总结、离线 embedding),顺序略有不同但思路一致:收到信号后先停止从队列拉新任务,再等手头这条跑完并正常 ack,最后退出。千万别在任务处理到一半时直接退——没 ack 的消息会被重新投递,而 LLM 任务重跑既费钱又可能产生重复写入。HTTP 服务靠摘流量,worker 靠停止拉取,本质都是"先关进口,再排存量"。

## 信号处理与 drain 的代码骨架

下面这段 Python 骨架把四步顺序落到实处。核心是用一个集合追踪所有 in-flight task,SIGTERM 来了先翻 readiness,睡几秒等负载均衡反应,再带上限地等待请求排空。

```python
import asyncio, signal
from contextlib import suppress

inflight = set()          # 正在处理的请求 task
shutting_down = False

async def handle(request):
    task = asyncio.current_task()
    inflight.add(task)
    try:
        return await run_llm(request)   # 20~90s 的长活
    finally:
        inflight.discard(task)

async def readiness():
    return 503 if shutting_down else 200   # 摘流量开关

async def on_sigterm():
    global shutting_down
    shutting_down = True             # 1. 先让 readiness 失败
    await asyncio.sleep(5)           # 2. 给 LB 留反应时间
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(      # 3. 带上限地 drain
            asyncio.gather(*inflight, return_exceptions=True),
            timeout=90)
    # 4. 仍未结束的强制收尾,并打日志便于复盘
```

关键点有三个:`sleep(5)` 不是偷懒,而是覆盖探活周期,少了这步就会漏流量;`wait_for` 的 timeout 要和你最长请求对齐,太短等于没做;`return_exceptions=True` 保证个别请求异常不会打断整体排空。别忘了在退出前记一条日志,写清这次 drain 等了多久、还剩几个请求被强杀,这是发布后复盘最有用的数据。

写完代码一定要验证,别指望它一次就对。最简单的办法是在预发环境构造一批长请求,然后手动 `kubectl delete pod` 触发 SIGTERM,盯着日志看:readiness 有没有立刻翻 503、已建立的请求是不是全部跑完、drain 时长有没有落在预期区间、有没有请求被 SIGKILL 强杀。把这套验证做成发布前的固定检查项,比线上出事再回滚划算得多。真正的优雅停机,是被反复演练出来的,不是配一次就一劳永逸。

## 负载均衡与 K8s:排水的两层配合

代码只是一半,另一半在编排层。真实链路里有两层排水:Kubernetes 把 Pod 从 Endpoints 摘除,和负载均衡把后端从转发列表摘除。这两层都不是瞬时的——Endpoints 变更要经过 kube-proxy 或 ingress controller 传播,LB 还有自己的 deregistration delay。preStop 里那几秒 sleep,正是用来盖住这段传播窗口。

具体到配置,这三个字段配错,再优雅的信号处理也白搭:

| 配置项 | 作用 | LLM 场景建议 |
| --- | --- | --- |
| terminationGracePeriodSeconds | SIGTERM 到 SIGKILL 的最大间隔 | ≥ 最长请求 + 缓冲,常设 120 |
| preStop hook | 退出前先 sleep,等 LB 摘流量 | sleep 5~10s 覆盖探活传播 |
| readinessProbe periodSeconds | 探活间隔,决定摘流量快慢 | 调到 2~3s,别让摘流量拖太久 |

还有个隐蔽的坑在长连接复用。LLM 网关前面常挂着 keep-alive 或 HTTP/2 连接,负载均衡即便把后端摘出转发列表,已建立的长连接仍可能继续复用发新请求。所以云上的 LB 通常有独立的 deregistration delay(ALB 默认 300 秒,nginx 靠 keepalive_timeout 控制),你得把它和 K8s 的 grace period 一起考虑,而不是只盯着 Pod 那一侧。两层参数对不齐,排水就会在某一层漏水。

最容易踩的坑是 `terminationGracePeriodSeconds` 用了默认的 30 秒。你代码里 drain 上限设了 90,可 K8s 到 30 秒就发 SIGKILL,前面的努力被系统直接抹掉。这两个值必须联动:grace period 一定要大于你的 drain deadline。另外用 `maxUnavailable: 0` 配合 `maxSurge`,让新 Pod 先就绪再退老 Pod,别一次退太多实例,否则排水期间容量骤降,活着的实例反而被挤爆。

## 流式请求与幂等的边界处理

就算给了充足时间,总有请求会撞上超时上限。对流式响应,别让连接无声断掉——在超时前主动发一个结束事件,比如 SSE 的 `data: [DONE]` 或一个带 `stop_reason: shutdown` 的收尾包,客户端就能优雅提示"生成被中断"而不是卡死转圈。这一步几乎零成本,体验差别却很大。

配套还要做幂等。既然断连会触发客户端重试,你就得保证同一个 request-id 重跑不会重复计费、不会双写数据库。做法是在入口用 request-id 查缓存:命中已完成结果就直接返回,命中处理中就复用同一个任务,而不是新起一个。停机排空和幂等设计是一对搭档,少了任何一个,发布日的账单和数据都会悄悄出问题。

客户端侧也得配合。服务端返回 503 或断连时,最好带上 `Retry-After`,让客户端按指数退避重试,而不是一收到错误就无脑重连,把刚起来的新实例又冲垮一遍。优雅停机从来不是服务端一家的事,是整条链路的默契。

踩坑清单,发布前逐条对一遍:

- grace period 用了默认 30 秒,drain 还没跑完就被 SIGKILL
- 翻了 readiness 却没 sleep 缓冲,负载均衡还在往这台转发
- drain 没设上限,一个卡死请求把整个 Pod 拖到永不退出
- 先关服务后摘流量,新请求源源不断,队列排不空
- 流式请求超时直接断连,不发结束事件,客户端卡死
- 没做幂等,客户端重试导致重复计费和脏数据
- 一次退太多 Pod,排水期间容量骤降,活着的实例被挤爆
