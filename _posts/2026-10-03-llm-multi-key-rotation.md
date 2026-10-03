---
layout: post
title: LLM 应用的多 Key 轮换实战 — 限流、成本归因和泄露止血一次搞定
date: 2026-10-03
topic: "工程实战"
tags: [LLM, 密钥管理, 限流]
excerpt: 单把 API Key 在生产里同时是限流天花板、泄露爆炸半径和成本黑洞；这篇用一个带健康状态的 Key 池把轮换、429 退避、密钥热更新和泄露止血讲透。
permalink: /posts/2026-10-03-llm-multi-key-rotation.html
---

你的 LLM 应用跑了三个月，某天上午突然全线 429。翻日志发现所有请求都压在同一把 API Key 上，而它的分钟级配额早就见顶。更糟的是上周有同事把这把 Key 提交进了 Git，现在你连换都不敢换，因为全站就靠它续命。单 Key 不是省事，是把一堆风险攒在一起等着集中爆发。这篇讲怎么用一个 Key 池，把限流、成本归因和泄露止血一次性解决。

## 一把 Key 撑不住生产的三个理由

单 Key 的第一个坑是限流天花板。provider 的 rate limit 按 Key 或组织维度计算，你后端实例扩到十个，请求最终还是全压在同一把 Key 上，TPM 和 RPM 一到顶就集体 429，水平扩容对这个瓶颈完全无效。举个真实数字：一把 Tier-2 的 Key 假设给你每分钟 80 万 token，看着很多，可一旦某个长文档批处理任务插进来，几十个并发请求几秒钟就把这个窗口吃干，线上对话全被挤到排队。

第二个坑是爆炸半径。这把 Key 一旦泄露、被风控封禁或额度烧光，全站服务瞬间归零，没有任何缓冲余地，你连灰度回退的机会都没有。单点依赖在别的系统里你会本能地加副本，到了 API Key 这里却常常忘了它同样是单点。第三个坑是成本归因：所有业务线、所有租户的流量混在一把 Key 下结算，月底账单来了你根本拆不开是推荐系统烧的还是客服机器人烧的，只能按感觉分摊，预算和限流策略也就无从谈起。

这里有个常被忽略的陷阱：有的 provider 把 rate limit 挂在组织而不是单个 Key 上，这种情况下你在同一个组织里狂建 Key 根本没用，配额是共享的，建再多也绕不开那个总闸。要真正提升天花板，得跨组织甚至跨账号申请 Key，再把它们汇进同一个池子统一调度。所以接入前先搞清楚你的 provider 到底按哪个维度限流，别白忙一场。

## Key 池：先把状态模型定清楚

解决思路是把多把 Key 抽象成一个带状态的池子。每把 Key 不只是一个字符串，它还带着权重、健康状态、冷却截止时间和用途标签。调度时在健康 Key 里按权重挑一把，遇到 429 就把它打进冷却，到点自动恢复。权重让你把配额大的 Key 多分流量，标签让你事后能按业务线算账，冷却让临时被限流的 Key 自动退场又自动归队。下面是最小可用的数据结构，刻意保持无共享状态，方便多实例各自随机加权、不必争一个全局计数器：

```python
import time, threading, random
from dataclasses import dataclass

@dataclass
class ApiKey:
    value: str
    weight: int = 1          # 配额大的 Key 给高权重
    tag: str = "default"     # 业务线/租户标签，用于成本归因
    cooldown_until: float = 0.0
    fails: int = 0

    @property
    def healthy(self) -> bool:
        return time.time() >= self.cooldown_until

class KeyPool:
    def __init__(self, keys):
        self._keys = keys
        self._lock = threading.Lock()

    def pick(self):
        with self._lock:
            live = [k for k in self._keys if k.healthy]
            if not live:
                raise RuntimeError("no healthy key")
            total = sum(k.weight for k in live)
            r, upto = random.uniform(0, total), 0
            for k in live:
                upto += k.weight
                if r <= upto:
                    return k
            return live[-1]

    def penalize(self, key, seconds):
        with self._lock:
            key.fails += 1
            key.cooldown_until = time.time() + seconds
```

这里用加权随机而不是严格轮询，是因为在多实例部署下，严格轮询需要一个共享游标才不会各转各的，而加权随机天然无状态，统计意义上就能把流量按权重摊平，省掉一个分布式协调点。

## 轮换策略：round-robin 不够用

最朴素的 round-robin 把请求均摊到每把 Key，但它对 Key 之间配额不均、某把临时被限流毫无感知，撞墙了也照撞不误。生产里你要的是 health-aware：优先按权重轮询正常 Key，碰到 429 立刻冷却并退避到其他 Key，冷却到期再自动放回池子。几种策略的取舍：

| 策略 | 均衡性 | 对 429 的反应 | 适用场景 |
|------|--------|---------------|----------|
| round-robin | 均摊 | 无感知，继续撞墙 | Key 配额一致、极少限流 |
| weighted | 按配额倾斜 | 无感知 | Key 配额差异大 |
| health-aware | 按权重+健康 | 冷却退避 | 生产默认 |
| least-latency | 偏向快的 | 间接规避 | 延迟敏感、多 region |

真正的难点在 429 之后怎么处理。provider 通常在响应头给 `Retry-After`，优先读它当冷却时长；没有就用指数退避，比如 2 的 fails 次方秒并封顶 60 秒，再叠一点随机 jitter，避免多实例同时恢复、同一瞬间又一起撞上去。连续失败超过阈值的 Key 要触发告警，因为那大概率是 Key 被封或额度耗尽，不是临时抖动，光靠自动冷却救不回来，得人工介入。

```python
def call_with_pool(pool, do_request, max_retry=4):
    for _ in range(max_retry):
        key = pool.pick()
        try:
            return do_request(key.value)
        except RateLimited as e:
            wait = e.retry_after or min(2 ** key.fails, 60)
            pool.penalize(key, wait)
        except AuthError:
            pool.penalize(key, 3600)   # Key 失效，长冷却并告警
    raise RuntimeError("all retries exhausted")
```

别忘了给每把 Key 单独打点监控：成功率、P99 延迟、429 次数、冷却时长。一旦某把 Key 的 429 曲线明显高于同伴，多半是它的额度被某个业务线悄悄吃掉了，这时按 tag 一查就能定位，比盲猜快得多。监控粒度下沉到单 Key，还能帮你发现哪把 Key 快被封、哪把常年闲置，为下一轮扩容或裁撤 Key 提供依据。

## 多租户：给大客户单独留一把 Key

当你的应用开始服务多个租户，Key 池还能顺手解决吵闹邻居问题。把若干把 Key 按 tag 分组，给付费的大客户单独绑一把或一组 Key，普通流量共用另一组，这样某个租户突发把配额打满，也只会拖垮它自己那一组，不会波及全站。实现上只要在 pick 时按租户的 tag 过滤出子池再加权挑选即可，池子结构完全复用。这一步还顺带把成本归因做实了：每把 Key 的账单天然对应一个租户，对账时一一对应，不用再从混在一起的日志里反推。

## 密钥不落地：从环境变量到热更新

Key 池解决了调度，但密钥本身的存储同样是工程问题。把 Key 硬编码或塞进镜像是红线，查出来就该返工；环境变量是及格线，可轮换一次就得重启全部实例；真正的生产做法是走 Secret Manager，比如 Vault、AWS Secrets Manager 或云厂商 KMS，应用启动时拉取、运行期定时刷新。关键是刷新要热更新：拿到新 Key 后原子替换池子里的列表，正在跑的请求继续用旧引用、新请求走新池，服务一秒都不中断。这样你轮换 Key 时根本不用发版，运维窗口直接消失。轮换时还要留一个重叠窗口，让新旧 Key 同时有效几分钟，等在途请求全部收口再吊销旧的，否则一刀切会打断那些还没返回的长请求。

## 泄露了怎么办：止血比追责重要

Key 池最大的隐藏收益是让泄露变得可控：单把 Key 泄露，你只需把它从池子里摘掉、再在 provider 后台吊销，剩下的 Key 继续扛流量，全站零停机。提前做三件事能救命：一是给每把 Key 打 tag，出事能立刻定位是哪条业务线、哪个仓库泄的；二是在 CI 里挂 secret 扫描，gitleaks 或 trufflehog 都行，提交阶段就把 Key 拦在外面；三是写好 rotation runbook，把吊销、生成、灌入 Secret Manager、池子热加载的每一步都脚本化，别等真出事了才现学现卖、手忙脚乱。演练也别省，每季度手动跑一次轮换流程，确认脚本还能用、权限没过期，真出事那天你才不会在半夜对着报错发懵。

踩坑清单：

- 别用进程内全局计数做轮询，多实例下各转各的，配额照样撞顶——轮换状态要么无状态随机加权，要么放 Redis 共享。
- `Retry-After` 可能是秒数也可能是 HTTP 日期，两种格式都得解析，只认一种迟早翻车。
- 冷却时间别设太短，429 刚恢复就猛打会二次触发限流，留足缓冲再放回。
- Key 的 tag 要落到计费日志里，否则上了多 Key 之后你比单 Key 时更算不清账。
- Secret Manager 刷新失败要 fail-open 继续用旧 Key，别因为拉密钥超时反把整个应用拖垮。

一句话总结：单 Key 把限流、成本、泄露三个风险拧成一股绳，多 Key 池把它们拆开各个击破——生产 LLM 应用，从你接入第二把 Key 开始才算真正上线。
