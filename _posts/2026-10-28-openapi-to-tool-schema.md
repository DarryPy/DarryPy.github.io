---
layout: post
title: 从 OpenAPI 到 Tool Schema — 让已有 API 自动变成 Agent 工具
date: 2026-10-28
topic: "Agent 与工具"
tags: [Agent, Tool-Use, OpenAPI]
excerpt: 手写 tool schema 迟早漂移。教你从 Python 函数签名和 OpenAPI 文档自动生成 Agent 工具定义,让接口改了 schema 自己跟着变。
permalink: /posts/2026-10-28-openapi-to-tool-schema.html
---

你的后端已经躺着几十个 REST 接口,产品又丢来一句"让 Agent 能查订单、改地址、发优惠券"。你第一反应多半是照着 function calling 文档,一个一个手写 JSON Schema。写到第五个就会发现:参数名、类型、必填项全是从现成接口里抄过来的,抄一遍不但枯燥,还容易漏掉一两个字段。等工具数量上到两位数,这份手抄工作就成了纯体力活。

更要命的是漂移。接口今天悄悄加了个参数,schema 没人同步,Agent 调用时参数对不上,线上静默报错还查不出原因。手写 tool schema 的本质问题是信息冗余——入参定义本就存在于 type hint、Pydantic model 或 OpenAPI 文档里,你却要手工再翻译一份给 LLM。只要同一份信息有两处独立维护,就迟早分叉。正解是让 schema 从唯一事实来源(single source of truth)自动派生,而不是靠人肉对齐。

自动派生有两条主路:工具是你自己写的本地函数,就从函数签名反射;工具是现成的 HTTP 接口,就从 OpenAPI 文档转。两条路殊途同归,终点都是一份不用手维护的 schema。下面分头拆开讲,最后再补上调用阶段怎么把参数还原回一个真实请求。

## 路线一:从 Python 函数签名反射生成

如果工具就是本地函数,最省事的方式是用反射读 type hint 和 docstring,运行时自动拼出 schema。下面这段极简实现能覆盖八成场景,也是大多数 Agent 框架内部 `@tool` 装饰器的核心逻辑:

```python
import inspect, typing

def to_tool(fn):
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    TYPE_MAP = {str: "string", int: "integer",
                float: "number", bool: "boolean"}
    props, required = {}, []
    for name, p in sig.parameters.items():
        t = TYPE_MAP.get(hints.get(name, str), "string")
        props[name] = {"type": t}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "name": fn.__name__,
        "description": (fn.__doc__ or "").strip(),
        "parameters": {"type": "object",
                       "properties": props,
                       "required": required},
    }
```

关键点在于,docstring 就是给 LLM 看的 description,写函数时顺手把用途和边界讲清楚,等于同时喂给了模型,一份注释两头用。参数有没有默认值直接决定它进不进 required,你不用再单独维护一份必填列表。缺点也很明显:嵌套对象、枚举、取值范围这类复杂约束,靠这套土办法根本表达不了,硬塞只会生成一堆无意义的 `string`。

真要上生产,别自己硬搓类型映射,直接把参数定义成 Pydantic model,调一下 `model_json_schema()` 就能拿到标准 JSON Schema,枚举、默认值、字段描述、取值范围全都带上。它天然处理嵌套和 `Optional`,还能在调用前顺手把模型吐回来的参数做一遍校验,等于把入参兜底和 schema 生成两件事合并成一件,维护成本直接砍半。一个小提醒:`Field(description=...)` 里写的字段说明会原样进 schema,这是你给每个参数补充语义的最佳位置,比全堆在 docstring 里更结构化,模型读起来也更准。

## 路线二:从 OpenAPI spec 批量转换

接口数量多、而且是别的团队维护的,就别去碰人家代码了,直接拿 OpenAPI(Swagger)文档转。OpenAPI 的每个 operation 几乎能一对一映射成一个 tool,字段对应关系很直白,一张表就能说清:

| OpenAPI 字段 | Tool schema 字段 | 说明 |
|---|---|---|
| operationId | name | 没有就用 method+path 拼一个 |
| summary / description | description | 给 LLM 的用途说明 |
| parameters + requestBody | parameters | 合并 query/path/body |
| required | required | 直接沿用 |

真正要动脑的是参数的"摊平"。OpenAPI 把参数拆在 path、query、header、requestBody 四个地方,而 tool schema 只认一个扁平的 `properties` 对象。所以生成时要把它们统统拍进同一层,同时另存一份"每个参数原来属于哪个位置"的映射表,供调用阶段还原。一份三百行的 OpenAPI 能在几秒内生成几十个工具,接口升级后把脚本重跑一遍即可,再也不用手动对齐。

举个例子,`GET /orders/{id}?include=items` 这个接口,operationId 是 getOrder,摊平后的 schema 就有两个参数:`id` 来自 path、必填,`include` 来自 query、可选,description 取自 summary。模型眼里只有一个干净的 getOrder 工具,完全不用知道 REST 那套 path 和 query 的区别,这些实现细节全被映射表吸收掉了。

有两类字段生成时要主动丢掉。一是 Authorization、API-Key 这种鉴权 header,它们属于调用层的基础设施,不该让模型看见,更不该让它填。二是分页游标、trace id 这类纯技术参数,暴露出来只会干扰模型判断。把过滤规则写死在转换脚本里,比事后一个一个手删干净得多,也不会漏。

## 生成之后:调用时的参数还原

生成 schema 只是上半场。模型吐回参数后,你还得把这堆扁平字段翻译成一个真实 HTTP 请求,靠的就是上面那份位置映射:命中 path 的塞进 URL 占位符,命中 query 的拼到问号后面,剩下的进 body。

```python
def dispatch(spec, args):
    url = spec["path"]
    query, body = {}, {}
    for name, val in args.items():
        loc = spec["loc"][name]        # path/query/body
        if loc == "path":
            url = url.replace("{%s}" % name, str(val))
        elif loc == "query":
            query[name] = val
        else:
            body[name] = val
    return requests.request(spec["method"], url,
                            params=query, json=body or None)
```

这层 dispatcher 最好集中写一处,所有自动生成的工具共用一套。接口再多,也只有这一份调用逻辑要维护,新增一个工具纯粹是数据层的事,不用改一行执行代码。顺带还能在这里统一塞鉴权头、超时和重试,集中管控比散落在每个工具里省心得多。

还有一件容易被忽略的事:请求回来的结果别原样回灌给模型。一个列表接口可能吐几百条记录,直接塞回去既爆 context 又让模型抓不住重点。在 dispatcher 出口统一做一次裁剪——只留模型用得上的字段、长列表截断并附上总数——这一步和参数还原同样重要,是自动生成的工具真正好用的前提。

## 转换时的三个坑

自动生成不是银弹,下面三个地方几乎每次都会咬你一口:

- **工具太多淹没模型**:一口气塞五十个工具进去,LLM 选错的概率飙升,还白烧 token。工具越多无关噪音越大,准确率反而往下掉。按业务场景分组,或先用一轮检索筛出相关工具再喂给模型,别指望它从一长串里精准点名。
- **description 太水**:OpenAPI 里 summary 经常只有"查询订单"四个字,模型根本分不清它和"查询退款"的区别。生成之后必须人工补充关键 description,写清楚什么场景该用、什么场景别碰、返回的是什么,这是整条链路里唯一没法全自动、也最值得花时间的环节。
- **危险操作混进来**:DELETE、转账、退款这类接口一旦被自动暴露,Agent 可能脑子一热就误调,还可能被 prompt injection 诱导着调。生成阶段就按 HTTP 方法或路径打标签,把写操作强制接到人工确认流上,读操作才放开让它自动跑。

## 落地清单

把上面几条拧成一张可以直接照着做的清单,从接到需求到工具上线基本就这几步:

- 选定唯一事实来源:本地函数走反射或 Pydantic,外部接口走 OpenAPI,别两头都维护
- 生成结果缓存到文件,CI 里校验 schema 与接口定义是否漂移,漂了就让流水线红给你看
- 调用层集中一处,统一处理鉴权、超时、重试,新增工具只动数据不动代码
- 结果回灌前先裁剪:只留有用字段、长列表截断附总数,别让一次调用就把 context 撑爆
- 给每个工具标 read / write 级别,写操作一律接确认流
- description 留人工兜底,别指望 summary 直接够模型用
- 工具数量超过二十个就上分组或检索,别一次性全塞进 context

说到底,接口定义才是源头,tool schema 只是它的一种投影。把这层关系理顺,你维护的永远只有接口那一份;手写 schema 的那套,早晚会在某次接口微调后悄悄失准。

一句话:tool schema 不该是你手写的文档,而该是从接口定义编译出来的产物。你只管维护接口,schema 自己跟着变,接口改一次、工具自动对齐一次,这才是能长期活下去的 Agent 工具层。
