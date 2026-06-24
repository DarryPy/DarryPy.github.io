---
layout: post
title: Agent 工具调用成本控制 — 每次 function call 都在花钱
date: 2026-06-24
topic: "Agent 与工具"
tags: [Agent, Tool Use, 成本控制, LLM]
excerpt: Agent 每调一次工具就是一次 LLM 往返，token 消耗、延迟、API 费用全在叠加。你的 Agent 到底烧了多少钱？怎么在不牺牲能力的前提下把成本降下来？
permalink: /posts/2026-06-24-agent-tool-cost-control.html
---

## 你的 Agent 在偷偷烧钱

你写了一个 Agent，功能很好用，跑起来流畅。然后某天你看了一眼账单，沉默了。

每次执行一个中等复杂任务：假设走了 8 次工具调用，每轮 LLM 往返平均消耗 3000 输入 token、800 输出 token。按常见旗舰模型的价格折算，一次完整任务大约花费 0.05 到 0.10 美元，听起来不多。但如果你的 Agent 每天跑 500 次，就是 25 到 50 美元一天，一个月轻松过千。

更糟糕的是，很多团队在上线 Agent 之前根本没有算过这笔账。等账单来了，才开始问："我们的 token 到底花在哪了？"

成本不是不能接受，而是你得搞清楚钱花在哪里、哪些开销是必要的、哪些是可以压缩的浪费。这篇文章就是帮你把这件事想明白。

---

## 工具调用的成本来自哪里

理解成本之前，先把 Agent 工具调用的完整链路拆开看：

第一步，用户输入 + 系统提示 + 所有工具的 schema 定义，一起组成这一轮的输入 prompt，发给 LLM。第二步，LLM 判断需要调用哪个工具，输出 tool call 的 JSON 结构。第三步，你的代码执行这个工具，拿到返回结果。第四步，把工具结果塞进对话历史，再次发给 LLM，让它继续推理。如此往复，直到 LLM 决定不再调用工具、输出最终答案。

每一轮循环都意味着一次完整的 LLM API 调用，输入和输出都要计费。而且随着对话轮次增加，对话历史本身也在不断变长，后面每一轮的输入 token 都会比前一轮更多。

成本来源可以归结为四类：

| 成本来源 | 具体说明 |
|---|---|
| 工具 schema 的固定开销 | 你定义的所有工具描述，每轮都要传入，轮次越多重复付费越多 |
| 对话历史的滚雪球效应 | 历史消息随轮次累积，后期每轮输入 token 大幅膨胀 |
| 工具返回结果的体积 | 搜索结果、文件内容、API 响应可能很长，直接塞进 context 代价很高 |
| 不必要的额外轮次 | 有些工具调用根本不需要 LLM 决策，却还是走了完整推理流程 |

其中最容易被忽视的是第一条。如果你定义了 20 个工具，每个工具的 schema 平均 200 token，那每一轮对话，光工具描述就固定占用了 4000 个输入 token。这 4000 个 token 不管你这一轮用不用那些工具，都要付钱。任务跑 10 轮，你就为这些 schema 付了 40000 token 的费用，哪怕有些工具一次都没被调用过。

---

## 五种控制成本的实战手段

### 手段一：按需加载工具，精简 schema 描述

不要把所有工具一口气全部注入。把工具按功能域分组，在 Agent 启动时根据任务类型动态决定加载哪一组：

```python
def get_tools_for_task(task_type: str) -> list[dict]:
    base_tools = [search_tool, read_file_tool]
    if task_type == "code_review":
        return base_tools + [execute_code_tool, lint_tool, diff_tool]
    if task_type == "research":
        return base_tools + [web_fetch_tool, summarize_tool, cite_tool]
    return base_tools
```

这样一来，代码相关任务不需要加载研究类工具，反之亦然。schema 数量从 20 个降到 5-8 个，每轮节省 2000-3000 输入 token，效果立竿见影。

同时，精简 schema 里的 description 字段。很多人习惯在工具描述里写三四句话，生怕 LLM 不理解。实际上，只要描述准确、参数命名清晰，一句话就够了。description 精简 50% 通常不会影响 LLM 的调用准确率，但 token 是实实在在省下来的。

### 手段二：截断和摘要工具返回结果

工具返回的内容体积差异巨大。一次网页抓取可能返回 5000 字，一次数据库查询可能返回几十行 JSON。如果你把原始结果直接塞进 context，后续每一轮的输入都会因此膨胀，而且这些冗长的内容在 context 里积累，LLM 的注意力也会被稀释。

在工具 wrapper 层加截断逻辑，是最直接的优化：

```python
def truncate_tool_result(result: str, max_chars: int = 2000) -> str:
    if len(result) <= max_chars:
        return result
    # 保留开头 1500 字 + 结尾 400 字，中间标记省略
    return result[:1500] + "\n\n...[内容较长，中间部分已省略]...\n\n" + result[-400:]
```

更精细的做法是让 LLM 在发起工具调用之前，先声明它需要结果里的哪个部分或哪类信息，然后在工具执行完之后，用一个轻量提取步骤只保留相关片段。这样进入后续轮次的内容更干净，token 消耗也更低。

如果工具返回的是结构化数据（比如 JSON），可以在 wrapper 里做字段过滤，只保留 Agent 当前步骤真正需要的字段，把不相关的键值对在进入 context 之前就过滤掉。

### 手段三：用路由层减少完整 Agent 流程的触发次数

不是每一个用户请求都需要跑完整的 Agent 推理链路。对于高频、结构简单的查询，你完全可以在 orchestrator 层做规则判断或轻量分类，直接执行确定性逻辑，完全绕过 LLM 决策环节：

```python
def handle_request(user_input: str) -> str:
    # 规则路由：天气查询直接走工具，不过 LLM
    if re.match(r"(今天|明天|后天).*(天气|气温|下雨)", user_input):
        location = extract_location(user_input)
        return get_weather(location)

    # 轻量模型分类：用 Haiku 级别的小模型做意图识别
    intent = classify_intent_with_small_model(user_input)
    if intent == "faq":
        return lookup_faq(user_input)

    # 兜底才走完整 Agent 流程
    return full_agent_run(user_input)
```

这种分层设计的好处在于：80% 的请求可能都是高频重复场景，规则路由或轻量分类就能搞定，只有剩下 20% 的复杂请求才真正需要完整的 Agent 能力。成本结构完全不同。

### 手段四：缓存工具调用结果

很多工具调用在短时间内会被重复触发：相同的搜索关键词、相同的文件路径、相同的 API 参数。如果每次都实际执行工具并把结果传给 LLM，不仅慢，而且浪费。

按工具名称 + 参数哈希做缓存，命中时直接返回，跳过工具执行和那一轮 LLM 往返：

```python
import hashlib, json

def get_cache_key(tool_name: str, args: dict) -> str:
    payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()

def call_tool_with_cache(tool_name: str, args: dict, ttl: int = 300):
    key = get_cache_key(tool_name, args)
    cached = cache.get(key)
    if cached:
        return cached
    result = execute_tool(tool_name, args)
    cache.set(key, result, ttl=ttl)
    return result
```

不同工具的 TTL 应该按数据新鲜度分别设置：天气数据可能 10 分钟，静态文档可能 1 小时，外部 API 结果可能 5 分钟。对于搜索类工具，缓存命中率通常能达到 30% 到 60%，这部分节省是真实的、持续的。

### 手段五：并行工具调用

如果某一步任务需要同时获取多个独立信息，串行调用意味着多次 LLM 往返：第一轮叫 LLM 决定调工具 A，等结果，第二轮叫 LLM 决定调工具 B，等结果。现代模型（Claude / GPT-4o 均支持）可以在一次输出中同时返回多个 tool call，让你并行执行，一次性把结果全部返回给 LLM：

```json
[
  {"name": "search_web", "args": {"query": "2026 年 RAG 最新进展"}},
  {"name": "get_arxiv_papers", "args": {"topic": "retrieval augmented generation"}},
  {"name": "get_current_date", "args": {}}
]
```

这三个工具从串行的 3 轮 LLM 往返，压缩成 1 轮。任务步骤越多、独立性越高，这个优化的效果越显著。如果你的 Agent 需要并行搜索多个来源，或者同时查询多个数据源，一定要确认你的实现支持并行 tool call，而不是强制串行。

---

## 成本监控实战清单

成本失控往往不是一次大爆炸，而是一点一点地悄悄变贵，等你发现的时候已经积累了不少浪费。建议把以下几件事纳入日常：

- [ ] 每次 Agent 执行后记录总 token 数，输入和输出分开统计，按任务类型分类汇总
- [ ] 统计每个工具的平均调用次数和每次调用带入 context 的 token 体积
- [ ] 给单次任务设置 token 上限，超出后自动中止或切换到降级路径，避免失控的 Agent 无限循环
- [ ] 定期抽查高消耗任务的完整 trace，看 LLM 有没有做出冗余的工具调用，或者在不必要的地方反复确认
- [ ] 每个月算一次工具 schema 的固定成本：`len(json.dumps(all_tools))` 换算 token，乘以任务次数，这是你的基础浪费

---

## 踩坑总结

工具调用成本控制最容易踩的坑是方向错了——很多人上来就优化单次 LLM 调用的 prompt 效率，但忽略了轮次数量才是最大的乘数。一个 10 轮的 Agent 任务，哪怕每轮只省 500 token，总体节省也有 5000 token；但如果你能把 10 轮压成 6 轮，节省的是整整 4 次 LLM 往返的费用，差距天壤之别。

先减轮次，再减单轮体积，最后上缓存和路由。顺序错了，优化的效率也会大打折扣。
