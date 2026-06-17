---
layout: post
title: Agent 上下文窗口管理 🗃️ — 长任务不爆 token 的六种策略
date: 2026-06-17
topic: "Agent 与工具"
tags: [Agent, 上下文管理, token, LLM, 工程实践]
excerpt: Agent 多步执行时，上下文会随每轮工具调用线性膨胀，最终触发 context_length_exceeded。本文拆解六种实用策略：工具截断、历史压缩、外置记忆、系统提示精简、思维链预算控制、流式 Map-Reduce，帮你把长任务的 token 消耗控制在窗口以内。
permalink: /posts/2026-06-17-agent-context-window-management.html
---

你跑了一个二十步的 Agent，第十八步突然报 `context_length_exceeded`。前面十七步积累的工具调用记录、观察结果、推理链全部堆在上下文里，模型撑不住了。

和单轮问答不同，Agent 的上下文是**累积增长**的。每轮追加一条 tool_call、一条 tool_result，如果开了 extended thinking 还要再加一块思维链。几十步下来，哪怕是 128K 的窗口也会被消耗殆尽。更麻烦的是，窗口里大量内容是低价值的——第三步取到的 3000 字 JSON 日志，到第十八步只需要其中一行结论，但它还是占着那些 token 一步都没走。

问题的本质不是"窗口多大"，而是"窗口里放什么"。有效管理上下文，不是靠等更大的模型出来，而是靠工程侧主动控制信息密度。下面六种策略可以叠加使用，一个普通 Agent 经过组合优化后能处理的有效步数通常可以提升 3-5 倍——前提是你得把这些事情真的做进去，而不只是知道有这回事。

## 策略一：工具返回截断

最直接的办法是在工具层控制输出粒度，不要把裁剪工作留给 LLM 来做。设计工具的时候，你写的返回格式决定了 Agent 的信息消费方式，工具返回越宽泛，Agent 读到的噪声越多，token 浪费越严重。

```python
def search_docs(query: str) -> str:
    results = vector_db.search(query, top_k=5)
    # 每条摘要截断到 300 字符，避免返回原始长文
    snippets = [r.content[:300] for r in results]
    return "\n---\n".join(snippets)
```

搜索结果只返回摘要段，不返回全文；API 响应只抽取关键字段，不透传原始 JSON；读取日志文件时只返回出错行和上下三行，不返回完整文件内容。工具要替 Agent 完成预处理，把"决策需要的最小信息"传回来，而不是把原始数据全量塞进上下文让 LLM 自己去找那一行关键内容。

记住这条原则：**工具是过滤器，不是传送带。** 每次你让工具返回的信息比 Agent 实际需要的多一倍，你就是在把窗口寿命砍半。

## 策略二：消息历史压缩

当累积 token 超过某个阈值（建议 60%-70% 窗口），把早期的 tool_result 替换成单句摘要，同时保留消息结构完整性，让对话历史看起来还是合法的 LLM 对话序列。

| 压缩时机 | 压缩策略 | 适用场景 |
|----------|----------|----------|
| 超过 60% 窗口 | 早期 tool_result 替换为一句总结 | 通用 Agent |
| 每 N 轮触发 | 调用 LLM 生成阶段摘要，清空旧历史 | 长流程 Agent |
| 检测到重复内容 | 去重合并同类工具结果 | 循环搜索 Agent |

实现时注意两点：第一，压缩时保留最近 3-5 轮不动，给 Agent 留住"刚才发生了什么"的近因感知，否则它在下一步会显得失忆；第二，tool_call 和 tool_result 要成对处理，只压 tool_result 不压 tool_call 的做法很常见，但实际上 tool_call 里的参数信息同样占 token，这个错误会让你以为压缩有效，但实际上没减少多少。

## 策略三：外置记忆 + 检索注入

不要把所有历史状态都塞进上下文——存到外面，按需取回。把 Agent 的"工作记忆"和"长期记忆"分离开来，是解决长任务 token 膨胀最彻底的方案。工作记忆就是当前推理窗口，长期记忆存在向量数据库或键值存储里。

```python
memory_store = {}

def remember(key: str, value: str):
    """Agent 主动把重要发现存入外置记忆"""
    memory_store[key] = value

def recall(query: str) -> str:
    """按需检索，只返回语义相关片段"""
    return semantic_search(memory_store, query, top_k=3)
```

上下文只存当前决策需要的内容，历史状态通过语义检索按需注入。这种模式叫 external memory + retrieval injection，对长时间运行的 Agent（五十步以上）或跨会话任务效果特别显著。需要注意的是检索延迟问题：高 QPS 场景下每步都触发向量检索，可能比窗口塞满还要慢，必须在应用层做热点缓存，把高频查询的结果暂存在内存里。

## 策略四：系统提示精简

System prompt 是每轮都占 token 的固定成本，它的体积直接决定了你的有效工作区。一个功能完整的 Agent 系统提示，很容易写到 3000-6000 token，相当于凭空缩小了十几次工具调用的可用空间，而且这个成本完全对 Agent 的推理不产生增量价值。

常见浪费点：把所有工具的完整用法和参数说明都写进系统提示；写了大量 few-shot 示例来教 LLM 如何选工具；加了"你是一个有用、无害、诚实的 AI 助手"这类对 Agent 行为没有实质影响的填充语。

优化方向：工具使用说明改成只在首轮注入，后续轮次不重复；few-shot 示例放进独立的 `get_examples` 工具，Agent 需要时主动调用；通用废话全部删除；工具描述控制在 50 字以内，参数说明用 JSON schema 就够，不要在 prompt 里再重复解释一遍。

## 策略五：思维链预算控制

如果你在用支持 extended thinking 的模型，thinking 块的 token 消耗往往超出预期，而且整个思维链会被追加进对话历史，每轮推理都要携带它。设置 `budget_tokens` 上限是控制这部分成本最直接的手段。

```json
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 1024
  }
}
```

更好的做法是分阶段控制：任务规划阶段和面对歧义时开 thinking，执行阶段（如循环调用固定工具、格式化输出）关掉 thinking。循环执行阶段的步骤本质上是分类或模板填充任务，根本不需要推理链，强行开着 thinking 只是在烧 token。streaming 场景下额外有一个好处：thinking 关掉之后首 token 延迟明显下降，用户感知的响应速度变快了。

## 策略六：流式 Map-Reduce 处理大文件

对于需要处理大文件的 Agent，绝对不要一次性把文件内容塞进上下文。改用 Map-Reduce 模式：把文件切块，每块独立处理，最后聚合结果带入主上下文。

```python
def process_large_file(path: str, chunk_size: int = 2000) -> str:
    summaries = []
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        summary = llm_call(f"请总结以下内容的关键信息：\n{chunk}")
        summaries.append(summary)
    # 只把聚合摘要带入主上下文
    return "\n".join(summaries)
```

Map 阶段可以使用更小更快的模型（比如 Haiku 类），降低每块的推理成本；Reduce 阶段才由主 Agent 模型做最终决策，整体成本和延迟都能压下来。各 chunk 之间没有依赖关系，Map 阶段完全可以并发处理，总耗时取决于最慢的那个块，而不是所有块串行加总，大文件处理速度能提升数倍。

## 踩坑清单

- **坑 1**：只压缩了 tool_result，没压 tool_call，两者要成对处理，否则 token 没减多少。
- **坑 2**：外置记忆检索延迟被算进 Agent 每步耗时，高频调用时反而比塞满上下文更慢，必须在热点查询上加缓存层。
- **坑 3**：system prompt 动态拼接工具列表，每次工具数量变化都导致 prompt cache 失效，要在工具列表固定后再锁定 prompt，避免每轮重新计算缓存。
- **坑 4**：摘要压缩本身消耗 token，压缩频率过高反而不划算，建议超过 70% 窗口才触发，不要每轮都压。
- **坑 5**：用字符数估算 token 数容易偏差 30%-40%，务必用官方 tokenizer 或 API 计数接口，不要靠感觉。
- **坑 6**：大文件 Map 阶段串行处理，块数多时总耗时可观，改并发处理 chunk 可以显著提速，但注意并发量别超过下游 API 的速率限制。

上下文窗口不是越大越好，越大意味着每次推理的计算开销越高、首 token 延迟越长。真正的工程水平在于：用恰好够用的上下文，装正好需要的信息。
