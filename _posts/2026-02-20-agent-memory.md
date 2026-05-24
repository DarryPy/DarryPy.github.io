---
layout: post
title: Agent Memory 管理 — 短期、长期、语义、过程
date: 2026-02-20
topic: "Agent 与工具"
tags: [AI, Agent, Memory]
excerpt: 没记忆的 agent 跟金鱼一样。短期 context、长期 vector store、过程性技能记忆，怎么搭层级才合理。
permalink: /posts/2026-02-20-agent-memory.html
---

## Agent 需要几种记忆

人类有不同记忆类型，agent 也需要分层：

| 类型 | 时间跨度 | 存哪 | 例子 |
|---|---|---|---|
| **Working Memory（短期）** | 当前会话 | LLM context | 这轮对话历史 |
| **Episodic Memory（事件）** | 几天-几周 | 数据库 / 向量库 | 跟某用户聊过什么 |
| **Semantic Memory（语义）** | 长期 | 向量库 + 知识图谱 | 学到的事实 / 偏好 |
| **Procedural Memory（过程）** | 长期 | 代码 / 工具库 | 怎么做某种任务的步骤 |

不分层 = 全塞 context 里，**贵且乱**。

## 1. Working Memory：context 管理

最直接的"记忆"——当前会话的 history。
但 context 不能无限增长：

### 滑动窗口

只保留最近 N 条消息。简单但丢老信息。

### 摘要压缩

老消息用 LLM 压缩成一段摘要替换：

```
[摘要] 过去 10 轮：用户在咨询信用卡换卡，已确认末四位 1234，
已下单新卡，等待发货。
[最近 3 轮原文]
...
```

### 检索式

老消息存进向量库，按当前 query 检索 top-K 塞回 context。
**长会话场景比全量塞省 70% token**。

## 2. Episodic Memory：跨会话回忆

agent 要记住"这个用户上次跟我聊过 X"：

```python
class EpisodicMemory:
    def add(self, user_id, session_summary):
        # 存进向量库
        vector_store.upsert(
            user_id=user_id,
            text=session_summary,
            embedding=embed(session_summary),
            timestamp=now(),
        )
    
    def recall(self, user_id, query, k=3):
        return vector_store.search(
            filter={"user_id": user_id},
            query=query,
            k=k,
        )
```

每个会话结束后压缩成 100-300 字摘要存起来，下次该用户来时按当前问题检索。

## 3. Semantic Memory：学到的事实

agent 在交互中学到的"通用知识"：

```
[Semantic Memory]
- 用户 user_42 偏好简短回答（不超过 100 字）
- 公司 X 的财年是 7 月-6 月
- "P0 bug" 在我们语境里指阻断性 bug
```

跟 Episodic 区别：Episodic 是"具体事件"，Semantic 是"抽象事实"。
存法：可以用键值对，也可以用向量库带高质量 metadata。

更新策略：每隔 N 个会话用 LLM 总结，提炼出新的 semantic 条目；陈旧的标记淘汰。

## 4. Procedural Memory：技能库

"怎么做某事"的步骤。Agent 反复在同类任务上跑后，可以**沉淀经验**：

```
[技能：处理用户退款请求]
1. 验证身份
2. 查询订单
3. 判断是否在退款窗口
4. 如可退：调 refund API
5. 如不可退：解释原因，引导客服
```

实现：每个技能一份 markdown，agent 遇到新任务前先检索相关技能加载。
让 agent **越用越强**，而不是每次从零。

## 实战：Mem0 / LangMem / 自建

- **Mem0**：开源 agent memory 框架，自动管理上面 4 种
- **LangMem**：LangChain 团队的内存方案
- **自建**：Postgres + pgvector 也够

90% 场景自建够用。轮子的优势是"开箱即用 + 内置最佳实践"。

## Context 优先级

每次组装 prompt 时按优先级塞：

```
1. System prompt（不变）
2. Semantic memory 中相关条目（少量、稳定）
3. Procedural skills（当前任务相关）
4. Episodic memory（跟当前 query 相关的过去会话）
5. Working memory（当前会话历史 / 摘要）
6. Current user message
```

每一层都有 token 预算，超出按优先级砍。

## 一个朴素结论

> Agent 没 memory = 每次都新人；
> 有简单 memory = 能记住几句；
> 有 4 层 memory = 真的"会学"。
>
> 长生命周期 agent 必须分层管 memory。
