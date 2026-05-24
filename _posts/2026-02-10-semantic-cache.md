---
layout: post
title: Semantic Cache — 让 LLM 应用省 30-50% 调用
date: 2026-02-10
topic: "工程实战"
tags: [AI, Cache, 性能]
excerpt: 用户问"今天天气怎么样"和"今天天气如何"——LLM 看是两个 query。Semantic Cache 让它们命中同一份结果。
permalink: /posts/2026-02-10-semantic-cache.html
---

## 普通缓存的盲区

传统缓存按字符串 hash：

```
key = hash("今天天气怎么样")
```

完全一样才命中。但 LLM 应用里用户问法多变：
- "今天天气怎么样"
- "今天天气如何"
- "今儿天儿咋样"
- "what's the weather today"

普通缓存命中率近 0%。

## Semantic Cache 的思路

按**语义相似度**而非字面匹配：

```
1. query 来了，先做 embedding
2. 在缓存里找相似度 > 0.95 的旧 query
3. 命中 → 返回缓存的 response
4. 没命中 → 调 LLM，写入缓存
```

实测在 FAQ / 客服场景能砍 30-50% LLM 调用。

## 最小实现

```python
class SemanticCache:
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.store = []  # [(embedding, query, response, ts)]
    
    def get(self, query):
        q_emb = embed(query)
        for emb, q, r, _ in self.store:
            if cosine(q_emb, emb) > self.threshold:
                return r
        return None
    
    def set(self, query, response):
        self.store.append((embed(query), query, response, time.now()))
```

生产用 Redis + RediSearch 或专门向量库（Qdrant / Milvus），不要 in-memory list。

## 阈值很重要

- 太松（0.85）：相似但不同含义的 query 误命中，**返回错答案**
- 太严（0.99）：几乎跟字符串 hash 一样，没意义
- 甜区：**0.92-0.96**

不同场景调不同：
- FAQ：0.95（容忍变化少）
- 创意 / 对话：缓存意义不大，别用
- 检索类：0.93-0.95

## 哪些 query 不该缓存

并不是所有 LLM 调用都该缓存：

| 类型 | 缓存？ |
|---|---|
| 静态事实问答（"什么是 RAG"）| ✅ |
| FAQ（产品 / 公司）| ✅ |
| 实时数据（"现在股价"）| ❌ |
| 用户个性化（"我的订单"）| ❌（带 user_id 区分）|
| 创意生成（"给我讲个笑话"）| ❌ |

加 metadata 控制：

```python
@cache(ttl=300, scope="static")  # 仅静态类用 5 分钟缓存
def llm_call_static(prompt): ...

@cache(ttl=0)  # 不缓存
def llm_call_creative(prompt): ...
```

## TTL 策略

- **静态知识**：1-7 天
- **新闻类**：1 小时
- **用户偏好**：会话期内
- **关键事实**（产品定价）：手动失效，不依赖 TTL

设错了 TTL 比不缓存更糟——用户拿到过期信息。

## 加 metadata 做精确缓存

很多时候不是纯语义相似就行——还要按 user / language / version 区分：

```python
def cache_key(query, user_id, model, system_prompt_version):
    embedding = embed(query)
    return {
        "embedding": embedding,
        "filter": {
            "user_id": user_id,
            "model": model,
            "spv": system_prompt_version,
        }
    }
```

向量库（如 Qdrant）支持 embedding + metadata filter 的组合查询。

## 现成工具

- **GPTCache**：开源 semantic cache 框架，集成 Redis / Milvus
- **LangChain Cache**：内置 semantic cache 选项
- **Helicone Cache**：商业 API，proxy 级缓存
- **自建**：pgvector + Redis 也够

## 一个朴素结论

> Semantic Cache 对 FAQ / 客服 / 文档问答场景**省钱效果立竿见影**。
> 但要小心：缓存出错比不缓存更糟——**用对 scope + TTL + filter**。
>
> 关键决策类调用永远不缓存。
