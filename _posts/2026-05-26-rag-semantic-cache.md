---
layout: post
title: RAG Semantic Cache 实战 — 让检索又快又省钱
date: 2026-05-26
topic: "RAG 与检索"
tags: [RAG, Semantic Cache, 向量检索, 工程优化]
excerpt: 生产环境中大量 RAG 查询其实高度相似，直接命中语义缓存可以跳过 embedding + 向量检索全流程，把 P99 latency 从秒级压到毫秒级，同时砍掉 40-60% 的 API 调用成本。这篇文章用可落地的代码告诉你怎么搭。
permalink: /posts/2026-05-26-rag-semantic-cache.html
---

你在生产上跑 RAG 一段时间之后，大概率会遇到一个让人头疼的现象：同一个知识库每天被问几百次"怎么申请报销""产品支持哪些语言""合同条款在哪里"，每次都完整跑一遍 embedding 调用 → 向量检索 → rerank → LLM 生成，钱花了、延迟高，结果还一模一样。你看着监控里每秒几十次的 API 调用，心里清楚至少一半是重复劳动，却不知道从哪里下手。这不是算法问题，是工程设计问题。

Semantic Cache（语义缓存）就是专门解这个问题的。它不是简单的字符串哈希缓存，而是把查询转成 embedding，用向量相似度判断"这个问题我之前回答过没有"，有就直接返回缓存结果，没有才走全流程。对无状态的事实型问答来说，这是性价比最高的优化手段之一——不改业务逻辑，不换模型，纯粹在入口处加一层语义过滤，就能显著降低成本和延迟。

---

## 为什么普通 Cache 不够用

传统 key-value 缓存依赖精确字符串匹配。"怎么申请报销"和"报销流程是什么"是两个完全不同的 key，命中率趋近于零。即便你做了 query 的小写标准化、去空格处理，面对中文的多样化表达，效果依然惨淡。

语义缓存的核心思路是：**把查询文本映射到 embedding 空间，用余弦相似度或内积判断语义等价性**。只要两个问题的 embedding 距离低于阈值，就视为同一问题，直接复用缓存的答案。"怎么申请报销""报销怎么操作""走报销流程需要哪些步骤"在 embedding 空间里彼此很近，一次正式回答可以服务这三个问题。

在不同 RAG 应用场景下，普通缓存和语义缓存的命中率差距相当显著：

| 场景 | 普通缓存命中率 | Semantic Cache 命中率 |
|------|--------------|----------------------|
| 客服 FAQ（问法多样） | < 5% | 40-70% |
| 代码文档问答 | 8-15% | 30-50% |
| 长尾知识库问答 | < 2% | 10-25% |
| 固定报表查询 | 60-80% | 80-95% |

命中率越高，收益越明显——embedding API 调用减少、向量检索跳过、LLM token 完全不消耗，只有一次小模型的 embedding 调用成本。客服场景如果命中率能做到 50%，相当于你的 RAG 运营成本直接减半，而用户感知到的响应速度提升更明显，因为缓存命中的路径几乎没有网络往返延迟。

---

## Semantic Cache 的工作原理

整个流程可以拆成三个阶段，理解清楚这三步，后面写代码就水到渠成。

**第一步：查缓存。** 用户 query 进来，先调用 embedding 模型生成 query vector，在缓存向量索引里做 ANN（近似最近邻）检索，找出最相似的历史 query 及其对应答案。这一步只有一次 embedding 调用，成本远低于完整 RAG 流程。

**第二步：判阈值。** 如果最高相似分 ≥ 设定阈值（比如 0.92），则认为语义等价，直接返回该 query 对应的缓存答案，整个 RAG pipeline 跳过。如果低于阈值，继续走后续流程。阈值是这套系统里最核心的超参数，它直接决定了准确性和命中率之间的平衡，下文会专门讲如何标定。

**第三步：写缓存。** 如果未命中，走正常 RAG 流程，拿到最终答案后，把 `(query_vector, answer, metadata)` 写入缓存向量索引，供后续相似问题复用。写入是异步操作，不影响当前请求的响应时间。

阈值的设置没有放之四海而皆准的数字。太低（< 0.85）容易误命中，把不相关问题的答案当作正确答案返回，这比慢更致命；太高（> 0.95）则命中率接近字符串缓存，失去了语义理解的意义。**0.90-0.93 是大多数中文知识问答场景的甜点区**，建议从 0.92 开始，根据真实业务数据迭代调整。

---

## 实战：基于 Redis + OpenAI Embedding 搭建 Semantic Cache

Redis Stack 内置了 HNSW 向量索引，是搭建 Semantic Cache 最省事的选择——你不需要额外部署一套向量数据库，Redis 同时承担缓存存储和向量检索两个角色，运维成本低。

**安装依赖与启动 Redis Stack：**

```bash
pip install redis openai numpy
# Redis Stack 包含向量搜索模块，不能用普通 redis-server 替代
docker run -d -p 6379:6379 redis/redis-stack-server:latest
```

**核心实现代码：**

```python
import json
import hashlib
import numpy as np
from openai import OpenAI
from redis import Redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

client = OpenAI()
r = Redis(host="localhost", port=6379, decode_responses=False)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CACHE_INDEX = "rag_semantic_cache"
SIMILARITY_THRESHOLD = 0.92


def create_cache_index():
    """初始化 Redis 向量索引，已存在则跳过"""
    try:
        r.ft(CACHE_INDEX).info()
    except Exception:
        schema = [
            TextField("query_text"),
            TextField("answer"),
            VectorField(
                "embedding",
                "HNSW",
                {"TYPE": "FLOAT32", "DIM": EMBED_DIM, "DISTANCE_METRIC": "COSINE"},
            ),
        ]
        r.ft(CACHE_INDEX).create_index(
            schema,
            definition=IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH),
        )


def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(input=text, model=EMBED_MODEL)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def cache_lookup(query: str) -> str | None:
    """查缓存，命中返回答案，未命中返回 None"""
    q_vec = get_embedding(query)
    q_bytes = q_vec.tobytes()

    redis_query = (
        Query("*=>[KNN 1 @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("query_text", "answer", "score")
        .dialect(2)
    )
    results = r.ft(CACHE_INDEX).search(redis_query, query_params={"vec": q_bytes})

    if results.total == 0:
        return None

    top = results.docs[0]
    # Redis COSINE 距离：0 = 完全相同，命中条件是 distance < (1 - threshold)
    distance = float(top.score)
    if distance < (1 - SIMILARITY_THRESHOLD):
        return top.answer

    return None


def cache_write(query: str, answer: str):
    """将 query + answer 写入缓存，设 7 天 TTL"""
    q_vec = get_embedding(query)
    key = f"cache:{hashlib.md5(query.encode()).hexdigest()}"
    r.hset(
        key,
        mapping={
            "query_text": query,
            "answer": answer,
            "embedding": q_vec.tobytes(),
        },
    )
    r.expire(key, 60 * 60 * 24 * 7)  # 7 天后自动失效


def rag_with_cache(query: str, rag_pipeline_fn) -> str:
    """包装函数：命中缓存直接返回，否则走 RAG 全流程并写缓存"""
    cached = cache_lookup(query)
    if cached:
        return cached

    answer = rag_pipeline_fn(query)
    cache_write(query, answer)
    return answer
```

这里有一个值得注意的细节：Redis 的 COSINE 距离返回的是**距离**（0 代表完全相同），而不是相似度（1 代表完全相同），所以命中判断条件是 `distance < (1 - threshold)`，初次接触容易搞反。`rag_pipeline_fn` 是你现有的检索加生成函数，完全不需要改动，Semantic Cache 以零侵入的方式包装在外层。

---

## 命中率优化与缓存失效策略

搭建起来只是第一步，真正决定收益的是持续优化命中率，以及在知识库更新时正确处理缓存失效。

**提升命中率的实用手段**

第一，对 query 做轻量级标准化，用小模型（`gpt-4o-mini` 或 `claude-haiku`）把口语化表达改写成标准问句，去掉指代词"这个""那个"，替换成具体实体名称。这个操作成本极低，但能有效减少因表达习惯差异导致的向量偏移，实测命中率提升 10-20%。

第二，分域建立独立缓存索引。产品文档、法务合规、财务报销各用一个 Redis 索引，避免跨域误命中——"合同条款"在法务和财务两个语境下语义相近但答案完全不同，放在同一个索引里风险很高。

第三，定期分析缓存未命中的 query 分布。如果某一类问题反复未命中，说明这个问法在知识库里是真正的新问题，需要补充内容；如果相似度始终在 0.88-0.91 之间徘徊，可能是阈值偏高，可以适当下调。把未命中率作为日常监控指标，设置告警阈值，当未命中率突然飙升时往往意味着用户意图发生了漂移或者知识库内容出现了空白，这是及时发现问题的好机会。

**知识库更新后的失效策略**

知识库更新是 Semantic Cache 最容易踩坑的场景，缓存里存的是旧知识的答案，用户拿到的是过期信息，比慢更糟糕。

对更新频率低的内容（法规文档、产品手册），用 TTL 兜底就够了——写入时设 3-7 天过期时间，定期全量更新知识库后不需要额外操作，旧缓存自然淘汰，实现成本极低。对更新频繁的内容，写入缓存时同时记录关联的文档 ID；知识库文档被修改时，主动查询并删除所有关联该文档 ID 的缓存条目，做到精准失效，不会影响其他正常条目的命中。实时性要求极高的场景（当日新闻、实时价格）不适合上 Semantic Cache，直接跑全流程更安全，正确性优先于速度。

---

## 踩坑清单

- **阈值没经过真实数据标定就上线**：抽 300-500 条真实查询对，人工标注"这两个问题是否等价"，画 precision-recall 曲线，找 F1 最优点再定阈值，不要拍脑袋
- **缓存了含动态信息的答案**："明天天气"、"当前汇率"、"今天有没有新公告"——这类 query 要在入口处做 intent 分类，动态 intent 绕过缓存直走 RAG
- **Embedding 模型升级后忘记清空缓存**：`text-embedding-3-small` 和 `text-embedding-3-large` 的向量空间不兼容，混着查会出奇怪的相似度分数，每次切模型必须清空索引重建
- **只缓存最终答案，不缓存检索来源**：部分场景需要展示引用来源文档，写缓存时把 `retrieved_chunks` 和文档链接一起存进去，否则来源追溯断掉
- **把 Semantic Cache 当银弹**：多跳推理、个性化问答、依赖用户历史上下文的场景，命中缓存反而带来错误答案，做好场景分层，不是所有 query 都适合走缓存

Semantic Cache 的本质是承认一个朴素的事实：**大多数用户问的是同样的问题**。接受这个事实，在工程层面利用它，你的 RAG 系统能在不改任何业务逻辑的情况下，把成本和延迟同时打下来一个数量级。
