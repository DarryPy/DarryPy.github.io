---
layout: post
title: RAG 索引热更新实战 — 动态知识库的增量同步与一致性保障
date: 2026-06-16
topic: "RAG 与检索"
tags: [RAG, 向量数据库, 索引更新, 知识库, 生产实践]
excerpt: 静态索引是 RAG 系统最容易被忽视的定时炸弹。文档更新了、索引没跟上，LLM 拿着过期知识作答，用户毫不知情。这篇文章拆解增量同步的三种策略、向量数据库的原生能力差异，以及如何在高频写入场景下保证查询一致性。
permalink: /posts/2026-06-16-rag-index-hot-update.html
---

你上线了一套 RAG 问答系统，平稳运行了两个月，然后用户开始反馈："这个答案是老黄历了。" 你一查，源文档三周前已经更新，但向量索引还停在初始化那天。这不是边缘 case，这是绝大多数生产 RAG 系统的真实状态。

静态索引适合文档几乎不变的场景：法律法规库、历史归档、产品手册第一版。一旦你的知识源是频繁变动的——产品迭代文档、内部 Wiki、实时新闻、故障手册——全量重建索引既慢又浪费，而且会有一段时间让用户拿着半新不旧的数据问答。增量热更新才是正路，但它本身有一套复杂度要处理。

## 为什么全量重建撑不住 🔄

全量重建的逻辑很直觉：每天凌晨把所有文档重新 embedding，清空旧索引，写入新的。对小规模（十万文档以内）、低更新频率（每天变动不超过 5%）的库，这方案没问题，简单可靠，出了事好排查。

但它有三个死角，在规模和频率上去之后就会暴露：

**时效性差**：重建窗口通常是小时级，甚至更长。文档白天更新了，要等到第二天凌晨才能被检索到。对实时性有要求的场景——故障排查手册、动态定价文档、合规政策更新——这个延迟是不可接受的。用户在凌晨两点遇到生产问题，打开 RAG 问答系统，拿到的是三天前的排查步骤，这才是真的定时炸弹。

**资源浪费严重**：如果你有一百万份文档，每天真正变化的可能只有两千份，但你要全部重新 embedding。对于调用 OpenAI embedding API 的团队，这是每天一次的账单燃烧；对于自建 GPU 服务的团队，这是每天一次的显卡占用。随着知识库规模增长，这个浪费会线性放大。

**重建期间存在一致性真空**：从旧索引开始清除，到新索引写满，中间有个时间窗口。短则几分钟，长则几小时。这期间的查询行为不确定：有时打到旧索引里已经删除一半的数据，有时打到还没写完的新索引。用户可能检索到文档片段缺失或语义错乱的结果，而你完全无法复现。

## 增量更新的三种策略

**策略一：事件驱动（推荐作为起点）**

给文档系统挂一个变更通知钩子。Confluence 有 Webhooks，Git 仓库有 post-receive hook，对象存储（S3、MinIO）有事件通知，数据库有 CDC（Change Data Capture）。文档一更新，触发消息进队列，消费者拉到消息后只对该文档做 re-embedding 并更新向量库中对应的记录。

```
Document Updated
      ↓
   Webhook / CDC
      ↓
Message Queue (Kafka / SQS)
      ↓
Embedding Worker Pool
      ↓
  Upsert to VectorDB
```

这个架构的延迟可以做到分钟级，甚至秒级。Kafka 队列天然提供削峰填谷：文档批量入库时，embedding worker 按自己的速度消费，不会把向量库写穿。缺点是依赖上游系统能正确发出事件，也需要保证消息不丢失（at-least-once delivery，消费端做幂等）。

**策略二：轮询 diff（稳健的补偿机制）**

定期扫描数据源，用 `last_modified` 时间戳或内容哈希找出变化的文档，只处理 diff 集合。这个策略不依赖上游推事件，适合数据源不支持 Webhook 或者事件通知不可靠的场景，也常作为事件驱动方案的补偿层——防止消息丢失导致漏更新。

```python
def find_changed_docs(since: datetime) -> list[Doc]:
    return db.query(
        "SELECT id, content, content_hash FROM docs WHERE updated_at > %s",
        (since,)
    )

def sync_changed(last_sync_time: datetime):
    changed = find_changed_docs(last_sync_time)
    for doc in changed:
        cached_hash = index_meta.get_hash(doc.id)
        if doc.content_hash != cached_hash:
            chunks = chunk(doc.content)
            vectors = [embed(c) for c in chunks]
            vector_db.upsert_batch(doc.id, vectors)
            index_meta.set_hash(doc.id, doc.content_hash)
```

轮询频率要根据业务时效需求定，不是越频繁越好。频繁轮询会持续占用数据库读资源，也会触发大量 embedding 调用哪怕什么都没变。通常 5 分钟到 1 小时是合理范围。

**策略三：版本化 shadow 索引（零停机切换）**

写入时不触碰现有索引，而是并行构建一个新版本索引。等新索引构建完成并通过健康检查（随机抽样查询，召回率 / 相关性达标），通过流量权重逐步切换，从旧索引灰度迁移到新索引。代价是短期双份存储开销，但查询侧完全无感知，没有任何一致性真空期。这种方案适合变更量大（比如大规模重新分块）或者切换 embedding 模型的场景。

## 向量数据库原生更新能力对比

不同向量数据库对增量更新的支持差异很大，选型前需要弄清楚几个关键维度：

| 数据库 | Upsert 支持 | 软删除 | 写入后可查延迟 | 一致性模型 |
|--------|------------|--------|--------------|----------|
| Qdrant | 原生 upsert，按 point_id | payload 标记过滤 | < 1 秒 | 最终一致 |
| Weaviate | object-level upsert | 软删除字段 | < 1 秒 | 最终一致 |
| Milvus | upsert（2.3+ 支持） | delete by expr | flush 后（默认 ~60s） | 最终一致 |
| Pinecone | 原生 upsert | 需标记后 metadata 过滤 | < 1 秒（serverless 略高） | 最终一致 |
| pgvector | ON CONFLICT DO UPDATE | PostgreSQL 标准软删除 | 事务提交即可查 | 强一致（MVCC）|

Milvus 的 flush 机制是最常踩的坑：数据写入后先进内存 buffer，flush 到 segment 才变成可查状态。默认 `autoFlushInterval` 可能到 60 秒。如果业务要求写入后立即可查，要显式调 `collection.flush()`，但这会降低写入吞吐，需要权衡。

pgvector 因为走 PostgreSQL MVCC，天然有事务隔离和强一致，是一致性要求最高场景的选择。缺点是纯向量检索性能相对专用向量库弱，适合中小规模（五百万向量以内），且与现有 PostgreSQL 业务库合并部署时有运维红利。

## 一致性保障：写入窗口与查询隔离

增量更新最棘手的是"写到一半"时查询进来怎么办。一篇文档切成多个 chunk，如果 chunk 3 已经更新但 chunk 1 还是旧的，同一次检索可能返回新旧混合的内容，LLM 拼出来的答案会前后矛盾。

**文档级原子性**：给每个 chunk 打上 `doc_version` 字段和 `status` 字段（pending / active）。新 chunk 写入时先标记 pending，全部写完后再原子切换为 active，同时把旧版本 chunk 删除。查询侧永远只查 `status == "active"` 的 chunk。

```python
# 写入阶段：先写新 chunk，状态 pending
for i, chunk in enumerate(new_chunks):
    vector_db.upsert(
        id=f"{doc_id}_v{new_version}_{i}",
        vector=embed(chunk),
        payload={
            "doc_id": doc_id,
            "version": new_version,
            "status": "pending",
            "text": chunk
        }
    )

# 全部写完后原子切换
vector_db.update_payload(
    filter={"doc_id": doc_id, "version": new_version},
    payload={"status": "active"}
)

# 删除旧版本
vector_db.delete(
    filter={"doc_id": doc_id, "version": {"$lt": new_version}}
)
```

**写入队列限流**：大批量更新时控制并发写入速率，建议不超过向量库额定写入 TPS 的 70%，留余量给同时进来的查询请求。Kafka consumer 的 `max.poll.records` 和 embedding worker 的并发线程数是两个主要调节旋钮。

**监控索引新鲜度**：不要只监控写入延迟，还要追踪一个更有业务意义的指标——**索引新鲜度**：当前索引中，各文档的向量版本与源文档最后更新时间之间的差值分布。P95 超过你的 SLA 阈值时告警。这比等用户反馈"答案是旧的"要早发现几个数量级。

## 踩坑清单

- **embedding 模型版本飘移**：升级 embedding 模型必须全量重建，不能局部更新。新旧模型的向量空间不兼容，混在一起检索会让召回率变成随机数。在 chunk payload 里记录 `embedding_model_version`，上线新模型前做 shadow 索引。

- **chunk ID 用位置偏移而非内容哈希**：文档改了几个字导致切分位置偏移，位置偏移生成的 ID 和旧 chunk 对不上，旧 chunk 没被删除，索引里同一内容有两份向量。改用 `sha256(doc_id + chunk_text[:64])` 做 chunk ID，内容不变 ID 就不变，天然幂等。

- **删除文档没联动删向量**：源文档删除不会更新 `last_modified`，轮询 diff 扫不到它。必须在文档删除事件里单独触发向量删除，或者定期做全量 ID 对账（向量库中存在但源库已删除的 doc_id 批量清理）。

- **写入失败没有重试死信队列**：embedding 调用超时或向量库写入报错，消息直接丢弃，该文档永久留在旧版本。消费端必须有重试机制（指数退避）和死信队列，人工排查和重投。

没有新鲜度监控的 RAG 系统，是在用静态快照假装实时检索。动态知识库的增量同步不是锦上添花，是让 RAG 在生产环境长期可信的底线工程。把事件驱动、一致性保障、新鲜度监控都建起来，你的系统才算真正脱离"demo 状态"。核心工程债就在这里——早建早省心。
