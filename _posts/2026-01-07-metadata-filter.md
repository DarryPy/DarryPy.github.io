---
layout: post
title: RAG Metadata Filter 设计 — 检索准不准的隐性瓶颈
date: 2026-01-07
topic: "RAG 与检索"
tags: [AI, RAG, Metadata]
excerpt: 90% 的检索查询其实是"按维度过滤 + 向量相似"。设计好 metadata schema，比换 embedding 模型涨点快。
permalink: /posts/2026-01-07-metadata-filter.html
---

## 90% 的检索是 hybrid query

业务里真实的检索很少是"全库找相似"：

- "找我团队 2025 年的文档" — 按 team + year 过滤
- "找官方 API 文档关于支付的" — 按 source + topic 过滤
- "找用户 X 上传的 PDF" — 按 user_id + file_type 过滤

向量相似 + metadata filter 缺一不可。
**filter 设计差**，再好的 embedding 也拉不回来。

## 必有的 metadata

每个 chunk 入库时至少要带：

```json
{
  "doc_id": "doc_abc",
  "chunk_id": "doc_abc_c3",
  "title": "...",
  "source": "official_docs",
  "language": "zh-CN",
  "tags": ["payment", "api"],
  "created_at": "2026-01-05T10:00:00Z",
  "updated_at": "2026-01-05T10:00:00Z",
  "author": "...",
  "section": "API > Authentication",
  "user_id": "...",  // 如果是用户私有数据
  "version": "v2.3"
}
```

设计原则：

### 1. 经常过滤的字段必须 indexed

```sql
CREATE INDEX ON chunks (user_id);
CREATE INDEX ON chunks (source);
CREATE INDEX ON chunks (language);
```

否则 filter 加在向量检索后变成全表扫，**比单纯向量检索还慢**。

### 2. 时间字段必备

让用户能查"最近 30 天"、"2025 年之后"。
**时间 filter 比向量信号有时更强**——用户说"最新"通常就是想要最新。

### 3. 来源 / 权威性

```json
{ "source_type": "official" | "community" | "user_generated" }
```

回答时优先官方文档，过滤掉低质量来源。

### 4. 多租户必备：tenant_id / user_id

```json
{ "tenant_id": "company_42", "user_id": "u_xyz" }
```

防止用户 A 检索到用户 B 的数据。**多租户场景的安全底线**。

## Filter 优先于向量

很多向量库支持 pre-filter（先过滤再向量）和 post-filter（先向量再过滤）：

- **pre-filter**：先 metadata filter 筛掉，再向量。当 filter 强（命中数 <10%）时**快**
- **post-filter**：先向量 top-K，再 filter。当 filter 弱时快

但 post-filter 有个坑：top-K 可能全被 filter 砍掉，最终拿不到任何结果。

Pinecone / Weaviate / Qdrant 都默认 pre-filter，这是对的。
自建（pgvector）注意 EXPLAIN 看执行计划。

## Hybrid Query 实例

```python
# Qdrant
client.search(
    collection_name="docs",
    query_vector=embed(query),
    query_filter=Filter(
        must=[
            FieldCondition(key="user_id", match=MatchValue(value="u_42")),
            FieldCondition(key="language", match=MatchValue(value="zh-CN")),
            FieldCondition(key="created_at", range=Range(gte=cutoff)),
        ]
    ),
    limit=10,
)
```

```sql
-- pgvector
SELECT id, content
FROM chunks
WHERE user_id = $1
  AND language = 'zh-CN'
  AND created_at >= $2
ORDER BY embedding <=> $3
LIMIT 10;
```

## 让 LLM 自动决定 filter

复杂场景：让 LLM 把自然语言 query 解析成 filter + 检索 query：

```python
def parse_query(user_query):
    prompt = f"""
把用户问题拆成:
1. 用于向量检索的核心 query
2. 元数据 filter

用户问题：{user_query}

输出 JSON: {{
  "search_query": "...",
  "filters": {{ "language": "...", "date_range": [...], "tags": [...] }}
}}
"""
    return llm.complete(prompt, response_format="json")
```

例：
"找去年发布的 Python 库文档" 
→ search_query: "Python 库", filters: {date_range: [2025-01-01, 2025-12-31], type: "library_doc"}

这种"query understanding" 是高级 RAG 的标配。

## 反模式

### 1. metadata 太丰富但没人用

20 个字段全存了，查询时一个都不用。**只是占空间**。
设计前先想"这字段会被 filter 吗？"

### 2. metadata 用动态 / 高基数字段

```
❌ "tags": ["user-uploaded-2026-01-07-abc-xyz"]   // 每条都不一样
```

filter 失去意义。**用归一化 / 受控词表**。

### 3. 把"内容"塞 metadata 当 filter

```
❌ filter: { "contains": "支付接口" }
```

这是检索功能（在向量层做）。
**metadata 应该是离散值或时间 / 数值类型**。

## 一份 metadata 设计 checklist

- [ ] 每个 chunk 有 doc_id + chunk_id
- [ ] 必要的 user_id / tenant_id 做多租户隔离
- [ ] 时间字段（created_at / updated_at）
- [ ] 来源字段（source_type / authority）
- [ ] 语言字段
- [ ] 关键 filter 字段 indexed
- [ ] tags 用控制词表
- [ ] 高基数字段不进 metadata

## 一个朴素结论

> 90% 的 RAG 检索是 hybrid query。
> Metadata schema 设计 = RAG 工程的隐性命门，**比换 embedding 模型涨点快**。
>
> 上线前想清楚"用户会按哪些维度筛"，再决定 metadata 字段。
