---
layout: post
title: RAG + Text2SQL 混合检索 — 结构化与非结构化数据一站式问答
date: 2026-07-21
topic: "RAG 与检索"
tags: [RAG, Text2SQL, 混合检索, 向量数据库, 结构化数据]
excerpt: 企业里的知识从来不只住在 PDF 里——订单量、GMV、库存这些数字藏在数据库里。把向量检索和 Text2SQL 拼在一起，才能让 RAG 真正回答"去年 Q4 哪个品类退货率最高"这种复合问题。
permalink: /posts/2026-07-21-rag-text2sql-hybrid-retrieval.html
---

你的用户问了一句话："上个月华东区的 TOP10 SKU 销售额是多少，跟产品说明书里的卖点对上了吗？"

前半句要查数据库——拿实际销售数字。后半句要检索文档——找产品介绍里写了什么卖点。纯 RAG 答不了前半句，纯 SQL 答不了后半句。这种复合问题在企业场景里比你想象的多得多，而且往往是最有价值的那类问题。RAG + Text2SQL 混合检索，就是专门为这类问题设计的架构模式。

## 为什么单打独斗都不够 🔀

向量检索的优势在于语义模糊匹配。你问"有没有关于退货流程的说明"，embedding 相似度搜一搜，能从几十份政策文档里找到相关段落。但你问"2026 年 6 月退货率超过 5% 的 SKU 有哪些"，向量数据库给不了答案——它存的是文本语义向量，不是行列事实，根本无法做精确数值过滤和聚合计算。

SQL 数据库反过来，数字绝对精确，但它听不懂自然语言。用户问"哪些商品卖得好"，你得先把这句话翻译成 `SELECT sku_id, SUM(amount) FROM orders GROUP BY sku_id ORDER BY SUM(amount) DESC LIMIT 10`，还得知道表叫什么、字段叫什么，用户根本不该学这些。

两种能力的边界非常清晰，而且恰好互补：

| 场景类型 | 向量检索 | Text2SQL |
|---|---|---|
| 政策 / 流程 / 文档问答 | ✅ 擅长 | ❌ 无法处理 |
| 精确数字与聚合查询 | ❌ 无法处理 | ✅ 擅长 |
| 时间范围筛选 | ❌ | ✅ 擅长 |
| 多字段模糊语义匹配 | ✅ 擅长 | ❌ |
| 结果需要引用原文来源 | ✅ 有出处 | ⚠️ 需额外处理 |
| 跨多表 JOIN 分析 | ❌ | ✅ 擅长 |

混合架构的核心思路是：**先判断这个问题走哪条路，或者两条路都走再合并结果**。听起来简单，工程细节却不少。

## 架构设计：Router + Executor + Merger

一个可落地的混合检索系统分三层，每层职责清晰，不要混在一起。

**Router（意图路由）**

用一个专门的 prompt 让 LLM 把用户 query 分类：`sql`、`vector`、`both`。分类依据是数据源的描述——路由 prompt 要带上数据库的 schema 摘要（表名 + 字段名 + 业务注释）和向量库的 collection 列表及说明，让 LLM 有实际依据做判断，而不是凭感觉猜。

```python
ROUTER_PROMPT = """
你是一个查询路由器。根据用户问题和以下数据源描述，决定走哪条检索路径。

数据库 schema（结构化数据）:
{db_schema_summary}

向量库 collections（非结构化文档）:
{vector_collection_desc}

用户问题：{query}

输出 JSON：
{{"route": "sql" | "vector" | "both", "confidence": 0.0-1.0, "reason": "..."}}
"""
```

`reason` 字段不要省——它既是调试的入口，也是后续 Merger 理解两路结果主次关系的依据。`confidence` 低于阈值时强制走 `both`，宁可多查一路。

**Executor（双路执行）**

SQL 路由走 Text2SQL：把 query 和精选后的 schema 传给 LLM 生成 SQL，沙箱执行，返回结构化行列结果。向量路由走标准 RAG pipeline：embedding 检索召回候选，Reranker 精排，取 top-k。

`both` 路由时，两路并行执行，不要串行——串行的话延迟是两路之和，并行只需等最慢那路：

```python
import asyncio

async def execute_both(query: str):
    sql_task = asyncio.create_task(run_text2sql(query))
    vec_task = asyncio.create_task(run_vector_retrieval(query))
    sql_result, vec_result = await asyncio.gather(
        sql_task, vec_task, return_exceptions=True
    )
    # 某一路出错不影响另一路
    return sql_result, vec_result
```

**Merger（结果合并）**

两路结果要合成一个连贯的回答，核心原则只有一条：**数字来自 SQL，语义解释来自向量，不允许 LLM 凭空编数字**。Merger prompt 里要明确标注每段内容的来源，防止 LLM 把向量文档里找到的"去年退货率约 3%"和 SQL 查出的"今年 5.2%"混在一起叙述，让用户以为是同一个数据。

## Text2SQL 落地的三个高频坑

Text2SQL 看上去直接，实际踩坑多。以下三个是最常见的。

**坑一：schema 太大，LLM 迷路**

企业数据库少则几十张表，多则几百甚至上千张。全量 schema 塞进 context，不仅超 token 限制，LLM 还会在大量无关表里迷路，生成错误的 JOIN 路径。解法是做 **schema 向量化**——把每张表的名称、业务注释、字段列表拼成一段文本，建向量索引；查询时先用 query 去检索最相关的 5-10 张表，只把这些表的 DDL 传给 LLM。

```python
# 建 schema 索引
for table in all_tables:
    doc = f"表名：{table.name}\n业务含义：{table.comment}\n主要字段：{', '.join(f'{c.name}({c.comment})' for c in table.columns)}"
    schema_index.upsert(text=doc, metadata={"table_name": table.name})

# 每次 Text2SQL 前先检索相关 schema
relevant_tables = schema_index.search(query, top_k=8)
filtered_ddl = "\n\n".join(get_ddl(t) for t in relevant_tables)
```

这一步能把传入 LLM 的 schema token 量从几万压到几百，同时显著提升生成 SQL 的准确率。

**坑二：生成的 SQL 执行报错，没有恢复机制**

LLM 生成的 SQL 有时字段名拼错、表名搞反、函数用法不对。如果只是报错返回给用户，体验很差。正确做法是加自动修复循环：捕获执行报错，把错误信息和原始 SQL 一起传回 LLM，让它自己分析改正，最多重试 2 次。2 次仍失败就走降级逻辑，返回"当前问题无法通过数据库查询回答，请尝试换个问法"。

```python
async def run_text2sql_with_retry(query: str, schema: str, max_retries: int = 2):
    sql = await generate_sql(query, schema)
    for attempt in range(max_retries + 1):
        try:
            return await execute_sql_safely(sql)
        except SQLError as e:
            if attempt == max_retries:
                return None  # 降级
            sql = await fix_sql(sql, error=str(e), schema=schema)
```

**坑三：SQL 注入风险被忽视**

LLM 生成 SQL 是动态拼接的过程，有人会故意在 query 里注入 `'; DROP TABLE orders; --`。防御措施有两层：第一，给 Text2SQL 配专用的只读数据库账号，最坏情况也只是读数据；第二，生成 SQL 后做静态分析，拒绝包含 `DROP`、`DELETE`、`UPDATE`、`INSERT`、`ALTER` 等写操作关键词的 SQL，直接报错不执行。这两层加一起，风险基本可控。

## 路由精度怎么提高

路由分错了，后面一切白费。几个实测有效的方法。

**规则前置兜底**：LLM 路由之前，先跑一层简单规则。query 里出现"多少""数量""金额""排名""同比""环比""增长""下降"等词，给路由 LLM 传 `hint: sql`；出现"怎么""为什么""说明""政策""规定""流程""介绍"等词，传 `hint: vector`。这些 hint 不覆盖 LLM 的最终判断，只是补充信号，实测能把明显分错的比例降低 30-40%。

**Few-shot 样本比 zero-shot 强很多**：路由 prompt 里放 15-20 条带正确路由标签的样本，要覆盖边界 case，比如"最畅销商品的用户好评说了什么"（答案是 `both`：SQL 查最畅销，vector 查好评内容）。实测 few-shot 准确率从 ~80% 提到 ~94%。

**置信度兜底**：路由输出带置信度，低于 0.7 的一律走 `both`，宁可多查一路多花点时间，不要因为路由摇摆就答错。

**持续收集错误样本**：把所有路由决策记录到日志，用户给差评的对话重点排查路由是否出错。发现新的错误模式就加进 few-shot 样本里，形成闭环。这是最便宜、收益最高的优化手段。

## 一个容易被忽略的细节：结果缓存分层

SQL 查询结果和向量检索结果的缓存策略是不一样的，不要用同一套逻辑统一处理。

SQL 结果的时效性取决于数据更新频率。库存、订单这类高频数据，缓存 TTL 设 30 秒到 2 分钟；产品上架状态、价格这类中频数据可以缓存 5-10 分钟；历史统计类（比如上月销售排名）基本不变，可以缓存几个小时甚至一天。缓存 key 用 SQL 语句的 hash 值，命中直接返回，不重复打数据库。

向量检索结果的缓存则要注意 query 的语义归一化。用户问"退货怎么办"和"怎么退货"，语义一样但文本不同，如果直接用 query 文本做 key，会缓存两份相同的结果。解法是先把 query 转成 embedding，在一个轻量的语义缓存索引里找相似的历史 query（cosine 相似度 > 0.95），命中就直接返回历史结果。这个技巧在 RAG Semantic Cache 里详细讲过，混合系统里同样适用。

两路结果的缓存层独立，不要共用同一个 Redis key 空间，否则 TTL 策略会互相干扰，很难排查问题。

## 生产部署清单

上线前逐条过一遍，每条都有真实的血泪教训：

- [ ] Text2SQL 使用专用只读数据库账号，禁止写操作权限
- [ ] SQL 执行设置超时上限（建议 5 秒），超时直接返回降级提示
- [ ] SQL 结果行数上限（超过 200 行必须聚合后再传给 LLM，不能全量传）
- [ ] 生成 SQL 的静态检查，过滤写操作关键词
- [ ] Schema 向量索引在表结构变更后自动更新（接 DDL 变更事件）
- [ ] 路由决策写入日志（query + route + confidence + reason）
- [ ] 向量检索 top-k 不超过 10，Reranker 精排后传 top-3 给 LLM
- [ ] Merger 回答里标注数据来源，数字来源标"数据库实时数据"
- [ ] 前端展示 SQL 查询语句（折叠展示），增加用户对数字的信任感
- [ ] 添加用户反馈按钮，收集路由误判样本用于持续优化

---

RAG + Text2SQL 最大的价值不是技术新颖，而是让 LLM 终于能接住企业里最高频的那类问题——"数字加解释"。路由做准，SQL 安全加固，剩下的都是迭代细节。别在架构还没稳的时候就急着堆功能，先把这两块打结实。
