---
layout: post
title: "知识图谱 + RAG 集成实战 — 结构化与非结构化信息的最强组合"
date: 2026-07-04
topic: "RAG 与检索"
tags: [AI, RAG, Knowledge Graph]
excerpt: 纯向量 RAG 搜不到关系型知识。"张三是哪家公司的 CEO？这家公司的竞争对手是谁？"——这类问题需要知识图谱。
permalink: /posts/2026-07-04-kg-rag-integration.html
---

## 纯向量 RAG 的盲区

向量相似度搜索本质上是**文本相似度**。它擅长找到语义相近的段落，但对关系型查询无能为力：

- "张三管理的团队里有哪些成员？" → 需要遍历关系边
- "这个产品的所有上下游依赖是什么？" → 需要图遍历
- "A 公司的竞争对手的最新新闻是什么？" → 需要先找竞争对手（图），再找新闻（向量）
- "哪些用户同时购买了 X 和 Y？" → 关系型聚合

这些问题在知识图谱里是简单的图查询，在向量数据库里几乎无解。

## 知识图谱基础

KG 的基本单元是**三元组（Triple）**：

```
(主体，关系，客体)
(Subject, Predicate, Object)

例子：
(马斯克, CEO_of, 特斯拉)
(特斯拉, 竞争对手, 比亚迪)
(特斯拉, 成立于, 2003年)
(比亚迪, 总部, 深圳)
```

所有三元组组成一张图。查询时，顺着关系边遍历。

## 集成方案一：KG 增强检索

**思路**：先查 KG 找到相关实体，把实体信息加入 RAG 上下文。

```
用户问题："介绍一下比亚迪的主要竞争对手及其最新动态"

Step 1: 实体识别
  → 识别出 "比亚迪"

Step 2: KG 查询
  → 查询 (比亚迪, 竞争对手, ?)
  → 得到：[特斯拉, 吉利, 广汽, 蔚来]

Step 3: 向量检索
  → 用 "特斯拉 最新动态"、"吉利 最新动态" 等检索文档库

Step 4: 合并上下文
  → KG 关系 + 检索到的文档 → 传给 LLM
```

```python
from neo4j import GraphDatabase
from openai import OpenAI
import anthropic

# Neo4j 连接
neo4j_driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)
oai = OpenAI()
claude = anthropic.Anthropic()

def query_kg_for_relations(entity: str, relation: str = None, depth: int = 1) -> list[dict]:
    """查询 KG 中实体的关系"""
    with neo4j_driver.session() as session:
        if relation:
            result = session.run(
                "MATCH (e:Entity {name: $name})-[r:RELATION {type: $rel}]->(target) "
                "RETURN target.name as target, r.type as relation",
                name=entity, rel=relation
            )
        else:
            result = session.run(
                "MATCH (e:Entity {name: $name})-[r]->(target) "
                "RETURN target.name as target, type(r) as relation "
                "LIMIT 20",
                name=entity
            )
        return [{"target": r["target"], "relation": r["relation"]} for r in result]

def extract_entities_from_query(query: str) -> list[str]:
    """用 LLM 从用户问题里提取实体"""
    response = claude.messages.create(
        model="claude-haiku-4-5",
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": f"""从下面的问题中提取需要查询的关键实体名称（人名、公司名、产品名等）。
只输出实体列表，用逗号分隔，不要解释。

问题：{query}

实体："""
        }]
    )
    entities_str = response.content[0].text.strip()
    return [e.strip() for e in entities_str.split(",") if e.strip()]

def embed(text: str) -> list:
    return oai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def kg_augmented_rag(user_query: str, doc_chunks: list[dict]) -> str:
    """
    KG 增强的 RAG：
    1. 从 query 提取实体
    2. 查 KG 得到关联实体
    3. 用所有实体做向量检索
    4. 合并 KG 关系 + 检索结果给 LLM
    """
    # Step 1: 提取实体
    entities = extract_entities_from_query(user_query)
    print(f"识别到实体: {entities}")

    # Step 2: 查 KG
    kg_facts = []
    expanded_entities = list(entities)
    for entity in entities:
        relations = query_kg_for_relations(entity)
        for r in relations:
            kg_facts.append(f"({entity}) --[{r['relation']}]--> ({r['target']})")
            expanded_entities.append(r['target'])
    expanded_entities = list(set(expanded_entities))

    # Step 3: 向量检索（用扩展后的实体列表）
    import numpy as np
    expanded_query = user_query + " " + " ".join(expanded_entities)
    query_emb = np.array(embed(expanded_query))

    scored_chunks = []
    for chunk in doc_chunks:
        chunk_emb = np.array(chunk["embedding"])
        sim = np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb))
        scored_chunks.append((sim, chunk["text"]))
    scored_chunks.sort(reverse=True)
    top_docs = [text for _, text in scored_chunks[:5]]

    # Step 4: 合并给 LLM
    kg_context = "\n".join(kg_facts) if kg_facts else "无相关图谱信息"
    doc_context = "\n\n---\n\n".join(top_docs)

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""请回答以下问题。

<knowledge_graph_facts>
{kg_context}
</knowledge_graph_facts>

<retrieved_documents>
{doc_context}
</retrieved_documents>

<question>
{user_query}
</question>

综合以上信息，给出详细准确的回答："""
        }]
    )
    return response.content[0].text
```

## 集成方案二：KG 约束生成（事实核查）

**思路**：LLM 生成答案后，用 KG 验证关键事实。

```python
def kg_grounded_generation(user_query: str, context: str) -> str:
    """用 KG 对 LLM 输出做事实验证"""
    # 第一步：让 LLM 生成答案
    draft = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": f"背景：{context}\n\n问题：{user_query}"}]
    ).content[0].text

    # 第二步：提取答案中的声明（claims）
    claims_response = claude.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"""从以下回答中提取可验证的事实声明，以 JSON 列表输出。
每条格式：{{"subject": "主体", "predicate": "关系", "object": "客体"}}

回答：{draft}

事实声明列表："""
        }]
    ).content[0].text

    import json
    try:
        claims = json.loads(claims_response)
    except:
        return draft  # 解析失败直接返回原答案

    # 第三步：逐条查 KG 验证
    verified = []
    unverified = []
    for claim in claims:
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (s:Entity {name: $sub})-[r:RELATION {type: $pred}]->(o:Entity {name: $obj}) "
                "RETURN count(*) as cnt",
                sub=claim.get("subject", ""),
                pred=claim.get("predicate", ""),
                obj=claim.get("object", "")
            )
            count = result.single()["cnt"]
            if count > 0:
                verified.append(claim)
            else:
                unverified.append(claim)

    # 第四步：如果有未验证的声明，标注或重新生成
    if unverified:
        caveat = f"\n\n注意：以下声明未能在知识库中验证，请自行核实：" + \
                 "\n".join(f"- {c['subject']} {c['predicate']} {c['object']}" for c in unverified)
        return draft + caveat

    return draft
```

## 集成方案三：从文本构建 KG（Text-to-KG）

**思路**：用 LLM 从文档中抽取三元组，构建 KG，然后查询。

```python
def extract_triples_from_text(text: str) -> list[dict]:
    """用 LLM 从文本提取三元组"""
    response = claude.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""从以下文本中提取实体关系三元组，以 JSON 数组输出。
每条格式：{{"subject": "主体", "predicate": "关系类型", "object": "客体"}}

只提取明确陈述的关系，不要推断。关系类型用简短动词短语描述。

文本：
{text}

三元组："""
        }]
    ).content[0].text

    import json, re
    try:
        # 提取 JSON 部分
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    return []

def build_kg_from_documents(documents: list[str]):
    """从文档列表构建 KG，存入 Neo4j"""
    all_triples = []
    for doc in documents:
        # 分段处理长文档
        chunks = [doc[i:i+1500] for i in range(0, len(doc), 1500)]
        for chunk in chunks:
            triples = extract_triples_from_text(chunk)
            all_triples.extend(triples)

    print(f"提取到 {len(all_triples)} 条三元组")

    # 写入 Neo4j
    with neo4j_driver.session() as session:
        for triple in all_triples:
            session.run(
                """
                MERGE (s:Entity {name: $subject})
                MERGE (o:Entity {name: $object})
                MERGE (s)-[r:RELATION {type: $predicate}]->(o)
                """,
                subject=triple.get("subject", ""),
                predicate=triple.get("predicate", ""),
                object=triple.get("object", "")
            )
    print("KG 构建完成")
```

## 使用 GraphRAG（微软开源方案）

微软的 GraphRAG 是 Text-to-KG + RAG 的完整实现：

```bash
pip install graphrag

# 初始化项目
mkdir my_rag && cd my_rag
python -m graphrag.index --init --root .

# 配置 settings.yml（填入 API key 等）

# 把文档放入 input/ 目录，然后建索引
python -m graphrag.index --root .

# 查询（两种模式）
# local: 精确检索，适合细节问题
python -m graphrag.query --root . --method local "特斯拉的主要竞争对手是谁？"

# global: 全局推理，适合总结性问题
python -m graphrag.query --root . --method global "这些公司的整体竞争格局如何？"
```

GraphRAG 的 global 查询是它的核心优势：通过图上的社区结构做全局摘要，回答"整体上…"类的问题，这是纯向量 RAG 做不到的。

## KG+RAG vs 纯 RAG：何时用哪个

| 查询类型 | 纯向量 RAG | KG+RAG |
|---|---|---|
| "解释 X 是什么" | 好 | 差不多 |
| "X 和 Y 有什么关系" | 差 | 好 |
| "X 的所有 Y 是什么" | 差 | 好 |
| "最近关于 X 的新闻" | 好 | 差不多 |
| "整体上，这个领域…" | 差 | 好（GraphRAG global）|
| 构建成本 | 低 | 高（需要抽取三元组）|
| 维护成本 | 低 | 高（需要更新 KG）|

**结论**：如果你的核心查询是关系型的，加 KG；如果主要是语义检索，纯向量就够。

## 一个朴素结论

> KG 和向量检索不是替代关系，是互补关系。
>
> 先建纯向量 RAG，遇到关系型查询答不好时，再考虑加 KG。
> 不要一开始就同时建两套，维护成本会压死你。
