---
layout: post
title: Vector Database 选型对比 — Pinecone / Weaviate / Qdrant / Milvus / pgvector
date: 2026-05-09
category: "RAG 与检索"
tags: [AI, 向量数据库, RAG, 工程实践]
excerpt: 主流向量数据库横评，按规模、运维、成本、生态打分，附一份决策表。
permalink: /posts/2026-05-09-vector-database-comparison.html
---

## 为什么向量数据库重要

RAG 的瓶颈在检索，检索的瓶颈在**向量库**。
选错了：要么慢、要么贵、要么扛不住量级。
2026 年市场上能用的有十来个，初看眼花，挑出来其实只有几个核心问题：

- 量级（百万 / 千万 / 亿）
- 部署模式（托管 / 自部署）
- 团队栈（Postgres / 独立服务 / k8s）
- 预算

## 主流选手

### 1. Pinecone（托管，省心款）

**特点**：
- 全托管 SaaS，几行 API 起步
- 自动扩缩容
- 多副本、跨区域
- 提供 hybrid search（dense + sparse 一站式）

**适合**：
- 小团队、不想运维向量库
- 初创 / MVP 阶段
- 量级 < 5000 万

**坑**：
- 贵——百万级以上账单很可观
- 数据出境（如果合规敏感不能用）
- 自定义能力有限

### 2. Weaviate（自部署 / 云，功能全）

**特点**：
- 开源 + 云托管两种
- GraphQL + REST 双接口
- 内置 hybrid search、modular embeddings
- 多租户支持很好

**适合**：
- 数据合规要求自部署
- 需要 hybrid search 但不想自己拼
- 中等规模（千万级）

**坑**：
- 资源占用偏大
- 索引重建相对慢

### 3. Qdrant（Rust，性能怪兽）

**特点**：
- Rust 写的，**单机性能在主流里最猛**
- 内置 payload filtering（结构化过滤 + 向量检索）
- 量化支持好（binary / int8）
- 部署轻量

**适合**：
- 自部署 + 性能敏感
- 中大规模（千万到亿级）
- 需要复杂 metadata filter

**坑**：
- 生态比 Weaviate 小一点
- 多机集群方案相对年轻

### 4. Milvus（云原生，大规模专用）

**特点**：
- 分布式架构（Kubernetes-native）
- 真正能扛十亿、百亿级
- 多种索引算法可选（HNSW / IVF / DiskANN）
- 完整的高可用方案

**适合**：
- 量级 > 1 亿
- 已经在 k8s 上跑业务
- 内部有数据基础设施团队

**坑**：
- 部署复杂（不适合小团队）
- 不当配置容易踩性能坑

### 5. pgvector（Postgres 插件）

**特点**：
- 直接在已有的 Postgres 加一个插件
- 不引入新的基础设施
- SQL 一把梭：`SELECT ... WHERE category = 'X' ORDER BY embedding <=> $1 LIMIT 10`
- 事务、备份、ACID 跟 Postgres 一致

**适合**：
- 已经在用 Postgres
- 量级 < 1000 万
- 团队 Postgres 经验丰富

**坑**：
- 量级上去后性能跟不上专用向量库
- HNSW 在 pgvector 里调优空间有限

### 6. ChromaDB（嵌入式，原型首选）

**特点**：
- 嵌入式数据库，零运维
- Python 一行 `pip install chromadb` 起步
- 适合本地开发 / 桌面应用

**适合**：
- 原型开发
- Jupyter notebook 试验
- 量级 < 100 万

**坑**：
- 不是生产级，扩展性有限

## 决策矩阵

| 场景 | 推荐 |
|---|---|
| 原型 / Demo / Jupyter | ChromaDB |
| 中小规模 + 不想运维 | Pinecone |
| 已经在用 Postgres + 不想多引入一套 | pgvector |
| 自部署 + 中等规模 + 功能要全 | Weaviate |
| 自部署 + 极致性能 | Qdrant |
| 亿级以上 + 有 k8s 运维 | Milvus |

## 容易被忽略的关键指标

挑选向量库不要只看"我能存多少"，更要看：

### 1. 写入吞吐

批量 ingestion 阶段，多大的 throughput？
百万级文档 embedding 入库，**有的库要 1 小时，有的库要一整天**。

### 2. 索引重建时间

embedding 模型升级后要全量重建索引——10 分钟 vs 12 小时的体验天差地别。

### 3. Metadata Filter 性能

实际查询里 90% 是"按 user_id / region / category 过滤 + 向量相似"。
**filter 实现差的库在这种 hybrid query 上慢到崩**。
要测：`WHERE user_id = X AND vector similarity` 的延迟。

### 4. 量化支持

千万级以上必上量化。
看库支不支持 binary / int8 量化，以及量化后的精度损失。

### 5. 监控 / 可观测性

prod 环境向量库必须有：
- p50 / p99 查询延迟
- 索引使用率
- 内存 / 磁盘占用

## 选型流程

1. **先估算量级**：documents 数 × embedding 维度 × bytes
2. **看团队栈**：有 Postgres？有 k8s？有数据团队？
3. **看预算**：托管 vs 自部署的 TCO 算一下
4. **跑 POC**：拿真实数据 50 万-100 万条做 benchmark
5. **测 hybrid query**：不要只测纯向量检索的延迟

## 一个朴素建议

**先用 pgvector 跑通，量级真起来再迁**。

很多团队一上来就上 Milvus，结果数据 10 万条，运维负担大于 10 倍。
等真的撑不住了再迁移，那个时候业务也长大了，迁移成本相对就低了。

技术选型最大的成本是**复杂度**。复杂度是负债，不是资产。
