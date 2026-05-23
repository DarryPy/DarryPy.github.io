---
layout: post
title: Vector DB 备份与恢复 — 别等服务挂了才想起来
date: 2025-12-26
topic: "工程实战"
tags: [AI, Vector DB, Backup]
excerpt: RAG 系统的向量库挂了 = 业务挂了。备份策略、跨区域复制、灾难恢复演练，一份运维必读。
permalink: /posts/2025-12-26-vector-db-backup.html
---

## 为什么单独讲

普通数据库备份大家都知道。
向量库备份有 3 个特殊性：

1. **数据量大**：1 千万向量 × 1024 维 × 4 字节 = 40GB，再加 metadata 翻倍
2. **重建成本高**：丢了重新 embedding 入库可能要几天 + 几千块成本
3. **索引大 + 重建慢**：HNSW / IVF 索引重建可能比导入数据本身更慢

不备份的代价比想象的高。

## 三个备份层次

### Layer 1: 原始数据备份（最优先）

向量库里存的向量是**派生数据**——可以从原始文档重新 embed。
所以最重要的备份是**原始文档 + embedding 配方**：

```
backup:
  - documents/             # 原始 PDF / Markdown / DB
  - chunks.jsonl           # 切片后的文本 + metadata
  - embeddings_config.yml  # embedding 模型 + 参数
```

最坏情况下从这里能 recreate 一切。

### Layer 2: 向量 + Metadata 备份

```
backup:
  - vectors.npy           # 向量本体
  - metadata.parquet      # 关联 metadata
  - id_mapping.json       # 跟原始文档的关系
```

不存索引（索引可重建），存数据本身。
恢复时重新 build index。

### Layer 3: 完整快照（含索引）

每家向量库的快照工具：

| 库 | 快照 |
|---|---|
| **Qdrant** | `qdrant_snapshot.tar` |
| **Weaviate** | filesystem-based / S3 backup module |
| **Milvus** | etcd + MinIO 快照 |
| **Pinecone** | "collection snapshot" 商业功能 |
| **pgvector** | Postgres dump |

完整快照恢复最快，但**体积最大**。

## 频率策略

```
[每日] 全量原始数据 (Layer 1) → 推 S3
[每周] 向量数据 (Layer 2) → 推 S3
[每月] 完整快照 (Layer 3) → 推 S3
[实时增量] write-ahead log / change streams
```

不同层不同频率，**总存储 = N × 单次 + 增量**。

## 跨区域复制

单地区备份还不够——区域级故障（电源 / 网络 / 自然灾害）。

```
主区域：us-east-1（生产）
备份区域：us-west-2 + eu-west-1（异地）
```

跨区域复制方案：

- **Pinecone**：付费选项 multi-region
- **Weaviate**：通过 S3 cross-region replication
- **自部署**：rsync / restic / cloud-native（Velero）

## 增量 vs 全量

大向量库每天全量备份不现实（几小时）。增量策略：

```
[每日] 增量：只备份新增/更新的向量
       记录哪些被删除（tombstone）

[每周] 合并：把 7 天增量合并成新基线
```

恢复时：基线 + 增量重放 → 完整状态。

## 灾难恢复演练（DR Drill）

**备份不演练 = 没备份**。

每季度模拟一次：
1. 拉一份最新备份到隔离环境
2. 完整恢复
3. 跑 eval 测试集：召回率 / 数据完整性
4. 测恢复时间：从命令开始到业务可用要多久

最低标准：
- **RPO (Recovery Point Objective)**：能接受丢多少数据 → 决定备份频率
- **RTO (Recovery Time Objective)**：能接受停多久 → 决定恢复速度

业务级 RTO < 4 小时 / RPO < 1 小时是基本要求。

## 监控

```yaml
metrics:
  - last_backup_timestamp        # 最近一次备份多久前
  - backup_size_bytes
  - backup_duration_sec
  - restore_test_success         # 最近一次恢复演练结果
  - replication_lag_sec          # 跨区域延迟

alerts:
  - last_backup > 48h            # 一天半没备份了
  - replication_lag > 1h         # 跨区域延迟过大
  - restore_test_failure         # 演练失败
```

## 一个朴素结论

> "我有备份" 跟 "我的备份能用" 是两件事。
>
> 没演练过的备份不算备份。
> 单一区域的备份不算备份。
> 只备向量不备原始数据，恢复也慢。
>
> RAG 系统的向量库要按生产数据库的标准做备份运维。
