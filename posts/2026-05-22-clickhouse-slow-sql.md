---
layout: post
title: 视频推荐慢 SQL 治理 — 从 MySQL 汇总查询迁移到 ClickHouse
date: 2026-05-22
tags: [后端, ClickHouse, 慢SQL, 数据迁移]
excerpt: 视频推荐里有几条按天/按维度的汇总查询，在 MySQL 上时延 5 秒打底、告警频繁。把它们整体迁到 ClickHouse 后，p99 从 5s 压到 200ms。
permalink: /posts/2026-05-22-clickhouse-slow-sql.html
---

## 问题长什么样

视频推荐场景里有几条**按天 + 多维度的汇总查询**，长得大致是：

```sql
SELECT vid, SUM(click), SUM(play), SUM(finish), AVG(duration)
FROM stats_video_day
WHERE day BETWEEN ? AND ?
  AND channel IN (...)
  AND category IN (...)
GROUP BY vid
ORDER BY SUM(click) DESC
LIMIT 5000;
```

表里**单分区上亿行**，跨 30 天就要扫几十亿。MySQL 上这条查询：

- p99 时延 **5s+**，p999 经常 10s+
- 高峰期把从库 CPU 顶到 80%，慢查日志告警每天几十条
- 业务侧的"小时榜 / 日榜 / 周榜"刷一次得等

走过的弯路：覆盖索引堆到 6 个字段也只压到 3s，分表也只能换汤不换药——**它本质上是 OLAP，被塞进了 OLTP 引擎**。

## 为什么挑 ClickHouse

OLAP 引擎几个候选：ClickHouse / Doris / StarRocks / Druid。挑 ClickHouse 的原因：

- **列存**：这种 GROUP BY 多维聚合的场景天然吃列存红利
- **MergeTree 排序**：按 `(day, channel, category)` 排序后，这种 WHERE 几乎是 zero-scan
- 单机就能撑出可观的吞吐，初期不用上分布式
- 公司栈里已经有运维经验

## 迁移方案

整体分**三步**走：

### 1. 数据双写（不切流）

业务侧的 stats 写入加一条**异步 sink 到 ClickHouse**，原 MySQL 写入完全不动。
跑一周确认双边数据一致（按天 checksum + 按 vid 抽样）。

### 2. 影子读对账

查询接口加一个 **ShadowRead 开关**：每条查询都同时打到 MySQL + ClickHouse，
返回 MySQL 的结果给业务（保安全），后台异步对比两边的差异写日志。

跑两天，差异率压到万分之 5 以下（差异基本都是 MySQL 主从延迟造成的瞬时不一致）。

### 3. 切流

按"接口 × 灰度比例"切：

- 5% → 验证
- 50% → 验证
- 100% → 完成

每一步留 24h 观察，没问题再升级。

## 几个关键的工程细节

**ClickHouse 表结构**：

```sql
CREATE TABLE stats_video_day_ck (
  day Date,
  vid String,
  channel LowCardinality(String),
  category LowCardinality(String),
  click UInt32,
  play UInt32,
  finish UInt32,
  duration UInt32
) ENGINE = MergeTree
ORDER BY (day, channel, category, vid)
PARTITION BY toYYYYMM(day)
TTL day + INTERVAL 13 MONTH;
```

几个点：

- `LowCardinality(String)`：channel / category 值少，省得多
- `ORDER BY` 顺序按"过滤选择性从高到低"排，命中前缀就能跳过整个 mark
- `TTL` 留 13 个月，过期自动清，省心

**调用方式**：起初用的是 native protocol，但 driver 行为不稳，后来统一换成 HTTP + JSON，配合自家的 SDK 做超时/重试/熔断，**用 GET 不用 POST**（方便日志和 trace）。

## 结果

- p99 时延：**5s → 200ms**
- p999：**10s+ → 600ms**
- MySQL 从库 CPU 高峰从 80% 降到 25%
- 业务侧"日榜/周榜"接口体感秒开
- ClickHouse 单机磁盘占用：约 MySQL 的 1/8（列存压缩）

## 踩过的坑

1. **不要把所有列都塞进 ORDER BY**。命中前缀才有用，全塞反而拖写入
2. **`final` 修饰符**：对 ReplacingMergeTree / CollapsingMergeTree 有用，但很贵。能避免就避免，靠业务层去重
3. **HTTP 调用的连接池**：Go 默认 HTTP client 有连接复用问题，要显式配 `MaxIdleConnsPerHost`
4. **避免大量 INSERT 小 batch**：ClickHouse 每次 INSERT 都是一个 part，太碎会触发 merge 压力，建议至少 1 万行 / 批

## 下一步

- 完成全量切流
- 沉淀**ClickHouse 调用 / 迁移规范**，复用到后续类似治理场景（订单统计、用户活跃榜单等）
- 把告警门禁加上：ClickHouse 查询 p99 / merge 队列长度

---

> 真正的优化不是"换数据库"，而是**搞清楚这条查询的本质属性**。OLAP 的事就交给 OLAP 引擎。
