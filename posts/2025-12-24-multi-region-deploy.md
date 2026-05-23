---
layout: post
title: AI 应用 Multi-region 部署 — 全球用户的延迟和可用性
date: 2025-12-24
topic: "工程实战"
tags: [AI, Deployment, Multi-region]
excerpt: 一个 us-east-1 region 部署 LLM 应用，欧亚用户延迟惨。多 region 部署的架构、数据同步、故障切换。
permalink: /posts/2025-12-24-multi-region-deploy.html
---

## 单 region 的痛点

LLM 应用部署在 us-east-1：
- 美国用户：50ms 网络延迟
- 欧洲：120ms
- 亚洲：250ms+
- 加上 LLM 推理 2-5s

亚洲用户 TTFT 体验**比美国差 2 倍**。
出 region 级故障（AWS us-east-1 一年挂 1-2 次），全球用户都受影响。

## 多 region 架构

```
                  ┌─ us-east-1
                  │   - App
Global LB (DNS) ──┼─ eu-west-1     ← 每个 region 独立完整 stack
                  │   - App
                  └─ ap-southeast-1
                      - App
```

每个 region：
- 应用服务（API / Worker）
- 向量库 / 缓存
- LLM provider 调用（就近）

用户根据**地理位置**（DNS / Anycast）路由到最近 region。

## LLM provider 的就近选择

主流 API 都有多区域 endpoint：

| Provider | 区域 |
|---|---|
| **OpenAI** | 默认 us，欧盟 / 日本 endpoint（企业版）|
| **Anthropic** | 跨多区域 / AWS Bedrock 多区域 |
| **Azure OpenAI** | 几十个 Azure region |
| **国产**（DeepSeek / Kimi / 通义）| 国内多区域 |

在 ap-southeast-1 部署的 app 调 LLM 时优先用亚太 endpoint，**节省 100-300ms**。

## 数据同步

应用是 stateless 还好，stateful 数据要同步：

### 1. 用户数据 / Auth

```
Global Postgres（如 AWS RDS Multi-region）/ CockroachDB
所有 region 读写
最终一致 + 强一致选项
```

### 2. 向量库

不太适合"全球同步写"——向量库还没全球分布式版。
策略：
- **主+只读副本**：写都到主 region，其他 region 只读
- **按租户分片**：每个用户的数据只在某个 region（home region）
- **CDN 化的 RAG**：高频文档复制到所有 region，低频跨 region 查询

### 3. 缓存

每 region 独立 Redis / Memcached，不跨区同步。
跨区数据由后端 lazy fetch。

### 4. 日志 / Trace

每 region 写本地日志，**异步汇总到全局分析仓库**（S3 + Athena / BigQuery）。

## Failover 策略

某 region 挂了：

### 方式 1: DNS 切换

```
api.yourapp.com
  健康检查每 30s
  → us-east-1 down → 30s 内 DNS 切到 us-west-2
```

简单但**DNS TTL 缓存**让客户端切换有延迟（几分钟）。

### 方式 2: Anycast IP

所有 region 共用一个 IP，BGP 自动路由到最近的。
故障时其他 region 自动接管。

代价高：需要 BGP / Anycast 基础设施（Cloudflare / AWS Global Accelerator）。

### 方式 3: 客户端 retry

应用 SDK 内置：

```javascript
const endpoints = [
  "https://us.api.com",
  "https://eu.api.com",
  "https://asia.api.com",
];
async function call(req) {
  for (const url of endpoints_sorted_by_distance) {
    try { return await fetch(url, req); }
    catch { continue; }
  }
}
```

最简单，但需要前端 / SDK 配合。

## 测试 Multi-region

要做**故障演练**：

- 每月模拟一次 region 故障
- 看流量自动切到其他 region 多快
- 看用户体验降级是否优雅
- 看数据一致性恢复

不演练 = 真出事时混乱。

## 部署成本

Multi-region = **基础设施 × N**。
3 个 region 部署，基础成本至少 2.5-3 倍。

权衡：
- 用户量大 / 收入高 → 值得
- MVP 阶段 → 单 region 配 CDN 缓存够

中间方案：**App 多 region，DB / 向量库单 region**——
延迟降一半，成本只多 30-50%。

## 实战 stack 参考

```
[CDN] Cloudflare（全球边缘）
  ↓
[App] AWS EKS in 3 regions
  ↓ (跨 region 路由)
[Cache] Redis per region
  ↓
[DB] AWS RDS Postgres multi-region
[Vector DB] Pinecone serverless multi-region
[LLM] Anthropic Bedrock multi-region endpoints
[Log] CloudWatch per region → 异步汇总到 S3 + Athena
```

## 一个朴素结论

> 全球用户 + LLM 应用 = 必须考虑 multi-region。
> 全部都做太贵——分层做：CDN 全球 + App 多区 + Stateful 数据集中 + LLM 就近调用。
>
> 投资回报率最高的是 **CDN + 多 region app**，stateful 数据可以慢慢迁移。
