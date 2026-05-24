---
layout: post
title: Golden Dataset 建立流程 — Eval 的命根子
date: 2026-03-12
topic: "评估与安全"
tags: [AI, Eval, Dataset]
excerpt: 没有黄金数据集，eval 就是空中楼阁。从采样、标注、版本化到迭代的完整流程。
permalink: /posts/2026-03-12-golden-dataset.html
---

## Golden Dataset 是什么

一份**人工标注好的 (input, expected output)** 列表，作为评估真值。
它是所有 eval 指标的基础——**没有它，accuracy / F1 / faithfulness 都无从计算**。

## 5 步建立流程

### 1. 从线上日志采样

不要凭空想样本。**直接从真实流量里采**：

```sql
SELECT query, response, user_feedback
FROM llm_logs
WHERE created_at > now() - INTERVAL '7 days'
ORDER BY RANDOM()
LIMIT 500;
```

随机采 500 条作为候选池。

### 2. 分层采样

避免长尾被忽视。按维度分层：

```
- 60%：高频常见 case（容易答对的）
- 25%：中等难度
- 15%：长尾 / 边界 / 容易答错的
```

如果只采高频，你会优化错地方——边界 case 永远在隐性失败。

### 3. 人工标注

每条样本标注：
- 期望答案（exact match / partial / 任意可接受 list）
- 关键事实点（用于 faithfulness 计算）
- 反例（"不能出现 X" 的负向要求）
- 难度分级（easy / medium / hard）

**多人交叉标注 + 分歧讨论达成共识**。
单人标注会带个人偏好。

### 4. 版本化

```
datasets/
  v1.0_2026-03-01.jsonl  ← 初版
  v1.1_2026-03-15.jsonl  ← 加了 50 个边界 case
  v2.0_2026-04-01.jsonl  ← 业务变化，重大更新
```

每次跑 eval 时记录用了哪个版本，**跨实验可比**。

### 5. 持续迭代

每月 / 每季度：
- 从生产日志新采样一批，加进来
- 把模型答错的 case 优先标注（hard negative mining）
- 业务变化时大版本升级

## 数据格式

JSONL，每行一个 case：

```json
{
  "id": "case_001",
  "query": "今天上海天气怎么样？",
  "expected_intents": ["query_weather"],
  "expected_entities": {"city": "上海", "date": "today"},
  "must_contain": ["天气", "上海"],
  "must_not_contain": ["明天"],
  "difficulty": "easy",
  "tags": ["intent_classification", "production"]
}
```

字段按任务定制——分类任务存 label，QA 存 expected_answer，RAG 存 expected_context_ids。

## 多少样本够

| 场景 | 推荐量 |
|---|---|
| 早期原型 | 50-100 |
| 上线门禁 | 200-500 |
| 长期监控 | 500-2000 |
| 大版本对比 | 2000+ |

不要超过 5000——超过这个量人工维护质量难保证，**质量 > 数量**。

## 数据治理

- **训练集和评估集严格隔离**：评估集不能进训练
- **版本控制**：用 git LFS / DVC 管理
- **审计**：标注的每一条要追溯到标注者 + 时间
- **隐私**：脱敏后再标注，PII 不能进 dataset

## 一份 checklist

发布 dataset 前确认：

- [ ] 至少 200 条样本
- [ ] 覆盖 60/25/15 难度分层
- [ ] 多人交叉标注，一致性 ≥ 80%
- [ ] 包含边界 case 和反例
- [ ] 严格隔离训练集
- [ ] PII 脱敏
- [ ] 版本化 + git 管理

## 一个朴素结论

> "eval 准不准" 80% 取决于 golden dataset 的质量。
> 在工具和指标上花的功夫，远不如在 dataset 上花的功夫值。
>
> 没建 golden dataset 之前，**所有 "改进了 X%" 都是错觉**。
