---
layout: post
title: LLM 应用的 A/B Testing 框架 — 怎么科学比较两个版本
date: 2026-01-17
topic: "工程实战"
tags: [AI, A/B Testing, Eval]
excerpt: 改 prompt / 换模型 / 改 RAG 真的更好吗？不做 A/B 测试都是猜。LLM 应用 A/B 的特殊考虑和工程实现。
permalink: /posts/2026-01-17-ab-testing-llm.html
---

## LLM 应用 A/B 跟普通不一样

普通 web A/B：看转化率 / CTR / 点击。
LLM A/B 额外要看：

- **回答质量**：好不好（主观）
- **任务完成率**：用户问题被解决了吗
- **后续 follow-up rate**：回答好的话用户不需要追问
- **拒绝率 / 错答率**
- **成本 + 延迟**：质量好但慢 10 倍 / 贵 10 倍能接受吗

不能只看一个指标。

## 流量分配

每个用户 ID 哈希到一个组，保证一致性：

```python
def assign_variant(user_id, experiment_id):
    h = hashlib.md5(f"{user_id}:{experiment_id}".encode()).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < 50:
        return "control"
    return "treatment"
```

同一用户每次都被分到同一组，**不会反复横跳**——这对 LLM 应用尤其重要（不同 prompt 输出差异大）。

## 指标体系

按重要性排：

### 1. North Star（业务核心）

每次实验都看的指标。例：
- "用户问题是否被解决"（thumbs up / down）
- "对话是否在 3 轮内结束"（结束 = 满意）
- "转化率"

这是判断"实验成功 / 失败"的最终指标。

### 2. Guardrail（不能崩）

不能让这些跌：
- 延迟 p99
- 成本 / 请求
- 错误率
- 安全事件（jailbreak / 不当输出）

如果 guardrail 崩了，**就算 north star 涨也得停**。

### 3. Diagnostic（诊断）

帮你理解为什么：
- 平均 token 数
- 重复率
- 工具调用次数
- 用户编辑提交内容的频率

## 实验设计

### 1. 假设清晰

```
H0：新 prompt 跟旧 prompt 用户满意度一样
H1：新 prompt 用户满意度 > 旧 prompt
```

### 2. 样本量 / 时长

跑多久才能下结论？计算公式：

```python
# 简化版
def required_sample_size(baseline_rate=0.5, mde=0.05, alpha=0.05, power=0.8):
    # 检测 5% 绝对提升，95% 置信，80% 功效
    # 用 statsmodels.stats.proportion.samplesize_proportions_2indep_onetail
    return ~1500  # 例
```

对 LLM 应用：**至少跑 1-2 周或 1000+ 用户/组**，避免周末效应等噪声。

### 3. 显著性检验

- 比例类（thumbs up rate）：z-test 或 chi-square
- 连续类（latency）：t-test 或 Mann-Whitney U
- 多指标：考虑 Bonferroni 修正

## 工具

| 工具 | 适合 |
|---|---|
| **LangSmith Experiments** | LangChain 栈，集成 trace |
| **Braintrust** | 跟 prompt 迭代深度整合 |
| **GrowthBook** | 通用 feature flag + A/B |
| **Statsig** | 商业级 + 自动 stats |
| **自建（Postgres + Python notebook）** | 完全可控 |

## LLM 特殊：离线 + 在线双层

LLM A/B 不应该只看在线：

```
Level 1: 离线 eval（pre-launch）
  跑 golden dataset → 比较核心指标 → 通过门禁
       ↓
Level 2: 灰度 A/B（5% → 50% → 100%）
  线上小流量 → 真实用户反馈 → 决定是否扩大
```

离线挡掉 80% 的"明显更差"，节省线上实验成本。

## 实战流程

```
[Day 0] 实验设计：假设 / 流量 / 指标 / 样本量
[Day 1] 离线 eval：跑 golden dataset
[Day 2] 5% 灰度上线
[Day 3-7] 看 guardrail，崩了停
[Day 8] 50% 灰度
[Day 9-14] 看 north star 显著性
[Day 15] 决策：扩大 100% / 回滚 / 重新设计
```

## 常见坑

### 1. Peeking（偷看）

每天看结果，看到显著就停 → **假阳性率高**。
解法：预先定好样本量，到了才看。或用顺序检验（sequential test）。

### 2. Novelty Effect

新版上线用户出于新鲜感互动多，假象"提升"。
解法：跑足够长时间（≥ 2 周），看是否稳定。

### 3. 维度切片

总体没差异，但某类用户差异很大。
解法：按用户分群分别看（地域 / 付费状态 / 使用频率）。

### 4. 缓存污染

控制组和实验组共用缓存 → 互相影响。
解法：缓存 key 加 variant 区分。

## 一个朴素结论

> 没 A/B 的 LLM 优化都是"自我感觉良好"。
> 真正的工程化：**离线 eval 挡 80% + 在线 A/B 验证最后 20%**。
>
> 建一套 A/B 框架的投入，会被未来每一次 prompt 改动反复收回。
