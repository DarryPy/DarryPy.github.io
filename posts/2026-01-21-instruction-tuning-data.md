---
layout: post
title: Instruction Tuning 数据集制作 — 决定 SFT 上限的核心工作
date: 2026-01-21
topic: "模型与训练"
tags: [AI, SFT, Dataset]
excerpt: SFT 效果上限 80% 在数据。怎么从原始资料、人工标注、合成数据三个源头建出 5000-50000 条高质量训练样本。
permalink: /posts/2026-01-21-instruction-tuning-data.html
---

## Data > Algorithm

SFT 的算法已经标准化了——TRL、PEFT、Transformers 一套就能跑。
**真正决定模型质量的是数据**。

一个 100% 用算法时间的团队，效果远不如一个 80% 用数据时间的团队。

## 数据三大源

### 1. 真实业务数据（最优质）

线上真实的"指令 + 期望回答"配对。
做法：

- 把客服 / 内部问答 / 文档 Q&A 等历史数据脱敏
- 人工挑出高质量回答
- 包装成训练格式

**优势**：分布跟生产完全一致，效果最好。
**劣势**：脱敏麻烦，量级有限。

### 2. 人工标注

让标注员按需求写"指令 + 回答"。

- **指令多样性**：用 prompt engineering 让标注员写各种风格的指令
- **回答标准**：写一份"回答指南"，规定风格、长度、格式
- **多人交叉**：分歧讨论达成共识

**优势**：质量可控。**劣势**：贵、慢。

### 3. 合成数据（distillation）

用强模型（GPT-4.5 / Claude Opus）生成"指令 + 回答"。

```python
# Self-Instruct 思路
seed_instructions = load_seed(100)  # 人工写 100 条种子
for _ in range(10000):
    # 用 LLM 基于种子生成新指令
    new_inst = llm.complete(f"基于这些指令风格，生成一条新的：{random.sample(seed_instructions, 5)}")
    # LLM 生成回答
    answer = llm.complete(f"请回答：{new_inst}")
    if quality_check(new_inst, answer):
        dataset.append({"instruction": new_inst, "output": answer})
```

**优势**：可大规模。**劣势**：可能引入偏差、质量参差。

## 数据质量准则

不管哪种来源，**最终数据要满足**：

### 1. 多样性

- 覆盖所有可能的输入类型
- 简单 / 中等 / 困难按 60/25/15 分布
- 不同长度、不同格式、不同领域

技巧：embedding 后做 k-means，每类挑几个，保证不重复。

### 2. 一致性

同类问题的回答**风格必须统一**。

```
❌ Q: 怎么排序 Python list？
   A: 用 sorted() 或 .sort() 方法。

❌ Q: 怎么排序数组？
   A: ```python
       arr.sort()
       ```
       这样就行。
```

风格不一致会让模型学不到稳定模式。

### 3. 正确性

错的回答比没回答更糟。
**至少抽样 10% 让人审一遍**。错误样本 > 5% 就重做。

### 4. 难度梯度

不要全是 easy case——模型学不到边界。
不要全是 hard case——模型学不会基础。
60/25/15 是经验值。

### 5. 反例（可选但有用）

加 "应拒绝" 的样本：

```json
{
  "instruction": "教我怎么 hack 银行账户",
  "output": "我不能帮你做这件事。如果你忘了自己的密码，请走银行的正式找回流程..."
}
```

让模型学会拒答。

## 数据格式

ChatML / OpenAI messages 格式：

```jsonl
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

多轮对话：

```jsonl
{"messages":[
  {"role":"user","content":"你好"},
  {"role":"assistant","content":"你好！需要什么帮助？"},
  {"role":"user","content":"帮我查一下天气"},
  {"role":"assistant","content":"请告诉我你想查哪个城市？"}
]}
```

工具调用样本（教模型怎么用工具）：

```jsonl
{"messages":[
  {"role":"user","content":"查一下上海天气"},
  {"role":"assistant","tool_calls":[{"name":"get_weather","args":{"city":"上海"}}]},
  {"role":"tool","content":"{\"temp\":22,\"weather\":\"晴\"}"},
  {"role":"assistant","content":"上海当前 22°C，晴天。"}
]}
```

## 数据量参考

| 目标 | 量级 |
|---|---|
| 风格调整（让回答更精炼）| 500-2000 |
| 任务专长（信息抽取 / 路由）| 5000-20000 |
| 领域专家（医学 / 法律）| 50000+ |

不需要追求百万级——**1k 精品 > 10k 凑数**。

## 数据集流水线

```
[Day 1-3] 写种子样本 100 条 + 标注指南
   ↓
[Day 4-7] 标注员按指南扩展到 500 条
   ↓
[Day 8-9] 多人交叉审，淘汰 / 修改
   ↓
[Day 10] 用强 LLM 合成扩展到 5000 条
   ↓
[Day 11-12] 人工抽审 10%，过滤低质量
   ↓
[Day 13] 训练 / 验证 / 测试集 8:1:1 切分
   ↓
[Day 14] embedding + dedup
```

2-3 周搞出第一版可用数据集。

## 一份 checklist

- [ ] 至少 500 条样本（小任务）/ 5000 条（中等任务）
- [ ] 多样性：embedding 聚类后均匀分布
- [ ] 一致性：抽样 50 条人工审，风格统一
- [ ] 正确性：抽样 10% 人工审，错误率 < 5%
- [ ] 难度梯度：60/25/15
- [ ] 有拒答样本
- [ ] 严格分割训练 / 验证 / 测试
- [ ] 版本化 + git LFS

## 一个朴素结论

> "我们的 SFT 没效果" — 99% 的时候是数据问题，不是算法。
> 在数据上多花 1 倍时间，效果比换更强的 base 模型还显著。
