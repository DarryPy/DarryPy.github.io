---
layout: post
title: Self-RAG 与 CRAG — 让 RAG 自纠错
date: 2026-02-24
topic: "RAG 与检索"
tags: [AI, RAG, Self-RAG, CRAG]
excerpt: 普通 RAG 检索回片段就直接喂模型，不管片段是否相关、是否够用。Self-RAG 和 CRAG 加上"自我评估 + 自纠正"环节。
permalink: /posts/2026-02-24-self-rag-crag.html
---

## 普通 RAG 的盲信

```
Query → Retrieval → top-K 片段 → LLM 生成
```

这里有 3 个隐含假设：
1. 检索到的片段一定相关
2. 一定够回答问题
3. LLM 一定基于片段回答

**实际上这 3 个都可能假**。Self-RAG / CRAG 就是给这 3 步加自我检验。

## Self-RAG

Cornell + Allen AI 2023 提出。让模型在生成时输出 **reflection tokens** 控制流程：

| 特殊 token | 含义 |
|---|---|
| `[Retrieve]` | 现在需要检索吗？ |
| `[IsRel]` | 检索到的片段相关吗？ |
| `[IsSup]` | 我的回答被片段支持吗？ |
| `[IsUse]` | 这段回答有用吗？ |

模型按需触发检索，每段输出后自评，不达标的内容会重新生成或检索。

```
User: 简介一下马斯克
Model: [Retrieve] yes
       → 检索"马斯克"
       [IsRel] yes
       生成: "马斯克是 SpaceX 和 Tesla 的 CEO..."
       [IsSup] yes
       [IsUse] yes
       → 继续
       [Retrieve] yes
       → 检索"马斯克 出生地"
       ...
```

需要专门 fine-tune 模型才能输出这些 token。

## CRAG (Corrective RAG)

CRAG 更轻量——**不改模型**，加一个评估器和纠错路径：

```
Query
  ↓
检索 → top-K 片段
  ↓
Retrieval Evaluator（轻量模型）评分:
  - Correct（相关且充分）→ 直接用
  - Ambiguous（可能不够）→ 用 + 触发网搜补充
  - Incorrect（无关）→ 弃用，纯靠网搜
  ↓
LLM 生成
```

关键组件：

### Retrieval Evaluator

一个小模型（比如 T5-large fine-tune 过的）打三档分数。
判断错误代价比让 LLM 用了错片段小得多。

### Web Search Fallback

本地知识库没找到 → 上网搜（DuckDuckGo / Brave / Tavily 等 API）。
保证总能拿到相关信息。

### Knowledge Refinement

把检索到的长片段进一步切碎、过滤、压缩，只保留跟 query 相关的句子。

## 实战代码骨架

```python
def crag(query):
    candidates = retrieve(query, top_k=5)
    score = retrieval_evaluator(query, candidates)
    
    if score == "correct":
        context = candidates
    elif score == "ambiguous":
        web = web_search(query)
        context = candidates + web
    else:  # incorrect
        context = web_search(query)
    
    # 进一步精炼
    refined = refine_knowledge(query, context)
    
    return llm_generate(query, refined)
```

## CRAG vs Self-RAG

| | Self-RAG | CRAG |
|---|---|---|
| 改模型 | 需要 fine-tune | 不需要 |
| 实现复杂度 | 高 | 中 |
| 效果 | 强但贵 | 接近 + 便宜 |
| 工程友好 | 一般 | 好 |

**生产首选 CRAG**——不用动模型，加一个评估器就能用。

## 工程坑

1. **评估器自己也会错**——不要 100% 信，留兜底
2. **网搜很贵**——只在必要时触发，加缓存
3. **延迟**：多一步评估 + 可能多一步网搜，整体多 500-1500ms
4. **网搜内容质量参差**——加一道过滤（黑名单网站 / 内容审核）

## 一个朴素结论

> 普通 RAG = "盲目相信检索结果"。
> Self-RAG / CRAG = "知道自己可能错，自检 + 自补"。
>
> 复杂 QA / 知识更新快的场景，**CRAG 比普通 RAG 强一个段位**。
