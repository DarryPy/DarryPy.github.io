---
layout: post
title: RAG 上下文压缩实战 — 用更少 Token 传递更多信息 🗜️
date: 2026-07-14
topic: "RAG 与检索"
tags: [RAG, 上下文压缩, LLMLingua, Token优化, 检索增强]
excerpt: 检索回来的 chunks 塞满上下文窗口，但真正有用的可能只有 30%。本文拆解四种 RAG 上下文压缩路线，带你用更少的 Token 喂给 LLM 更干净的信息，降成本同时提升回答质量。
permalink: /posts/2026-07-14-rag-context-compression.html
---

你从向量库里取回 Top-5 chunks，每块 512 tokens，一共 2560 tokens 的上下文。结果 LLM 给出的回答平平，还偶尔出现幻觉——问题往往不是"检索没检到"，而是"塞进去的废话太多"。

上下文压缩（Context Compression）就是解决这个问题的一类技术：在把 chunks 送给 LLM 之前，先过滤、精简、裁剪，只留下真正与问题相关的片段。这不仅能降低推理成本，实测还能提升回答准确率——因为 LLM 的注意力被迫集中到更干净的信息上。

## 为什么 Top-K 检索结果充满"噪音"？

理解压缩之前，先搞清楚噪音从哪来。

一个典型的 RAG 管道把原始文档切成固定长度的 chunk（通常 256～512 tokens），embedding 之后存向量库。用户提问时取 Top-K 最相近的 chunks 拼进 prompt。这个流程有两个结构性问题：

**问题一：相关但冗余**。同一个知识点可能在文档里重复出现三次，Top-5 里有两块几乎一样的内容。你在为重复信息付双倍 Token 费用。

**问题二：相关但不精准**。chunk 是按字数切的，一块 512 tokens 里，与问题真正相关的句子可能只有 3～4 句，其余都是前后文过渡语、章节标题、版权声明之类的干扰内容。

实验数据来自 LangChain 团队在 2023 年的测试：在标准 QA 任务上，对 Top-3 chunks 做上下文压缩后，回答准确率平均提升 8～12 个百分点，同时 prompt Token 数降低 40%～60%。

## 四种压缩路线对比

| 方案 | 原理 | 压缩率 | 延迟增量 | 适用场景 |
|---|---|---|---|---|
| LLM 提取式过滤 | 用小 prompt 让 LLM 抽取相关句子 | 40%～70% | +200～500 ms | 精度优先，预算宽裕 |
| LLMLingua 令牌级压缩 | 用小语言模型打分删 token | 50%～80% | +100～300 ms | 长文档，Token 成本敏感 |
| Reranker + 截断 | 重排后只保留 Top-N 句段 | 30%～60% | +50～150 ms | 已有 reranker 管道 |
| 规则式过滤 | 正则 / 关键词匹配删无关段落 | 20%～40% | <10 ms | 结构化文档，低延迟要求 |

四种方案不互斥，工程上通常是**规则过滤 → Reranker 截断 → LLM 精提取**三层串联，越靠后的步骤越贵，但处理的候选集也越小。

## LLMLingua 实战：用小模型给 prompt 做减法

LLMLingua 是微软研究院在 2023 年开源的令牌级压缩方案。核心思路是用一个参数量小的语言模型（比如 GPT-2、Llama-7B）给每个 token 打一个"条件概率分"，概率低意味着这个 token 对当前问题来说信息量不大，可以删除。

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="cpu",
)

chunks_text = "\n\n".join(retrieved_chunks)  # 你的 RAG 检索结果
result = compressor.compress_prompt(
    chunks_text,
    question=user_query,
    target_token=512,          # 压缩到目标 token 数
    condition_compare=True,
    condition_in_question="after",
    reorder_context="sort",
)

compressed_context = result["compressed_prompt"]
print(f"原始 tokens: {result['origin_tokens']}, 压缩后: {result['compressed_tokens']}")
# 典型输出：原始 tokens: 2048, 压缩后: 498，压缩率 75.7%
```

用 `llmlingua-2` 的 BERT 系小模型做打分，速度比用 GPT 系快得多，CPU 上通常 100 ms 以内能处理一个 prompt。中文支持用 `multilingual` 版本，实测中文场景压缩质量略逊于英文，但在技术类文档上通常可接受。

需要注意的是，令牌级压缩可能会破坏 Python 代码或 JSON 结构。如果你的知识库里有代码片段，建议先把代码块整体提取出来单独保留，只对散文部分做压缩。

## Reranker + 句段截断组合拳

如果你已经有 reranker 管道（比如 Cohere Rerank 或 BGE-reranker），有一个成本极低的压缩方案：**按句子粒度重排，然后截取 Top-N 句**。

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_cohere import CohereRerank

# 方案一：用 LLM 提取相关句子
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
)

# 方案二：reranker 打分后只取高分段落
reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)

docs = compression_retriever.invoke(user_query)
context = "\n\n---\n\n".join([d.page_content for d in docs])
```

Reranker 方案的优势是延迟增量小、不依赖额外模型部署，缺点是它只能在 chunk 粒度做取舍，无法进一步删除 chunk 内部的冗余句子。如果每个 chunk 内部噪音很高，还是需要配合 LLM 提取或 LLMLingua。

## 工程落地：三层过滤流水线

下面是一个生产可用的三层压缩架构：

```
用户提问
   │
   ▼
向量检索 Top-10 chunks
   │
   ▼
[第一层] 规则过滤
  - 删除纯目录行、版权声明
  - 删除 < 50 字的碎片 chunk
  → 剩余约 7 chunks
   │
   ▼
[第二层] Reranker 截断
  - BGE-reranker-v2 打分
  - 只保留 Top-4，分数 < 0.3 的丢弃
  → 剩余 3～4 chunks
   │
   ▼
[第三层] LLMLingua 令牌压缩
  - target_token = 600
  - 跳过代码块
  → 最终 context ≈ 600 tokens
   │
   ▼
LLM 生成回答
```

三层下来，初始 5000 tokens 的候选内容最终只有 600 tokens 进 LLM，而关键信息保留率在 85% 以上（需要用你自己的 QA 数据集验证，不要盲信）。

## 踩坑清单

- **不要在流式输出场景里同步等待 LLMLingua**：LLMLingua 是阻塞调用，加进 streaming 管道会让首 token 延迟从 200 ms 跳到 500 ms+，用户感知明显；改用异步或缓存预压缩。
- **压缩率不是越高越好**：target_token 设得太激进（压缩 80% 以上），中文语义容易被破坏，实测 50%～65% 是质量与压缩比的甜点区间。
- **代码和 JSON 不要压缩**：令牌级压缩会删掉括号、缩进，代码直接废了；检测到代码块就整体保留，只压散文。
- **压缩结果要加来源标注**：压缩后的 context 很难人工 debug，生产环境建议在每段前标注原始 chunk ID，方便追溯是哪个文档的哪一段出了问题。
- **冷门领域慎用 LLMLingua**：LLMLingua 的打分模型在通用语料上训练，对高度专业的医疗、法律、金融术语理解有限，可能把关键专有名词当"低概率无用词"删掉。

RAG 的本质不是"塞更多内容进去"，而是"把最相关的信息精准递到 LLM 面前"。上下文压缩是让这个精准度从 60 分到 90 分的那把刀。
