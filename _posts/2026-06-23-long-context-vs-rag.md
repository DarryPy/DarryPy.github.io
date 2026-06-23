---
layout: post
title: 长上下文 LLM vs RAG — 200 万 token 之后，检索还有意义吗 🔍
date: 2026-06-23
topic: "RAG 与检索"
tags: [RAG, LLM, 长上下文, 向量检索, 架构设计]
excerpt: Gemini 和 Claude 的上下文窗口突破百万 token，让"直接塞文档"变成现实选项。但长上下文真的能取代 RAG 吗？本文从成本、延迟、更新频率、可解释性四个维度拆解两者差异，并给出最适合落地的混合架构方案。
permalink: /posts/2026-06-23-long-context-vs-rag.html
---

你有没有想过这个问题：Gemini 1.5 Pro 把上下文窗口推到了 200 万 token，Claude 稳在 200K，GPT-4o 也有 128K。按每个汉字约 1.5 token 换算，200 万 token 能装下大约 130 万字——相当于一整部《红楼梦》加上《三国演义》再加上《西游记》。

这让不少团队开始质疑：我们花了几个月搭的 RAG pipeline，是不是白搭了？很多人的逻辑是：既然模型能"记住"整本手册，还需要什么向量检索、什么 chunk 切分？直接把文档塞进去让模型自己找不就行了？

这个想法并不蠢，但它在生产环境里会踩进几个你可能没预料到的坑。答案是：RAG 没白搭，但你需要清楚地知道它解决的是哪些长上下文解决不了的问题。

## 先把差异摆出来

与其抽象讨论，不如直接上对比表。这六个维度基本覆盖了生产环境里真正重要的考量：

| 维度 | 长上下文 LLM | RAG |
|------|-------------|-----|
| 单次推理成本 | 随 context 线性增长，百万 token 约 $2-5 | 检索 + 小 prompt，通常 < $0.05 |
| 首 token 延迟 | 长 context TTFT 可达 15-40s | 检索 + 短推理，通常 < 3s |
| 知识库更新 | 需重新拼接全量 context，无法增量 | 向量库增量写入，秒级可用 |
| 可解释性 | 无法精准定位答案来源段落 | 返回 source chunk，可逐段溯源 |
| 规模上限 | 受 context 窗口硬限制 | 理论无上限，取决于向量库 |
| 中段注意力 | "Lost-in-the-middle"衰减明显 | 检索保证相关段落排在 prompt 前端 |

表格里没有任何一个维度是长上下文全面胜出的。即便是它看起来最擅长的"全文理解"，也被注意力衰减问题打了折扣。

## 长上下文的三块真实天花板

**第一块：成本。**

用 Gemini 1.5 Pro 处理 100 万 token 的 context，单次调用约需 $2-4。如果你的应用有 1000 个日活用户，每人每天问 5 个问题，仅 context 成本就是 $10,000-20,000 每天。RAG 架构下，同等规模的检索 + 短推理大约 $0.03 每次，日成本约 $150。差距是两个数量级。Prefix caching 可以缓解重复前缀的费用，但变化的用户 query 部分照样按全价计。

**第二块：Lost-in-the-middle 不是谣言。**

斯坦福 2023 年的实验（"Lost in the Middle: How Language Models Use Long Contexts"）发现，当关键信息放在 context 中间段时，多数主流模型的准确率会显著下降，有时比把信息放在头尾低 20-30 个百分点。100 万 token 的文档，中间那 95 万字几乎是衰减盲区。这就是为什么"把整个知识库塞进去"在 benchmark 上看起来不错，但真实用户投诉问题依然频发。

**第三块：实时更新场景根本不适合长上下文。**

企业的产品手册、合规政策、价格表每天都在变。用长上下文的做法是每次请求前重新拼接最新文档——这意味着每次请求都要传输数 MB 的文本，延迟和成本同时爆炸。RAG 的向量库支持增量写入，新文档几秒完成 embedding 入库，下一条查询立刻可见。两者的工程运维复杂度也完全不在一个量级。

还有一个常被忽视的问题：**context 不是存储**。你每次把文档塞进 context，下一次请求时还要再塞一遍，API 对每个 token 都重新计算注意力，没有任何持久化缓存。RAG 一旦把文档向量化写入数据库，后续检索的计算量是 O(log N) 级别，扩展到百万文档依然流畅。

## RAG 真正的护城河

三个长上下文替代不了的场景，值得牢记：

**TB 级私有知识库。** 金融、医疗、法律行业的内部文档动辄数百 GB。向量化存储 + 语义检索是唯一可行路径。即便未来 context 窗口继续扩大，把全量私有数据每次都传给 API 也会面临带宽、延迟、数据泄露风险等多重限制，RAG 的分离架构在这里没有替代品。

**答案溯源与合规审计。** 医院问答系统要求每条答案都能指回具体诊疗指南的某一段落；金融投顾系统要求每条建议附上依据条款编号。RAG 返回的 source chunk 天然满足这个需求，而长上下文只能说"根据上传的文档……"，无法给出可验证的精准定位。

**多租户数据隔离。** SaaS 产品里每个企业客户有独立的知识库，绝对不能互相泄露。RAG 通过 namespace / tenant\_id 过滤天然支持硬隔离；长上下文方案则需要每次按用户重新拼接 context，既有数据混入风险，又放大了成本和延迟。

## 最优解：两阶段混合架构

长上下文 LLM 和 RAG 不是非此即彼，而是互补关系。目前生产环境里表现最好的方案是**先检索、再推理**——RAG 负责找到最相关的几段，长上下文模型负责在这几段里做精细推理：

```python
from openai import OpenAI
from qdrant_client import QdrantClient

client = OpenAI()
qdrant = QdrantClient(url="http://localhost:6333")

def two_stage_qa(question: str, collection: str, top_k: int = 8) -> dict:
    # Stage 1: 语义检索——从百万文档里找最相关的 top_k 段
    query_vec = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding

    hits = qdrant.search(
        collection_name=collection,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True
    )
    sources = [h.payload for h in hits]
    context_text = "\n\n".join(
        f"[{i+1}] {h['text']}" for i, h in enumerate(sources)
    )

    # Stage 2: 长上下文推理——在检索结果上做深度跨段理解
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "你是知识库问答助手，只基于提供的上下文回答。"
                "回答时引用段落编号，如[1][3]。"
            )},
            {"role": "user", "content": (
                f"上下文：\n{context_text}\n\n问题：{question}"
            )}
        ]
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": sources
    }
```

这个设计的核心逻辑：把需要模型处理的 context 从 200 万 token 压缩到 4000-8000 token，成本降低 99%，TTFT 从 30s+ 降到 2s 以内，同时保留了长上下文模型强大的跨段推理和指令跟随能力。检索窗口小，模型注意力集中，Lost-in-the-middle 问题也随之消失。

这个架构还有一个隐性好处：**可调试性**。当答案出错时，你可以先看检索阶段——是召回的 chunk 根本不对，还是检索对了但模型推理出了岔？两阶段的错误是可以分离定位的；纯长上下文方案出错时，你往往只能猜是"文档本身的问题"还是"模型的问题"。

## 什么时候可以考虑纯长上下文

以下场景可以减少或绕过 RAG，直接走长上下文：

- 文档总量 **< 50 万 token**，且问题需要跨全文综合推理（合同整体分析、代码库全局重构、学术论文深度问答）
- **一次性分析任务**，不需要复用，不介意成本
- 文档结构极度离散，没有明显语义分块边界，向量检索召回质量本来就差

即便在这些场景，也建议先做一轮粗粒度检索作为前置过滤，再把命中文档完整放入 context——既减少无关噪音，又利用长上下文的整合推理能力。两者不是对立关系，是可以叠加的。值得一提的是，Claude 和 GPT-4o 都支持 prefix caching，当你的 system prompt + 背景文档固定不变时，重复的 token 可以享受大幅折扣，这让"半静态长上下文"的成本变得相对可控。

## 踩坑清单

- **top\_k 别贪多**。检索 20 个 chunk 不代表质量更高，超过 10 个后信噪比急剧下降，模型开始在高度相似的段落间混淆，答案反而变得模糊。从 5-8 个开始调，先用 MRR@K 和 NDCG 量化检索质量，确认提升后再决定是否加。
- **长上下文不是数据库**。上下文不持久化，每次请求都要重传，没有索引、没有事务。Prefix caching 能节省重复前缀的 token 费用，但前提是 system prompt + 文档前缀要严格固定，任何一个字的变动都会使缓存失效。
- **embedding 模型和生成模型要对齐**。检索用 BGE-M3，生成用 GPT-4o——两者对"语义相关"的定义未必同步，会出现检索命中但 LLM 回答"上下文没有提到"的情况。解决方式：用同一家的 embedding + LLM 组合，或在检索后加 cross-encoder reranker 做第二层相关性校准。
- **长上下文不能根治幻觉**。能缓解，不能根治。模型在长 context 里依然会虚构数字、日期、引用来源，且虚构的内容往往因为混进了真实文档的行文风格而听起来更可信。加 citation grounding——要求模型在回答里必须引用原文句子并标注段落编号，再用程序校验引用是否真实存在——是目前最有效的防幻觉手段。

一句犀利总结：**长上下文是更大的推理空间，RAG 是更精准的信息入口；两者的关系不是替代，而是流水线上的前后两道工序。**
