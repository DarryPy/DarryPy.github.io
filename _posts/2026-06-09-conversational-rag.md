---
layout: post
title: Conversational RAG 实战 — 对话历史与向量检索的优雅融合
date: 2026-06-09
topic: "RAG 与检索"
tags: [RAG, 对话系统, LangChain, 检索增强, LLM]
excerpt: 单轮 RAG 遇到多轮对话就会"失忆"——用户的每一轮追问都失去上下文，检索到的是牛头不对马嘴的文档。本文拆解三种主流 Conversational RAG 方案，带代码对比优劣，帮你搞清楚什么场景用哪种，以及最容易踩的几个坑。
permalink: /posts/2026-06-09-conversational-rag.html
---

你搭了一套 RAG 问答系统，单轮测试效果不错，结果用户一用就投诉："我刚才问了 A，现在问它的优缺点是什么，你给我检索了一堆不相关的东西。"原因很简单——你的检索用的是用户当轮的裸文本"它的优缺点是什么"，没有任何上下文，向量数据库完全不知道"它"是谁。这就是 Conversational RAG 要解决的核心问题。

这类问题在内部演示时很少暴露，因为演示时总是问完整的独立问题；但真实用户的对话习惯完全不同，他们像和人说话一样追问、省略、指代，把你精心搭建的单轮系统逼得毫无招架之力。Conversational RAG 不是锦上添花，而是让 RAG 产品真正可用的最后一块拼图。

## 🔍 问题根源：检索层的"单轮近视"

标准 RAG 管道的检索步骤大致是：用户输入直接 embedding，然后在向量数据库里找最近邻，召回文档之后才喂给 LLM 生成回答。多轮对话时，问题出在这个链路的最前端——embedding 之前。

用户的每一轮输入往往是省略了前文的代词指代或追问。"它的主要功能是什么""那性能上有什么差距""再具体说说第二点"这类表达在自然对话里极其常见，但这些句子单独拿出来进行语义检索，向量和文档库里任何一篇文章都对不上号。

LLM 生成阶段可以拿到 `chat_history` 作为上下文，所以最终回答不会出错——但前提是检索层得先把正确的文档召回来。而检索层早在 LLM 介入之前就已经跑完了，这时候 LLM 的理解能力帮不上忙。问题定位清楚之后，解法方向也就明确了：在检索之前，先把当前用户问题改写成一个携带足够上下文的独立查询。

这个步骤在学术界叫做 Query Reformulation 或 Query Condensation，在工程实现里有三种落地方式，适合不同的场景。值得一提的是，这个问题并不局限于 RAG——任何依赖语义检索的多轮交互系统，比如代码搜索助手、文档导航工具，都会遇到同样的困境。解决思路是相通的。

## 三种主流方案对比

**方案一：Query Condensation（查询压缩）**

用一次独立的 LLM 调用，把对话历史加上当前问题，改写成一句语义完整的独立问题。好处是改写质量高，检索结果的相关性接近单轮效果；坏处是每轮多一次 LLM 调用，在对话轮次密集时会明显增加延迟和成本。适合对答案质量要求较高的知识库问答、客服场景。

**方案二：Memory Buffer（滑动窗口记忆）**

直接把最近 N 轮对话历史拼接在当前问题前面，整体作为检索 query。好处是零额外 LLM 调用，实现极简；坏处是 query 变长会稀释检索信号，当历史窗口超过六轮时，embedding 向量里有效的"当前意图"信息被大量历史文本稀释，召回质量下降明显。适合对话轮次短、问题跨度小的场景，比如文档阅读助手里的逐段追问。

**方案三：Summary Memory（摘要记忆）**

用 LLM 持续维护一个对话摘要，每轮对话结束后更新摘要，检索时用摘要加当前问题构成查询。好处是无论对话多长，输入给检索层的文本量始终可控；坏处是摘要本身可能失真，尤其是话题频繁切换时，摘要质量显著下降，且维护摘要本身也有额外 LLM 成本。适合长流程助手，比如覆盖整个项目生命周期的需求分析工具。

| 方案 | 延迟开销 | 实现复杂度 | 最佳场景 |
|------|----------|-----------|---------|
| Query Condensation | +300-800ms | 低 | 知识库问答、客服 |
| Memory Buffer | 几乎为零 | 极低 | 短对话、文档阅读 |
| Summary Memory | +400-600ms/轮 | 中 | 长流程助手 |

## Query Condensation 代码实战

以下用 LangChain 演示 Query Condensation 方案，这也是生产中最常用的一种。整个管道分两条链：第一条链负责改写查询，第二条链负责基于检索结果生成最终回答。两条链共用同一个 `chat_history`，但承担不同职责，不要混淆。

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 查询改写 prompt
condense_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "根据对话历史，将用户最新问题改写成一个完整独立的问题，不能有代词指代。"
     "只输出改写后的问题，不要解释。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 最终回答 prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业助手。根据以下上下文回答用户问题。\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

condense_chain = condense_prompt | llm | StrOutputParser()

def get_condensed_query(inputs):
    # 第一轮无历史时跳过改写，省一次 LLM 调用
    if inputs.get("chat_history"):
        return condense_chain.invoke(inputs)
    return inputs["input"]

rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(
            retriever.invoke(get_condensed_query(x))
        )
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

# 调用示例
chat_history = []
answer = rag_chain.invoke({
    "input": "它的主要优缺点是什么？",
    "chat_history": chat_history
})
```

这里有个细节值得注意：`get_condensed_query` 做了一个短路判断——第一轮对话不走改写，直接检索，省掉一次 LLM 调用。只有 `chat_history` 非空才触发 Query Condensation。这个细节在高并发场景里可以节省可观的 API 成本。

另外，改写 prompt 的措辞对效果影响非常大。"不能有代词指代"是一条非常有效的硬性约束，强迫模型输出一个任何人拿到都能理解的完整问题。如果改写结果里仍然出现"它""这个"之类的代词，说明你的 prompt 约束还不够强，可以加上具体示例（few-shot）来引导模型。

## 多轮检索结果的去重处理

Conversational RAG 还有一个容易被忽视的问题：用户在第三轮追问时，改写后的 query 语义和第一轮非常接近，检索到的文档可能大面积重叠。把同样的段落反复喂给 LLM，既浪费 token，也可能让模型在生成时过度重视某一段内容，产生偏差。

处理方式是在召回后加一层去重逻辑。最轻量的做法是用文档内容的哈希指纹去重；更优雅的方式是开启 LangChain retriever 内置的 MMR（Maximal Marginal Relevance）算法，它在召回时同时考虑相关性和多样性，天然抑制重复文档：

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7}
)
```

`lambda_mult` 控制相关性与多样性的权衡，0.7 偏向相关性，0.3 偏向多样性，根据你的知识库特点调整。对于内容高度同质化的知识库（比如大量重复的 FAQ 文档），适当降低 `lambda_mult` 效果更好。

除了文档去重，还需要考虑对话状态的管理方式。在 Web 服务场景下，`chat_history` 通常不能放在内存里——每次请求都是无状态的，历史需要从 Redis 或数据库里读取。建议用 session ID 作为键，把每轮的用户输入和 AI 回复序列化存储，每次请求读取最近 N 轮。这一层的设计如果做得不好，很容易出现不同用户的对话历史串行的 bug，测试时务必覆盖并发场景。

## 踩坑清单

- **改写模型别用太弱的**：Query Condensation 依赖 LLM 的指代消解能力，能力不足的模型在处理复杂上下文时改写失败率明显上升，改出来的问题语义偏移，导致后续检索全盘崩坏，且这种错误很难在日志里直观发现。
- **chat_history 长度要截断**：不要把整个对话历史都塞进 condense_prompt，超过十轮后改写质量反而下降，且 token 成本剧增。滑动窗口取最近六到八轮已经足够，更早的对话对当前问题的影响微乎其微。
- **改写后的 query 必须记录日志**：上线初期一定要把改写结果单独打到日志里，这是排查"问题问得好但答案不对"这类检索质量 bug 最直接的线索，省掉大量猜测时间。
- **流式输出时注意时序**：改写是同步阻塞的，必须等改写完才能检索，再检索完才能流式生成，整体首 token 延迟比单轮 RAG 高出三百到八百毫秒，需要在前端给用户"思考中"的视觉反馈，避免体验下降。
- **不要把 Conversational RAG 当万能药**：如果你的产品场景九成是单轮问答，加 Query Condensation 只会增加成本和延迟，没有任何收益。评估多轮对话占比，再决定是否引入这套机制，别为了技术完整性做无用功。

真正的难点不是实现，而是定义"什么叫改写成功"——建议在上线前建立一个包含二十到三十条多轮对话样本的评测集，人工标注每一条的期望改写结果，用精确匹配率作为改写链的质量门禁。没有这个门禁，你永远不知道改写模型在什么时候悄悄退化了。
