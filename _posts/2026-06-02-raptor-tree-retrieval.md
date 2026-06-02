---
layout: post
title: RAPTOR 树状摘要检索 — 打破 RAG 对长文档的理解天花板
date: 2026-06-02
topic: "RAG 与检索"
tags: [RAG, RAPTOR, 长文档, 摘要检索, embedding]
excerpt: 传统 RAG 切片后丢失了文档的宏观结构，RAPTOR 用递归摘要树让检索同时覆盖细节与全局语义。从原理到代码，带你看懂 RAPTOR 如何分层聚类、逐层摘要，以及在真实项目里的落地取舍。
permalink: /posts/2026-06-02-raptor-tree-retrieval.html
---

## 你的 RAG 为什么在长文档上总翻车

你把一本两百页的技术规范喂进 RAG，然后问"这套系统的整体设计原则是什么"——大概率拿回一段东拼西凑的废话。问题不是 LLM 不够聪明，而是检索压根没有召回正确的内容。

传统 RAG 的工作方式决定了这种局限：把文档切成每段五百 token 左右的 chunk，分别做 embedding，存进向量库。每一个 chunk 在向量空间里表达的是"某页第三段说了什么"，而不是"整本书在讲什么逻辑"。当用户提出全局性问题时，答案往往分散在几十个不同的 chunk 里，Top-K 检索能召回其中几条，但 LLM 拿到这些碎片根本无法形成完整认知——就像你要判断一棵树长什么样，却只给你十片随机摘下来的叶子。

这个问题在短文档上不明显，文档越长、结构越复杂，这个缺陷就越致命。一份几十页的产品需求文档、一份年度研究报告、一套多章节的接口手册，只要用户习惯问跨段落的综合性问题，传统 RAG 就会频繁失效。

问题的根本在于：标准 RAG 只有一个粒度，就是原始 chunk。它在局部细节上表现还行，在全局理解上几乎无能为力。RAPTOR 就是为了补这个坑而生的。

真实场景里这类失败随处可见。用户问"这份合同的核心风险点有哪些"，系统只能返回包含"风险"关键词的几段条款，却无法将分散在第三条、第七条和附件二里的违约责任、免责声明、争议解决机制整合成一个完整的风险画像。用户问"这个开源框架和竞品相比优势在哪里"，系统只能返回架构章节的片段，却无法横跨性能测试、案例章节和设计理念章节给出综合回答。这不是工程调参能解决的问题，是索引结构本身的缺陷。

---

## 🌳 RAPTOR 的核心结构：递归摘要树

RAPTOR（Recursive Abstractive Processing for Tree-Organized Retrieval）是斯坦福大学 2024 年提出的检索增强方案。它的核心思路在于：在原始 chunk 之上，额外构建若干层摘要节点，形成一棵自底向上的树形索引，让检索可以同时命中叶节点（细节）和任意层级的父节点（摘要与宏观结构）。

**构建过程分三步，循环执行：**

第一步，切片成叶节点。和标准 RAG 一样，把原始文档切成固定大小的 chunk，每个 chunk 就是树的叶节点，也叫 level 0。这一步没有任何变化。

第二步，语义聚类。对当前层所有节点做 embedding，然后先用 UMAP 把高维向量降维到十到二十维，再跑 GMM（混合高斯模型）做无监督聚类。为什么不直接用 K-Means？因为 GMM 对不规则形状的聚类更鲁棒，在语义空间里同一主题的 chunk 往往呈现非球形分布，GMM 能更自然地把它们归为一组。

第三步，按组摘要。把同一聚类组的所有 chunk 拼合起来，调 LLM 生成一段两百字以内的摘要。这段摘要就成为上一层（level 1）的父节点。

然后对 level 1 的所有摘要节点重复上面的第二、三步，生成 level 2，如此递归，直到节点数不足以继续聚类，或达到预设的最大层数。最终你得到这样一棵树：

```
           [全文摘要 — 根节点 level 2]
          /                           \
 [主题A摘要 level 1]          [主题B摘要 level 1]
  /           \                  /            \
[chunk1]   [chunk2]         [chunk3]       [chunk4]
 level 0    level 0          level 0        level 0
```

构建完成后，所有层级的节点——叶节点加上每一层的摘要节点——统一向量化，全部写进同一个向量库。检索时不区分层级，Top-K 会自然命中不同粒度的节点：问具体实现细节，命中叶节点；问整体设计思路，命中高层摘要节点。多粒度检索这件事，在索引阶段就已经建好了，无需检索时做任何额外处理。

这里有一个细节值得关注：父节点的 embedding 是对摘要文本做的，而不是对子节点 embedding 取平均。摘要是经过 LLM 重新生成的自然语言，语义更紧凑、更具代表性。取平均的方式在数学上看起来合理，但实际上容易把相互矛盾的语义抵消掉，得到一个没什么辨识度的中间向量。用摘要文本做 embedding 这一步，是 RAPTOR 效果好于简单向量聚合方案的关键原因之一。

---

## 核心代码：搭一个最小可用的 RAPTOR 索引

下面是精简后的 Python 实现，依赖 `openai`、`scikit-learn`、`umap-learn` 三个库：

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from umap import UMAP
from openai import OpenAI

client = OpenAI()

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-3-small", input=texts
    )
    return np.array([d.embedding for d in resp.data])

def summarize_group(texts: list[str]) -> str:
    combined = "\n\n".join(texts)
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"请对以下内容做简洁摘要（200字以内）：\n\n{combined}"
        }]
    ).choices[0].message.content

def build_raptor_tree(chunks: list[str], max_levels: int = 3) -> list[dict]:
    all_nodes = [{"text": c, "level": 0} for c in chunks]
    current = chunks

    for level in range(1, max_levels + 1):
        if len(current) < 4:   # 节点太少，不值得继续聚
            break
        embs = embed(current)
        n_dim = min(10, len(embs) - 1)
        reduced = UMAP(n_components=n_dim, metric="cosine",
                       random_state=42).fit_transform(embs)
        n_clusters = max(2, len(current) // 5)
        labels = GaussianMixture(n_components=n_clusters,
                                 random_state=42).fit_predict(reduced)
        summaries = []
        for cid in range(n_clusters):
            group = [current[i] for i, l in enumerate(labels) if l == cid]
            if group:
                s = summarize_group(group)
                summaries.append(s)
                all_nodes.append({"text": s, "level": level})
        current = summaries

    return all_nodes  # 全部节点统一向量化后存入向量库
```

`build_raptor_tree` 返回的 `all_nodes` 列表，每项包含 `text` 和 `level` 两个字段。把全部节点向量化后批量写入 Qdrant 或 pgvector 即可。存入时记得把 `level` 作为 metadata 字段一并保存，方便日后分析或按层级过滤。

---

## 与标准 RAG 的对比

用一张表把核心差异说清楚：

| 维度 | 标准 RAG | RAPTOR |
|---|---|---|
| 局部细节问题 | 良好 | 良好（叶节点命中） |
| 全局跨段落问题 | 差 | 显著改善 |
| 索引构建成本 | 低 | 中高（额外 LLM 摘要调用） |
| 存储量 | 1x | 约 1.3–2x |
| 最适合的文档类型 | 短文档、FAQ | 长报告、手册、合同、研报 |

论文里在 QuALITY 和 QASPER 两个基准上的测试结果是：对于需要跨段落综合推理的问题，RAPTOR 相比 baseline RAG 的 F1 得分平均提升超过百分之二十，并且文档越长、问题越综合，提升幅度越大。这个结论在实际业务中也比较一致：文档页数不超过十页、用户问题大多是局部细节查询时，标准 RAG 已经够用；一旦文档超过三十页且充满跨章节的关联逻辑，RAPTOR 就值得认真考虑。

另一个维度是用户问题的类型分布。如果你的产品里用户问的绝大多数都是"第三章第五节怎么配置"这类精确查找，加不加 RAPTOR 差异不大；如果经常有用户问"帮我梳理一下整个架构的数据流"或"这套方案的主要局限是什么"，那 RAPTOR 几乎是绕不开的选项。可以先在问题日志里做一轮统计，再决定是否值得投入索引重建的成本。

---

## 工程落地的踩坑清单

- **层数不是越多越好**：超过三层之后，摘要质量快速退化，内容开始变成"本章主要介绍了……"这类废话，embedding 的区分度也随之下降。实践中两到三层是甜点区，超过这个阈值反而有害。

- **聚类数量要随规模动态调整**：把 `n_clusters` 写死成五在文档规模相差十倍的场景下效果差异极大。推荐用 `len(chunks) // 5` 做基准，或者用 GMM 的 BIC/AIC 分数自动搜索最优的聚类数，每次都比固定值稳得多。

- **摘要成本要提前做预算**：一份一百页的 PDF 约有五百个 chunk，建两层树大约需要额外处理一百组摘要调用，token 成本是纯向量化的十到三十倍。这个成本离线预建一次性完全可以接受，但绝对不能放在实时写入的路径上，否则用户上传文档时会超时。

- **降维是必须步骤，不能省**：直接拿 1536 维的 embedding 跑 GMM 会遭遇维度诅咒——高维空间里所有点之间的距离趋向相等，聚类完全失效。UMAP 降到十到二十维是必要的预处理，不是可选优化。

- **更新成本远高于新建**：当文档内容发生变化时，因为聚类和摘要都依赖全量节点的相对关系，局部增量更新很难做到准确，通常需要整棵树重建。要规划好索引重建的触发策略，避免在高频变更的场景下产生性能问题。

标准 RAG 在局部检索上已经够稳，RAPTOR 是它在全局理解维度上的垂直扩展。选不选它，核心判断标准就一条：你的用户会不会问那种需要综合全文才能回答的问题。
