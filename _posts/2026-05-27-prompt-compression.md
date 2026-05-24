---
layout: post
title: "Prompt 压缩技术 — 省 Token、不丢信息的实用方法"
date: 2026-05-27
topic: "Prompt 与推理"
tags: [AI, Prompt Engineering, Token Optimization]
excerpt: Token 就是钱，还是延迟。Prompt 压缩不是删字，是把信息密度提上去。四种实用方法和它们的适用场景。
permalink: /posts/2026-05-27-prompt-compression.html
---

## 为什么要压缩 Prompt

三个现实原因：

1. **成本**：Claude claude-opus-4 输入 $15/MTok，一个 8000 token 的 RAG prompt 每次调用 $0.12，一天 10 万次 = $12000
2. **延迟**：输入越长，prefill 越慢，TTFT 越高
3. **上下文窗口**：再大也有上限，RAG 塞了一堆文档，有效内容反而被淹没

压缩的目标：**更少 token，不损失（或少损失）信息量**。

---

## 技术一：LLMLingua — 选择性 Token 删除

LLMLingua 的思路：用一个小模型对 prompt 里每个 token 算条件概率，删掉信息量低的 token。

```python
# pip install llmlingua
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
    device_map="cpu",
)

original_prompt = """
Context: The quarterly financial report shows that revenue increased by 23.5% 
compared to the same period last year. Operating expenses grew by only 8.2%, 
leading to a significant improvement in operating margins. The company's EBITDA 
reached $450 million, up from $312 million in Q3 2024. Customer acquisition 
costs decreased by 15% while customer lifetime value increased by 28%.

Question: What was the EBITDA in Q3 2024?
"""

compressed = compressor.compress_prompt(
    original_prompt,
    rate=0.5,          # 压缩到原来 50%
    force_tokens=['\n', '.'],   # 强制保留这些 token
)

print(f"Original tokens: ~{len(original_prompt.split()) * 1.3:.0f}")
print(f"Compressed: {compressed['compressed_prompt']}")
print(f"Ratio: {compressed['ratio']:.2f}x")
```

输出示例：
```
Original tokens: ~105
Compressed: Context: quarterly financial report revenue increased 23.5% 
Operating expenses grew 8.2% EBITDA reached $450 million up $312 million Q3 2024.
Question: EBITDA Q3 2024?
Ratio: 0.48x
```

**适合**：长上下文、说明性文档、法律文本。  
**不适合**：代码（删 token 会破坏语法）、精确数字（可能删掉小数点）。

---

## 技术二：摘要蒸馏

用 LLM 把长文档压缩成摘要，再用摘要作为 context。

```python
from anthropic import Anthropic

client = Anthropic()

def summarize_for_context(document: str, query: str, max_tokens: int = 500) -> str:
    """
    针对特定问题对文档做摘要压缩。
    比通用摘要效果好——只保留和 query 相关的信息。
    """
    resp = client.messages.create(
        model="claude-haiku-4-5",  # 用小模型做摘要，省钱
        system="你是一个信息提取助手。根据用户问题，从文档中提取最相关的信息，精简输出。",
        messages=[{
            "role": "user",
            "content": f"问题：{query}\n\n文档：\n{document}\n\n请提取和问题相关的关键信息，100字以内。"
        }],
        max_tokens=max_tokens,
    )
    return resp.content[0].text

# 使用示例
long_doc = """... 一份 5000 字的产品文档 ..."""
query = "产品的退款政策是什么？"

compressed_context = summarize_for_context(long_doc, query)
# compressed_context: "退款政策：7天无理由退款，需保证产品完好..."
# 从 5000 字压缩到 50 字
```

**优点**：压缩率极高，语义保持好。  
**缺点**：多一次 LLM 调用（但用 Haiku 成本低），可能丢失细节。

**进阶**：多文档分层摘要。

```python
def hierarchical_compress(documents: list[str], query: str) -> str:
    """先对每个文档摘要，再对所有摘要做二次摘要"""
    summaries = [summarize_for_context(doc, query, max_tokens=200) for doc in documents]
    
    combined = "\n\n".join(summaries)
    if len(combined.split()) > 500:
        # 二次压缩
        return summarize_for_context(combined, query, max_tokens=400)
    return combined
```

---

## 技术三：检索增强的上下文裁剪

RAG 的常见问题：检索到 10 个 chunk，把全部塞进 prompt，80% 是噪声。

**做法：先检索，再过滤，只保留高相关的**。

```python
from sentence_transformers import CrossEncoder
import numpy as np

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def trim_context(query: str, chunks: list[str], top_k: int = 3, 
                 score_threshold: float = 0.3) -> list[str]:
    """
    用 Cross-Encoder 重排，只保留高分 chunk。
    比直接截断更聪明——保留的都是真正相关的。
    """
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    
    ranked = sorted(zip(scores, chunks), reverse=True)
    
    # 双重过滤：top_k AND 分数阈值
    selected = [
        chunk for score, chunk in ranked[:top_k]
        if score > score_threshold
    ]
    
    return selected

# 使用示例
chunks = retrieve_from_vector_db(query, top_k=10)  # 检索 10 个
trimmed = trim_context(query, chunks, top_k=3)     # 只用 3 个

# 原来：10 个 chunk = ~3000 tokens
# 现在：3 个 chunk = ~900 tokens，且更相关
```

**与 LLMLingua 结合**：先裁 chunk，再对剩余 chunk 做 token 级压缩。

```python
def compress_rag_context(query, chunks):
    # Step 1: 裁剪 chunk 数量
    selected_chunks = trim_context(query, chunks, top_k=3)
    
    # Step 2: 对每个 chunk 做 token 级压缩
    compressed_chunks = []
    for chunk in selected_chunks:
        result = compressor.compress_prompt(chunk, rate=0.6)
        compressed_chunks.append(result["compressed_prompt"])
    
    return "\n\n".join(compressed_chunks)
```

---

## 技术四：符号化压缩

把重复的长字符串替换成符号，在 prompt 顶部定义。

```python
def symbolic_compress(prompt: str) -> str:
    """
    自动识别重复短语，用符号替换。
    适合有大量重复术语的专业场景。
    """
    from collections import Counter
    import re
    
    # 提取所有 3-6 词的短语
    words = prompt.split()
    phrases = []
    for size in range(3, 7):
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i:i+size])
            phrases.append(phrase)
    
    # 找出出现 3 次以上的短语
    counts = Counter(phrases)
    repeated = [(p, c) for p, c in counts.items() if c >= 3]
    repeated.sort(key=lambda x: len(x[0]) * x[1], reverse=True)
    
    # 替换
    definitions = []
    compressed = prompt
    for i, (phrase, count) in enumerate(repeated[:10]):
        symbol = f"[P{i}]"
        savings = (len(phrase) - len(symbol)) * (count - 1)
        if savings > 20:  # 值得替换
            definitions.append(f"{symbol}={phrase}")
            compressed = compressed.replace(phrase, symbol)
    
    if definitions:
        header = "定义：" + "；".join(definitions) + "\n\n"
        return header + compressed
    return prompt

# 示例
original = """
在处理自然语言处理任务时，我们需要考虑自然语言处理的各种挑战。
自然语言处理的核心问题包括语义理解。在自然语言处理领域，...
"""
# 重复"自然语言处理" → 替换为 [P0]
```

---

## Token 压缩效果对比

| 技术 | 压缩率 | 质量损失 | 速度 | 适用场景 |
|------|--------|----------|------|----------|
| LLMLingua | 50-70% | 低-中 | 快 | 说明文档、对话历史 |
| 摘要蒸馏 | 80-95% | 中 | 慢（多一次 LLM） | 长文档 QA |
| 检索裁剪 | 60-80% | 低 | 快 | RAG |
| 符号压缩 | 10-30% | 极低 | 极快 | 高重复专业文本 |

---

## 什么时候不该压缩

**不要压缩 few-shot examples**：示例的格式、措辞本身就是信息，删掉会让模型不知道该输出什么格式。

**不要压缩精确指令**：`"只输出JSON，不要任何解释文字"` 这句话删短了会出问题。

**不要在高精度任务上过度压缩**：医疗、法律、合同——宁可多花钱，别丢细节。

**对话历史用摘要代替硬截断**：直接截掉早期对话不如先摘要再附后。

---

## 一个朴素结论

> 大多数 RAG prompt 有 40-60% 是噪声——重复的文档片段、不相关的 chunk、啰嗦的指令。
>
> 先做检索裁剪（成本最低，收益最高），再考虑 token 级压缩。
>
> **压缩 prompt 不是优化，是清洁代码——把不该在那里的东西清掉。**
