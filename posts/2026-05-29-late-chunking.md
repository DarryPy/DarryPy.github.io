---
layout: post
title: "Late Chunking 与 Sentence Window Retrieval — 更聪明的 RAG 切片"
date: 2026-05-29
topic: "RAG 与检索"
tags: [AI, RAG, Chunking]
excerpt: 固定大小切片会让 chunk 失去上下文——"它"指代什么，检索时完全不知道。Late Chunking 和 Sentence Window Retrieval 是两种成本不高但效果显著的改进方案。
permalink: /posts/2026-05-29-late-chunking.html
---

## 固定切片的根本问题

最朴素的 RAG 做法：按 512 token 切文档，每段独立 embed。

问题来了：

```
# 原文（500字）
...苹果公司发布了新款 MacBook Pro。这款笔记本配备了 M4 芯片，
性能提升 40%。它的续航时间达到了 22 小时...

# 切片后的 chunk
chunk_1: "...这款笔记本配备了 M4 芯片，性能提升 40%。"
chunk_2: "它的续航时间达到了 22 小时..."
```

chunk_2 里的"它"，脱离上下文之后根本不知道指什么。Embedding 也无法捕获这个语义。

用户问"MacBook Pro 的续航"→ 检索到 chunk_2 → LLM 看到"它的续航达到 22 小时"→ 不知道"它"是什么。

---

## 方案一：Late Chunking

### 原理

不提前切分，先用 embedding 模型对**整段文档**做 token-level embedding，然后再按 chunk 边界做 mean pooling。

好处：每个 token 的 embedding 已经包含了全文的上下文信息（因为 transformer 的注意力是跨全文的），pooling 出来的 chunk embedding 就带有全局语义。

```
传统做法：
Doc → split → [chunk1, chunk2, chunk3] → embed each → [emb1, emb2, emb3]
                ↑ 上下文在切割时丢失

Late Chunking：
Doc → embed(全文) → token embeddings → pool by chunk boundary → [emb1, emb2, emb3]
           ↑ 每个 token 都见过全文，pooling 保留了跨 chunk 语义
```

### 代码实现

```python
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

class LateChunkingEmbedder:
    def __init__(self, model_name="jinaai/jina-embeddings-v3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
    
    def embed_with_late_chunking(
        self, 
        text: str, 
        chunk_size: int = 256,
        overlap: int = 32
    ) -> list[tuple[str, np.ndarray]]:
        """
        返回 [(chunk_text, chunk_embedding), ...]
        每个 embedding 都包含全文上下文。
        """
        # 1. 全文 tokenize
        tokens = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        input_ids = tokens["input_ids"]
        offset_mapping = tokens["offset_mapping"][0]  # [(start, end), ...]
        
        # 2. 全文前向传播，得到每个 token 的 embedding
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]
        
        # 3. 按 chunk 边界做 mean pooling
        seq_len = input_ids.shape[1]
        chunks = []
        
        for start in range(1, seq_len - 1, chunk_size - overlap):  # 跳过 [CLS] 和 [SEP]
            end = min(start + chunk_size, seq_len - 1)
            
            # 取这段 token 的 mean embedding
            chunk_emb = token_embeddings[start:end].mean(dim=0).numpy()
            
            # 映射回原文 char 位置
            char_start = offset_mapping[start][0].item()
            char_end = offset_mapping[end - 1][1].item()
            chunk_text = text[char_start:char_end]
            
            chunks.append((chunk_text, chunk_emb))
            
            if end >= seq_len - 1:
                break
        
        return chunks
    
    def build_index(self, documents: list[str]) -> list[dict]:
        """构建带 late chunking 的索引"""
        index = []
        for doc_id, doc in enumerate(documents):
            chunks = self.embed_with_late_chunking(doc)
            for chunk_idx, (chunk_text, chunk_emb) in enumerate(chunks):
                index.append({
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "text": chunk_text,
                    "embedding": chunk_emb,
                    "full_doc": doc,  # 可选：保留原文引用
                })
        return index
```

**注意**：Late Chunking 要求模型支持长文本（上下文窗口 > 8k），适合 jina-embeddings-v3 这类模型。  
对于只支持 512 token 的 BERT 类模型，文档超长就没法用了。

---

## 方案二：Sentence Window Retrieval

### 原理

**embed 小单位（句子），返回大单位（窗口）**。

检索时用精准的句子 embedding 匹配，命中后返回该句前后 N 句作为 LLM 的 context。

```
索引时：
句子1 → emb1
句子2 → emb2  ← 命中
句子3 → emb3

检索返回：
句子1 + 句子2 + 句子3（窗口大小 = 前后1句）→ 给 LLM
```

### 代码实现

```python
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np

nltk.download("punkt_tab", quiet=True)

class SentenceWindowRetriever:
    def __init__(self, window_size: int = 3, embed_model: str = "BAAI/bge-m3"):
        self.window_size = window_size  # 返回命中句前后各 window_size//2 句
        self.embedder = SentenceTransformer(embed_model)
        self.sentences = []
        self.embeddings = None
    
    def index(self, text: str):
        """切成句子，只对句子做 embedding"""
        self.sentences = nltk.sent_tokenize(text)
        self.embeddings = self.embedder.encode(
            self.sentences, 
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self
    
    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        返回 top_k 个命中的"窗口"，每个窗口包含命中句前后各若干句。
        """
        query_emb = self.embedder.encode([query], normalize_embeddings=True)
        
        # 余弦相似度（已 normalize，dot product = cosine）
        scores = (query_emb @ self.embeddings.T)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        half_window = self.window_size // 2
        
        for idx in top_indices:
            window_start = max(0, idx - half_window)
            window_end = min(len(self.sentences), idx + half_window + 1)
            
            window_text = " ".join(self.sentences[window_start:window_end])
            
            results.append({
                "matched_sentence": self.sentences[idx],
                "window_text": window_text,
                "score": float(scores[idx]),
                "sentence_idx": int(idx),
                "window_range": (window_start, window_end),
            })
        
        return results

# 使用示例
retriever = SentenceWindowRetriever(window_size=5)  # 命中句 ± 2句
retriever.index(long_document)

results = retriever.retrieve("MacBook Pro 的续航时间")
for r in results:
    print(f"Score: {r['score']:.3f}")
    print(f"Matched: {r['matched_sentence']}")
    print(f"Context window: {r['window_text']}")
    print()
```

---

## 方案三：Parent-Child Chunk

比 Sentence Window 更灵活：索引时存两级 chunk，检索命中小 chunk 后返回其父 chunk。

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Chunk:
    id: str
    text: str
    embedding: Optional[np.ndarray]
    parent_id: Optional[str]
    children_ids: list[str]

class ParentChildChunker:
    def __init__(
        self, 
        parent_size: int = 1024, 
        child_size: int = 128,
        embedder=None
    ):
        self.parent_size = parent_size
        self.child_size = child_size
        self.embedder = embedder
        self.chunks: dict[str, Chunk] = {}
    
    def chunk_document(self, text: str, doc_id: str) -> list[Chunk]:
        words = text.split()
        all_chunks = []
        
        # 切父 chunk
        for p_idx, p_start in enumerate(range(0, len(words), self.parent_size)):
            parent_words = words[p_start:p_start + self.parent_size]
            parent_text = " ".join(parent_words)
            parent_id = f"{doc_id}_p{p_idx}"
            
            parent_chunk = Chunk(
                id=parent_id,
                text=parent_text,
                embedding=None,   # 父 chunk 不做 embedding，不参与检索
                parent_id=None,
                children_ids=[]
            )
            
            # 切子 chunk
            for c_idx, c_start in enumerate(range(0, len(parent_words), self.child_size)):
                child_words = parent_words[c_start:c_start + self.child_size]
                child_text = " ".join(child_words)
                child_id = f"{parent_id}_c{c_idx}"
                
                child_emb = self.embedder.encode([child_text])[0] if self.embedder else None
                
                child_chunk = Chunk(
                    id=child_id,
                    text=child_text,
                    embedding=child_emb,
                    parent_id=parent_id,
                    children_ids=[]
                )
                
                parent_chunk.children_ids.append(child_id)
                self.chunks[child_id] = child_chunk
                all_chunks.append(child_chunk)
            
            self.chunks[parent_id] = parent_chunk
            all_chunks.append(parent_chunk)
        
        return all_chunks
    
    def retrieve_with_parent(self, query: str, top_k: int = 3) -> list[str]:
        """检索子 chunk，返回父 chunk 文本"""
        query_emb = self.embedder.encode([query])[0]
        
        # 在子 chunk 中检索
        child_chunks = [c for c in self.chunks.values() if c.parent_id is not None]
        scores = [
            (c, np.dot(query_emb, c.embedding) / 
             (np.linalg.norm(query_emb) * np.linalg.norm(c.embedding)))
            for c in child_chunks if c.embedding is not None
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回命中子 chunk 对应的父 chunk
        seen_parents = set()
        results = []
        for chunk, score in scores:
            parent_id = chunk.parent_id
            if parent_id not in seen_parents:
                parent_text = self.chunks[parent_id].text
                results.append(parent_text)
                seen_parents.add(parent_id)
            if len(results) >= top_k:
                break
        
        return results
```

---

## 三种方案对比

| 方案 | 索引复杂度 | 检索质量 | 适用文档 | 主要限制 |
|------|-----------|----------|----------|----------|
| Late Chunking | 高（全文前向传播）| 高 | 任意长文档 | 需要支持长上下文的 embed 模型 |
| Sentence Window | 低 | 中-高 | 叙述性文本 | 句子切分对中文不友好 |
| Parent-Child | 中 | 中-高 | 结构化文档 | 索引存储翻倍 |

中文场景建议：用 jieba 或 pkuseg 做句子切分替代 nltk。

---

## 一个朴素结论

> 固定切片是 RAG 的原罪——切掉了指代关系，切掉了上下文，再好的 embedding 模型也救不了。
>
> Sentence Window 实现最简单，今天就能用；Late Chunking 效果最好，但依赖模型支持长上下文。
>
> **先把 Sentence Window 做了，再考虑 Late Chunking 进阶——别一开始就追求最优解。**
