---
layout: post
title: 多模态 RAG 实战 — 文本 / 图像 / 表格一起检索
date: 2026-02-02
topic: "RAG 与检索"
tags: [AI, RAG, 多模态]
excerpt: 用户问 "帮我找财报里那张毛利率柱状图" — 纯文本 RAG 做不到。多模态 RAG 的架构和实战。
permalink: /posts/2026-02-02-multimodal-rag.html
---

## 普通 RAG 处理不了的场景

- "找出产品手册里关于安装步骤的那张图"
- "总结一下这份 PDF 里 5 个图表说了什么"
- "对比这两张架构图"
- "财报中 Q3 营收柱状图是多少"

文本 RAG 不知道**图片里讲的是什么**，纯靠文本描述召回率惨。

## 多模态 RAG 的两条路

### 路线 A：把图像转文本

```
PDF / 文档
  ↓ 抽取
文字片段 + 图片
  ↓ 每张图喂给 Vision LLM 生成描述
"图1：2024 Q1-Q4 营收柱状图，显示..."
  ↓ embed 描述
向量库
```

检索时只对文本描述检索，找到对应图片 ID 后**回显原图给 LLM**。

**优势**：实现简单，复用现有文本 RAG 栈。
**劣势**：图片描述质量决定召回上限；丢失视觉信号。

### 路线 B：多模态 embedding

直接把图片 embed 成向量（用 CLIP / SigLIP / Voyage multimodal 等）：

```
图片 → CLIP image encoder → 向量
文本 → CLIP text encoder → 向量（同空间）
```

文本和图片**同一向量空间**——可以用文本 query 直接检索图片。

**优势**：保留视觉信号，跨模态检索能力强。
**劣势**：CLIP 模型对**精细文本**（图表数字、长文）理解有限。

### 实战：A + B 组合

```
检索时：
  1. 文本 query → 同时找文本片段 + 图像描述
  2. 多模态 embedding → 找视觉相似的图

生成时：
  把文本片段 + 图片原图 + 图像描述一起喂给 Vision LLM
```

最稳的工程方案。

## Voyage multimodal-3

2025 年发布的多模态 embedding 标杆：
- 文本 + 图片 + 表格 同空间
- 比 CLIP 在文档检索任务上强 20%+
- API 调用直接 embed

```python
import voyageai
vo = voyageai.Client()
result = vo.multimodal_embed(
    inputs=[
        {"content": "查找毛利率图表"},          # 文本 query
        {"content": "data:image/png;base64,..."}, # 图片
    ]
)
```

## ColPali / ColQwen 路线

2024 末新派：**整页 PDF 直接 embed 成多向量**，跳过文本抽取 / 图片识别。

ColPali = Vision Encoder 处理 PDF 截图 + multi-vector retrieval。
对图表 / 表格 / 排版复杂的文档效果惊人，**比传统 OCR + 文本 RAG 强一截**。

劣势：每页存几十到几百个向量，**存储 + 检索成本高**。

## 表格的特殊处理

表格在 PDF 里被 OCR 抽出来通常是乱的。处理思路：

1. **保留为 HTML / Markdown 表格**：embed 时连结构一起编码
2. **小表直接转 JSON 存 metadata**
3. **大表用 Text-to-SQL**：表存数据库，query 时让 LLM 写 SQL 查

ColPali 这种 vision-based 路线对表格友好——直接看像素，不依赖 OCR 准不准。

## 实战架构

```
PDF / 网页
  ↓
[1] PyMuPDF 抽文字 → 文本 chunks
[2] PyMuPDF 抽图片 → 图片 + Vision LLM 描述
[3] (可选) ColPali → 整页向量
  ↓
向量库（文本 + 图片描述 + 多模态向量分库）
  ↓
查询：query embedding → 在 3 个分库都检索 → 合并 rerank
  ↓
top-K 喂 Vision LLM 生成答案
```

## 工程坑

- **图片 embedding 贵**：批量入库时控制成本
- **存储暴涨**：vision-based 路线 1 页可能 50KB 向量
- **延迟**：检索 + 抓图 + 喂 Vision LLM，总延迟 2-5s
- **图片版权**：商业场景注意原图引用

## 一个朴素结论

> 文档复杂、图表多的场景，**多模态 RAG 是必选**。
> 简单方案：A 路线（图转文）；高质量方案：A + B（多模态 embed）；
> 极致质量：ColPali（全 vision）但贵。
