---
layout: post
title: RAG 文档预处理实战 — PDF / 表格 / OCR 的坑与解法
date: 2026-07-07
topic: "RAG 与检索"
tags: [RAG, PDF解析, OCR, 文档预处理, LLM]
excerpt: 检索质量的天花板，往往不是 embedding 模型，而是你喂进去的文档有多烂。这篇把 PDF、扫描件、表格这三类最常见的文档预处理难题拆开来讲，给出可直接落地的解法清单。
permalink: /posts/2026-07-07-rag-document-preprocessing.html
---

你花了大量时间调 embedding 模型、优化 reranker、设计 hybrid search 策略，但 RAG 系统的召回质量就是上不去——这种情况大概率不是检索算法的问题，而是文档在进入向量库之前就已经面目全非了。

文档预处理是 RAG 管道里最容易被忽视、但对最终效果影响最大的环节之一。本文聚焦三类最常踩的坑：PDF 解析、扫描件 OCR、结构化表格，并给出每类问题的实战解法。

## PDF 解析：看起来容易，实则一地碎片

PDF 格式本质上是一种渲染描述语言，不是语义文档。它存的是"在哪个坐标画什么字符"，而不是"这一段话讲了什么"。这导致用基础工具解析 PDF 时，最常见的问题有三类：

**跨栏文本错乱**：双栏排版（学术论文最常见）被解析成左栏第一行 + 右栏第一行交替拼接的乱序文本。向量化之后，语义完全破碎。

**页眉页脚污染**：页码、公司 logo 文字、"机密"水印被混入正文段落，embedding 后成为噪声。

**表格变平铺文本**：原本有行列关系的表格，被解析成一坨没有结构的数字和标签，任何检索都无法正确理解其含义。

```python
# 常见错误做法：直接用 pdfplumber 提取全文
import pdfplumber

with pdfplumber.open("report.pdf") as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages)
# 结果：双栏文本乱序 + 表格变流水账 + 页眉混入正文
```

更好的做法是分层解析：用 `pdfplumber` 或 `pymupdf` 获取文本块的坐标信息，再按 x 坐标分栏、按 y 坐标排序，而不是无脑 `extract_text()`。

```python
import pdfplumber

def extract_single_column(page):
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    # 按 y 坐标排序，过滤页眉（top < 60）和页脚（top > page.height - 60）
    words = [w for w in words if 60 < w["top"] < page.height - 60]
    words.sort(key=lambda w: (round(w["top"] / 12), w["x0"]))
    return " ".join(w["text"] for w in words)
```

对于排版复杂的 PDF，更推荐直接上 **layout-aware 解析器**，比如 `unstructured`（开源）或 `LlamaParse`（付费但效果强）。它们会识别文档区域类型（标题 / 正文 / 表格 / 图注），分类处理后再输出结构化结果。

| 工具 | 适用场景 | 双栏处理 | 表格识别 | 成本 |
|------|---------|---------|---------|------|
| pdfplumber | 简单数字 PDF | 需手写逻辑 | 弱 | 免费 |
| pymupdf | 速度优先 | 需手写逻辑 | 中 | 免费 |
| unstructured | 通用文档管道 | 较好 | 好 | 免费/企业版 |
| LlamaParse | 复杂学术 / 报告 PDF | 好 | 强 | 按量付费 |
| Azure Document Intelligence | 企业扫描件 + 表格 | 好 | 强 | 按量付费 |

## 扫描件 OCR：质量差一点，错误多一倍

扫描件没有"文本层"，必须走 OCR。Tesseract 是最常见的开源选择，但它有一个隐蔽的特性：**识别错误不均匀分布**——某些字符对（比如"0"和"O"、"1"和"l"）的混淆会在后续语义搜索中制造大量假阴性。

更严重的问题是：OCR 结果的置信度分布是长尾的。90% 的文字识别准确率听起来不错，但如果文档有 1000 个词，就有 100 个识别错误，每一段 chunk 里都有几个乱码，整体语义质量非常差。

```python
import pytesseract
from PIL import Image
import cv2
import numpy as np

def preprocess_for_ocr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 去噪 + 二值化
    img = cv2.fastNlMeansDenoising(img, h=10)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 矫正倾斜
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    (h, w) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))
    return img

processed = preprocess_for_ocr("scanned_contract.png")
text = pytesseract.image_to_string(processed, lang="chi_sim+eng")
```

除了预处理图像质量，还有两个实践上容易遗漏的点：

**后处理纠错**：用正则清理 OCR 常见错误（连字符断行、全角符号、无意义空格），再对低置信度词做词典匹配纠正。对中文文档，jieba + 自定义词典可以修复相当一部分业务术语识别错误。

**多模型投票**：Tesseract + PaddleOCR 对同一段图像各出一版，取字符级别的多数投票结果。实测在低质量扫描件上，准确率可提升 8-15 个百分点，代价是处理速度减半。

对精度要求高的场景（合同 / 法规 / 财务报告），直接用云端 OCR（Azure Document Intelligence、Google Document AI）更划算，它们自带版面分析，输出结果已经按段落和表格分好结构。

## 表格：最容易毁掉语义的数据结构

表格是 RAG 预处理里最难处理的结构，因为它的语义依赖行列的交叉关系——"2024 年 Q3 的净利润是多少"这个问题，需要同时知道行标签（Q3）和列标签（净利润）才能定位到正确的单元格。

把表格直接 flatten 成文本流，再切成 chunk，会发生什么？行标签和列标签极大可能被切断，落在不同的 chunk 里，导致检索时任何一个 chunk 单独看都缺少上下文，无法正确回答问题。

实战中有三种处理策略，根据表格复杂度选择：

**策略一：表格转自然语言描述（适合简单二维表）**

```python
def table_to_nl(df, table_title=""):
    rows = []
    for _, row in df.iterrows():
        desc = "、".join(f"{col}为{row[col]}" for col in df.columns)
        rows.append(desc)
    return f"{table_title}：" + "；".join(rows) + "。"

# 输入：{Q1: 100万, Q2: 120万, Q3: 95万}
# 输出：季度业绩：Q1为100万、Q2为120万、Q3为95万。
```

这种方法简单可控，但对有合并单元格或多级表头的复杂表格会失效。

**策略二：保留 Markdown 表格格式，整表作为一个 chunk**

对中小型表格（行数 < 30），直接把整张表转成 Markdown 格式，作为一个不可切割的 chunk 存入向量库。表头 + 数据行都在同一个 chunk 里，embedding 模型能感知到行列关系。

**策略三：表格拆分为单元格级别的三元组（适合大宽表）**

对列数超过 20、行数超过 50 的大表格，把每个单元格展开为 `（行标签, 列标签, 值）` 三元组形式，每个三元组是一个独立 chunk。这样任何一个检索命中都自带完整上下文。

```python
def table_to_triples(df, row_key_col):
    triples = []
    for _, row in df.iterrows():
        row_label = row[row_key_col]
        for col in df.columns:
            if col == row_key_col:
                continue
            triples.append(f"{row_label} 的 {col} 是 {row[col]}")
    return triples
```

## 按文档类型分流，而不是用一套逻辑打天下

现实项目里，文档库往往是混杂的：有原生 PDF、扫描 PDF、Word 文档、Excel 表格、HTML 网页，甚至 PPT 幻灯片。用同一套预处理逻辑处理所有类型，是最常见的工程错误之一。

更好的做法是在 ingestion 入口做类型路由：

```python
from pathlib import Path

def route_document(file_path: str):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        # 先检测是否有文本层
        if has_text_layer(file_path):
            return process_digital_pdf(file_path)
        else:
            return process_scanned_pdf(file_path)  # OCR 路径
    elif suffix in [".xlsx", ".csv"]:
        return process_spreadsheet(file_path)
    elif suffix in [".docx", ".doc"]:
        return process_word(file_path)
    elif suffix in [".html", ".htm"]:
        return process_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def has_text_layer(pdf_path: str) -> bool:
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text() or ""
        return len(text.strip()) > 20
```

对于 HTML 文档，记得用 `trafilatura` 或 `readability-lxml` 提取主体内容，过滤掉导航栏、广告、footer 等干扰元素。原始 HTML 直接喂进去，embedding 会被大量标签和重复的页面模板污染。

分流路由还带来另一个好处：不同类型的文档可以有不同的 chunk 策略和 metadata 附加逻辑。PDF 的 chunk 附带页码，Excel 的 chunk 附带 sheet 名和行号，HTML 的 chunk 附带来源 URL 和标题层级——这些 metadata 在检索后过滤和来源引用时都非常有价值。

## 构建可观测的预处理管道

一个容易被忽略的教训是：文档预处理的错误往往是沉默的。解析出了乱码，向量化照样跑完，检索照样返回结果，只是答案质量悄悄变差了，你不一定知道问题出在哪里。

好的预处理管道需要内置质量检查点：

```python
def quality_check(chunks):
    issues = []
    for i, chunk in enumerate(chunks):
        # 乱码检测：非打印字符占比
        non_printable = sum(1 for c in chunk if not c.isprintable()) / len(chunk)
        if non_printable > 0.05:
            issues.append(f"chunk {i}: 疑似乱码，非打印字符占比 {non_printable:.1%}")
        
        # 过短 chunk 检测
        if len(chunk) < 50:
            issues.append(f"chunk {i}: 内容过短（{len(chunk)} 字），可能是页眉/页码")
        
        # 重复检测
        if chunks.count(chunk) > 1:
            issues.append(f"chunk {i}: 重复内容")
    
    return issues
```

把这个检查嵌入 ingestion pipeline，在写入向量库之前先看报告，能拦住大量低质量 chunk 入库。

更进一步，可以把预处理指标打进监控系统。每次 ingestion 结束后，记录这几个关键数字：总 chunk 数、被过滤的低质量 chunk 数、平均 chunk 长度、各文档类型的解析耗时。当某个文档批次的低质量 chunk 比例突然升高，就是文档格式发生变化的信号——比如供应商换了一套 PDF 模板，或者扫描件分辨率下降了。早发现，早处理，而不是等用户投诉之后再回溯排查。

可观测性还包括：为每个 chunk 附加 `source_doc`、`page_num`、`parse_method` 三个 metadata 字段。当 LLM 给出一个奇怪的答案，你可以直接追溯到是哪个文档哪一页哪种解析方式生成的那个 chunk，而不是在数十万条向量里盲目排查。

## 踩坑清单

- **坑 1**：用 `pdfplumber.extract_text()` 处理双栏 PDF，文字顺序完全错乱，改用坐标排序或 unstructured
- **坑 2**：OCR 前不做图像预处理，识别准确率比预处理后低 20-30%
- **坑 3**：表格直接 flatten 后按固定 token 数切块，行标签和数据被切断落入不同 chunk
- **坑 4**：页眉页脚未过滤，embedding 空间里充满"第 X 页"和公司名称的无效向量
- **坑 5**：没有质量检查，乱码和空 chunk 静默入库，只能靠用户反馈才发现
- **坑 6**：对所有文档用同一套预处理逻辑，没有按文档类型分流处理

检索管道的质量上限，在文档进入向量库的那一刻就已经决定了。
