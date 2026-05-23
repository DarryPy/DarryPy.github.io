---
layout: post
title: 多模态 LLM 实战 — Vision / Audio / 文档理解
date: 2026-04-27
tags: [AI, 多模态, Vision, Audio]
excerpt: GPT-4V / Claude Vision / Gemini 都能看图听音了。多模态实战要点、PDF 解析、OCR 替代、视频理解的工程模式。
permalink: /posts/2026-04-27-multimodal-llm.html
---

## 2026 的多模态现状

文本模型已经是过去时。
现在主流模型（Claude 4 / GPT-4.5 / Gemini 2.x）**默认就是多模态的**：

- 看图（Vision）：截图、照片、PDF 截图、图表
- 听音（Audio）：语音、音乐、环境音
- 看视频：抽帧 + 时间轴理解（Gemini 直接吃视频流）

这一波最大的变化是：**很多以前要专门 OCR / ASR / CV pipeline 的工作，一个 LLM 调用就搞定**。

## Vision 实战：图像理解

### 最常见用法

```python
from anthropic import Anthropic
import base64

client = Anthropic()
with open("chart.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
            {"type": "text", "text": "这张图表说明了什么？提取所有数据点。"}
        ]
    }]
)
```

### 实战场景

- **图表 / 表格解析**：从截图直接提取结构化数据
- **PDF 处理**：把每一页转图片，让 LLM 看（比传统 OCR 准）
- **手写笔记 OCR**：以前要 Google Vision，现在 Claude 直接读
- **UI 截图理解**：用户截图问"这个按钮在哪"，直接答
- **设计图比对**：两张截图比较差异
- **从图片填表**：发票、身份证、菜单 → 结构化数据

### Vision 的几个坑

1. **分辨率有上限**：Claude 长边 1568px，超过会被压缩。要 OCR 小字时分块上传
2. **图像 token 成本不便宜**：一张 1k×1k 图相当于 ~1500 tokens
3. **精确坐标 / 像素级测量**做不准——LLM 看的是语义，不是几何
4. **嵌入式手写公式 / 复杂图表**还是会错，关键场景要人审

## PDF 解析：传统 vs LLM 时代

传统 pipeline：

```
PDF → pdfminer / PyMuPDF 抽文字 → 正则 / NLP 抽字段 → 结构化数据
```

痛点：
- 表格抽取永远是大坑（合并单元格、跨页表格）
- 扫描件要先 OCR（Tesseract / PaddleOCR）
- 多列布局、页眉页脚干扰
- 图片里的文字漏掉

LLM 时代的新流程：

```
PDF → 每页转 PNG → 喂给 Vision LLM → 结构化输出
```

优点：
- 表格、图片、手写一次性搞定
- 直接拿到字段级 JSON
- 不需要复杂 pipeline

缺点：
- 单页处理成本约 $0.01-0.02（Vision token）
- 大批量 OCR 还是用传统方案更省
- 极致精度场景，传统 + LLM 互补更稳

**建议**：低频、复杂、需要语义理解的 PDF 用 Vision LLM；高频、规整、只要字符的用传统 OCR。

## Audio：语音 → 文本，以及更多

主流方案分两类：

### 1. ASR 然后传文本

```
audio → Whisper / Deepgram → 文本 → LLM
```

成熟、便宜、可控。适合**只关心说了什么**的场景。

### 2. 原生音频输入

OpenAI Realtime API、Gemini Live API 直接吃音频。
模型不仅听到"说了什么"，还能感知：

- 语气、情绪（生气、犹豫、兴奋）
- 背景音（环境、距离）
- 多说话人区分
- 停顿、笑声、犹豫词

适合**实时语音助手、客服质检、情绪分析**等场景。

### 工程注意

- Whisper Large v3 还是开源 ASR 的标杆
- 实时场景延迟敏感：用 streaming ASR
- 中文识别 Deepgram 略弱，国内有阿里 / 腾讯 ASR 可选
- 多模态实时 API 价格较贵，按秒计费

## 视频理解

Gemini 2.x 直接吃视频文件。两种用法：

### 1. 抽帧理解（通用）

每 N 秒抽一帧，对帧调 Vision LLM，按时间轴汇总。
适合**视频分类、关键时刻定位**。

### 2. 原生视频输入（Gemini）

直接传 mp4，模型一次性看完：

```python
import google.generativeai as genai
video = genai.upload_file("meeting.mp4")
response = model.generate_content([video, "总结这个会议的关键决策点和时间戳"])
```

适合**整段视频理解、会议纪要、视频内容审核**。

## 多模态 RAG

RAG 不只能检索文本：

- **图像 embedding**：把图片 embed 到向量库，可以"按描述找图"
- **跨模态检索**：用文本查询找到相关图片 / 视频片段
- **混合**：一份产品文档里既有文字也有截图，全部 embed 后联合检索

实战：CLIP / SigLIP 是经典图像 embedding；Cohere multimodal embed、Voyage multimodal 等托管选项也开始普及。

## 一些常被忽略的工程细节

1. **图片预处理**：太大的图先 resize（保持长边 < 1500px），省 token 也快
2. **token 成本预估**：图片不是"一张 = 一个 token"，按像素算法不同模型不同
3. **批处理**：单 prompt 内多张图比分多次调便宜（减少基础 prompt 开销）
4. **缓存**：图片 base64 + 任务的组合可以 hash 缓存，避免重复调用
5. **降级策略**：贵的 multimodal 失败时 fallback 到便宜的 OCR + 文本 LLM

## 一个朴素观察

**很多以前需要专门 ML 团队的工作**，现在多模态 LLM 几行代码就能解决：

- 财报截图 → 结构化数据
- 用户上传身份证 → 自动提取字段
- 客服录音 → 自动质检 + 情绪分析
- 设计稿 → 评估可访问性

不是说传统 CV / OCR 没用了，而是**门槛被砸下去了**。
小团队也能做以前要十人月才做出来的多模态产品。
