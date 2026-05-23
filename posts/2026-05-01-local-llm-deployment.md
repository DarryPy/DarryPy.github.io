---
layout: post
title: 本地 LLM 部署实战 — Ollama / vLLM / llama.cpp 怎么选
date: 2026-05-01
topic: "工程实战"
tags: [AI, Local LLM, 部署, 自托管]
excerpt: 数据敏感、API 太贵、想离线跑？三种主流本地 LLM 部署方案的实战对比与选型建议。
permalink: /posts/2026-05-01-local-llm-deployment.html
---

## 为什么要自托管

调用 API 是省事，但有几种场景必须自部署：

- **数据合规**：金融 / 医疗 / 政务，数据不能出境 / 不能给第三方
- **成本压制**：高频调用下 API 账单 > 自己的 GPU 折旧
- **延迟敏感**：本机推理比远程 API 低 100-500ms
- **离线 / 边缘**：没有稳定网络的场景
- **完全自由**：能 fine-tune 自己的模型、控制全部参数

代价是：**运维变成你的事**。

## 三种主流方案

### 1. Ollama（最低门槛）

**定位**：本地跑 LLM 的"app store"。

```bash
brew install ollama  # 或者下安装包
ollama pull llama3.3:70b
ollama run llama3.3:70b
```

跑起来就能用，自动 GGUF 量化，自动管理内存。

**优点**：
- 一行命令起步
- Mac / Linux / Windows 都能跑
- 自动管理多个模型，磁盘上像 docker images
- 提供 OpenAI 兼容 API

**缺点**：
- 并发性能差（一个请求占满 GPU）
- 不适合生产服务
- 量化版本有时落后社区

**适合**：个人开发、Mac 上跑 demo、写代码时本地 copilot。

### 2. llama.cpp（性能怪兽）

**定位**：纯 C++ 推理引擎，GGUF 格式标准的发明者。

```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
./llama-cli -m models/llama-3.3-70b-q4.gguf -p "你好"
./llama-server -m ... # 起个 OpenAI 兼容 API server
```

**优点**：
- 极致优化，CPU 上也能跑得动
- GPU / Metal / Vulkan / CUDA 全支持
- 量化方案最丰富（q2 到 q8、IQ 系列）
- 嵌入式 / 树莓派 / iPhone 都能跑

**缺点**：
- 命令行偏极客
- 自己管理服务、并发、监控
- 不像 Ollama 那么"开箱即用"

**适合**：性能敏感场景、边缘部署、写自己的客户端集成。

### 3. vLLM（生产级吞吐）

**定位**：高性能 GPU 推理服务器，**生产首选**。

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4
```

**优点**：
- PagedAttention：内存利用率极高
- continuous batching：高并发不掉吞吐
- 多 GPU 张量并行
- 提供完整 OpenAI 兼容 API
- 生产级 metrics / health check

**缺点**：
- 必须 NVIDIA GPU（CUDA）
- 没 GPU 装都装不上
- 内存占用大（要装得下原始权重）

**适合**：生产 API 服务、内部团队共享模型、大并发场景。

## 决策矩阵

| 场景 | 推荐 |
|---|---|
| Mac 个人开发 / Demo | Ollama |
| Linux / 服务器自托管小流量 | Ollama 或 llama.cpp |
| 极致性能 / 边缘 / 嵌入式 | llama.cpp |
| 生产服务 / 多用户共享 | vLLM |
| 没 GPU / CPU only | llama.cpp |
| 想跑量化模型测试效果 | Ollama 或 llama.cpp |

## 量化：本地推理的关键

模型原始权重通常太大，30B 模型 FP16 要 60GB 显存。
**量化**把权重从 FP16 压到 4-bit / 5-bit / 8-bit，大幅降低内存需求：

| 量化等级 | 显存需求（70B 模型） | 质量损失 |
|---|---|---|
| FP16（原版） | 140 GB | 0% |
| Q8 | 70 GB | ~1% |
| Q5_K_M | 50 GB | ~2-3% |
| Q4_K_M（最常用） | 40 GB | ~3-5% |
| Q3 | 30 GB | ~7-10% |
| Q2 | 24 GB | 显著退化 |

实战建议：

- **能上 Q5_K_M 就上**：质量损失基本不可感知
- **Q4_K_M 是甜点**：质量 OK + 显存友好
- **避开 Q3 / Q2**：除非真的没显存

## 硬件参考

跑常见模型需要的显存（Q4_K_M 量化）：

| 模型 | 显存 | 适合的 GPU |
|---|---|---|
| 7B / 8B | 5-6 GB | 任意现代 GPU / 集显也行 |
| 13B | 8-10 GB | RTX 3060 / 4060 |
| 30B / 32B | 18-22 GB | RTX 3090 / 4090 |
| 70B | 38-42 GB | A6000 / 双 4090 / 单 H100 |
| 120B+ | 70+ GB | 多卡 / H100 80G |

普通开发机配 4090 能跑 30B 流畅，70B 紧巴巴。要稳定跑 70B 需要 A6000 或更高。

## 工程上的几个坑

### 1. 第一次加载慢

70B 模型从 SSD 加载到 VRAM 要 30-60 秒。
**用 vLLM / Ollama 都会把模型常驻**，所以是冷启动慢，热请求快。

### 2. 并发性能差异巨大

同样硬件：

- Ollama：1 个请求占用全部 GPU
- vLLM：continuous batching 能同时处理几十个请求，每个延迟相似

**生产场景一定用 vLLM**，不要拿 Ollama 跑共享服务。

### 3. 量化选型要测

不要看 benchmark 选，**用你自己的真实业务 prompt 测**。
不同任务对量化损失的敏感度天差地别。

### 4. 监控不能少

自部署最大的隐性成本是**没人值守**。至少要监控：

- GPU 利用率 / 显存占用
- 请求延迟 p50 / p99
- 错误率 / OOM 次数
- 排队队列长度

## 一份起步配置（Linux + 单 A6000）

```bash
# 装 vLLM
pip install vllm

# 起服务
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --quantization awq \
  --max-model-len 8192 \
  --port 8000

# 客户端用 OpenAI SDK 直接接
client = OpenAI(base_url="http://localhost:8000/v1", api_key="placeholder")
```

10 分钟跑通一个内部 LLM API。

## 一句话总结

> 个人玩 Ollama，工程师玩 llama.cpp，生产用 vLLM。
> 量化锚定 Q4_K_M / Q5_K_M。
> 选型先看 GPU，再看场景，最后看模型大小。

本地 LLM 已经从 2023 年的"玩具"变成了 2026 年的"生产组件"——只要愿意运维。
