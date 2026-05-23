---
layout: post
title: "LLM 量化实战 — GPTQ / AWQ / GGUF 怎么选"
date: 2026-06-04
topic: "模型与训练"
tags: [AI, Quantization, Inference]
excerpt: 把 70B 模型塞进一张消费级显卡，量化是唯一出路。GPTQ、AWQ、GGUF 三种方案各有适用场景，选错了性能和效果都会打折。
permalink: /posts/2026-06-04-quantization.html
---

## 为什么量化

Llama-3 70B 的 FP16 权重：70B × 2 bytes = **140 GB** VRAM。
8 张 A100（80GB）才够装。

量化到 INT4：70B × 0.5 bytes ≈ **35 GB**。
两张 A100、或者一张 H100，可用。

量化到 4-bit GGUF：约 **40 GB**，可以纯 CPU 运行（慢但能跑）。

三个收益：**VRAM 降、推理快、成本低**。代价：一定程度的精度损失。

---

## 量化方案对比（一张表）

| 格式 | 精度 | VRAM（70B） | 速度 | 质量损失 | 适合硬件 |
|------|------|------------|------|----------|---------|
| FP16 | 基准 | 140 GB | 基准 | 无 | 多卡 A100/H100 |
| INT8（bitsandbytes） | INT8 | 70 GB | 略慢 | 极低 | 高端单卡 |
| GPTQ INT4 | INT4 | 35 GB | 快 | 低 | A100/3090 |
| AWQ INT4 | INT4 | 35 GB | 快 | 极低 | A100/3090 |
| GGUF Q4_K_M | INT4 | ~40 GB | 慢（CPU）| 低 | CPU/Apple Silicon |
| GGUF Q8_0 | INT8 | ~70 GB | 慢 | 极低 | CPU（内存大） |

---

## GPTQ — 后训练量化，GPU 推理首选

GPTQ（Post-Training Quantization）：用一小批校准数据，逐层优化量化误差。

```python
# pip install auto-gptq transformers accelerate
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 校准数据（128 条就够）
calibration_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    # ... 更多样本，最好和实际使用场景类似
]

calibration_data = [
    tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    for text in calibration_texts
]

# 量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,                    # INT4
    group_size=128,            # 每 128 个权重一组，组内共享 scale
    desc_act=False,            # True 效果更好但更慢
)

# 量化（需要 GPU，耗时约 10-30 分钟 for 7B）
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config,
)
model.quantize(calibration_data)

# 保存
model.save_quantized("./llama3-8b-gptq-int4")
tokenizer.save_pretrained("./llama3-8b-gptq-int4")

print("量化完成！")
```

加载量化后的模型：

```python
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

model = AutoGPTQForCausalLM.from_quantized(
    "./llama3-8b-gptq-int4",
    device="cuda:0",
    use_triton=True,           # Triton kernel，更快
)
tokenizer = AutoTokenizer.from_pretrained("./llama3-8b-gptq-int4")

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0]))
```

**GPTQ 适合**：有 NVIDIA GPU，追求推理速度，对质量有一定要求。

---

## AWQ — Activation-Aware，质量更好

AWQ 的改进点：不是均匀量化所有权重，而是**保留对激活值影响大的权重的精度**。

```python
# pip install autoawq
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3-8B"
quant_path = "./llama3-8b-awq"

# 量化配置
quant_config = {
    "zero_point": True,   # 对称量化
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"     # 或 "GEMV"（小 batch 更快）
}

# 加载并量化
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model.quantize(tokenizer, quant_config=quant_config)

# 保存
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

加载：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

model = AutoAWQForCausalLM.from_quantized(
    quant_path,
    fuse_layers=True,          # 融合算子，更快
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True)

# 推理
inputs = tokenizer("解释一下量子纠缠", return_tensors="pt").to("cuda")
model.generate(
    **inputs,
    streamer=streamer,
    max_new_tokens=512,
    temperature=0.7,
)
```

**AWQ 适合**：质量优先，GPU 环境，有 HuggingFace 上的 AWQ 预量化版本可直接下载。

---

## GGUF — llama.cpp，CPU 友好

GGUF 是 llama.cpp 的模型格式，支持 CPU 推理（也支持 GPU offload）。适合没有或只有小 GPU 的环境。

量化级别（常用）：

| Level | 描述 | 质量 | 大小（7B） |
|-------|------|------|-----------|
| Q2_K | 2-bit | 很差 | ~2.8 GB |
| Q4_0 | 4-bit 简单 | 一般 | ~3.8 GB |
| Q4_K_M | 4-bit KQuants（推荐）| 好 | ~4.1 GB |
| Q5_K_M | 5-bit KQuants | 很好 | ~5.0 GB |
| Q8_0 | 8-bit | 接近 FP16 | ~7.2 GB |

```python
# pip install llama-cpp-python
# 如果有 NVIDIA GPU：CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python

from llama_cpp import Llama

llm = Llama(
    model_path="./llama-3-8b.Q4_K_M.gguf",
    n_ctx=4096,            # 上下文窗口
    n_gpu_layers=35,       # 把多少层 offload 到 GPU（0=纯 CPU）
    n_threads=8,           # CPU 线程数
    verbose=False,
)

# 推理
response = llm(
    "Q: 什么是量化？\nA:",
    max_tokens=512,
    stop=["Q:", "\n\n"],
    echo=False,
)
print(response["choices"][0]["text"])

# 流式输出
for chunk in llm(
    "用 Python 写一个快速排序",
    max_tokens=1024,
    stream=True,
):
    print(chunk["choices"][0]["text"], end="", flush=True)
```

从 HuggingFace 格式转换成 GGUF：

```bash
# 需要 llama.cpp 源码
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

# 转换
python convert_hf_to_gguf.py /path/to/hf-model --outfile model.gguf --outtype f16

# 量化到 Q4_K_M
./llama-quantize model.gguf model.Q4_K_M.gguf Q4_K_M
```

---

## 效果基准测试

用 perplexity 衡量量化质量损失（越低越好，FP16 = 基准）：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def calculate_perplexity(model, tokenizer, dataset_name="wikitext", num_samples=100):
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split="test")
    
    total_loss = 0
    total_tokens = 0
    
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        
        encodings = tokenizer(sample["text"], return_tensors="pt", max_length=512, truncation=True)
        input_ids = encodings.input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        total_loss += loss.item() * input_ids.shape[1]
        total_tokens += input_ids.shape[1]
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

# 实测参考（Llama-3-8B，wikitext-2）
# FP16:    PPL ≈ 6.14   （基准）
# AWQ INT4: PPL ≈ 6.43  （+4.7%）
# GPTQ INT4: PPL ≈ 6.55 （+6.7%）
# GGUF Q4_K_M: PPL ≈ 6.45 （+5.0%）
# GGUF Q2_K: PPL ≈ 9.11  （+48%，质量明显下降）
```

---

## 选型决策树

```
有 NVIDIA GPU？
├── 是
│   ├── VRAM >= 35 GB per card（70B 场景）？
│   │   └── 是 → AWQ INT4 or GPTQ INT4（GPU 推理最优）
│   └── VRAM < 35 GB？
│       └── GGUF + n_gpu_layers 部分 offload
└── 否（纯 CPU / Apple Silicon）
    └── GGUF Q4_K_M（CPU 推理，内存要求 ~2x 模型大小）

对质量要求极高？
├── 是 → AWQ > GPTQ > GGUF Q8_0 > GGUF Q4_K_M
└── 一般 → GGUF Q4_K_M（通用，社区资源最丰富）
```

---

## 一个朴素结论

> 量化不是黑魔法，是工程权衡：VRAM / 速度 / 质量三选二。
>
> 90% 的场景，Q4_K_M GGUF（开发测试）或 AWQ INT4（GPU 生产）已经足够好。
>
> **别在量化格式选型上纠结太久。先跑起来，再用 perplexity 和你的 eval dataset 验证实际效果。**
