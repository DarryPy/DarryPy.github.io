---
layout: post
title: LLM Fine-tuning 实战 — LoRA / QLoRA / 全参微调怎么选
date: 2026-04-25
category: "模型与训练"
tags: [AI, Fine-tuning, LoRA, 模型训练]
excerpt: 真要微调的那 10% 场景，三种主流方案到底差在哪。数据集准备、训练、评估的完整流程。
permalink: /posts/2026-04-25-llm-fine-tuning.html
---

## 微调能解决什么

之前的文章说过：**90% 的"想 fine-tune"其实是 prompt 没写好**。
但还有 10% 真的需要微调：

- 任务的**风格 / 格式**难以靠 prompt 复现（品牌语气、特定的输出结构）
- 想用**小模型替代大模型**，省成本（7B 微调后达到 70B 通用模型的 80% 效果，单请求成本 1/20）
- 需要让模型**学会新的特殊行为**（特殊领域术语、内部 DSL、独有推理范式）
- 对**延迟敏感**——本地小模型比远程 API 快

注意：**微调不能教模型新事实**——那是 RAG 的事。

## 三种主流方案

### 1. 全参微调（Full SFT）

更新模型**所有参数**。

**优点**：效果上限最高，模型能完全学到新行为。
**缺点**：
- 显存需求巨大（7B 模型全参微调要 ~150GB 显存）
- 训练慢、贵
- 容易 catastrophic forgetting（学了新任务忘了旧能力）

**适合**：训练自己的基础模型、有充足 GPU 资源、对效果有极致要求。

### 2. LoRA（Low-Rank Adaptation）

只训练一组**低秩的适配矩阵**，原模型参数冻结。

**原理**：在每个 attention 层旁边插入两个小矩阵 `A (d × r)` 和 `B (r × d)`，其中 r 远小于 d（通常 r=8~64）。
训练时只更新 A 和 B；推理时把 `A·B` 加回到原权重。

**优点**：
- 显存需求降低 10x 以上（7B LoRA 微调一张 24GB GPU 够用）
- 训练 5-10 倍快
- 一个底模 + 多个 LoRA adapter 灵活切换（每个任务一个 adapter）
- 不破坏底模能力（adapter 关掉就是原模型）

**缺点**：
- 效果略弱于全参（差距 1-5%）
- 对底模质量依赖大

**适合**：99% 的微调场景。

### 3. QLoRA（Quantized LoRA）

LoRA 之上再加量化——**把底模量化到 4-bit 存储**，LoRA adapter 保持 FP16 训练。

**优点**：显存再省 4x。70B 模型 QLoRA 一张 48GB GPU 能跑。
**缺点**：训练略慢（量化反向传播开销）；效果损失小但有。

**适合**：单卡训练大模型、个人 / 小团队预算紧。

## 决策矩阵

| 场景 | 推荐 |
|---|---|
| 7-13B 模型 + 一张 24-48GB GPU | LoRA |
| 30-70B 模型 + 一张 80GB GPU | QLoRA |
| 大集群 + 极致效果 | Full SFT |
| 多任务复用同一底模 | LoRA（每个任务一个 adapter） |

## 数据集准备：最重要的一步

**效果上限 80% 在数据，20% 在算法**。

### 数据格式

通用对话微调用 ChatML / OpenAI messages 格式：

```jsonl
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
{"messages":[...]}
```

每行一个完整对话样本。

### 数据量

- **少**：500-2000 条，能学会**风格 / 格式**
- **中**：5000-20000 条，能学会**任务级行为**
- **大**：50000+，能学会**领域级专长**

不要"数据越多越好"。**质量永远比数量重要**。

### 数据质量准则

1. **多样性**：覆盖所有可能输入的边角
2. **一致性**：同类问题答案风格统一（最难做到的）
3. **正确性**：错的数据比没有数据更糟
4. **难度分布**：80% 中等、15% 困难、5% 边界
5. **避免数据泄漏**：评估集严格隔离训练集

### 常见数据来源

- 人工标注（最贵但最准）
- 真实业务数据脱敏（数据集质量最高）
- 用大模型生成 + 人工修正（distillation）
- 公开数据集（FLAN / Alpaca / OpenOrca）

## 训练流程概要

用 [`transformers`](https://github.com/huggingface/transformers) + [`trl`](https://github.com/huggingface/trl) + [`peft`](https://github.com/huggingface/peft) 是标准栈：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir="./out",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        save_strategy="epoch",
    ),
)
trainer.train()
```

## 几个关键超参

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `r`（LoRA rank） | 8-64 | 越大表达力越强但参数多；常用 16 |
| `lora_alpha` | 2×r | scaling factor，控制 LoRA 输出强度 |
| `target_modules` | q/k/v/o + gate/up/down | 覆盖更全效果更好 |
| `learning_rate` | 1e-4 ~ 5e-4 | LoRA 比全参微调要高一两个数量级 |
| `num_train_epochs` | 2-5 | 多了过拟合 |
| `batch_size × gradient_accumulation` | effective 32-128 | 看 GPU 显存 |

## 评估：必须从训练阶段就准备

不要等训练完了才想"怎么知道学得好不好"：

1. **训练集 / 验证集 / 测试集 严格隔离**
2. **训练时跑 eval loss**（每 epoch 或每 N steps）
3. **训练后用独立的 evaluation set 跑业务指标**（accuracy / F1 / LLM-as-judge）
4. **回归测试**：训练后用通用 benchmark（MMLU / GSM8K）测有没有遗忘

## 部署

LoRA 训练好后，推理有两种模式：

1. **运行时加载 adapter**：用 PEFT 库，底模 + adapter 分别存。优势：一个底模配多个 adapter 灵活切换。
2. **Merge 后导出**：把 `A·B` 合并回原权重，导出一个完整模型。优势：推理时无 adapter overhead，可直接用 vLLM / TGI / Ollama 部署。

生产单一任务用 merge，多任务复用底模用 adapter。

## 一个朴素结论

**90% 微调用 LoRA / QLoRA 就够**。
全参微调留给真正的研发场景。
数据准备时间应该占整体 70%，训练只占 20%，评估占 10%。

数据没准备好就开始训练 = 烧 GPU 钱写错代码。
