---
layout: post
title: "Speculative Decoding — 让大模型推理速度翻倍的技巧"
date: 2026-06-16
topic: "模型与训练"
tags: [AI, Inference, Speculative Decoding]
excerpt: 自回归解码一次只生成一个 token，GPU 计算能力被严重浪费。Speculative Decoding 用小模型批量"猜"多个 token，用大模型并行验证，大幅提升吞吐。
permalink: /posts/2026-06-16-speculative-decoding.html
---

## 自回归解码慢在哪里

Transformer 生成文本是自回归的：

```
输入：[今天天气] → 输出 token 1：很
输入：[今天天气很] → 输出 token 2：好
输入：[今天天气很好] → 输出 token 3：，
...
```

每次只能生成一个 token，然后才能生成下一个。

这意味着：
1. **GPU 计算不饱和**：生成每个 token 的计算量远小于 GPU 的计算能力，GPU 大部分时间在等待
2. **内存带宽瓶颈**：每次 forward pass 都要从显存加载全部模型权重，而权重加载速度（内存带宽）才是真正的瓶颈

对 70B 模型，生成速度往往被内存带宽而非算力限制。

---

## Speculative Decoding 原理

核心思路：**用一个小的 draft model 批量预测 N 个 token，然后用大的 target model 一次性并行验证这 N 个 token**。

```
传统解码（70B 模型，生成 4 个 token）：
[前缀] → 大模型 → token1
[前缀+1] → 大模型 → token2   → 4 次大模型调用
[前缀+2] → 大模型 → token3
[前缀+3] → 大模型 → token4

Speculative Decoding：
[前缀] → 小模型 → t1, t2, t3, t4（猜测 4 个）
[前缀, t1, t2, t3, t4] → 大模型 → 并行验证所有位置  → 1 次大模型调用
```

大模型验证时：
- 如果 t1 被接受，继续验证 t2
- 如果 t2 被拒绝，丢弃 t2 及之后所有 token，从 t2 位置重新开始

关键：大模型对 [前缀, t1, t2, t3, t4] 的一次 forward，**等价于同时计算了每个位置的输出**。验证 N 个 token 的代价 ≈ 生成 1 个 token 的代价（因为计算并行化了）。

---

## 接受率（Acceptance Rate）和加速比

设接受率为 α（每个 draft token 被接受的概率），draft 长度为 K：

```
每轮期望生成的 token 数 = K × α + 1（最后至少接受 1 个大模型自己的 token）

加速比 ≈ (K × α + 1) / c
其中 c = 大模型调用 1 次 / 小模型调用 K 次 的相对成本比
```

实际数据（Llama-70B + Llama-7B draft，K=4）：
- α ≈ 0.75（通用文本）
- 期望生成 token 数 ≈ 4 × 0.75 + 1 = 4
- 加速比 ≈ 2.5-3x

α 越高，加速效果越好。α 取决于：
- Draft model 和 target model 的"相似程度"（通常是同系列模型）
- 任务的可预测性（重复性高的文本 α 高）

---

## 简单实现（教学用）

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def speculative_decode(
    target_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    draft_k: int = 4,           # 每轮 draft 多少个 token
    temperature: float = 1.0,
) -> str:
    """
    Speculative Decoding 实现。
    target_model: 大模型（验证）
    draft_model: 小模型（猜测）
    draft_k: 每轮 draft 的 token 数
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()
    
    accepted_count = 0
    rejected_count = 0
    
    while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
        # Step 1: 用 draft model 生成 K 个候选 token
        draft_tokens = []
        draft_probs = []
        
        draft_input = generated.clone()
        with torch.no_grad():
            for _ in range(draft_k):
                draft_out = draft_model(draft_input)
                logits = draft_out.logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                
                next_token = torch.multinomial(probs, num_samples=1)
                draft_tokens.append(next_token)
                draft_probs.append(probs[0, next_token.item()].item())
                
                draft_input = torch.cat([draft_input, next_token], dim=1)
        
        # Step 2: target model 一次性处理所有 draft token
        candidate_ids = torch.cat([generated] + draft_tokens, dim=1)
        
        with torch.no_grad():
            target_out = target_model(candidate_ids)
        
        target_logits = target_out.logits / temperature
        
        # Step 3: 验证每个 draft token（rejection sampling）
        n_accepted = 0
        
        for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
            # target model 在位置 i 的输出（对应下一个 token 的概率）
            pos = generated.shape[1] + i - 1
            target_probs = F.softmax(target_logits[0, pos, :], dim=-1)
            target_prob = target_probs[draft_token.item()].item()
            
            # 接受概率 = min(1, p_target / p_draft)
            accept_prob = min(1.0, target_prob / (draft_prob + 1e-8))
            
            if torch.rand(1).item() < accept_prob:
                # 接受这个 token
                n_accepted += 1
                accepted_count += 1
            else:
                # 拒绝，从 target distribution 重采样
                # 修正分布：max(0, p_target - p_draft) / Z
                corrected = torch.clamp(target_probs - torch.tensor(draft_prob), min=0)
                corrected = corrected / corrected.sum()
                new_token = torch.multinomial(corrected, num_samples=1).unsqueeze(0)
                draft_tokens[i] = new_token
                rejected_count += 1
                break
        
        # Step 4: 追加接受的 token
        accepted_tokens = draft_tokens[:n_accepted + 1]  # +1 for the corrected/target token
        for token in accepted_tokens:
            generated = torch.cat([generated, token], dim=1)
            if token.item() == tokenizer.eos_token_id:
                break
        
        # 如果全部被接受，还需要追加 target 的额外 token
        if n_accepted == draft_k:
            last_pos = generated.shape[1] - 2
            bonus_probs = F.softmax(target_logits[0, last_pos, :], dim=-1)
            bonus_token = torch.multinomial(bonus_probs, num_samples=1).unsqueeze(0)
            generated = torch.cat([generated, bonus_token], dim=1)
    
    total = accepted_count + rejected_count
    print(f"Acceptance rate: {accepted_count}/{total} = {accepted_count/max(total,1):.1%}")
    
    return tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
```

---

## 生产级用法

不需要自己实现，主流推理框架都支持：

### vLLM

```python
from vllm import LLM, SamplingParams

# Speculative Decoding with eagle drafter
llm = LLM(
    model="meta-llama/Llama-3-70b-instruct",
    speculative_model="meta-llama/Llama-3-8b-instruct",  # draft model
    num_speculative_tokens=5,       # K
    speculative_draft_tensor_parallel_size=1,
    tensor_parallel_size=4,
)

sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=500,
)

outputs = llm.generate(["解释一下量子纠缠"], sampling_params)
print(outputs[0].outputs[0].text)
```

### HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Target model
target_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70b",
    device_map="auto",
    torch_dtype=torch.float16,
)

# Draft model（同系列，更小）
draft_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8b",
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-70b")

inputs = tokenizer("解释量子纠缠", return_tensors="pt").to("cuda")

# HF 原生 speculative decoding
outputs = target_model.generate(
    **inputs,
    assistant_model=draft_model,    # 就这一行！
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 变体：Medusa 和 EAGLE

### Medusa

不需要独立的 draft model，而是在 target model 的最后一层上加多个额外的"头"，每个头预测后续不同位置的 token。

```
target model hidden state → head1 → token at position +1
                          → head2 → token at position +2  
                          → head3 → token at position +3
                          → head4 → token at position +4
```

优点：不需要维护两个模型，内存开销更小。  
缺点：需要专门 fine-tune Medusa 头（几千步 SFT）。

### EAGLE

EAGLE 在 Medusa 基础上改进：用轻量级自回归 draft 头，接受率更高（通常 > 0.8）。

```python
# EAGLE with vLLM
llm = LLM(
    model="meta-llama/Llama-3-70b-instruct",
    speculative_model="[lmsys/vicuna-7b-v1.3]",  # EAGLE head
    speculative_model_uses_v1=True,
    num_speculative_tokens=5,
)
```

---

## 什么时候 Speculative Decoding 效果最好

**高 α（接受率高）**：
- Draft 和 target 是同系列模型（Llama-3 8B draft + 70B target）
- 任务内容可预测（代码生成、模板填写）
- 低 temperature（greedy / temperature < 0.5）

**低 α（效果差，不建议用）**：
- Draft 和 target 是不同系列
- 高创意任务（high temperature > 1.0）
- 极短输出（< 10 token，overhead 不值得）

---

## 速度基准

| 模型 | 方法 | Token/s（A100） | 加速比 |
|------|------|----------------|--------|
| Llama-3 70B | 标准解码 | ~25 | 1x |
| Llama-3 70B | Spec (8B draft, K=4) | ~55 | 2.2x |
| Llama-3 70B | EAGLE (K=5) | ~65 | 2.6x |
| Llama-3 70B | Medusa (K=5) | ~58 | 2.3x |

---

## 一个朴素结论

> Speculative Decoding 本质是用 GPU 的并行计算能力来弥补自回归解码的顺序瓶颈。
>
> 对于延迟敏感的生产场景，vLLM + Speculative Decoding 是今天能拿到手的最高性价比优化。
>
> **先算你的 acceptance rate，如果 > 0.7，上 speculative decoding 几乎没有副作用，只有提速。**
