---
layout: post
title: "KV Cache 优化 — 提速省钱的底层技术"
date: 2026-07-10
topic: "模型与训练"
tags: [AI, KV Cache, Inference]
excerpt: KV Cache 是 LLM 推理最重要的优化之一。理解它的工作原理，才知道怎么设计 prompt 结构来最大化缓存命中率，省 60% 的成本不是夸张。
permalink: /posts/2026-07-10-kv-cache-optimization.html
---

## KV Cache 是什么

Transformer 模型在生成每个新 token 时，需要对**所有已处理的 token** 计算 Attention。

计算 Attention 需要 Key 和 Value 矩阵：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

对于已经处理过的 token，它们的 K 和 V 是固定的，不会变。

**KV Cache 就是把这些已计算过的 K 和 V 存下来，下次生成新 token 时直接读取，而不是重新计算。**

没有 KV Cache 时：
- 生成第 100 个 token：需要计算前 99 个 token 的 K 和 V
- 生成第 101 个 token：又要重新计算前 100 个 token 的 K 和 V
- 复杂度：O(n²) — 非常慢

有 KV Cache 时：
- 生成第 100 个 token：读缓存
- 生成第 101 个 token：只计算第 100 个 token 的 K 和 V，其余读缓存
- 复杂度：O(n) — 快很多

## KV Cache 的成本结构

KV Cache 很大。对于一个 7B 模型：

```
每个 token 的 KV Cache 大小
= 层数 × 2 (K+V) × 注意力头数 × 头维度 × 精度字节数
= 32层 × 2 × 32头 × 128维 × 2字节(fp16)
= 32 × 2 × 32 × 128 × 2 = 524,288 bytes ≈ 0.5MB

32k context 的 KV Cache
= 32768 tokens × 0.5MB/token = 16GB
```

这就是为什么**长 context 既慢又贵**：不只是计算量，内存也不够。

## Anthropic Prompt Caching

Anthropic 提供的 Prompt Caching 功能允许**跨请求复用 KV Cache**。

原理：如果两个请求的 prompt 前缀相同，第二个请求直接复用第一个请求已经计算好的 KV Cache。

**价格对比（claude-sonnet-4-6）：**
| 类型 | 价格 |
|---|---|
| 普通 input tokens | $3 / 1M tokens |
| Cache write（首次写入缓存）| $3.75 / 1M tokens |
| Cache read（命中缓存）| $0.30 / 1M tokens |

命中缓存比普通输入便宜 **10 倍**。

### 如何使用

```python
import anthropic

client = anthropic.Anthropic()

# 长系统 prompt（适合缓存的内容）
LONG_SYSTEM_PROMPT = """
你是一个专业的法律文件分析助手。你熟悉中国合同法、劳动法、公司法的相关条款。

在分析合同时，你需要：
1. 识别主要条款和特殊条款
2. 标注潜在的法律风险
3. 比对行业惯例，指出明显偏离标准条款的内容
4. 对模糊措辞给出解读建议

你不提供法律意见，只做文件分析和信息整理。如需法律建议，建议咨询专业律师。

[此处假设还有很多业务规则，共 5000 tokens...]
""" * 20  # 模拟长系统 prompt

def analyze_contract_with_cache(user_question: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": LONG_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}  # 标记为可缓存
            }
        ],
        messages=[
            {"role": "user", "content": user_question}
        ]
    )

    # 检查缓存命中情况
    usage = response.usage
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Cache write tokens: {getattr(usage, 'cache_creation_input_tokens', 0)}")
    print(f"Cache read tokens: {getattr(usage, 'cache_read_input_tokens', 0)}")

    cache_read = getattr(usage, 'cache_read_input_tokens', 0)
    if cache_read > 0:
        print(f"✓ 缓存命中！节省约 {cache_read * 0.0027 / 1000:.4f} 美元")

    return response.content[0].text

# 第一次调用：写入缓存（贵一点）
r1 = analyze_contract_with_cache("分析这份合同的违约条款")
# 第二次调用：命中缓存（便宜 10 倍）
r2 = analyze_contract_with_cache("这份合同的保密条款有什么风险？")
```

### 多个缓存断点

可以设置多个缓存位置：

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": BASE_SYSTEM_PROMPT,  # 所有请求共享
            "cache_control": {"type": "ephemeral"}
        },
        {
            "type": "text",
            "text": USER_SPECIFIC_CONTEXT,  # 每用户的背景信息
            "cache_control": {"type": "ephemeral"}
        },
        {
            "type": "text",
            "text": CURRENT_DOCUMENT,  # 当前处理的文档（多轮对话中保持不变）
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[{"role": "user", "content": user_question}]
)
```

## OpenAI 的 Prompt Caching

OpenAI 的 GPT-4o 系列也支持自动缓存，无需手动标记：

```python
from openai import OpenAI

openai_client = OpenAI()

# OpenAI 会自动缓存超过 1024 tokens 的 prompt 前缀
# 不需要特殊标记，但 prompt 结构对命中率至关重要
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": LONG_STABLE_SYSTEM_PROMPT},  # 这部分会被缓存
        {"role": "user", "content": user_question}  # 这部分每次变化
    ]
)

# 检查缓存使用
usage = response.usage
print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Cached tokens: {usage.prompt_tokens_details.cached_tokens}")
```

## vLLM 的 KV Cache 管理（PagedAttention）

vLLM 的核心创新是 **PagedAttention**：把 KV Cache 分成固定大小的 page（类似操作系统内存页），按需分配，避免内存碎片。

```python
from vllm import LLM, SamplingParams

# 启动时配置 KV Cache
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.90,  # GPU 内存的 90% 用于 KV Cache
    max_model_len=32768,          # 最大支持的 context 长度
    enable_prefix_caching=True,   # 开启前缀缓存（跨请求复用）
    # block_size=16,              # KV Cache 块大小（tokens）
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

# 前缀缓存：相同前缀的请求会复用 KV Cache
prompts_with_same_prefix = [
    "你好，请分析：用户A的情况是..." ,
    "你好，请分析：用户B的情况是...",
    "你好，请分析：用户C的情况是...",
]

# vLLM 自动检测共同前缀并复用
outputs = llm.generate(prompts_with_same_prefix, sampling_params)
```

## 最大化缓存命中率的实践技巧

### 技巧一：把稳定内容放前面，变化内容放后面

```python
# 错误的结构：变化内容在前（每次都不同，没法缓存）
def bad_prompt(user_id: str, user_question: str) -> str:
    return f"用户{user_id}的问题：{user_question}\n\n[5000字系统说明]"

# 正确的结构：稳定内容在前，变化内容在后
def good_prompt(user_question: str) -> list:
    return [
        {
            "type": "text",
            "text": "[5000字稳定系统说明]",
            "cache_control": {"type": "ephemeral"}
        },
        {
            "type": "text",
            "text": f"用户问题：{user_question}"  # 变化内容在最后
        }
    ]
```

### 技巧二：对话历史的缓存策略

在多轮对话中，把整个对话历史标记缓存：

```python
def multi_turn_with_cache(conversation_history: list[dict], new_message: str) -> str:
    """多轮对话中利用 prompt caching"""
    # 把历史消息都标记为可缓存
    messages_with_cache = []
    for i, msg in enumerate(conversation_history):
        if i == len(conversation_history) - 1:
            # 最后一条历史消息加缓存标记
            messages_with_cache.append({
                "role": msg["role"],
                "content": [
                    {
                        "type": "text",
                        "text": msg["content"],
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            })
        else:
            messages_with_cache.append(msg)

    # 加入新消息
    messages_with_cache.append({"role": "user", "content": new_message})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=messages_with_cache
    )
    return response.content[0].text
```

### 技巧三：批量请求共享前缀

```python
def batch_with_shared_prefix(documents: list[str], task: str) -> list[str]:
    """批量处理时，把公共 prompt 缓存"""
    SHARED_INSTRUCTION = f"请完成以下任务：{task}\n\n任务要求：..."

    results = []
    for doc in documents:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=[{
                "type": "text",
                "text": SHARED_INSTRUCTION,
                "cache_control": {"type": "ephemeral"}
                # 第一个请求写入缓存，后续所有请求命中缓存
            }],
            messages=[{"role": "user", "content": f"文档内容：{doc}"}]
        )
        results.append(response.content[0].text)
    return results
```

## 缓存命中率监控

```python
class CacheMetrics:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_cache_write = 0
        self.total_cache_read = 0
        self.request_count = 0

    def record(self, usage):
        self.request_count += 1
        self.total_input_tokens += usage.input_tokens
        self.total_cache_write += getattr(usage, 'cache_creation_input_tokens', 0)
        self.total_cache_read += getattr(usage, 'cache_read_input_tokens', 0)

    def report(self):
        cacheable = self.total_cache_write + self.total_cache_read
        hit_rate = self.total_cache_read / cacheable if cacheable > 0 else 0

        # 计算节省的成本
        saved = self.total_cache_read * (3.0 - 0.30) / 1_000_000  # 每百万 token 节省 $2.70

        print(f"请求数: {self.request_count}")
        print(f"缓存命中率: {hit_rate:.1%}")
        print(f"节省成本: ${saved:.4f}")
        print(f"总 input tokens: {self.total_input_tokens}")
        print(f"其中命中缓存: {self.total_cache_read} ({hit_rate:.1%})")
```

## 局限性

- **内存占用大**：KV Cache 随 context 长度线性增长，长 context 很快撑满 GPU 内存
- **缓存有 TTL**：Anthropic 的 ephemeral cache 默认 5 分钟过期
- **只缓存前缀**：如果变化的内容在 prompt 中间，缓存就失效了
- **模型版本变化会失效**：升级模型后所有缓存需要重建

## 一个朴素结论

> KV Cache 的核心原则只有一句：**把不变的内容放前面，把变化的内容放后面**。
>
> 遵守这个原则，加上 Anthropic/OpenAI 的 Prompt Caching，
> 高复用场景（RAG 系统、多轮对话、批量处理）能节省 40-70% 的 token 成本。
> 这是成本最低、收益最高的 LLM 性能优化。
