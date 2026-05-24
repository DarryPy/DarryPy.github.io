---
layout: post
title: "LLM API 压测 — 怎么找出生产系统的真实极限"
date: 2026-06-18
topic: "工程实战"
tags: [AI, Load Testing, Performance]
excerpt: LLM 压测和普通 HTTP 压测完全不同：请求慢、token 计费、streaming 响应。不搞清楚这些，压出来的数字没有意义。
permalink: /posts/2026-06-18-load-testing-llm.html
---

## 为什么 LLM 压测不一样

普通 HTTP 接口压测：请求 20ms，并发 500，算 QPS。

LLM 接口：
- 单请求 2-30 秒（取决于输入长度和输出长度）
- 计费按 token，不按请求数
- streaming 响应：第一个 token 和最后一个 token 时间差巨大
- 上下文长度的分布会极大影响整体吞吐

普通压测工具可以用，但**你盯的指标必须换掉**。

## 核心指标

| 指标 | 含义 | 目标 |
|---|---|---|
| TTFT | Time to First Token，第一个 token 多久 | < 1s (用户感知) |
| TPS | Tokens per second，生成速度 | 取决于模型和硬件 |
| p95 延迟 | 95% 请求完成时间 | 看业务 SLA |
| 错误率 | 429 / 500 / 超时 | < 0.1% |
| 每请求成本 | input tokens + output tokens × 单价 | 看商业预算 |
| 并发下的 TTFT 退化 | 并发增加时 TTFT 如何增长 | 找出拐点 |

**TTFT 比 p95 更重要**——用户盯着屏幕等第一个字出来。

## 压测前先搞清楚你的流量模型

你需要知道生产流量的：
- 平均 input token 数
- 平均 output token 数
- 是否有少量超长请求（拖累整体）
- 用户是否会并发

真实生产中，token 分布往往是**长尾的**：
- 80% 请求 < 500 input tokens
- 15% 请求 500-2000 tokens
- 5% 请求 2000+ tokens（可能引发限流）

压测时必须用**真实分布**，不能全用同一个 prompt。

## 工具选择

### 方案一：Locust（推荐，灵活）

```python
# locustfile.py
import random
from locust import HttpUser, task, between
import json

PROMPTS = [
    ("短问题", "What is 2+2?", 20),
    ("中等问题", "Explain the difference between TCP and UDP in 3 sentences.", 80),
    ("长分析", "Analyze the pros and cons of microservices vs monolith architecture for a startup with 5 engineers. Consider team size, deployment complexity, and scalability needs.", 200),
]

class LLMUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def chat(self):
        # 按权重采样，模拟真实分布
        weights = [50, 35, 15]
        label, prompt, expected_tokens = random.choices(PROMPTS, weights=weights)[0]

        start_first_token = None
        first_token_received = False
        total_tokens = 0

        payload = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}]
        }

        with self.client.post(
            "/v1/messages",
            json=payload,
            stream=True,
            headers={"Content-Type": "application/json"},
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    data = line[6:]
                    if data == b"[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        # 记录 TTFT
                        if not first_token_received:
                            delta = chunk.get("delta", {})
                            if delta.get("type") == "text_delta":
                                first_token_received = True
                                # 用自定义事件上报 TTFT
                                self.environment.events.request.fire(
                                    request_type="TTFT",
                                    name=f"ttft_{label}",
                                    response_time=...,  # 从请求开始到此刻
                                    response_length=0,
                                    exception=None,
                                )
                        # 统计 token
                        usage = chunk.get("usage", {})
                        if usage:
                            total_tokens = usage.get("output_tokens", 0)
                    except json.JSONDecodeError:
                        pass

            response.success()
```

### 方案二：k6（JavaScript，适合 CI/CD 集成）

```javascript
// llm-load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate, Counter } from 'k6/metrics';

const ttft = new Trend('ttft_ms', true);
const tokenThroughput = new Trend('tokens_per_second');
const errorRate = new Rate('error_rate');
const totalTokens = new Counter('total_tokens');

export const options = {
  stages: [
    { duration: '2m', target: 10 },   // 爬坡
    { duration: '5m', target: 10 },   // 稳定
    { duration: '2m', target: 30 },   // 加压
    { duration: '5m', target: 30 },   // 稳定观察
    { duration: '2m', target: 0 },    // 降温
  ],
  thresholds: {
    'ttft_ms': ['p(95)<2000'],        // TTFT p95 < 2s
    'http_req_duration': ['p(95)<30000'],  // 总时间 p95 < 30s
    'error_rate': ['rate<0.01'],      // 错误率 < 1%
  },
};

export default function () {
  const payload = JSON.stringify({
    model: 'claude-sonnet-4-6',
    max_tokens: 512,
    messages: [{ role: 'user', content: 'Explain what a transformer model is in 2 paragraphs.' }],
  });

  const start = Date.now();
  const res = http.post('https://api.anthropic.com/v1/messages', payload, {
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': __ENV.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
    },
    timeout: '60s',
  });

  const duration = Date.now() - start;

  check(res, {
    'status 200': (r) => r.status === 200,
  }) ? errorRate.add(0) : errorRate.add(1);

  if (res.status === 200) {
    const body = res.json();
    const outTokens = body?.usage?.output_tokens || 0;
    totalTokens.add(outTokens);
    if (outTokens > 0) {
      tokenThroughput.add((outTokens / duration) * 1000);
    }
  }

  sleep(1);
}
```

### 方案三：自定义脚本（最精确的 TTFT 测量）

```python
import asyncio
import time
import statistics
import anthropic

client = anthropic.AsyncAnthropic()

async def measure_single_request(prompt: str) -> dict:
    start = time.perf_counter()
    ttft = None
    output_tokens = 0

    async with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for event in stream:
            if ttft is None and hasattr(event, 'type'):
                if event.type == 'content_block_delta':
                    ttft = time.perf_counter() - start

        final = await stream.get_final_message()
        output_tokens = final.usage.output_tokens

    total_time = time.perf_counter() - start
    return {
        "ttft": ttft,
        "total_time": total_time,
        "output_tokens": output_tokens,
        "tps": output_tokens / total_time if total_time > 0 else 0,
    }

async def run_load_test(concurrency: int, duration_sec: int):
    results = []
    deadline = time.time() + duration_sec

    async def worker():
        while time.time() < deadline:
            try:
                r = await measure_single_request(
                    "Explain gradient descent in machine learning."
                )
                results.append(r)
            except Exception as e:
                results.append({"error": str(e)})

    await asyncio.gather(*[worker() for _ in range(concurrency)])

    ttfts = [r["ttft"] for r in results if "ttft" in r and r["ttft"]]
    totals = [r["total_time"] for r in results if "total_time" in r]
    errors = [r for r in results if "error" in r]

    print(f"\n=== 并发 {concurrency} 的压测结果 ===")
    print(f"总请求数: {len(results)}")
    print(f"错误数: {len(errors)} ({len(errors)/len(results)*100:.1f}%)")
    if ttfts:
        print(f"TTFT p50: {statistics.median(ttfts)*1000:.0f}ms")
        print(f"TTFT p95: {sorted(ttfts)[int(len(ttfts)*0.95)]*1000:.0f}ms")
    if totals:
        print(f"总延迟 p95: {sorted(totals)[int(len(totals)*0.95)]:.1f}s")
    print(f"TPS: {sum(r.get('tps', 0) for r in results if 'tps' in r)/len(results):.1f} tokens/s avg")

asyncio.run(run_load_test(concurrency=10, duration_sec=60))
```

## 常见瓶颈在哪里

### 1. 连接池耗尽

症状：并发上去之后，大量请求卡在建连阶段，TTFT 飙升。

```python
# 错误做法：每次请求新建 client
def bad_call():
    client = anthropic.Anthropic()  # 每次建连
    return client.messages.create(...)

# 正确做法：复用 client（复用连接池）
client = anthropic.Anthropic(
    max_retries=2,
    http_client=httpx.Client(
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
        timeout=60.0,
    )
)
```

### 2. 超长请求拖死队列

5% 的 2000+ token 请求占用连接很久，让后面的短请求排队。

解法：给长请求单独的资源池，短请求走快速通道：

```python
import asyncio

# 两个 semaphore，短请求用一个，长请求用另一个
short_sem = asyncio.Semaphore(20)
long_sem = asyncio.Semaphore(5)

async def call_llm(prompt: str):
    token_estimate = len(prompt.split()) * 1.3
    sem = long_sem if token_estimate > 1500 else short_sem
    async with sem:
        return await actual_api_call(prompt)
```

### 3. 429 限流级联

API 返回 429 之后，如果你立刻重试，会引发雪崩。

```python
import random

async def call_with_backoff(prompt, max_retries=4):
    for attempt in range(max_retries):
        try:
            return await actual_api_call(prompt)
        except anthropic.RateLimitError:
            if attempt == max_retries - 1:
                raise
            # 指数退避 + 抖动
            wait = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait)
```

## 压测结果怎么读

找三个拐点：

1. **TTFT 开始劣化点**：并发从 N 涨到 N+1，TTFT p95 明显跳升
2. **错误率开始出现点**：并发多少时开始出现 429/500
3. **吞吐量饱和点**：增加并发但每秒完成请求数不再增加

安全生产并发 = **TTFT 劣化点的 70%**，留出余量。

## 一个朴素结论

> LLM 压测的核心不是"能撑多少 QPS"，而是"TTFT 在什么并发下开始变差"。
>
> 找到那个拐点，乘以 0.7，就是你的安全生产水位。
> 超过这个数字，加一台机器或者上 API gateway 限流，别硬撑。
