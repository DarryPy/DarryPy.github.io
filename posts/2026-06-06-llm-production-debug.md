---
layout: post
title: "LLM 应用生产排障 — 出了问题怎么快速定位"
date: 2026-06-06
topic: "工程实战"
tags: [AI, Debugging, Observability]
excerpt: LLM 应用出了问题和传统服务出问题不一样——日志有了，但输出是自然语言，你不知道哪里出错了。一套系统的排障方法论。
permalink: /posts/2026-06-06-llm-production-debug.html
---

## LLM 应用的特殊故障模式

传统服务出问题：HTTP 500、超时、数据库连接失败——有明确的错误码。

LLM 应用出问题：

- **幻觉**：返回 HTTP 200，输出看起来正常，但内容是错的
- **质量退化**：输出变差了，没有任何错误信号
- **成本飙升**：某个请求用了 50000 token，你没察觉
- **延迟回归**：p99 从 3s 变成 8s，原因不明
- **安全事件**：模型说了不该说的话，已经被截图传播了

这些问题，没有系统性排障流程，很难快速定位。

---

## 排障方法论：Trace → Isolate → Reproduce → Fix

### Step 1：Trace（找到问题请求）

每次 LLM 调用必须有一个唯一的 `trace_id`，贯穿整个调用链。

```python
import uuid
import structlog
from contextvars import ContextVar

# 用 contextvars 在请求全生命周期内携带 trace_id
current_trace_id: ContextVar[str] = ContextVar("trace_id", default="")

def generate_trace_id() -> str:
    return str(uuid.uuid4())

log = structlog.get_logger()

class LLMClient:
    def __init__(self, client):
        self.client = client
    
    def create_message(self, **kwargs) -> dict:
        trace_id = current_trace_id.get() or generate_trace_id()
        
        start_time = time.monotonic()
        
        try:
            response = self.client.messages.create(**kwargs)
            
            latency_ms = int((time.monotonic() - start_time) * 1000)
            
            log.info(
                "llm_call_success",
                trace_id=trace_id,
                model=kwargs.get("model"),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
                stop_reason=response.stop_reason,
            )
            
            return response
            
        except Exception as e:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            log.error(
                "llm_call_failed",
                trace_id=trace_id,
                error_type=type(e).__name__,
                error_msg=str(e),
                latency_ms=latency_ms,
            )
            raise

# FastAPI 中间件：为每个请求注入 trace_id
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def inject_trace_id(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id") or generate_trace_id()
    token = current_trace_id.set(trace_id)
    
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    
    current_trace_id.reset(token)
    return response
```

### Step 2：Isolate（隔离问题）

收到告警后，第一步是缩小范围：

```python
# 排障查询示例（假设日志在 ClickHouse）

# 1. 延迟回归：找出慢的请求
"""
SELECT 
    trace_id,
    model,
    prompt_template_id,
    input_tokens,
    output_tokens,
    latency_ms,
    timestamp
FROM llm_calls
WHERE timestamp > now() - INTERVAL 1 HOUR
  AND latency_ms > 5000
ORDER BY latency_ms DESC
LIMIT 50
"""

# 2. 成本飙升：找出高 token 用量请求
"""
SELECT 
    user_id,
    sum(input_tokens + output_tokens) as total_tokens,
    count() as request_count,
    sum(cost_usd) as total_cost
FROM llm_calls
WHERE date = today()
GROUP BY user_id
ORDER BY total_cost DESC
LIMIT 20
"""

# 3. 幻觉/质量问题：关联用户反馈
"""
SELECT 
    lc.trace_id,
    lc.model,
    lc.prompt_template_id,
    lc.input_tokens,
    fb.rating,
    fb.comment
FROM llm_calls lc
JOIN user_feedback fb ON lc.trace_id = fb.trace_id
WHERE fb.rating = 'negative'
  AND lc.timestamp > now() - INTERVAL 24 HOUR
ORDER BY lc.timestamp DESC
"""
```

### Step 3：Reproduce（复现）

找到问题请求后，把完整的 prompt + response 捞出来，在本地复现。

```python
class DebugReproducer:
    """从生产日志捞 trace，本地复现"""
    
    def __init__(self, trace_store):
        self.trace_store = trace_store
    
    def fetch_full_trace(self, trace_id: str) -> dict:
        """获取完整 trace，包括所有中间步骤"""
        return self.trace_store.get_trace(trace_id)
    
    def reproduce_locally(self, trace_id: str, client) -> str:
        trace = self.fetch_full_trace(trace_id)
        
        print(f"=== Reproducing trace: {trace_id} ===")
        print(f"Model: {trace['model']}")
        print(f"Prompt template: {trace['prompt_template_id']}")
        print(f"\n--- System Prompt ---\n{trace['system_prompt']}")
        print(f"\n--- User Message ---\n{trace['user_message']}")
        print(f"\n--- Original Response ---\n{trace['response']}")
        
        # 本地重新调用
        resp = client.messages.create(
            model=trace["model"],
            system=trace["system_prompt"],
            messages=[{"role": "user", "content": trace["user_message"]}],
            max_tokens=trace.get("max_tokens", 1024),
        )
        
        print(f"\n--- Reproduced Response ---\n{resp.content[0].text}")
        return resp.content[0].text
```

---

## 案例一：排查延迟飙升

**场景**：p99 延迟从 3s 飙到 12s，持续 2 小时。

**排查步骤**：

```python
# 1. 看是否和时间相关（模型 API 本身问题）
"""
SELECT 
    toStartOfMinute(timestamp) as minute,
    quantile(0.99)(latency_ms) as p99,
    quantile(0.95)(latency_ms) as p95,
    count() as rps
FROM llm_calls
WHERE timestamp > now() - INTERVAL 3 HOUR
GROUP BY minute
ORDER BY minute
"""
# 如果 p99 在某个时间点突然跳升，大概率是外部 API 问题

# 2. 看延迟的构成（prefill vs generation）
"""
SELECT 
    quantile(0.99)(ttft_ms) as p99_ttft,           -- 首 token 延迟（prefill）
    quantile(0.99)(latency_ms - ttft_ms) as p99_gen, -- 生成延迟
    quantile(0.99)(input_tokens) as p99_input_tokens
FROM llm_calls
WHERE timestamp > now() - INTERVAL 1 HOUR
"""
# p99_ttft 高 → input 太长，或者服务端 prefill 慢
# p99_gen 高 → output 太长，或者 TPS 下降

# 3. 如果是 input 太长导致
"""
SELECT 
    prompt_template_id,
    quantile(0.99)(input_tokens) as p99_tokens,
    count() as cnt
FROM llm_calls  
WHERE latency_ms > 8000
  AND timestamp > now() - INTERVAL 1 HOUR
GROUP BY prompt_template_id
ORDER BY p99_tokens DESC
"""
# 发现：某个 template 的 input_tokens 从 2000 增加到 15000
# 原因：RAG 系统返回了太多 chunk，都塞进了 prompt
```

**根因定位**：RAG 检索的 `top_k` 参数被某次变更从 3 改成了 15，导致 prompt 暴增。

**修复**：
```python
# 在 RAG pipeline 中加输入长度保护
MAX_CONTEXT_TOKENS = 6000

def build_rag_prompt(query: str, chunks: list[str]) -> str:
    selected_chunks = []
    total_tokens = 0
    
    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk)
        if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
            break
        selected_chunks.append(chunk)
        total_tokens += chunk_tokens
    
    return format_prompt(query, selected_chunks)
```

---

## 案例二：排查幻觉集群

**场景**：用户反馈"AI 给了错误的产品价格"，一天内收到 15 条类似反馈。

**排查步骤**：

```python
# 1. 聚类负面反馈，找共同特征
negative_traces = query_negative_feedback(last_24h=True, category="factual_error")

# 2. 分析这些 trace 的共同点
common_features = {}
for trace in negative_traces:
    common_features["prompt_template"] = Counter(trace["prompt_template_id"])
    common_features["user_query_type"] = analyze_query_type(trace["user_message"])
    common_features["retrieved_docs"] = trace.get("retrieved_doc_ids", [])

# 发现：85% 的错误 trace 都在问"最新价格"，且检索到的文档日期是 2024 年的

# 3. 验证假设：检索到了过期文档
for trace_id in [t["trace_id"] for t in negative_traces[:5]]:
    trace = get_trace(trace_id)
    for doc_id in trace["retrieved_doc_ids"]:
        doc = get_document(doc_id)
        print(f"Doc {doc_id}: last_updated={doc['updated_at']}, content_preview={doc['content'][:100]}")

# 输出：
# Doc doc_4521: last_updated=2024-03-15, content_preview=产品A价格：¥299...
# Doc doc_4522: last_updated=2024-06-01, content_preview=产品A限时优惠：¥249...
```

**根因**：价格相关文档没有设置 TTL，过期数据仍在向量库中，被检索到。

**修复**：

```python
# RAG 检索时加时效性过滤
def retrieve_with_freshness_filter(
    query: str,
    max_doc_age_days: int = 30,
    metadata_filter: dict = None
) -> list[dict]:
    cutoff_date = datetime.now() - timedelta(days=max_doc_age_days)
    
    filter_expr = {
        "updated_at": {"$gte": cutoff_date.isoformat()},
        **(metadata_filter or {})
    }
    
    return vector_db.query(
        query_text=query,
        filter=filter_expr,
        top_k=5,
    )

# 定期清理过期文档
def cleanup_stale_documents(max_age_days: int = 90):
    cutoff = datetime.now() - timedelta(days=max_age_days)
    deleted = vector_db.delete(filter={"updated_at": {"$lt": cutoff.isoformat()}})
    log.info("stale_docs_cleaned", count=deleted)
```

---

## 必备排障工具

```
Langfuse（开源）: LLM trace 存储 + 查询 + 评估
  → 自部署，完整 trace，支持 prompt 版本追踪

Phoenix（Arize）: 可视化 trace，幻觉检测
  → 本地可用，适合开发阶段

ClickHouse: 结构化日志存储 + 快速分析查询
  → 日志量大时必用

Grafana: 实时 dashboard + 告警
  → 接 Prometheus 或 ClickHouse
```

---

## 排障 Checklist

出现问题时，按顺序执行：

```
[ ] 确认问题范围（所有用户/部分用户/特定 endpoint）
[ ] 查最近 1h 的 p50/p95/p99 latency 趋势
[ ] 查最近 1h 的错误率和错误类型
[ ] 查 input_tokens 分布是否有异常（暴增说明 prompt 变了）
[ ] 找到最近的 prompt 或代码变更时间，是否和问题时间吻合
[ ] 捞出 5-10 个问题 trace，人工检查 prompt + response
[ ] 确认是模型问题、prompt 问题、数据问题还是代码问题
[ ] 如果是外部 API 问题，查 provider 的 status page
```

---

## 一个朴素结论

> LLM 应用排障的核心是：trace_id + 结构化日志 + 可查询的 trace 存储。
>
> 没有这三样，出了问题就是大海捞针。有了这三样，80% 的问题 30 分钟内能定位到。
>
> **今天就把 trace_id 接入，日志落到可查询的地方。别等出了问题再后悔。**
