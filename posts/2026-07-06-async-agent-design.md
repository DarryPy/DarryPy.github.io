---
layout: post
title: "异步 Agent 设计 — 长任务、人在回路、多 Agent 协调"
date: 2026-07-06
topic: "Agent 与工具"
tags: [AI, Agent, Async]
excerpt: Agent 任务可能跑 5 分钟，用户等不了也不能一直等。异步设计不是可选项，是生产级 Agent 的必须项。
permalink: /posts/2026-07-06-async-agent-design.html
---

## 为什么 Agent 必须异步

同步 Agent 的问题：
- HTTP 请求超时一般 30-60s，但 Agent 任务可能要 5-30 分钟
- 网络断开就丢失进度
- 无法在中途等待人工审批
- 无法并行执行多个子任务

**你不能用一个 HTTP 连接跑一个 15 分钟的 Agent 任务。**

异步模式的核心：**把"提交任务"和"获取结果"分离**。

## 模式一：Job Queue（提交 + 轮询）

最简单的异步模式。

```
用户 → POST /tasks → 返回 task_id
用户 → GET /tasks/{task_id} → 查询状态（pending/running/done/failed）
用户 → GET /tasks/{task_id}/result → 获取结果（done 后）
```

```python
# 使用 FastAPI + Redis + Celery
from fastapi import FastAPI, BackgroundTasks
from celery import Celery
import redis
import uuid
import json
from datetime import datetime
import anthropic

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
celery_app = Celery('agent', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

@celery_app.task(bind=True)
def run_agent_task(self, task_id: str, user_query: str):
    """后台 Celery 任务：运行 Agent"""
    client = anthropic.Anthropic()

    def update_status(status: str, message: str = "", result: str = None):
        data = {
            "task_id": task_id,
            "status": status,
            "message": message,
            "updated_at": datetime.now().isoformat(),
        }
        if result:
            data["result"] = result
        redis_client.set(f"task:{task_id}", json.dumps(data), ex=3600)

    update_status("running", "Agent 启动中")

    try:
        # 模拟多步 Agent 执行
        steps = []

        # Step 1
        update_status("running", "正在分析任务...")
        plan_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": f"分析以下任务并列出执行步骤：{user_query}"}]
        )
        plan = plan_response.content[0].text
        steps.append({"step": "planning", "output": plan})

        # Step 2（模拟工具调用）
        update_status("running", "正在执行子任务...")
        import time
        time.sleep(2)  # 模拟耗时操作
        steps.append({"step": "execution", "output": "任务执行完成"})

        # Step 3
        update_status("running", "正在整理结果...")
        final_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"原始任务：{user_query}\n\n执行步骤：{json.dumps(steps, ensure_ascii=False)}\n\n请整理最终结果："
            }]
        )
        result = final_response.content[0].text

        update_status("done", "任务完成", result=result)

    except Exception as e:
        update_status("failed", f"任务失败：{str(e)}")

# API 端点
@app.post("/tasks")
async def submit_task(query: str):
    task_id = str(uuid.uuid4())[:8]
    # 初始化状态
    redis_client.set(f"task:{task_id}", json.dumps({
        "task_id": task_id,
        "status": "pending",
        "message": "任务已提交，等待执行",
        "created_at": datetime.now().isoformat()
    }), ex=3600)

    # 异步执行
    run_agent_task.delay(task_id, query)
    return {"task_id": task_id, "status": "pending"}

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    data = redis_client.get(f"task:{task_id}")
    if not data:
        return {"error": "task not found"}
    return json.loads(data)

@app.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str):
    data = redis_client.get(f"task:{task_id}")
    if not data:
        return {"error": "task not found"}
    task = json.loads(data)
    if task["status"] != "done":
        return {"error": f"task not ready, current status: {task['status']}"}
    return {"result": task.get("result", "")}
```

**客户端轮询示例**：
```python
import time
import requests

def submit_and_wait(query: str, poll_interval: int = 3, timeout: int = 300) -> str:
    # 提交
    resp = requests.post("http://localhost:8000/tasks", params={"query": query})
    task_id = resp.json()["task_id"]
    print(f"任务已提交：{task_id}")

    # 轮询
    deadline = time.time() + timeout
    while time.time() < deadline:
        status_resp = requests.get(f"http://localhost:8000/tasks/{task_id}")
        status = status_resp.json()

        print(f"状态：{status['status']} - {status.get('message', '')}")

        if status["status"] == "done":
            result_resp = requests.get(f"http://localhost:8000/tasks/{task_id}/result")
            return result_resp.json()["result"]
        elif status["status"] == "failed":
            raise RuntimeError(f"任务失败：{status.get('message', '')}")

        time.sleep(poll_interval)

    raise TimeoutError(f"任务超时（{timeout}s）")
```

## 模式二：Webhook 回调

任务完成后主动推送，不需要客户端轮询。

```python
@celery_app.task
def run_agent_with_webhook(task_id: str, user_query: str, callback_url: str):
    """完成后发 webhook"""
    try:
        # ... 执行 Agent ...
        result = "最终结果"

        # 任务完成，推送结果
        import httpx
        httpx.post(callback_url, json={
            "task_id": task_id,
            "status": "done",
            "result": result,
        }, timeout=10)

    except Exception as e:
        httpx.post(callback_url, json={
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
        }, timeout=10)

@app.post("/tasks/webhook")
async def submit_with_webhook(query: str, callback_url: str):
    task_id = str(uuid.uuid4())[:8]
    run_agent_with_webhook.delay(task_id, query, callback_url)
    return {"task_id": task_id}
```

## 模式三：Streaming 进度推送（SSE）

用 Server-Sent Events 实时推送进度。

```python
from fastapi.responses import StreamingResponse
import asyncio

@app.get("/tasks/{task_id}/stream")
async def stream_task_progress(task_id: str):
    """SSE 端点，实时推送任务进度"""
    async def event_generator():
        while True:
            data = redis_client.get(f"task:{task_id}")
            if data:
                task = json.loads(data)
                yield f"data: {json.dumps(task, ensure_ascii=False)}\n\n"

                if task["status"] in ("done", "failed"):
                    break
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

前端接收：
```javascript
const evtSource = new EventSource(`/tasks/${taskId}/stream`);
evtSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`状态: ${data.status} - ${data.message}`);
    if (data.status === 'done') {
        console.log('结果:', data.result);
        evtSource.close();
    }
};
```

## 模式四：人在回路（Human-in-the-Loop）

对于需要人工审批的关键步骤：

```python
import asyncio
from enum import Enum

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class HumanInLoopAgent:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.client = anthropic.Anthropic()

    def request_approval(self, task_id: str, action: str, details: dict) -> str:
        """暂停任务，等待人工审批"""
        approval_id = str(uuid.uuid4())[:8]
        self.redis.set(f"approval:{approval_id}", json.dumps({
            "approval_id": approval_id,
            "task_id": task_id,
            "action": action,
            "details": details,
            "status": ApprovalStatus.PENDING.value,
            "created_at": datetime.now().isoformat(),
        }), ex=86400)  # 24h 过期

        # 通知（实际项目里发 Slack/邮件/钉钉）
        print(f"[需要审批] approval_id={approval_id}\n操作: {action}\n详情: {details}")
        return approval_id

    def wait_for_approval(self, approval_id: str, timeout: int = 3600) -> bool:
        """阻塞等待审批结果"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self.redis.get(f"approval:{approval_id}")
            if data:
                approval = json.loads(data)
                status = approval["status"]
                if status == ApprovalStatus.APPROVED.value:
                    return True
                elif status == ApprovalStatus.REJECTED.value:
                    return False
            time.sleep(5)
        raise TimeoutError(f"审批超时（{timeout}s）")

    def run_financial_agent(self, task_id: str, instruction: str):
        """带人工审批的财务 Agent"""
        # 正常步骤，无需审批
        analysis = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": f"分析以下财务操作的合理性：{instruction}"}]
        ).content[0].text

        # 关键操作：转账金额超过阈值时需要审批
        if "转账" in instruction and "10000" in instruction:
            approval_id = self.request_approval(
                task_id=task_id,
                action="大额转账",
                details={"amount": "10000", "instruction": instruction, "analysis": analysis}
            )
            approved = self.wait_for_approval(approval_id)
            if not approved:
                return "操作已被审批人拒绝，任务终止。"

        # 审批通过后继续
        return "操作已完成。"

# 审批端点（供人工操作）
@app.post("/approvals/{approval_id}")
async def handle_approval(approval_id: str, approved: bool):
    data = redis_client.get(f"approval:{approval_id}")
    if not data:
        return {"error": "approval not found"}

    approval = json.loads(data)
    approval["status"] = ApprovalStatus.APPROVED.value if approved else ApprovalStatus.REJECTED.value
    approval["handled_at"] = datetime.now().isoformat()
    redis_client.set(f"approval:{approval_id}", json.dumps(approval), ex=86400)
    return {"status": "updated"}
```

## 模式五：多 Agent 并行协调

```python
import asyncio
import anthropic

client = anthropic.AsyncAnthropic()

async def sub_agent(agent_id: str, task: str) -> dict:
    """子 Agent 异步执行子任务"""
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": task}]
    )
    return {
        "agent_id": agent_id,
        "task": task,
        "result": response.content[0].text
    }

async def orchestrator(main_task: str) -> str:
    """主 Agent：分解任务，并发执行子任务，汇总结果"""
    # 分解任务
    decompose_resp = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"将以下任务分解为3个可独立并行执行的子任务，以 JSON 数组输出：{main_task}"
        }]
    )

    import json, re
    sub_tasks_text = decompose_resp.content[0].text
    json_match = re.search(r'\[.*\]', sub_tasks_text, re.DOTALL)
    if not json_match:
        return "任务分解失败"
    sub_tasks = json.loads(json_match.group())

    print(f"分解为 {len(sub_tasks)} 个子任务，并发执行...")

    # 并发执行所有子任务
    sub_results = await asyncio.gather(*[
        sub_agent(f"agent_{i}", task)
        for i, task in enumerate(sub_tasks)
    ])

    # 汇总
    results_summary = "\n\n".join(
        f"子任务 {r['agent_id']}：{r['task']}\n结果：{r['result']}"
        for r in sub_results
    )

    final_resp = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"主任务：{main_task}\n\n各子任务结果：\n{results_summary}\n\n请综合以上结果给出最终答案："
        }]
    )
    return final_resp.content[0].text

# 运行
result = asyncio.run(orchestrator("分析比亚迪、特斯拉、蔚来在2026年Q1的竞争态势"))
```

## 超时和失败处理

```python
@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def robust_agent_task(self, task_id: str, query: str):
    """带重试和超时的健壮 Agent 任务"""
    try:
        # 设置任务级超时
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Agent 执行超时")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(900)  # 15 分钟超时

        # 执行 Agent
        result = run_agent_logic(query)
        signal.alarm(0)  # 取消超时

        redis_client.set(f"task:{task_id}", json.dumps({
            "status": "done",
            "result": result
        }))

    except TimeoutError:
        redis_client.set(f"task:{task_id}", json.dumps({
            "status": "failed",
            "message": "任务执行超时，已记录断点，可重新提交"
        }))
    except Exception as exc:
        # 如果是可重试的错误（如网络问题），自动重试
        if "rate limit" in str(exc).lower() or "timeout" in str(exc).lower():
            raise self.retry(exc=exc)
        redis_client.set(f"task:{task_id}", json.dumps({
            "status": "failed",
            "message": str(exc)
        }))
```

## 一个朴素结论

> 异步 Agent 的设计就三件事：提交任务、跟踪状态、获取结果。
>
> 从 Job Queue 开始，够用就不要加复杂度。
> 只有当你真的需要"等人审批"或者"多 Agent 并行"时，
> 再加对应的模式。不要一开始就设计完整的异步架构。
