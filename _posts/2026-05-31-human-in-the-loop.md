---
layout: post
title: "Human-in-the-Loop Agent — 什么时候必须让人介入"
date: 2026-05-31
topic: "Agent 与工具"
tags: [AI, Agent, HITL]
excerpt: Agent 有了工具调用能力就能做"真实世界"的事——删文件、发邮件、付款。正因如此，它也能造成真实的损害。什么时候必须暂停让人确认？
permalink: /posts/2026-05-31-human-in-the-loop.html
---

## 为什么 Agent 需要人介入

自主 Agent 的问题在于**它不知道它不知道什么**。

三类高风险场景：

**不可逆操作**：删除数据、发送通知、付款、部署代码。做错了没有后悔药。

**低置信场景**：模型对当前情况不确定，但还是会选一个"最可能"的答案继续执行。

**超出预期范围**：用户说"帮我清理旧邮件"，Agent 决定删除所有 2023 年之前的邮件——这是个越权决策。

---

## 四种 HITL 模式

### 模式一：行动前确认（Confirm Before Action）

最简单的 HITL：执行高风险工具前，暂停等待用户批准。

```python
from anthropic import Anthropic
import asyncio, json
from typing import Any

client = Anthropic()

# 高风险工具清单
HIGH_RISK_TOOLS = {
    "delete_file",
    "send_email", 
    "execute_sql",
    "deploy_code",
    "charge_payment",
}

async def request_human_approval(action: str, params: dict) -> bool:
    """请求人工确认。生产环境接 Slack/飞书/webhook。"""
    print(f"\n[需要人工确认]")
    print(f"操作: {action}")
    print(f"参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
    
    # 实际生产：发送到审批系统，等待回调
    # 这里用命令行模拟
    response = input("是否批准？(y/n): ").strip().lower()
    return response == "y"

async def execute_tool_with_hitl(tool_name: str, tool_input: dict) -> Any:
    """
    工具执行层：高风险工具先请求人工确认。
    """
    if tool_name in HIGH_RISK_TOOLS:
        approved = await request_human_approval(tool_name, tool_input)
        if not approved:
            return {"status": "rejected", "message": "用户拒绝了此操作"}
    
    # 实际执行工具
    return await dispatch_tool(tool_name, tool_input)

async def run_agent_with_hitl(user_message: str):
    messages = [{"role": "user", "content": user_message}]
    tools = get_tool_definitions()
    
    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )
        
        if response.stop_reason == "end_turn":
            # Agent 完成
            return response.content[0].text
        
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # 这里做 HITL 检查
                    result = await execute_tool_with_hitl(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
            
            messages.append({"role": "user", "content": tool_results})
```

### 模式二：置信度阈值触发（Confidence Threshold）

Agent 评估自身置信度，低于阈值时主动请求人工协助。

```python
CONFIDENCE_PROMPT = """
在执行任何操作之前，先评估你对当前决策的置信度（0-100）。

输出格式（JSON）：
{
  "confidence": <0-100的整数>,
  "reasoning": "为什么你有/没有把握",
  "action": "你准备做什么",
  "risks": ["风险1", "风险2"]
}

如果置信度 < 70，说明你在以下情况：
- 用户意图不明确
- 操作可能有歧义
- 情况超出你的正常处理范围
"""

async def agent_with_confidence_check(task: str) -> str:
    # 第一步：评估置信度
    assessment_resp = client.messages.create(
        model="claude-opus-4-5",
        system=CONFIDENCE_PROMPT,
        messages=[{"role": "user", "content": task}],
        max_tokens=512,
    )
    
    try:
        assessment = json.loads(assessment_resp.content[0].text)
        confidence = assessment["confidence"]
        
        print(f"置信度: {confidence}/100")
        print(f"分析: {assessment['reasoning']}")
        
        if confidence < 70:
            # 置信度不足，请求人工澄清
            print(f"\n[置信度不足，需要人工确认]")
            print(f"计划操作: {assessment['action']}")
            print(f"风险: {assessment.get('risks', [])}")
            
            clarification = input("请提供更多信息或确认操作: ")
            task = f"原始任务: {task}\n用户补充说明: {clarification}"
    
    except json.JSONDecodeError:
        pass  # 评估失败，继续正常执行
    
    # 第二步：实际执行
    return await run_agent(task)
```

### 模式三：异步审批流（Escalate-and-Pause）

适合不需要实时响应的场景（如批量操作、定时任务）。

```python
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class ApprovalRequest:
    id: str
    agent_task_id: str
    action: str
    params: dict
    status: ApprovalStatus
    created_at: datetime
    expires_at: datetime
    approver_notes: str = ""

class ApprovalSystem:
    def __init__(self):
        self.requests: dict[str, ApprovalRequest] = {}
    
    def create_request(self, agent_task_id: str, action: str, params: dict, 
                       ttl_hours: int = 24) -> str:
        req_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        req = ApprovalRequest(
            id=req_id,
            agent_task_id=agent_task_id,
            action=action,
            params=params,
            status=ApprovalStatus.PENDING,
            created_at=now,
            expires_at=now + timedelta(hours=ttl_hours),
        )
        self.requests[req_id] = req
        
        # 发送审批通知（Slack、邮件、飞书）
        self._notify_approvers(req)
        
        return req_id
    
    def _notify_approvers(self, req: ApprovalRequest):
        # 实际接 Slack / 飞书 webhook
        print(f"[审批通知] ID={req.id} 操作={req.action}")
        print(f"  批准: POST /approvals/{req.id}/approve")
        print(f"  拒绝: POST /approvals/{req.id}/reject")
    
    async def wait_for_approval(self, req_id: str, poll_interval: int = 10) -> ApprovalStatus:
        """轮询直到审批完成或超时"""
        while True:
            req = self.requests[req_id]
            
            if datetime.utcnow() > req.expires_at:
                req.status = ApprovalStatus.EXPIRED
                return ApprovalStatus.EXPIRED
            
            if req.status != ApprovalStatus.PENDING:
                return req.status
            
            await asyncio.sleep(poll_interval)
    
    def process_approval(self, req_id: str, approved: bool, notes: str = ""):
        """审批人调用此接口处理审批"""
        req = self.requests[req_id]
        req.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        req.approver_notes = notes

approval_system = ApprovalSystem()

async def agent_with_async_approval(task: str):
    task_id = str(uuid.uuid4())
    
    # Agent 决策需要执行某操作
    action = "send_bulk_email"
    params = {"recipient_count": 5000, "subject": "产品更新通知"}
    
    # 创建审批请求，暂停等待
    req_id = approval_system.create_request(task_id, action, params)
    print(f"已创建审批请求 {req_id}，等待审批...")
    
    status = await approval_system.wait_for_approval(req_id)
    
    if status == ApprovalStatus.APPROVED:
        return await execute_action(action, params)
    else:
        return f"操作被拒绝或超时，状态: {status.value}"
```

### 模式四：全量审批（Full Approval Flow）

适合关键业务流程：Agent 只负责起草方案，人审批整个方案后再一次性执行。

```python
async def agent_plan_and_approve(goal: str) -> str:
    """先生成完整执行计划，人工审批后再执行"""
    
    # Step 1: 生成执行计划（不实际执行任何工具）
    plan_resp = client.messages.create(
        model="claude-opus-4-5",
        system="生成执行计划，列出所有需要调用的工具和参数。不要实际执行，只描述计划。",
        messages=[{"role": "user", "content": goal}],
        max_tokens=1024,
    )
    
    plan = plan_resp.content[0].text
    print(f"\n[执行计划]\n{plan}")
    
    # Step 2: 人工审批整个计划
    approved = input("\n批准此计划并执行？(y/n): ").strip().lower() == "y"
    
    if not approved:
        return "计划被拒绝"
    
    # Step 3: 审批通过后执行
    return await run_agent(goal)
```

---

## 审计日志

所有 HITL 事件必须记录：

```python
import logging
from dataclasses import dataclass, asdict

@dataclass
class HITLEvent:
    event_type: str          # "approval_requested", "approved", "rejected", "expired"
    timestamp: str
    agent_task_id: str
    action: str
    params: dict
    approver_id: str = ""
    notes: str = ""
    latency_ms: int = 0

def log_hitl_event(event: HITLEvent):
    logging.info(json.dumps({
        "event": "hitl",
        **asdict(event)
    }))
```

---

## UX 注意事项

**让审批简单**：一个按钮批准/拒绝，不要让审批人写理由（可选）。

**提供足够上下文**：审批通知里要包含 Agent 的推理过程，不只是"需要删除文件"。

**设置合理超时**：审批请求不能无限等待。超时后 Agent 应该选择安全默认行为（通常是不执行）。

**避免审批疲劳**：如果每次都要审批，人就会开始无脑点确认。只把真正高风险的操作设为需要审批。

---

## 一个朴素结论

> Agent 越自主，监管越重要。不是不信任模型，是不信任任何系统在边缘情况下的行为。
>
> HITL 不是让 Agent 变废，而是给 Agent 划一个安全边界——边界内全自动，边界外暂停。
>
> **从"执行前确认"开始，只把真正不可逆的操作设为需要确认。运行一段时间后，根据数据收紧或放宽。**
