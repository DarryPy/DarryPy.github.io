---
layout: post
title: "Agent 状态机设计 — 用有限状态管好复杂 Agent 流程"
date: 2026-06-12
topic: "Agent 与工具"
tags: [AI, Agent, State Machine]
excerpt: Agent 代码一旦复杂起来就容易变成意大利面。用有限状态机（FSM）来组织 Agent 流程，让每个状态职责明确、每次转换可追溯、每个错误有兜底。
permalink: /posts/2026-06-12-agent-state-machine.html
---

## 为什么 Agent 需要状态机

最简单的 Agent 代码是 while 循环 + if/else：

```python
while not done:
    response = llm.call(messages)
    if response.stop_reason == "tool_use":
        results = execute_tools(response.tool_calls)
        messages.append(results)
    else:
        done = True
```

这在简单场景可以，但一旦加入：
- 人工审批步骤
- 重试逻辑
- 并发子任务
- 错误回滚
- 审计日志

代码就开始乱。状态分散在各个 if/else 里，很难追踪 Agent 当前在做什么、为什么在做。

**状态机的收益**：
- **可追溯**：任何时刻知道 Agent 处于哪个状态
- **可测试**：每个状态转换单独测试
- **可调试**：日志精确到每次状态转换
- **可暂停/恢复**：状态可序列化，支持长时间任务

---

## 核心概念

```
States（状态）：Agent 可能处于的确定性阶段
Transitions（转换）：从一个状态到另一个状态的条件
Guards（守卫）：转换的前置条件检查
Actions（动作）：进入 / 退出状态时执行的副作用
Entry/Exit hooks：状态切换时的回调
```

---

## 示例：文档审核 Agent（5 个状态）

这是一个自动审核文档、必要时请人工介入的 Agent。

### 状态定义

```
IDLE → ANALYZING → REVIEWING → AWAITING_APPROVAL → COMPLETED
                                    ↓                    ↑
                                 REJECTED              APPROVED
                    
每个状态的职责：
IDLE:              等待任务输入
ANALYZING:         LLM 分析文档内容，提取关键信息
REVIEWING:         根据分析结果做初步决策
AWAITING_APPROVAL: 有疑问/风险，等待人工审批
COMPLETED:         任务完成（通过或拒绝）
```

### Python 实现

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import json, time, logging
from anthropic import Anthropic

log = logging.getLogger(__name__)
client = Anthropic()

class State(Enum):
    IDLE = auto()
    ANALYZING = auto()
    REVIEWING = auto()
    AWAITING_APPROVAL = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass
class AgentContext:
    """Agent 运行时上下文，可序列化用于持久化"""
    task_id: str
    document: str
    state: State = State.IDLE
    
    # 分析结果
    analysis: Optional[dict] = None
    risk_score: float = 0.0
    decision: Optional[str] = None  # "approve" | "reject" | "escalate"
    
    # 审批相关
    approval_request_id: Optional[str] = None
    approval_result: Optional[bool] = None
    approver_notes: str = ""
    
    # 错误处理
    error: Optional[str] = None
    retry_count: int = 0
    
    # 审计日志
    state_history: list[dict] = field(default_factory=list)
    
    def log_transition(self, from_state: State, to_state: State, reason: str = ""):
        self.state_history.append({
            "from": from_state.name,
            "to": to_state.name,
            "reason": reason,
            "timestamp": time.time(),
        })
        log.info(f"[{self.task_id}] {from_state.name} → {to_state.name}: {reason}")

class DocumentReviewAgent:
    """
    文档审核 Agent，使用有限状态机管理流程。
    """
    
    # 状态转换表：{当前状态: [(目标状态, 守卫函数, 动作函数)]}
    TRANSITIONS = {
        State.IDLE: [
            (State.ANALYZING, lambda ctx: ctx.document is not None, None),
        ],
        State.ANALYZING: [
            (State.REVIEWING, lambda ctx: ctx.analysis is not None, None),
            (State.FAILED, lambda ctx: ctx.error is not None, None),
        ],
        State.REVIEWING: [
            (State.COMPLETED, lambda ctx: ctx.decision in ["approve", "reject"], None),
            (State.AWAITING_APPROVAL, lambda ctx: ctx.decision == "escalate", None),
        ],
        State.AWAITING_APPROVAL: [
            (State.COMPLETED, lambda ctx: ctx.approval_result is not None, None),
        ],
    }
    
    def __init__(self, approval_system=None):
        self.approval_system = approval_system
    
    def transition_to(self, ctx: AgentContext, new_state: State, reason: str = "") -> bool:
        """执行状态转换"""
        allowed_transitions = self.TRANSITIONS.get(ctx.state, [])
        
        for target_state, guard, action in allowed_transitions:
            if target_state == new_state:
                if guard and not guard(ctx):
                    log.warning(f"Guard failed for transition {ctx.state.name} → {new_state.name}")
                    return False
                
                old_state = ctx.state
                ctx.log_transition(old_state, new_state, reason)
                ctx.state = new_state
                
                if action:
                    action(ctx)
                
                return True
        
        log.error(f"Invalid transition: {ctx.state.name} → {new_state.name}")
        return False
    
    # ---- 各状态的处理逻辑 ----
    
    def _handle_analyzing(self, ctx: AgentContext):
        """分析文档，提取结构化信息"""
        try:
            resp = client.messages.create(
                model="claude-opus-4-5",
                system="""你是一个文档审核助手。分析文档并返回 JSON：
{
  "document_type": "合同/报告/申请/其他",
  "key_points": ["要点1", "要点2"],
  "risk_factors": ["风险1", "风险2"],
  "risk_score": 0.0到1.0,
  "recommended_action": "approve/reject/escalate",
  "reasoning": "推理过程"
}""",
                messages=[{
                    "role": "user",
                    "content": f"请审核以下文档：\n\n{ctx.document}"
                }],
                max_tokens=1024,
            )
            
            analysis_text = resp.content[0].text
            ctx.analysis = json.loads(analysis_text)
            ctx.risk_score = ctx.analysis.get("risk_score", 0.0)
            
            self.transition_to(ctx, State.REVIEWING, "Analysis completed")
            
        except json.JSONDecodeError as e:
            ctx.error = f"Failed to parse analysis: {e}"
            self.transition_to(ctx, State.FAILED, "JSON parse error")
        except Exception as e:
            ctx.error = str(e)
            ctx.retry_count += 1
            if ctx.retry_count < 3:
                log.warning(f"Retrying analysis ({ctx.retry_count}/3)")
                self._handle_analyzing(ctx)  # 重试
            else:
                self.transition_to(ctx, State.FAILED, "Max retries exceeded")
    
    def _handle_reviewing(self, ctx: AgentContext):
        """根据分析结果做决策"""
        if ctx.analysis is None:
            ctx.error = "No analysis available"
            self.transition_to(ctx, State.FAILED, "Missing analysis")
            return
        
        recommended = ctx.analysis.get("recommended_action", "escalate")
        risk_score = ctx.risk_score
        
        # 自动决策逻辑（高置信度情况）
        if risk_score < 0.2:
            ctx.decision = "approve"
            self.transition_to(ctx, State.COMPLETED, f"Auto-approved (risk={risk_score:.2f})")
        elif risk_score > 0.8:
            ctx.decision = "reject"
            self.transition_to(ctx, State.COMPLETED, f"Auto-rejected (risk={risk_score:.2f})")
        else:
            # 中等风险，需要人工审批
            ctx.decision = "escalate"
            self.transition_to(ctx, State.AWAITING_APPROVAL, f"Escalated (risk={risk_score:.2f})")
    
    def _handle_awaiting_approval(self, ctx: AgentContext):
        """请求人工审批，等待结果"""
        if self.approval_system is None:
            # 无审批系统，模拟交互
            print(f"\n[需要人工审批]")
            print(f"文档类型: {ctx.analysis.get('document_type')}")
            print(f"风险评分: {ctx.risk_score:.2f}")
            print(f"风险因素: {ctx.analysis.get('risk_factors', [])}")
            print(f"AI 推荐: {ctx.analysis.get('recommended_action')}")
            
            decision = input("批准 (y) 还是拒绝 (n)？: ").strip().lower()
            ctx.approval_result = decision == "y"
        else:
            # 异步审批系统
            req_id = self.approval_system.create_request(
                task_id=ctx.task_id,
                data=ctx.analysis,
            )
            ctx.approval_request_id = req_id
            # 等待审批（实际场景中这里是异步 wait）
            ctx.approval_result = self.approval_system.wait_for_result(req_id)
        
        reason = "Human approved" if ctx.approval_result else "Human rejected"
        self.transition_to(ctx, State.COMPLETED, reason)
    
    # ---- 主运行循环 ----
    
    def run(self, document: str, task_id: str = None) -> AgentContext:
        """运行 Agent 直到完成或失败"""
        import uuid
        ctx = AgentContext(
            task_id=task_id or str(uuid.uuid4()),
            document=document,
        )
        
        # 启动
        self.transition_to(ctx, State.ANALYZING, "Starting analysis")
        
        # 状态驱动循环
        max_steps = 10
        step = 0
        
        while ctx.state not in (State.COMPLETED, State.FAILED) and step < max_steps:
            step += 1
            
            if ctx.state == State.ANALYZING:
                self._handle_analyzing(ctx)
            elif ctx.state == State.REVIEWING:
                self._handle_reviewing(ctx)
            elif ctx.state == State.AWAITING_APPROVAL:
                self._handle_awaiting_approval(ctx)
            else:
                log.warning(f"Unhandled state: {ctx.state}")
                break
        
        if step >= max_steps:
            ctx.error = "Max steps exceeded"
            ctx.state = State.FAILED
        
        return ctx
```

---

## 状态持久化（支持长时间任务）

```python
import pickle, redis

class PersistentAgentRunner:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.agent = DocumentReviewAgent()
    
    def save_state(self, ctx: AgentContext):
        """序列化并保存 Agent 状态"""
        key = f"agent:state:{ctx.task_id}"
        data = {
            "task_id": ctx.task_id,
            "document": ctx.document,
            "state": ctx.state.name,
            "analysis": ctx.analysis,
            "risk_score": ctx.risk_score,
            "decision": ctx.decision,
            "approval_request_id": ctx.approval_request_id,
            "approval_result": ctx.approval_result,
            "error": ctx.error,
            "retry_count": ctx.retry_count,
            "state_history": ctx.state_history,
        }
        self.redis.setex(key, 86400, json.dumps(data))
    
    def load_state(self, task_id: str) -> Optional[AgentContext]:
        """恢复 Agent 状态，继续执行"""
        key = f"agent:state:{task_id}"
        raw = self.redis.get(key)
        if not raw:
            return None
        
        data = json.loads(raw)
        ctx = AgentContext(
            task_id=data["task_id"],
            document=data["document"],
            state=State[data["state"]],
            analysis=data["analysis"],
            risk_score=data["risk_score"],
            decision=data["decision"],
            approval_request_id=data["approval_request_id"],
            approval_result=data["approval_result"],
            error=data["error"],
            retry_count=data["retry_count"],
            state_history=data["state_history"],
        )
        return ctx
    
    def resume(self, task_id: str) -> AgentContext:
        """从保存的状态恢复并继续执行"""
        ctx = self.load_state(task_id)
        if ctx is None:
            raise ValueError(f"No saved state for task {task_id}")
        
        log.info(f"Resuming task {task_id} from state {ctx.state.name}")
        return self.agent.run_from_context(ctx)
```

---

## 用 LangGraph 实现（更简洁）

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class ReviewState(TypedDict):
    document: str
    analysis: dict
    risk_score: float
    decision: str
    approved: bool
    error: str

def analyze_node(state: ReviewState) -> ReviewState:
    # ... LLM 分析
    return {**state, "analysis": analysis, "risk_score": risk_score}

def review_node(state: ReviewState) -> ReviewState:
    risk = state["risk_score"]
    if risk < 0.2:
        return {**state, "decision": "approve"}
    elif risk > 0.8:
        return {**state, "decision": "reject"}
    return {**state, "decision": "escalate"}

def approval_node(state: ReviewState) -> ReviewState:
    # 人工审批
    approved = request_human_approval(state["analysis"])
    return {**state, "approved": approved}

def route_after_review(state: ReviewState) -> str:
    if state["decision"] == "escalate":
        return "await_approval"
    return END

# 构建图
workflow = StateGraph(ReviewState)
workflow.add_node("analyze", analyze_node)
workflow.add_node("review", review_node)
workflow.add_node("await_approval", approval_node)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "review")
workflow.add_conditional_edges("review", route_after_review)
workflow.add_edge("await_approval", END)

app = workflow.compile()
```

---

## 一个朴素结论

> Agent 代码的复杂度不来自 LLM 调用，来自状态管理。
>
> 在你感觉到"if 嵌套太深，搞不清楚现在在做什么"的时候，就是引入状态机的时候了。
>
> **不需要上来就用 LangGraph。先用 Python 的 Enum + dataclass 做一个简单的 FSM，能工作就是好设计。**
