---
layout: post
title: Agent 安全沙箱设计 — 让 Agent 干活但不能干坏事
date: 2026-01-05
topic: "Agent 与工具"
tags: [AI, Agent, 安全]
excerpt: Agent 能调工具、改文件、执行代码——也意味着能搞破坏。多层沙箱设计：权限、网络、文件系统、API、人工审批。
permalink: /posts/2026-01-05-agent-sandbox.html
---

## Agent 时代的新威胁

API agent：模型生成文本，最多说错话。
Tool / Code / Browser agent：能执行操作，**说错话变成做错事**：

- agent 帮你写代码 → 顺手 `rm -rf /`
- agent 帮你处理邮件 → 把账户密码发到外部
- agent 帮你查数据 → 顺手 `DROP TABLE`

不是科幻，是 2025 年已经发生过的真实事故。

## 5 层沙箱模型

最小权限 + 多层防御：

```
[Layer 5] Human Approval     ← 不可逆操作
[Layer 4] API Whitelist      ← 工具/参数白名单
[Layer 3] Network Isolation  ← 限网
[Layer 2] Filesystem Sandbox ← 限文件
[Layer 1] OS-level Container ← Docker / VM
```

## Layer 1: 容器隔离

Code execution agent 必须在容器里跑：

```python
import docker
client = docker.from_env()
container = client.containers.run(
    "python:3.11-slim",
    command=f"python -c '{user_code}'",
    network_mode="none",     # 无网
    mem_limit="512m",
    cpus="1",
    read_only=True,          # 文件系统只读
    tmpfs={"/tmp": "size=100M"},
    detach=True,
    auto_remove=True,
)
```

Docker / gVisor / Firecracker 哪个都行，**就是不能跑在 host 上**。

## Layer 2: 文件系统沙箱

agent 能读写哪些路径？

```python
ALLOWED_READ = ["/workspace/", "/data/public/"]
ALLOWED_WRITE = ["/workspace/output/"]
DENIED = ["/etc/", "/.ssh/", "/.env"]

def fs_access(path, mode):
    abs_path = os.path.realpath(path)  # 防 symlink 攻击
    if mode == "read":
        return any(abs_path.startswith(p) for p in ALLOWED_READ)
    if mode == "write":
        return any(abs_path.startswith(p) for p in ALLOWED_WRITE)
```

实战注意：
- 用 `realpath` 解决 symlink / `..` 路径欺骗
- 默认 deny，白名单 allow
- 隔离敏感目录（`.ssh`、`.env`、`.aws`）

## Layer 3: 网络隔离

agent 能访问哪些网络？

```
✅ 允许：
- 公司内部 API（指定 host + port）
- 已批准的第三方 API

❌ 拒绝：
- 任意外部 IP
- 内部管理端口（DB / Redis 直连）
- 本地 metadata server（云上 169.254.169.254 这种）
```

Docker 用 `--network` 限制；K8s 用 NetworkPolicy。

## Layer 4: API / Tool 白名单

哪些工具暴露给 agent，每个工具能用什么参数：

```python
TOOLS = [
    {
        "name": "read_user_profile",
        "level": "safe",  # 自由调
    },
    {
        "name": "send_email",
        "level": "approval_required",  # 必须人工批
        "args_constraints": {
            "to": "pattern: @yourcompany\\.com$",  # 只能内部
        }
    },
    {
        "name": "execute_sql",
        "level": "approval_required",
        "args_constraints": {
            "sql": "regex: ^SELECT",  # 只能 SELECT，不能 DROP/DELETE
        }
    },
    {
        "name": "drop_table",
        "level": "disabled",  # 干脆不暴露
    },
]
```

## Layer 5: Human-in-the-loop

危险操作必须人审：

```python
async def call_tool(name, args, user):
    tool = TOOLS[name]
    if tool.level == "approval_required":
        approval = await human_approve(
            user=user,
            tool=name,
            args=args,
            reason=agent_reasoning(),  # 让 agent 说为什么要做
        )
        if not approval:
            return {"error": "user_rejected", "message": "操作被用户拒绝"}
    return tool.execute(args)
```

UI 设计：

```
🤖 Agent 想执行：
  工具：send_email
  参数：
    to: external@notyourdomain.com   ⚠️ 外部地址
    subject: ...
  
  Agent 解释：用户要求发邮件到这个地址
  
  [ ✓ 批准 ]  [ ✗ 拒绝 ]  [ 详情 ]
```

危险参数（外部地址 / 大金额）要**视觉警告**。

## Code Execution Agent 专项

跑 LLM 生成的代码风险最大。专门防御：

1. **必须容器化**（前面说过）
2. **CPU / 内存 / 时间限制**：防死循环
3. **环境变量过滤**：不要把生产 secret 暴露给容器
4. **包安装限制**：白名单 pip 包
5. **输出大小限制**：防止 1GB 输出撑爆系统

Apple / Anthropic / OpenAI 的 code interpreter 都跑在严格沙箱里。

## 监控 + 审计

agent 每个 tool call 记录：

```
[2026-01-05 10:30:15]
agent: claude-opus-4-7
session: sess_abc
user: u_42
tool: send_email
args: { to: "...", subject: "..." }
approved_by: u_42 (human)
result: success
```

事后能查谁干了啥。

## 一份发布 checklist

- [ ] Code execution 在容器
- [ ] 文件系统白名单
- [ ] 网络白名单
- [ ] 工具按 safe/approval/disabled 分级
- [ ] 危险操作 human approval
- [ ] 所有 tool call 落审计日志
- [ ] 跑过红队工具滥用测试

## 一个朴素结论

> 强大的 agent = 危险的 agent。
> **能力越强，沙箱要越严**。
>
> 不要相信 agent "懂事"——它什么都信，包括恶意指令。
> 工具层 + 沙箱层把真实的"做坏事"通道堵死，才是 agent 时代的安全姿态。
