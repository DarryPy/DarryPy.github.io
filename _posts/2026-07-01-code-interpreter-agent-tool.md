---
layout: post
title: Code Interpreter 工具实战 — Agent 自己写代码、自己跑、再看结果
date: 2026-07-01
topic: "Agent 与工具"
tags: [Agent, Code Interpreter, 工具调用, Python, 沙箱]
excerpt: Code Interpreter 让 Agent 真正"动手算"而不只是"嘴巴说"。本文拆解它的工作原理、沙箱隔离方案、IPython kernel 状态管理，以及最常踩的五个坑：包缺失、输出爆炸、状态污染、死循环、路径硬编码。
permalink: /posts/2026-07-01-code-interpreter-agent-tool.html
---

你让 LLM 计算一道复杂的统计题，它很可能给你一个"看起来合理"的答案——但不一定真的对。语言模型本质上在做 token 预测，不是在执行计算。面对数据分析、算法验证、文件处理这类任务，依赖模型"推断"出答案远不如交给真实的 Python 运行时跑一遍。这就是 Code Interpreter 工具的核心价值：把执行权从模型脑子里拿出来，放到真实的运行时里。

ChatGPT 的 Code Interpreter 功能让这个模式广为人知，但在自己的 Agent 里复现它，有一套需要认真对待的工程细节。

## 三层架构：接口、执行、状态

一个可用的 Code Interpreter 工具由三层组成，缺一不可。

**接口层**是 Agent 调用的 tool schema，接收代码字符串，返回 stdout / stderr / 执行结果。这一层决定模型"如何描述"它想做的事。schema 要写清楚工具能干什么、不能干什么，包括可用的包、工作目录约定、超时限制——这些信息直接影响模型生成代码的质量。

**执行层**是真正跑代码的运行时，可以是本机 subprocess、Docker 容器、或 E2B / Modal 之类的托管沙箱。这一层决定安全边界在哪里，是整个方案里安全风险最集中的地方。

**状态层**负责维护跨调用的会话状态——变量、import 过的库、已加载的 DataFrame。这是最容易被忽视的一层。如果每次 tool call 都开一个全新解释器，Agent 就没法在第一次调用里 `import pandas as pd`，然后在第二次调用里直接用 `df = pd.read_csv(...)`。复杂的数据分析任务往往需要跨五六次 tool call 逐步完成，状态层不通，任务就只能退化成一次性脚本。

```python
# tool schema 示例（Claude / OpenAI 格式通用）
{
    "name": "execute_python",
    "description": (
        "在 Python 沙箱里运行代码，返回 stdout 和执行结果。"
        "可用包：pandas、numpy、matplotlib、scikit-learn、requests。"
        "文件读写限 /workspace/ 目录。单次超时 30 秒。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "合法的 Python 代码片段，支持多行"
            }
        },
        "required": ["code"]
    }
}
```

Agent 把代码字符串塞进 tool call，执行层跑完把 stdout 或异常堆栈作为 tool result 回传，模型读到结果再决定下一步：是修 bug、接着计算，还是直接汇报结论。这个"写代码 → 跑 → 看结果 → 再写"的循环，正是 Code Interpreter 工具的核心交互模式。

## 沙箱方案对比

| 方案 | 隔离强度 | 冷启动延迟 | 适用场景 |
|------|----------|------------|----------|
| `exec()` 内联 | 无 | 极低 | 只用于本地调试，绝不上生产 |
| `subprocess` + timeout | 进程级 | 低（毫秒）| 内网受信任的自有环境 |
| Docker 容器 | 容器级 | 中（秒级）| 大多数自部署场景 |
| E2B / Modal 托管 | VM 级 | 中高（1-3s）| SaaS、多租户、代码来源不可信 |

生产环境的底线是**进程隔离 + 强制超时**。哪怕只是把代码写到临时文件再用 `subprocess.run(['python', tmpfile], timeout=30, capture_output=True)` 跑，也能防住死循环和内存爆炸。

如果代码来自用户输入，进程级隔离远远不够。裸 `exec()` 里任何 `os.system('rm -rf /')` 都会直接在你的主机上执行。这种场景必须上容器或 VM——容器隔离文件系统和进程命名空间，VM 则在 hypervisor 层彻底切断。

## 状态管理：复用 IPython kernel

跨调用保活状态，最自然的方案是复用一个 IPython kernel，在整个 Agent 会话期间一直保持它运行：

```python
from jupyter_client import KernelManager

km = KernelManager()
km.start_kernel()
kc = km.client()
kc.start_channels()

def execute_code(code: str, timeout: int = 30) -> dict:
    kc.execute(code)
    outputs, error = [], None
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=timeout)
        except Exception:
            km.restart_kernel()         # 超时后重启，否则 kernel 卡死
            return {"error": "执行超时，kernel 已重启"}
        t = msg["msg_type"]
        if t == "stream":
            outputs.append(msg["content"]["text"])
        elif t == "error":
            error = "\n".join(msg["content"]["traceback"])
            break
        elif t == "status" and msg["content"]["execution_state"] == "idle":
            break
    if error:
        return {"error": error[:2000]}  # 截断超长堆栈
    raw = "".join(outputs)
    if len(raw) > 3000:
        raw = raw[:3000] + f"\n[已截断，原始输出 {len(raw)} 字符]"
    return {"output": raw}
```

kernel 在对话周期里保活，变量在多次 tool call 之间共享，就像 Jupyter notebook 一格一格往下跑。任务结束或出错后重启 kernel，下一个独立任务从干净状态开始。

## 五个高频坑

**坑 1：包缺失**。镜像里没装 `scikit-learn`，Agent 的 `import sklearn` 直接 `ModuleNotFoundError`。对策有两个：一是在工具描述里列出可用包白名单，让模型知道能用什么；二是构建镜像时预装所有常用依赖，不依赖运行时 `pip install`（安全且快）。

**坑 2：输出爆炸**。Agent 在循环里打印调试信息，轻松输出十万行，全量回传直接把 context window 塞满，后续 token 全花在读输出上。对策：工具层硬截断 stdout 到 3000 字符，超出部分告知模型"已截断，原始长度 N 字符"——模型知道有遗漏，会主动缩减输出量。

**坑 3：状态污染**。第一步把 `df` 的某一列清洗掉了，第二步以为 `df` 还是原始数据然后出了奇怪的结果。对策：在 Agent 的系统 prompt 里提示"每个子任务开始前应声明自己依赖哪些变量，并在必要时重新加载数据"；或者让工具在每次调用前 dump 当前 kernel 里所有变量名供模型参考。

**坑 4：死循环与 kernel 卡死**。`while True` 进了死循环，外层 timeout 是最后一道闸。subprocess 方案靠 `timeout` 参数强杀进程；IPython kernel 方案要在外层设线程级超时，超时后必须 `restart_kernel()`——光 `interrupt_kernel()` 有时不够，某些 IO 阻塞无法被信号打断。

**坑 5：路径硬编码**。Agent 写出 `/home/ubuntu/data.csv`，沙箱根本没这个路径。在工具描述里明确约定工作目录（比如统一用 `/workspace/`），用户上传的文件挂载到这里，Agent 读写文件一律走这个路径，杜绝猜测。

## 踩坑清单

- [ ] 沙箱选型：内网可用 subprocess；对外或代码不可信必须容器 / VM
- [ ] 超时设置：每次 tool call 独立 timeout，超时后重启 kernel 而非只 interrupt
- [ ] 输出截断：stdout 硬限 3000 字符，超出告知模型有遗漏
- [ ] 包白名单：镜像预装，schema 里写清楚，不依赖运行时安装
- [ ] 状态隔离：独立任务开新 kernel，同一任务内共享 kernel
- [ ] 工作目录约定：路径写死在工具描述里，文件挂载保持一致

Code Interpreter 让 Agent 从"预测答案"变成"验证答案"——代价是你得把沙箱、超时、输出截断这三件事做扎实，否则它从"帮你算"变成"帮你炸服务器"只需要一行代码。
