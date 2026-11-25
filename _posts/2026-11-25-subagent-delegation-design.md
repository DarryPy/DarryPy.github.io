---
layout: post
title: Sub-Agent 委派设计 — 主 Agent 怎么派活、隔离上下文、验收结果
date: 2026-11-25
topic: "Agent 与工具"
tags: [Agent, Sub-Agent, 上下文隔离]
excerpt: 主 Agent 把子任务交给 Sub-Agent，看似省 token，实则藏着上下文丢失、结果失真、递归失控三类坑，这篇讲清派活、隔离和验收的完整设计。
permalink: /posts/2026-11-25-subagent-delegation-design.html
---

你大概遇到过这种场景：主 Agent 接了一个大任务，要查二十个网页、读五个代码仓库，再汇总成报告。全塞进一个上下文，跑到一半 token 就爆了，前面读过的内容也被挤得模模糊糊。于是你想到拆出 Sub-Agent，让它们各自干活，只把结论交回来。

思路没错，但我第一次这么做时翻车得很彻底：子 Agent 不知道任务背景，查了一堆无关资料；回来的总结写着“已完成”，其实文件根本没写；还有一个子 Agent 又派了孙 Agent，最后整棵树跑了四十分钟。这篇把委派这件事拆成三块讲：派活、隔离、验收。

## 什么时候值得委派

委派不是免费的。每拉起一个 Sub-Agent，就要重新付一遍系统提示词和任务说明的 token，还要多一轮模型推理的延迟。所以判断标准很简单：子任务的中间过程很脏、很长，但最终结论很短，这时委派才划算，否则直接用工具调用就够了。

| 场景 | 直接调用工具 | 委派 Sub-Agent |
| --- | --- | --- |
| 查一个天气、读一个文件 | 合适 | 浪费 |
| 读完整个仓库找某个 bug 根因 | 上下文被代码淹没 | 合适，只回传结论 |
| 并行调研五个竞品 | 串行太慢 | 合适，可并行 |
| 需要和用户来回确认 | 合适 | 不合适，子 Agent 问不了人 |
| 一步写操作，如发邮件 | 合适 | 风险大，难以审计 |

我的经验是，凡是“读得多、写得少、结论可压缩”的任务，都适合委派。反过来，涉及写操作或者需要人工确认的步骤，留在主 Agent 手里更安全，因为主 Agent 持有完整上下文，也更容易被人类监督。

还有一个常被忽视的信号是可并行性。如果几个子任务之间没有依赖，比如分别调研五家公司的定价，委派后可以同时跑，总耗时接近单个任务。可如果后一步严重依赖前一步的细节，拆出去反而要反复传递中间结果，不如留在主 Agent 里顺序执行。

## 派活：子 Agent 什么都不知道

最容易忽略的一点是，Sub-Agent 是从零开始的，它看不到主 Agent 和用户聊过的任何内容。你在主对话里说过“只看 2025 年以后的数据”“报告用中文”，子 Agent 一概不知。派活时必须把这些约束显式写进任务说明里，不能指望它猜。

一份合格的任务说明至少包含四样东西：目标、背景、约束、产出格式。目标要具体到可以判断完成与否；背景给出文件路径、错误信息这类原始材料；约束写清楚语言、时间范围和禁止动作；产出格式最好给一个 JSON Schema，方便主 Agent 程序化解析。

产出格式里我还会固定加两个字段：status 和 evidence。status 只允许 done、partial、failed 三个值，逼着子 Agent 对完成度表态；evidence 是证据列表，放链接、行号或命令输出摘要。有了这两个字段，主 Agent 不用再从一大段自然语言里猜结论，汇总逻辑也能写成确定的代码。

```python
from dataclasses import dataclass, field

@dataclass
class DelegateTask:
    goal: str                 # 可验收的目标，例如“找出登录接口 500 的根因”
    context: str              # 文件路径、报错原文、用户已确认的约束
    constraints: list = field(default_factory=list)
    output_schema: dict = field(default_factory=dict)
    max_steps: int = 20       # 子 Agent 的步数上限
    can_delegate: bool = False  # 默认禁止再派孙 Agent

def build_brief(task: DelegateTask) -> str:
    lines = ["## 目标", task.goal, "## 背景", task.context, "## 约束"]
    lines += ["- " + c for c in task.constraints]
    lines += ["## 产出", "严格按下面的 JSON Schema 返回，不要附加解释："]
    lines.append(str(task.output_schema))
    return "\n".join(lines)
```

注意 `can_delegate` 默认是 False。如果不加这个开关，子 Agent 看到任务复杂，很可能也学主 Agent 去拆分委派，结果就是一棵失控的递归树。我一般只允许一层委派，确实需要两层时，也要在主 Agent 侧统一限制总并发和总步数。

写任务说明时，我会刻意站在一个完全不了解项目的新同事视角重读一遍：只看这段文字，他能不能开工？如果需要追问“哪个仓库”“哪个环境”“用什么格式交付”，说明背景还不够。宁可多写两百字，也比子 Agent 跑偏一轮再返工便宜得多。

## 隔离：上下文、工具、权限分开管

委派的核心价值就是上下文隔离：子 Agent 的中间过程留在它自己的会话里，主 Agent 只收到最终总结。这意味着子 Agent 读了十万 token 的日志，主 Agent 也只多出几百 token。但隔离不止于上下文，工具集和权限也应该按任务裁剪，而不是原样继承。

我的做法是给每类子任务配一个工具白名单。调研类只给搜索和网页读取，代码排查类只给只读文件和终端里的只读命令，任何子 Agent 都不给发消息、推送代码、删除文件这类工具。这样即使子 Agent 被网页里的 prompt injection 带偏，它也做不出不可逆的事情。

如果某个子任务确实需要写操作，我会让子 Agent 只产出“待执行的操作清单”，比如要改哪几个文件、每处改成什么，由主 Agent 审核后再统一执行。这相当于把子 Agent 降级为提案者，执行权始终收在一个地方，出问题时也只需要追查主 Agent 这一条链路。

还有一个细节是会话状态。子 Agent 用到的终端、浏览器会话最好也是独立的，否则两个并行的子 Agent 在同一个浏览器标签页里互相跳转，抓回来的内容会张冠李戴。共享状态只放在主 Agent 那一层，子 Agent 之间不直接通信，所有协调都经主 Agent 中转。

预算也要隔离。每个子 Agent 都应有独立的步数上限和超时时间，到点就强制收尾，并要求它在结束前交出已经拿到的部分结论。这样单个子任务卡死时，主 Agent 仍能拿到其余结果继续汇总，而不是整棵任务树陪着一个慢节点一起等待。

## 验收：子 Agent 的总结只是自述

这是我踩过最痛的坑。子 Agent 回来说“文件已写入 /tmp/report.md”，主 Agent 照单全收告诉用户，结果文件根本不存在。原因很简单：子 Agent 的总结是它自己生成的文本，不是事实。模型在步数用完或者工具报错时，依然倾向于写一个看起来完成了的总结。

所以验收要分两层。第一层是格式校验：用 output_schema 检查返回结构，不合格就带着错误信息让它重试一次，再不行就标记失败。第二层是事实校验：凡是声称产生了外部副作用的，比如写了文件、建了工单、提交了代码，主 Agent 必须拿到可验证的句柄，比如路径、ID、URL，再亲自去查一遍。

```python
import json, os

def verify_result(raw: str, schema_keys: list) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"ok": False, "reason": "非法 JSON: " + str(e)}
    missing = [k for k in schema_keys if k not in data]
    if missing:
        return {"ok": False, "reason": "缺少字段: " + ",".join(missing)}
    path = data.get("artifact_path")
    if path and not os.path.exists(path):
        return {"ok": False, "reason": "声称写入但文件不存在: " + path}
    return {"ok": True, "data": data}
```

另外，让子 Agent 在结果里附带证据来源，比如引用的 URL、代码行号、命令输出片段。主 Agent 汇总时可以抽查一两条，发现编造就整条结论作废。多个子 Agent 结论冲突时，不要让主 Agent 凭感觉二选一，而是把冲突点单独列出来，必要时再派一个子 Agent 专门核对。

最后，主 Agent 给用户汇报时要区分三种状态：已核实、未核实、失败。已核实的结论附上证据，未核实的明确标注来源只是子 Agent 自述，失败的写清楚卡在哪一步。宁可让报告显得不那么漂亮，也不能把一个没验证过的“成功”包装成事实交出去。

## 踩坑清单

- 任务说明里没写语言和时间范围，子 Agent 交回来一份英文的过时资料。
- 子 Agent 继承了主 Agent 全部工具，被网页内容诱导后差点调用了发消息工具。
- 没限制委派深度，一个调研任务裂变成十几个 Agent，账单和耗时一起爆炸。
- 并行子 Agent 共用一个浏览器会话，抓取结果互相串台。
- 直接相信“已完成”的自述，没核对文件或 ID，把假结果交给了用户。
- 子 Agent 需要用户确认的信息却无人可问，只能瞎猜，结论建立在错误前提上。

一句话总结：委派省的是主 Agent 的上下文，不是你的验收责任。
