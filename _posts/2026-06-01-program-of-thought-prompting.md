---
layout: post
title: Program-of-Thought Prompting — 让模型写代码来推理，把计算交给解释器
date: 2026-06-01
topic: "Prompt 与推理"
tags: [Prompt, 推理, PoT, LLM, Python]
excerpt: Chain-of-Thought 让模型把推理写出来，但数值计算还是靠"猜"——Program-of-Thought 换了一条路：让模型输出 Python 代码，把计算本身交给解释器执行，彻底消除算术错误。
permalink: /posts/2026-06-01-program-of-thought-prompting.html
---

你有没有遇到过这种情况：让大模型用 Chain-of-Thought 解一道有点复杂的数学题，推理步骤写得头头是道，一步步看下来逻辑严密，最后答案就是错了。不是偶尔，是你只要题目够绕，十次里能碰到好几次。

原因其实很直白：模型在预测下一个 token，不是在做竖式加法。它"心算"进位的时候会出错，多步骤乘法更是重灾区。模型有很强的推理能力，但它从来不是一台计算器，强迫它扮演计算器只会暴露它最弱的那一面。很多人以为扩大上下文窗口或者换更大的模型就能解决，但本质问题没变——只要计算还是由 token 预测来完成，算术错误就始终存在。

Chain-of-Thought 解决了"怎么想"的问题，但没解决"怎么算"的问题。Program-of-Thought（PoT）Prompting 是对这个痛点的直接回应：**让模型只负责把推理逻辑翻译成代码，把计算本身交给 Python 解释器执行**。分工明确，各司其职。

## CoT vs PoT：核心差别在哪里

两者的出发点一致——都是让模型在给出最终答案之前先"慢下来"，把中间步骤显式写出来，而不是直接蹦出答案。区别在于"慢下来"之后做什么。

Chain-of-Thought 让模型用自然语言写推理链，整个过程从理解题目到最终答案都由模型独立完成。这在逻辑推导类问题上效果很好，但涉及具体数值计算时，模型依然是在"猜"那个数字——它看过大量的数学文本，知道乘法应该怎么做，但它在做 token 预测，不在做真正的乘法运算。

Program-of-Thought 让模型把推理过程写成 Python 代码，然后把这段代码交给解释器实际执行，得到的执行结果才是最终答案。模型的职责从"算出答案"变成了"写出正确的逻辑"，计算这件事完全转移给了不会犯算术错误的机器。

| 维度 | Chain-of-Thought | Program-of-Thought |
|------|------------------|--------------------|
| 输出格式 | 自然语言推理步骤 | Python 代码 |
| 计算执行者 | 模型自己"心算" | Python 解释器实际运行 |
| 典型错误来源 | 逻辑错 + 算术错 | 只剩逻辑错，算术不会错 |
| 适合场景 | 常识推理、逻辑判断 | 数学计算、数据处理、统计 |
| 可复现性 | 相同 prompt 结果可能不同 | 代码确定，执行结果确定 |

## 一个最小可运行的示例

先看 PoT 的 prompt 大概长什么样。你告诉模型输出规则，然后给它一道题：

```text
请用 Python 代码解决下面的问题。
规则：
- 只输出纯 Python，不要任何解释文字
- 把最终答案赋值给变量 answer
- 不要使用 input()，所有数值直接硬编码

题目：小明买了 37 件商品，每件定价 128 元，享受 8.5 折优惠，
      另外用优惠券再减 50 元，实际支付多少元？
```

模型输出的代码大概是这样：

```python
unit_price = 128
quantity = 37
discount = 0.85
coupon = 50

subtotal = unit_price * quantity * discount
answer = subtotal - coupon
```

你把这段代码交给解释器跑，得到 `answer = 3972.8`，精确无误。模型没有"心算"任何数字，只是把解题逻辑翻译成了几行语句。这几行语句本身的正确性，是模型负责保证的；执行这几行语句、算出结果，是 Python 负责保证的。把这两件事分清楚，整个流程的可靠性就高了一个层次。

## 用 Python 搭一条 PoT 流水线

理解了原理，搭一条端到端的流水线并不复杂。核心逻辑只有三步：发请求给模型、提取代码块、执行代码并读取 `answer`。

```python
import re
import subprocess
import textwrap
import anthropic

client = anthropic.Anthropic()

POT_SYSTEM = """
你是一个数学推理助手。收到问题后，输出能解决该问题的 Python 代码。
要求：
1. 只输出代码块，不要任何解释性文字
2. 最终答案必须赋值给变量 answer
3. 使用英文变量名，不要用 import 引入外部库（math 和 decimal 除外）
"""

def extract_code(text: str) -> str:
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text

def pot_solve(problem: str) -> str:
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=POT_SYSTEM,
        messages=[{"role": "user", "content": problem}]
    )
    code = extract_code(resp.content[0].text)
    runnable = textwrap.dedent(code) + "\nprint(answer)"

    result = subprocess.run(
        ["python3", "-c", runnable],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        return f"执行出错：{result.stderr.strip()}"
    return result.stdout.strip()

print(pot_solve("一列火车以 80 km/h 行驶 2.5 小时，另一列以 120 km/h 行驶 1.75 小时，哪列更远，相差多少 km？"))
```

`extract_code` 函数是个小但不能省的细节。模型有时会在代码块外夹一两句自然语言解释，直接 exec 整段输出会报语法错误。用正则把代码块单独提出来，是最稳定的做法。

`subprocess.run` 的 `timeout=10` 也不是可选项。如果模型生成的代码里有死循环，没有超时保护会让你的进程永远挂在那里。

## 什么时候用 PoT，什么时候还是用 CoT

选择 PoT 还是 CoT，核心判断标准是：**这道题的答案，是靠计算得来的，还是靠推导得来的？**

计算得来的答案，意味着有明确的数值运算步骤，结果是一个可以被验证的数字或结构。这种情况用 PoT 更可靠，模型只需要把逻辑写对，Python 负责算对。

推导得来的答案，比如逻辑谜题、常识判断、"谁的方案更合理"这类问题，本质上不涉及精确计算，强行用代码表达反而绕了弯路，这种情况 CoT 更合适。

**适合 PoT 的典型场景：**
- 金融计算：复利、折扣、税率叠加
- 物理题：速度、加速度、功率涉及多步公式
- 数据处理：从一组数里找最大值、求平均、按条件筛选
- 日期时间计算：N 天后是星期几、两个日期相差多少工作日

**继续用 CoT 更合适的情况：**
- 推理谁是凶手、条件能否同时满足
- 常识性判断，答案无法用数字或代码量化
- 需要模型解释"为什么"，输出本身是自然语言

两者也可以组合使用：先用 CoT 理清解题思路，再用 PoT 执行计算部分。这种 **Hybrid Reasoning** 做法在竞赛级数学题上的效果，比单独使用任何一种都要好。具体实现上，你可以先发一条 CoT 请求让模型分析题目结构、拆解子问题，然后把这份分析作为上下文附在第二条 PoT 请求里，让模型根据已有的推理框架写出代码。两轮请求的成本并不高，换来的是可靠性的双重保障。

## 沙箱安全：生产环境必须做的防护

直接把模型输出的代码交给 `exec()` 或 `subprocess.run()` 执行，在实验和内部工具里没问题，但在面向用户的生产系统里是有风险的。用户可以通过精心构造的输入，诱导模型生成包含危险操作的代码，比如读写文件、发起网络请求、执行系统命令。

最轻量的防护是在执行前做静态检查，用 `ast` 模块解析代码，拒绝包含可疑 import 的代码：

```python
import ast

ALLOWED_MODULES = {"math", "decimal", "fractions", "statistics", "datetime"}

def is_safe_code(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name.split(".")[0] for alias in node.names]
            if any(m not in ALLOWED_MODULES for m in modules):
                return False
    return True
```

更彻底的方案是使用 **E2B** 这类代码沙箱服务，把执行环境隔离到一个无法访问主机文件系统和网络的容器里。对于对外提供服务的产品，这基本上是必选项，不是加分项。值得一提的是，沙箱方案除了安全隔离，还附带了一个实用的副产品：你可以给每次执行设置独立的内存和 CPU 限制，防止某次复杂计算把你的整个服务器打满。

## 踩坑清单

- **模型忘记给 `answer` 赋值**：prompt 里强调不止一次，可以加"如果没有赋值 answer，代码视为无效，必须重写"
- **代码里夹了解释文字**：不要直接 exec 整段输出，一定要用正则提取代码块
- **浮点精度坑**：金融场景换用 `decimal.Decimal`，用 float 算钱会碰到 0.1 + 0.2 != 0.3 的经典坑
- **中文变量名**：Python 3 语法上允许，但会让代码提取和解析工具出问题，prompt 里明确要求用英文命名
- **超时不设**：只需要一次死循环，你的服务就会永远挂着，`timeout=10` 是强制要求不是建议

PoT 不是 CoT 的竞争者，是给 CoT 装上了计算器。让模型做它擅长的事，让工具做它擅长的事——这条分工原则在 AI 工程里到处都成立，PoT 只是其中一个具体的落地形式。
