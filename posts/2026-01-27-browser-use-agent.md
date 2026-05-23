---
layout: post
title: Browser-Use / Computer-Use Agent 实战 — 让 AI 真的"用电脑"
date: 2026-01-27
topic: "Agent 与工具"
tags: [AI, Agent, Browser, Computer Use]
excerpt: 2024 末 Anthropic Computer Use、2025 OpenAI Operator 把 agent 从"调 API" 推向了 "用 GUI"。原理、工具、可用性现状。
permalink: /posts/2026-01-27-browser-use-agent.html
---

## 从 API agent 到 GUI agent

API agent：用结构化 API 完成任务（调 SaaS / 数据库）。
GUI agent：**像人一样用浏览器 / 桌面**——点击、滚动、填表、截图判断。

为什么需要 GUI agent？因为**很多系统没有 API**：
- 内部老系统
- 第三方网站
- 桌面软件
- 政务办公平台

GUI agent 是这些场景的最后一公里。

## 主流方案

### 1. Anthropic Computer Use

Claude 4.x 内置 `computer` 工具，能：
- 截屏
- 移动鼠标 + 点击
- 输入键盘
- 看到屏幕做下一步

```python
response = client.messages.create(
    model="claude-opus-4-7",
    tools=[{
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": 1024,
        "display_height_px": 768,
    }, ...],
    messages=[{"role": "user", "content": "帮我在浏览器订一张明天去上海的高铁"}],
)
```

**优势**：模型原生支持，规划能力强。
**劣势**：还在 beta，慢、贵（截图 token 很多）、偶尔点错。

### 2. OpenAI Operator

GPT-4.5 之上的浏览器 agent。专门跑在云端虚拟浏览器里。
**优势**：托管，无须自己搭沙箱。**劣势**：仅 Pro 用户可用，每月限次。

### 3. 开源框架

| 框架 | 特点 |
|---|---|
| **browser-use** | 简单易用，纯浏览器自动化 |
| **Skyvern** | 商业 + 开源，企业级 |
| **Playwright + LLM** | 自己搭，最灵活 |
| **AgentQL** | 提取数据特化 |

## 工作流（以浏览器为例）

```
[Loop]
  截图当前页面
    ↓
  LLM 看截图 + 任务，决定下一步
  （"我看到搜索框，输入 '北京 上海 高铁' 然后点搜索"）
    ↓
  执行动作（输入 / 点击）
    ↓
  等待页面响应
    ↓
  循环
```

每步约 3-10 秒，复杂任务可能要 30+ 步，**总耗时 5-20 分钟很常见**。

## 工程坑

### 1. 截图 token 暴炸

每步都截图，每张图相当于 1500-3000 tokens。
跑 30 步 = 50k-100k token，**单次任务 $1-5 很正常**。

缓解：
- 降低分辨率（1024×768 足够）
- 只在关键步骤截图
- 用 DOM 替代（如果 web）

### 2. 元素定位不稳

页面跳变 / 异步加载 / iframe → agent 看着不对，点错。
- 加 wait 等待元素出现
- 失败时重试 + 重新规划

### 3. 反爬虫

Cloudflare / reCAPTCHA / 设备指纹检测会把 agent 当机器人。
- 用真人浏览器 + Playwright stealth
- 控制点击间隔（别太机械）
- 接受会有部分场景搞不定

### 4. 安全

GUI agent 能做的事远多于 API agent——能买东西、发邮件、删数据。
**必须有 human approval 在不可逆操作前**：

```
Agent: 准备点 "确认支付 ¥500"
→ 弹给人确认
→ 人点 yes → 真执行
→ 人点 no → 不执行 + 让 agent 解释为什么
```

## 当前可用性

| 任务复杂度 | 成功率 |
|---|---|
| 简单（搜索 / 表单填写）| 70-85% |
| 中等（订票 / 下单）| 40-60% |
| 复杂（多页流程 / 跨网站）| 10-30% |

**2026 年还不算"生产可用"**，但简单场景已经能省人力。
24-26 月会进化到 "60-80% 中等场景可用"。

## 实战建议

- **简单任务**：可以用，配合 human-in-the-loop 兜底
- **关键业务流程**：仍然写专门集成（爬虫 / API / RPA）
- **探索性场景**：值得投入，未来 1-2 年会大爆发

## 一个朴素结论

> GUI agent 是 LLM 应用的"最终形态"——能用电脑就能干一切。
> 但 2026 年还在早期，**简单场景能用，复杂场景靠人**。
>
> 投资学习这套技术，但不要急于全栈替代。
