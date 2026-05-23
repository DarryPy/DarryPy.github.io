---
layout: post
title: LLM 流式响应实战 — SSE / WebSocket / fetch streaming 怎么选
date: 2026-05-03
category: "工程实战"
tags: [AI, Streaming, SSE, 前端]
excerpt: "等几秒才出字" 的 LLM 体验跟 ChatGPT 那种打字机感觉差天差地。流式实现怎么做，三种协议怎么选。
permalink: /posts/2026-05-03-streaming-sse.html
---

## 为什么必须做流式

LLM 一次完整回答可能要 5-15 秒。
用户盯着一个"思考中..."等 10 秒，**体验是灾难**。
但如果第一个字 1 秒内出现，后面字一个一个跳出来——**等同样的 10 秒，体感完全不一样**。

流式输出不是"锦上添花"，是 LLM 应用的**基础体验**。

## 三种主流协议

### 1. SSE（Server-Sent Events）

主流模型 API 的默认流式协议。

**协议层**：HTTP 长连接，服务器持续往响应里写 `data: ...\n\n` 格式的事件。
**优点**：基于 HTTP，简单；浏览器原生 `EventSource` API；服务器只发不收，单向通信契合 LLM 输出模型。
**缺点**：单向；老 IE / 某些代理处理不好。

### 2. WebSocket

全双工通信。

**优点**：双向；可以中途 cancel / interrupt / send 额外指令。
**缺点**：协议复杂；防火墙 / 代理可能挡；连接管理麻烦。

### 3. fetch + ReadableStream（HTTP/2）

现代浏览器 `fetch` 返回的 Response 可以读流：

```js
const res = await fetch('/api/chat', { method: 'POST', body });
const reader = res.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value);
  // 处理 chunk
}
```

**优点**：用普通 fetch；可以传 POST 数据（SSE 只能 GET）；HTTP/2 多路复用。
**缺点**：要自己解析 chunk 边界。

## 怎么选

| 场景 | 推荐 |
|---|---|
| 浏览器对接 OpenAI / Anthropic API 风格的 streaming | fetch + ReadableStream |
| 需要中途打断（用户点"停止生成"） | WebSocket 或 fetch.signal AbortController |
| 简单展示 / 老浏览器 / GET 接口 | SSE |
| 多用户并发 + 实时双向 | WebSocket |

实战中**fetch + ReadableStream 是 web 应用首选**——简单、灵活、可中断。

## 一个最小后端实现（Node.js）

```js
import express from 'express';
import Anthropic from '@anthropic-ai/sdk';

const app = express();
const client = new Anthropic();

app.post('/api/chat', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const stream = await client.messages.stream({
    model: 'claude-sonnet-4-6',
    max_tokens: 1024,
    messages: req.body.messages,
  });

  for await (const event of stream) {
    if (event.type === 'content_block_delta') {
      res.write(`data: ${JSON.stringify({ text: event.delta.text })}\n\n`);
    }
  }
  res.write('data: [DONE]\n\n');
  res.end();
});
```

## 最小前端消费

```js
async function streamChat(messages) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n\n');
    buffer = lines.pop(); // 留下不完整的最后一行

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') return;
      const obj = JSON.parse(data);
      updateUI(obj.text); // 把 text 追加到当前消息
    }
  }
}
```

## 工程上要处理的坑

### 1. 中断与取消

用户点"停止"，前后端都要响应：

- 前端用 `AbortController` 取消 fetch
- 后端在写 chunk 前检查 `req.aborted`，发现就提前 `stream.controller.abort()`
- LLM API 那边收到 abort 后停止计费

### 2. 错误处理

流中途出错怎么办？

- 错误时发一个 `data: {"error": "...."}\n\n` 给前端
- 前端 reader 抛错时清晰展示错误（不是干掉之前的字）

### 3. 反向传输：心跳

代理 / CDN 中间层有时会因为"长时间没数据"主动断连。
解决：每 10-15 秒发一个 SSE 注释行 `:\n\n`（comment）作心跳。

### 4. 缓冲 / 节流

模型可能一次吐出来"几个 token"或者"一个完整句子"。
有时一秒钟 100 个 delta 会让 UI 卡。
前端做一层节流（requestAnimationFrame）或者按句号 / 逗号断句再 flush。

### 5. SSR / 边缘部署的兼容性

部署在 Vercel Edge / Cloudflare Workers / Lambda 时，注意：

- 有的平台要 `Cache-Control: no-store` 才不缓存
- 有的 buffer flush 时机不同（Nginx 默认会缓冲，要 `X-Accel-Buffering: no`）
- 有的有最大执行时长（Edge 30s / Lambda 15min）

### 6. 工具调用怎么流

Tool use 的 stream 比文本复杂——同一个 stream 里可能交替出现：
- 文本 delta
- tool_use 块开始
- tool_use 输入 JSON 增量
- tool_use 块结束
- ...

前端要按 event type 路由，不要混在一起。

## 一个朴素观察

流式不是"高级特性"，是**最低体验门槛**。
如果你做了 LLM 应用还是同步 wait 完整 response，**用户会觉得这东西比 ChatGPT 差一个时代**。

实现起来不复杂，关键是抗住几个工程细节（中断 / 心跳 / 缓冲）。
做完一次，所有 LLM 应用都能复用。
