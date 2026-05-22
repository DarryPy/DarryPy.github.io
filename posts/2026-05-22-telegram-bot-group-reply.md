---
layout: post
title: Telegram bot 在群里 @ 它却不回话？从隐私模式到 visibleReplies 的排查链
date: 2026-05-22
tags: [Telegram, Bot, AI Agent, 调试]
excerpt: 群里 @机器人，它给 👀 但不出文字。看上去是配置问题，挖到最后却是系统提示里那条"casual banter 保持静默"的规则在作祟。
permalink: /posts/2026-05-22-telegram-bot-group-reply.html
---

## 现象

群里 `@qinqiong_agent_win_bot 你好`，bot **给了一个 👀 表情反应，但没有文字回复**。
DM 一切正常。

## 第一层：Telegram 隐私模式

Telegram bot 默认开启 `privacy mode`，调用 `getMe` 看：

```json
{"can_read_all_group_messages": false}
```

这种状态下 bot 在群里**只能看到**：

- 真实的 `@bot_username` mention
- 对 bot 自己消息的回复
- `/cmd@bot_username` 命令

输入 `@中文昵称` 之类的不是真 mention，不会推给 bot。
但日志里看到 mention 是收到的（👀 印证了），所以这一层 OK。

## 第二层：`visibleReplies` 配置

OpenClaw 配置里有这么一段：

```json
"messages": {
  "groupChat": {
    "visibleReplies": "message_tool"
  }
}
```

`message_tool` 模式下，agent 普通的文字回复**默认不发到群里**，必须显式调 `message(action=send)` 才会出文字。
切到 `automatic` 应该就好了——重启网关验证。

群里再 @，结果……**还是没回复**。

## 第三层：reply 真的被生成了吗

打开 `~/.claude/projects/.../<session>.jsonl` 看本地 Claude CLI 的 transcript，能看到 agent 生成的回复：

```json
{
  "role": "assistant",
  "content": [{
    "type": "text",
    "text": "[[reply_to_current]] 在的在的～宝贝儿在线，主人请说🌷"
  }]
}
```

**回复确实生成了**，只是没被投递到 Telegram。
`[[reply_to_current]]` 是 OpenClaw 自家的回复线程标签，跟 `channels.telegram.replyToMode` 有关。
看了一眼源码——`replyToMode: off` 默认时这个标签依然会被尊重，所以**不是它的锅**。

## 第四层：再被同样的 bot 测一次

让用户再 @ 一次。这次看转录文件，agent 的回复内容变成了：

```
NO_REPLY
```

这就是答案了。在 OpenClaw 的源码里：

```js
if (isSilentReplyText(text, "NO_REPLY")) return { skip: true };
```

`NO_REPLY` 是**静默令牌**——agent 明确选择"这条不回"，gateway 收到后直接 skip。

那 agent 为什么会自己选不回？看 workspace 里的 `AGENTS.md`：

```markdown
**Stay silent (HEARTBEAT_OK) when:**
- It's just casual banter between humans
- Your response would just be "yeah" or "nice"
- The conversation is flowing fine without you
```

agent 把 "测试" 这种 mention 判定为"无意义闲聊"，自动输出 `NO_REPLY` 静默掉了。

## 真正的修法

不是配置层，是**规则层**——给 `AGENTS.md` 加一条 hard override：

```markdown
**Hard override — always reply, never NO_REPLY / HEARTBEAT_OK:**

- 任何 @ 我的消息（无论内容是什么——`测试`/`你好`/单字 ping 都算）
- 即使一个字的 "hi"，也至少回一个简短问候
```

同时把这条规则写进 agent 的**长期 memory**，避免 AGENTS.md 哪天被改回又出问题。

再 @，bot 立刻回话。

## 调试这种问题的几个心法

1. **先确认信号到没到**。👀 表情说明 gateway 收到了 mention，这一层 OK
2. **再确认 agent 生成没生成**。看本地 transcript，能直接看到 LLM 输出
3. **看输出文本里有没有特殊标记**。`NO_REPLY` / `HEARTBEAT_OK` 这种静默令牌很容易被忽略
4. **`reply_text` 和 `delivery_status` 分别看**——能生成 ≠ 能投递
5. **改配置之前，先看现有配置能不能解释行为**。如果不能，问题大概率在更上层（规则、prompt、训练）

---

> 配置调了好几次都没修好的问题，往往真正的故障点在你以为"已经解决了"的更上层。
