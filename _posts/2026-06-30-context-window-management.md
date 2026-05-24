---
layout: post
title: "Context Window 管理策略 — 长对话不丢信息的 5 种方法"
date: 2026-06-30
topic: "工程实战"
tags: [AI, Context Window, Memory]
excerpt: 对话越来越长，context 撑不住。5 种策略各有适用场景，照着实现，不用每次都从头设计。
permalink: /posts/2026-06-30-context-window-management.html
---

## 问题有多大

Claude 3.5 Sonnet 有 200k token context，听起来很大。但：
- 200k tokens ≈ 15 万汉字 ≈ 一本小说
- 每轮对话可能产生 1000-5000 tokens
- 50-200 轮对话就撑到极限
- 更关键：context 越长，**成本越高、延迟越大**

100k tokens 的请求比 5k tokens 的请求：
- API 成本高 20 倍
- 延迟高 3-5 倍
- 还不算模型在超长 context 里注意力分散导致的质量下降

需要主动管理。

## 策略一：滑动窗口

**原理**：只保留最近 N 轮对话。

```python
from anthropic import Anthropic

client = Anthropic()

def sliding_window_chat(
    conversation_history: list[dict],
    user_message: str,
    max_turns: int = 20,
    system_prompt: str = ""
) -> tuple[str, list[dict]]:
    # 加入新消息
    conversation_history.append({"role": "user", "content": user_message})

    # 截取最近 max_turns 轮（一轮 = 1 user + 1 assistant）
    # 确保保留偶数条，维持 user/assistant 交替结构
    if len(conversation_history) > max_turns * 2:
        conversation_history = conversation_history[-(max_turns * 2):]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=conversation_history,
    )

    reply = response.content[0].text
    conversation_history.append({"role": "assistant", "content": reply})

    return reply, conversation_history
```

**优点**：实现简单，绝对不会超限。
**缺点**：丢失早期上下文，用户如果引用"之前说的 X"，模型不知道。

**适用**：闲聊机器人，每轮对话相对独立，历史不重要。

## 策略二：摘要压缩

**原理**：早期对话用 LLM 摘要压缩，只保留精华。

```python
import anthropic

client = anthropic.Anthropic()

def summarize_old_turns(old_messages: list[dict]) -> str:
    """把旧的对话轮次压缩成摘要"""
    conversation_text = ""
    for msg in old_messages:
        role = "用户" if msg["role"] == "user" else "助手"
        conversation_text += f"{role}：{msg['content']}\n\n"

    response = client.messages.create(
        model="claude-haiku-4-5",  # 用便宜快速的模型做摘要
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""请将以下对话历史压缩为简洁摘要，保留关键信息、用户需求、已达成的结论，省略重复和冗余内容。

<conversation>
{conversation_text}
</conversation>

摘要："""
        }]
    )
    return response.content[0].text

class SummarizingConversation:
    def __init__(self, system_prompt: str = "", window_size: int = 10):
        self.system_prompt = system_prompt
        self.window_size = window_size
        self.summary = ""           # 历史对话的压缩摘要
        self.recent_messages = []   # 最近 N 轮完整保留

    def _build_messages(self) -> list[dict]:
        messages = []
        # 如果有历史摘要，作为第一条 user 消息注入
        if self.summary:
            messages.append({
                "role": "user",
                "content": f"[对话历史摘要]\n{self.summary}"
            })
            messages.append({
                "role": "assistant",
                "content": "明白，我已了解之前的对话背景。"
            })
        messages.extend(self.recent_messages)
        return messages

    def chat(self, user_message: str) -> str:
        self.recent_messages.append({"role": "user", "content": user_message})

        # 超出窗口时压缩旧消息
        if len(self.recent_messages) > self.window_size * 2:
            # 把前一半移入摘要
            to_summarize = self.recent_messages[:self.window_size]
            self.recent_messages = self.recent_messages[self.window_size:]

            new_summary_content = summarize_old_turns(to_summarize)
            if self.summary:
                # 合并新旧摘要
                self.summary = f"{self.summary}\n\n【后续补充】\n{new_summary_content}"
            else:
                self.summary = new_summary_content

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=self.system_prompt,
            messages=self._build_messages(),
        )

        reply = response.content[0].text
        self.recent_messages.append({"role": "assistant", "content": reply})
        return reply
```

**优点**：历史信息不完全丢失，保留关键内容。
**缺点**：摘要有损，细节会丢；增加一次额外 LLM 调用。

**适用**：需要对话连贯性的客服/助手场景。

## 策略三：RAG-based 记忆

**原理**：把历史对话存入向量库，每次对话时检索相关历史。

```python
from openai import OpenAI
import numpy as np
from datetime import datetime

openai_client = OpenAI()

class VectorMemory:
    def __init__(self):
        self.messages = []  # [(text, embedding, timestamp)]

    def add(self, role: str, content: str):
        text = f"{role}: {content}"
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
        self.messages.append({
            "text": text,
            "embedding": embedding,
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
        })

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.messages:
            return []

        query_emb = np.array(openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding)

        scored = []
        for msg in self.messages:
            emb = np.array(msg["embedding"])
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            scored.append((sim, msg))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [msg for _, msg in scored[:top_k]]

class RAGConversation:
    def __init__(self, system_prompt: str = "", recent_window: int = 6):
        self.system_prompt = system_prompt
        self.recent_window = recent_window
        self.memory = VectorMemory()
        self.all_messages = []

    def chat(self, user_message: str) -> str:
        # 从历史中检索相关内容
        relevant = self.memory.retrieve(user_message, top_k=4)

        # 构建 messages
        messages = []

        # 加入检索到的历史（排除最近窗口里已有的）
        recent_texts = {m["content"] for m in self.all_messages[-self.recent_window:]}
        relevant_context = [r for r in relevant if r["content"] not in recent_texts]

        if relevant_context:
            context_str = "\n".join(f"- {r['text']}" for r in relevant_context[:3])
            messages.append({
                "role": "user",
                "content": f"[相关历史对话]\n{context_str}"
            })
            messages.append({
                "role": "assistant",
                "content": "好的，我参考了相关历史对话。"
            })

        # 加入最近窗口
        messages.extend(self.all_messages[-self.recent_window:])
        messages.append({"role": "user", "content": user_message})

        response = anthropic.Anthropic().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=self.system_prompt,
            messages=messages,
        )

        reply = response.content[0].text

        # 存入记忆
        self.memory.add("user", user_message)
        self.memory.add("assistant", reply)
        self.all_messages.append({"role": "user", "content": user_message})
        self.all_messages.append({"role": "assistant", "content": reply})

        return reply
```

**优点**：能跨越很长时间找到相关历史。
**缺点**：实现最复杂，需要向量库；检索不一定找到最相关内容。

**适用**：长期个人助手，用户会跨会话引用历史。

## 策略四：结构化记忆提取

**原理**：从对话中主动提取关键事实，存入结构化记忆，而不是保存原文。

```python
import json
from anthropic import Anthropic

client = Anthropic()

def extract_memory_updates(conversation_turn: dict, existing_memory: dict) -> dict:
    """从一轮对话中提取需要更新的记忆条目"""
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""从下面这轮对话中，提取需要长期记住的用户信息，以 JSON 格式输出。
只提取明确、具体的事实，不要推测。如果没有新信息，返回空 JSON {{}}。

<existing_memory>
{json.dumps(existing_memory, ensure_ascii=False)}
</existing_memory>

<conversation>
用户：{conversation_turn['user']}
助手：{conversation_turn['assistant']}
</conversation>

请输出需要新增或更新的记忆条目（JSON格式）："""
        }]
    )

    try:
        text = response.content[0].text.strip()
        # 提取 JSON 部分
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        updates = json.loads(text)
        return updates
    except:
        return {}

class StructuredMemoryConversation:
    def __init__(self):
        self.memory = {
            "user_preferences": {},
            "user_facts": {},
            "project_context": {},
            "decisions_made": [],
        }
        self.recent_messages = []
        self.recent_window = 8

    def chat(self, user_message: str) -> str:
        messages = []

        # 把结构化记忆注入 system prompt
        memory_str = json.dumps(self.memory, ensure_ascii=False, indent=2)

        # 最近对话
        messages.extend(self.recent_messages[-self.recent_window:])
        messages.append({"role": "user", "content": user_message})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=f"你是一个私人助手。以下是关于用户的已知信息，请参考：\n{memory_str}",
            messages=messages,
        )
        reply = response.content[0].text

        # 更新记忆
        updates = extract_memory_updates(
            {"user": user_message, "assistant": reply},
            self.memory
        )
        # 深度合并
        for key, val in updates.items():
            if key in self.memory and isinstance(self.memory[key], dict):
                self.memory[key].update(val)
            else:
                self.memory[key] = val

        self.recent_messages.append({"role": "user", "content": user_message})
        self.recent_messages.append({"role": "assistant", "content": reply})

        return reply
```

**适用**：个人助手、客服系统，需要跨会话记住用户信息。

## 策略五：分层 Context

**原理**：把 context 分成三层，分别处理。

```
┌─────────────────────────────┐
│  第一层：系统角色 + 核心指令   │  永远保留，稳定
│  (~500 tokens)               │
├─────────────────────────────┤
│  第二层：会话摘要 + 关键事实   │  定期压缩更新
│  (~1000 tokens)              │
├─────────────────────────────┤
│  第三层：最近 N 轮完整对话     │  滑动窗口
│  (~3000 tokens)              │
└─────────────────────────────┘
```

```python
class HierarchicalContext:
    def __init__(self, system_prompt: str):
        self.layer1_system = system_prompt      # 不变
        self.layer2_summary = ""                # 定期压缩
        self.layer3_recent = []                 # 滑动窗口
        self.layer3_max_turns = 6
        self.turn_count = 0
        self.summarize_every = 10               # 每 10 轮更新一次摘要

    def _compress_layer2(self, old_messages: list[dict]):
        """把旧消息压缩进 layer2"""
        old_text = "\n".join(
            f"{'用户' if m['role']=='user' else '助手'}：{m['content']}"
            for m in old_messages
        )
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content":
                f"现有摘要：{self.layer2_summary}\n\n新对话：{old_text}\n\n请合并更新为简洁摘要："}]
        )
        self.layer2_summary = response.content[0].text

    def chat(self, user_message: str) -> str:
        self.turn_count += 1

        # 超出 layer3 窗口时压缩
        if len(self.layer3_recent) > self.layer3_max_turns * 2:
            to_compress = self.layer3_recent[:-self.layer3_max_turns]
            self.layer3_recent = self.layer3_recent[-self.layer3_max_turns:]
            self._compress_layer2(to_compress)

        # 构建最终 messages
        messages = []
        if self.layer2_summary:
            messages.append({"role": "user", "content": f"[背景摘要] {self.layer2_summary}"})
            messages.append({"role": "assistant", "content": "了解。"})
        messages.extend(self.layer3_recent)
        messages.append({"role": "user", "content": user_message})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=self.layer1_system,
            messages=messages,
        )
        reply = response.content[0].text
        self.layer3_recent.append({"role": "user", "content": user_message})
        self.layer3_recent.append({"role": "assistant", "content": reply})
        return reply
```

## 策略选择指南

| 场景 | 推荐策略 |
|---|---|
| 短期闲聊，历史不重要 | 滑动窗口 |
| 需要保持话题连贯 | 摘要压缩 |
| 用户长期使用，偶尔引用历史 | RAG 记忆 |
| 需要记住用户偏好/信息 | 结构化记忆 |
| 复杂助手，需要最强连贯性 | 分层 Context |

## 一个朴素结论

> Context 管理没有银弹，但有一条原则：**越靠近现在的内容越完整保留，越久远的内容越要压缩**。
>
> 从滑动窗口开始，够用就不要过度设计。
> 当你真的遇到"用户提到了三天前的事但模型不记得"的问题时，再上 RAG 记忆。
