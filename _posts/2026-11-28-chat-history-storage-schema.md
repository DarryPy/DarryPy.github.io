---
layout: post
title: LLM 对话存储的数据模型设计 — 消息、分支与重新生成怎么落库
date: 2026-11-28
topic: "工程实战"
tags: [数据库设计, 对话存储, LLM 工程]
excerpt: 一张 messages 表撑不住重新生成、编辑分支和工具调用。本文给出树形消息模型、表结构、读取路径和迁移时最容易踩的坑，帮你把对话历史稳稳落库。
permalink: /posts/2026-11-28-chat-history-storage-schema.html
---

你做第一个聊天应用时，大概率建了一张 messages 表：conversation_id、role、content、created_at，按时间排序一读就完事。上线两周后产品说要加「重新生成」，再过一周要加「编辑上一条提问」，然后工具调用、附件、多模型对比接踵而至。我见过不少团队在这里推倒重来，因为线性表从根上就表达不了分支。

## 为什么线性表撑不住

线性模型默认一段对话是一条直线，但真实交互是一棵树。用户点重新生成，同一个提问下就有两个回答；用户编辑第三条提问，从第三条开始就分出一条新路径，旧路径还得保留，方便用户切回去看。如果你用覆盖写法，旧内容直接丢失，审计和回放都做不了。

更隐蔽的问题是排序。按 created_at 排序在并发写入时会乱，流式回答先建记录、后补全文，工具调用结果可能比助手消息更早落库。时间戳只是写入时刻，不是对话中的逻辑位置。一旦你把逻辑顺序和物理时间绑在一起，后面每加一个功能都要打补丁。

还有上下文拼装。发给 LLM 的 messages 数组必须是一条从根到当前节点的完整路径，中间不能混进其他分支的内容。线性表里你只能靠各种 is_deleted、version 字段去过滤，条件越写越复杂，最后谁也说不清某次请求到底带了哪些历史。

最后是合规和排障。用户投诉模型说了不该说的话，你需要还原那一刻送进模型的完整输入，包括被用户后来编辑掉的提问。覆盖写的线性表做不到这一点，只能对着日志猜，而日志往往已经被脱敏或者过了保留期限。

## 树形消息模型

我现在固定用的方案是每条消息记录 parent_id，对话表记录 current_leaf_id。重新生成就是给同一个父节点挂一个新的子节点，编辑提问就是给原提问的父节点挂一个新的用户消息。读取时从 current_leaf_id 沿 parent_id 往上走到根，再反转，就是当前分支的完整上下文。

```sql
CREATE TABLE conversations (
  id              BIGINT PRIMARY KEY,
  user_id         BIGINT NOT NULL,
  title           VARCHAR(200),
  current_leaf_id BIGINT,
  model           VARCHAR(64),
  created_at      TIMESTAMP NOT NULL,
  updated_at      TIMESTAMP NOT NULL
);

CREATE TABLE messages (
  id              BIGINT PRIMARY KEY,
  conversation_id BIGINT NOT NULL,
  parent_id       BIGINT,
  role            VARCHAR(16) NOT NULL,  -- system/user/assistant/tool
  status          VARCHAR(16) NOT NULL,  -- streaming/done/failed/aborted
  content         JSONB NOT NULL,        -- 多段内容：文本、图片、工具调用
  model           VARCHAR(64),
  input_tokens    INT,
  output_tokens   INT,
  sibling_index   INT NOT NULL DEFAULT 0,
  created_at      TIMESTAMP NOT NULL
);

CREATE INDEX idx_msg_conv_parent ON messages (conversation_id, parent_id);

-- 读取当前分支：从叶子递归回溯到根
WITH RECURSIVE path AS (
  SELECT m.*, 0 AS depth FROM messages m
  WHERE m.id = (SELECT current_leaf_id FROM conversations WHERE id = :cid)
  UNION ALL
  SELECT p.*, path.depth + 1 FROM messages p
  JOIN path ON p.id = path.parent_id
)
SELECT * FROM path ORDER BY depth DESC;
```

content 用 JSON 数组而不是纯文本，是为了容纳多段内容：一条助手消息可能同时包含文字、工具调用请求和引用来源。sibling_index 给前端渲染「2/3」这种分支切换器用，status 字段让流式中断的半成品也能被识别和清理。

切换分支的操作也很简单：前端点「上一个回答」时，后端找到目标兄弟节点，再沿着它一路往下找最新的子孙作为新叶子，写回 current_leaf_id。这个向下查找可以按 created_at 取最新，也可以在每个节点上记一个 last_active_child_id，让每条分支都记住用户上次停留的位置。

## 三种存储方案对比

树形结构不是唯一选择，你需要根据读写模式权衡。下面是我在几个项目里实际用过的三种做法，核心差别在于读取当前路径的成本和分支操作的复杂度。

| 方案 | 读取当前分支 | 重新生成/编辑 | 适合场景 |
|------|------------|-------------|---------|
| 线性表 + 覆盖写 | 一次顺序查询，最快 | 丢历史，无法切回 | 一次性问答、客服工单 |
| parent_id 树 + 递归查询 | 递归 CTE，深度几十层无压力 | 插入一个节点即可 | 通用聊天、需要分支切换 |
| 整段对话存一个 JSON 文档 | 一次读取，前端自行解析 | 整体重写，有并发覆盖风险 | 短对话、低并发、原型期 |

大部分对话深度在一百轮以内，递归查询完全不是瓶颈。真遇到超长对话，可以在消息上冗余一个 depth 字段，或者把当前路径的 id 列表缓存在 conversations 上，切换分支时重算一次。别为了想象中的性能问题提前上图数据库。

还有一种折中做法是事件溯源：只追加事件，不更新任何记录，读取时重放事件得到当前状态。它在审计上最干净，但查询和调试成本都高，团队里没人熟悉这套模式时我不建议首选。先用 parent_id 树跑起来，等真有强审计需求再演进也来得及。

## 写入路径与流式落库

流式回答的落库时机决定了异常时数据是否干净。我的做法是收到第一个 token 前就插入一条 status 为 streaming 的助手消息，同时更新 current_leaf_id；流式过程中按固定间隔或固定字节数批量更新 content，结束时写入最终内容、token 用量并把状态改为 done。

不要每个 token 都写一次数据库，那样写放大会非常夸张。每隔五百毫秒或者攒够两百个字符刷一次就够了，前端展示走内存里的流，数据库只负责兜底。如果进程中途崩溃，定时任务把超过若干分钟还停在 streaming 的消息标成 aborted，前端据此显示「生成中断，点击重试」。

工具调用要拆成独立消息。助手发起调用是一条 role 为 assistant、content 里带 tool_use 段的消息，工具返回是一条 role 为 tool 的子消息，再往下才是助手的最终回答。这样拼上下文时结构天然正确，也方便你单独统计每个工具的耗时和失败率。

失败也要落库。上游返回错误或者触发内容安全拦截时，别直接丢掉这条助手消息，而是标成 failed 并在 content 里记下错误码和原因。用户点重试时生成一个新的兄弟节点，失败那条保留下来，你事后才能统计不同模型、不同时段的真实失败率。

current_leaf_id 的更新必须和消息插入在同一个事务里，否则并发场景下会出现叶子指向一条不存在或者属于别的会话的消息。用户在两个标签页同时发消息是真实存在的，我的做法是用乐观锁：更新时带上旧的 leaf_id 作为条件，影响行数为零就提示用户刷新。

## 上线前的实战清单

模型设计好之后，真正决定稳定性的是这些细节。下面这份清单是我每次评审对话存储方案时会逐条对照的，大多数线上事故都能在这里找到影子，建议你直接拿去当评审模板。

其中最容易被忽略的是上下文拼装和路径的一致性。很多团队表结构设计得很好，最后却在拼上下文时偷懒，按会话全量查询再在内存里过滤，结果某次分支切换后把两条路径的内容混在了一起，模型回答前言不搭后语，排查了整整一天才找到原因。

- 发给模型的上下文只从当前路径拼装，绝不按 conversation_id 全量捞取。
- system prompt 版本号记在消息或会话上，排查问题时才能复现当时的完整输入。
- 删除默认软删除，但用户要求彻底删除时，必须连同所有分支和附件一起物理清理。
- 附件只存对象存储的 key，不把 base64 塞进 content，否则单行体积会失控。
- 记录每条助手消息的 model 和 token 用量，计费、对账和模型对比都靠它。
- 会话标题异步生成，失败不影响主流程，也不要阻塞第一条回答的返回。
- 导出和迁移脚本要按树遍历，按时间排序导出会把分支混在一起，迁移后根本对不上。

一句话总结：对话不是一条直线而是一棵树，你越早用 parent_id 承认这件事，后面重新生成、编辑和工具调用就越不需要返工。
