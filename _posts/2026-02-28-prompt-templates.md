---
layout: post
title: Prompt 模板库与版本管理 — 像管代码一样管 Prompt
date: 2026-02-28
topic: "Prompt 与推理"
tags: [AI, Prompt Engineering, 工程]
excerpt: Prompt 散在代码各处是技术债。集中管理 + 版本化 + 灰度 + 回滚才是生产姿态。
permalink: /posts/2026-02-28-prompt-templates.html
---

## 散落的 prompt = 技术债

刚起步时，prompt 都内嵌在代码里。
项目长大后会出现：

- 改一个 system prompt 要改 8 个地方
- 不知道线上跑的是哪个版本
- A/B 测两个 prompt 没法控制流量比例
- prompt 回滚困难
- 跨团队复用没希望

**Prompt 是生产配置**，跟代码一样需要工程化。

## 4 个层次的解决方案

### 1. 抽出到独立文件

最朴素：每个 prompt 一个 `.md` 或 `.j2` 文件。

```
prompts/
  classification/
    intent_v1.md
    intent_v2.md
  rag/
    qa_assistant_v3.md
```

代码加载时按 key 找：

```python
prompt = render_template("classification/intent_v2.md", {"query": q})
```

至少不再散落代码里。

### 2. 加版本号 + 灰度

```python
# config.yaml
prompts:
  intent_classifier:
    versions:
      v1: { weight: 0.0 }
      v2: { weight: 0.2 }    # 20% 灰度
      v3: { weight: 0.8 }    # 80% 主力

def get_prompt(name, user_id):
    cfg = prompt_config[name]
    version = weighted_choice(cfg.versions, key=hash(user_id))
    return load_prompt(f"{name}/{version}.md")
```

按用户 ID 哈希分流，保证同用户每次跑同个版本。

### 3. 加 metadata + diff

每个 prompt 文件加 front matter：

```yaml
---
name: intent_classifier
version: v3
author: qinqiong
created: 2026-02-15
description: 优化了否定句识别
based_on: v2
changes:
  - 加了 5 个否定句示例
  - system prompt 改成 XML 结构
eval_scores:
  - dataset: intent_v1
    accuracy: 0.92
    f1: 0.89
---

你是一个意图分类器...
```

每次升级有据可查，**故障时知道回滚到哪**。

### 4. Prompt Management 平台

更进一步用平台：

| 工具 | 优势 |
|---|---|
| **LangSmith** | Prompt + Eval + Trace 一站 |
| **Braintrust** | UX 现代，diff view 强 |
| **Langfuse** | 开源自部署 |
| **PromptLayer** | 历史版本 + 团队协作 |
| **HumanLoop** | 重视 prompt eng 工作流 |

平台一般支持：版本控制、模板变量、A/B 实验、运行追踪、按用户回滚。

## 灰度发布流程

```
1. 写新版 prompt → v_new
2. 跑 eval（用 golden dataset）→ 比 v_current 好？
3. 5% 灰度 → 24h 监控
4. 50% → 24h
5. 100% → 保留 v_current 一周可秒回
```

不要直接 100% 切——AI 输出有随机性，**线上才会发现 bug**。

## 模板渲染推荐用 Jinja2

```python
from jinja2 import Template

template = Template("""
你是 {{ role }}。
{% if examples %}
示例：
{% for ex in examples %}
- {{ ex }}
{% endfor %}
{% endif %}

问题：{{ question }}
""")

prompt = template.render(role="法律顾问", examples=[...], question=...)
```

比字符串拼接安全（不会 prompt injection 字段名）、可控（条件 / 循环 / 默认值）。

## 一份 prompt 工程化 checklist

发布前确认：

- [ ] Prompt 抽出到独立文件
- [ ] 有版本号
- [ ] 有 front matter metadata
- [ ] 已用 golden dataset eval
- [ ] 灰度策略明确
- [ ] 老版本可秒回
- [ ] 改动有 changelog
- [ ] Code review 看过

5 个满足你的 prompt 就有"代码"的工程严谨度。

## 一个朴素结论

> Prompt 不是字符串，是**配置 + 业务逻辑**。
> 跟代码一样的版本控制 + eval + 灰度 + 回滚，才能让 AI 应用稳得住。
