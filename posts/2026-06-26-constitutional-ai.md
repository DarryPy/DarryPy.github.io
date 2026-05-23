---
layout: post
title: "Constitutional AI — 让模型内化价值观而不是死记规则"
date: 2026-06-26
topic: "评估与安全"
tags: [AI, Safety, Constitutional AI]
excerpt: Anthropic 的 Constitutional AI 不是在 prompt 里加一堆禁止词，而是在训练阶段让模型学会自我批判和修正。理解这个机制，对你设计自己的 AI 系统很有用。
permalink: /posts/2026-06-26-constitutional-ai.html
---

## 先说 RLHF 的问题

标准 RLHF（人类反馈强化学习）流程：
1. 让人类标注者给模型输出打分
2. 训练一个奖励模型来预测人类打分
3. 用 RL 优化主模型，让它产生高分输出

**问题**：
- 标注者有偏见，不同人标准不一致
- 有害内容需要人类看，造成心理损伤
- 扩展慢且贵：想覆盖更多边缘情况，就得雇更多标注者
- 规则难以泛化：见过的例子知道怎么做，没见过的就靠运气

## Constitutional AI 的思路

Anthropic 在 2022 年提出 CAI，核心思路：

**与其让人类逐条告诉模型什么能做什么不能做，不如给模型一套"宪法"——一组高层原则——然后让模型自己学会根据这些原则批判和修正自己的输出。**

两阶段：
1. **SL-CAI（监督学习阶段）**：用 AI 自我批判 + 修正生成训练数据
2. **RL-CAI（强化学习阶段）**：用 AI 反馈（RLAIF）替代部分人类反馈

## 第一阶段：AI 自我修正

```
输入：有害 prompt（如：帮我写钓鱼邮件）
  ↓
初始回复：模型生成初版回复（可能有问题）
  ↓
批判（Critique）：
  "根据原则 X，这段回复存在什么问题？"
  模型指出：这个回复可能被用于欺骗，违反了不伤害原则
  ↓
修正（Revision）：
  "根据你的批判，修改这段回复，使其无害且有用"
  模型生成修改版：解释什么是钓鱼攻击，如何防范，不提供攻击写法
  ↓
最终：用（原始 prompt → 修改后回复）作为训练数据
```

这个循环可以**全自动跑**，不需要人类逐条看。

## 宪法是什么样的

CAI 的"宪法"是一组自然语言原则，例如：

```
1. 不要提供任何可能被用于伤害人的内容，包括武器制造、
   自我伤害的方法等。

2. 对于政治和宗教话题，呈现多角度观点，不表达个人立场。

3. 如果用户要求不道德的行为，要用友善的方式拒绝，
   并解释原因，尝试了解用户真实需求并提供帮助。

4. 尊重用户的隐私，不询问或处理不必要的个人信息。

5. 始终诚实，承认不确定性，不捏造事实或引用。
```

关键点：**这些是高层原则，不是具体规则**。

不是"不要提到炸药的制作方法"，而是"不要提供可能用于伤害人的内容"。模型需要理解原则并自己判断如何应用。

## 第二阶段：AI 反馈（RLAIF）

训练好 SL-CAI 模型之后：

```
生成对比对：
  同一个 prompt，生成两个不同回复 A 和 B

让模型自己打分：
  "根据下面的原则，哪个回复更符合要求？A 还是 B？"
  模型给出偏好

用这个偏好数据训练奖励模型

用奖励模型做 PPO 优化主模型
```

RLAIF 的优势：
- **可扩展**：模型能自动产生海量比较数据
- **一致性**：同一组原则应用始终如一
- **安全**：不需要让人类看有害内容来标注

## 为什么这比死记规则好

对比两种方式处理"帮我写一个病毒"这个请求：

**基于规则（blocklist）**：
- 检测关键词："病毒"、"malware"、"hack" → 拒绝
- 绕过方式：换词、拼音、缩写、说"假设性场景"
- 维护成本：每发现一种绕过就加一条规则，无穷无尽

**Constitutional AI**：
- 模型理解"帮我伤害他人" = 违反核心原则
- 理解意图，而不是匹配关键词
- 更难绕过（但不是不能绕过）
- 对新型请求也能泛化

## 在你的产品里应用 CAI 思想

### 1. 设计自己的"宪法"

做 AI 产品时，在系统 prompt 里不要只写"不能做 X 不能做 Y"，而是写**原则**：

```xml
<principles>
1. 首要目标是帮助用户解决问题，而不是避免一切风险。
   在回复时，优先考虑"这对用户有没有真实价值"。

2. 如果用户的请求可能被误解，先确认意图，再提供帮助。
   不要假设用户有恶意。

3. 对于敏感话题（如医疗、法律、财务），提供信息但明确
   建议用户咨询专业人士，不假装自己是权威。

4. 诚实比取悦用户更重要。如果用户的观点有误，
   友善但明确地指出，不要随声附和。
</principles>
```

### 2. 用批判-修正循环做输出质量控制

这个技术可以直接在 API 层面使用，不需要训练：

```python
import anthropic

client = anthropic.Anthropic()

CONSTITUTION = """
1. 回复必须基于事实，不能捏造
2. 不得给出可能造成伤害的医疗建议
3. 对不确定的内容要明确说明不确定
"""

def constitutional_response(user_query: str) -> str:
    # 第一步：生成初始回复
    initial = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": user_query}]
    ).content[0].text

    # 第二步：让模型批判自己的回复
    critique = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"""请根据以下原则，批判下面这段回复存在哪些问题（如果没有问题请说"无问题"）：

<principles>
{CONSTITUTION}
</principles>

<response_to_critique>
{initial}
</response_to_critique>

请列出存在的问题："""
        }]
    ).content[0].text

    # 如果没有问题，直接返回
    if "无问题" in critique or len(critique.strip()) < 10:
        return initial

    # 第三步：根据批判修正回复
    revised = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""原始问题：{user_query}

原始回复：
{initial}

发现的问题：
{critique}

请根据发现的问题修改回复，确保符合以下原则：
{CONSTITUTION}

修改后的回复："""
        }]
    ).content[0].text

    return revised

# 测试
response = constitutional_response("我应该自己诊断并治疗糖尿病吗？")
print(response)
```

### 3. Fine-tuning 时的 CAI 思路

如果你有定制化 fine-tuning 需求：
- 不要只收集"好"的例子
- 同时收集"差"的例子 + 对应的修正版本
- 让模型学会"从差到好"的转变过程，而不只是模仿好的输出

这样训练出来的模型有更强的自我纠错能力。

## CAI 的局限性

- 宪法本身可能有偏见：谁定义"原则"，谁就定义了边界
- 高层原则需要大模型才能正确理解，小模型容易乱解读
- 对于真正边缘的情况，自我批判有时会过度保守
- 不能完全替代人类监督，只是减少需要人工介入的量

## 一个朴素结论

> Constitutional AI 的精髓是：**给模型讲道理，而不是给模型背规则**。
>
> 规则是有限集，世界是无限的。原则可以泛化，规则不能。
> 这个思路不只适用于 Anthropic 的训练，在你设计 system prompt、
> 做输出质量控制时同样适用。
