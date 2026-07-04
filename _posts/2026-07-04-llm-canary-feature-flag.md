---
layout: post
title: LLM 应用灰度发布实战 🚦
date: 2026-07-04
topic: "工程实战"
tags: [灰度发布, Feature Flag, Canary, LLM, 工程实战]
excerpt: LLM 的输出是非确定的，staging 测好了上生产照样翻车。这篇讲怎么用 Feature Flag 和 Canary 流量切分，把新 prompt 或新模型的风险控在 5% 的流量里，出问题 30 秒回滚。
permalink: /posts/2026-07-04-llm-canary-feature-flag.html
---

你上线过一次"只改了几个词的 prompt"然后线上质量直接崩掉吗？或者换了个新模型版本，staging 跑得好好的，生产一放量用户就开始投诉？LLM 应用和普通后端服务最大的区别就在这里——非确定性让你很难在上线前把所有问题都测出来。灰度发布不是可选项，是标配。

## 为什么 LLM 应用更需要灰度

普通服务的灰度逻辑很清晰：新版本代码逻辑不变，测好了上线风险可控。LLM 应用不一样，有几个特殊性：

- **输出非确定**：同一个 prompt 跑 100 次，你会得到 100 个语义相近但措辞不同的结果，staging 环境的样本覆盖永远不够
- **用户行为多样**：真实用户的输入分布和你的测试集有偏差，边界 case 在生产才会暴露
- **质量难以前置量化**：代码的单测覆盖率可以达到 90%，LLM 输出质量靠什么？人工评估成本高、速度慢
- **模型版本静默变更**：API 提供商有时会悄悄更新底层模型，你的 prompt 可能在某天之后就不如以前稳定

结论：**你必须在小流量上先验证，再扩量**。

## Feature Flag：控制谁看到新版本

Feature Flag 是灰度的开关层。最简单的实现不需要引入外部服务，一个带权重的随机分配就够用：

```python
import hashlib

def get_variant(user_id: str, flag_name: str, rollout_pct: float) -> str:
    """确定性分配：同一个 user_id 每次拿到相同 variant"""
    key = f"{flag_name}:{user_id}"
    hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
    bucket = hash_val % 100
    return "treatment" if bucket < rollout_pct else "control"

# 使用
variant = get_variant(user_id="u_12345", flag_name="new_prompt_v2", rollout_pct=5)
if variant == "treatment":
    response = call_llm(new_prompt_template, messages)
else:
    response = call_llm(old_prompt_template, messages)
```

用 `user_id` 而不是随机数的好处是**同一个用户始终进同一个组**，避免体验割裂（今天看到新版本，明天又回到旧版本）。

如果团队规模更大，可以接入 LaunchDarkly、Growthbook 或 Flagsmith 这类专业服务，支持按用户属性（地区、付费等级、设备）分组，管理界面也更直观。

## Canary 流量切分的 5 个阶段

灰度不是一次性把流量从 0 拨到 100，要分阶段走，每个阶段验证一组指标再往下推：

| 阶段 | 流量比例 | 验证时长 | 通过条件 |
|------|---------|---------|---------|
| 内部测试 | 1%（仅内部用户）| 1-2 天 | 无 P0 错误，输出格式合规 |
| 金丝雀 | 5% | 2-3 天 | 质量分 ≥ 对照组 -2%，延迟 p99 ≤ 对照组 ×1.2 |
| 小批量 | 20% | 3-5 天 | 用户投诉率无显著上升，成本符合预期 |
| 扩量 | 50% | 2-3 天 | 各指标平稳 |
| 全量 | 100% | — | 完成，关闭旧版本 Flag |

每个阶段的"通过"不是手动判断，要写成自动化决策脚本，否则推进节奏会依赖个人记忆，很容易被跳过。

## 关键指标：你要看什么

指标分两层：**工程层**和**质量层**。

工程层的指标和普通服务一样，接入现有监控体系就行：
- 错误率（API 调用失败、超时、格式解析失败）
- 延迟（p50 / p95 / p99）
- Token 消耗（cost per request）

质量层是 LLM 特有的，更难量化：
- **LLM-as-judge 评分**：用一个评估模型给 treatment 和 control 的输出打分，自动对比均值
- **用户行为信号**：点赞/踩、复制率、追问率、会话中止率
- **下游任务成功率**：如果 LLM 输出要被后续逻辑处理，看解析成功率和任务完成率

质量层的指标不用很精确，能区分"明显变差"就够了。5% 的流量跑 2 天，样本量通常足以发现系统性问题。

## 回滚：30 秒内要能完成

灰度有意义的前提是出问题能快速回滚。回滚不应该是"修代码 → CI/CD → 重新部署"，那至少要 10-30 分钟。正确做法是：

**把版本切换做成配置，而不是代码**。Feature Flag 的值存在 Redis 或配置中心，回滚 = 把 `rollout_pct` 改回 0，服务不重启，实时生效。

```python
# 从 Redis 读 rollout 配置，热更新
import redis

r = redis.Redis()

def get_rollout_pct(flag_name: str) -> float:
    val = r.get(f"ff:{flag_name}:pct")
    return float(val) if val else 0.0
```

运维侧准备一个简单的管理脚本或 Dashboard，紧急情况下任何人都能一键把灰度比例归零，不需要等有权限的工程师来操作。

## 踩坑清单

- **不要对"只改了注释"的 prompt 跳过灰度**：注释也会影响模型输出，LLM 没有"安全改动"这个概念
- **不要用时间分割代替用户分割**：比如"8-9 点走新版本，其他时间走旧版本"，这会把时区、用户活跃时段的差异混进来，结论不可信
- **不要只看 p50 延迟**：LLM 的长尾延迟（p99）往往比平均值高 3-5 倍，用 p50 评估会漏掉最差体验
- **不要在 staging 全量测完才上灰度**：staging 和生产的流量分布天然不同，灰度本身就是最好的测试环境
- **模型提供商更新后要主动触发小流量验证**：OpenAI / Anthropic 的模型有时会静默 patch，你的监控要能感知到基线漂移

灰度不是银弹，但它是你在 LLM 应用不确定性面前少数几个能主动控制的杠杆之一。把风险锁在 5% 的流量里，出问题 30 秒收手，这才是工程上的正确姿势。
