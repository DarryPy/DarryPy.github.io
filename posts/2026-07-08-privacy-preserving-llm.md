---
layout: post
title: "隐私保护 LLM — 处理敏感数据的工程方案"
date: 2026-07-08
topic: "评估与安全"
tags: [AI, Privacy, Security]
excerpt: 用户数据发给 LLM API 之前你想清楚了吗？日志、训练、泄露——每个环节都有风险。四种工程方案，按威胁级别选。
permalink: /posts/2026-07-08-privacy-preserving-llm.html
---

## 威胁模型先搞清楚

在谈解决方案之前，先想清楚你的威胁是什么：

**威胁一：数据被 API 供应商记录用于训练**
- Anthropic、OpenAI 的 API 默认不用商业 API 数据训练（有合同保障）
- 但你在测试/开发阶段用的免费 tier 可能没有这个保障
- 风险等级：商业 API 中低，免费 tier 中

**威胁二：数据在传输中被截获**
- HTTPS 已经处理了这个问题
- 风险等级：低（如果使用 HTTPS）

**威胁三：你自己的日志和存储泄露**
- 你把含 PII 的 prompt 写进了 ELK、Datadog、或者 S3
- 员工可以访问这些日志
- 风险等级：高（这是实际上最常见的泄露源）

**威胁四：模型输出中的隐私信息扩散**
- 模型把 A 用户的信息混入给 B 用户的回复（共享上下文 bug）
- 风险等级：取决于你的架构

**对应的解法是不同的。** 不要混在一起。

## 方案一：发送前 PII 脱敏

**适用**：不能私有化部署，但需要降低数据泄露风险。

```python
# pip install presidio-analyzer presidio-anonymizer spacy
# python -m spacy download zh_core_web_sm

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import re

# 初始化（支持中英文）
configuration = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "zh", "model_name": "zh_core_web_sm"},
        {"lang_code": "en", "model_name": "en_core_web_sm"}
    ]
}
provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()

analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["zh", "en"])
anonymizer = AnonymizerEngine()

class PIIScrubber:
    # 中文手机号、身份证、银行卡的正则补充
    CHINESE_PATTERNS = [
        (r'1[3-9]\d{9}', 'PHONE_NUMBER'),
        (r'\d{15}|\d{18}|\d{17}[xX]', 'ID_CARD'),
        (r'[1-9]\d{15,18}', 'BANK_CARD'),
        (r'\d{3}-\d{4}-\d{4}', 'PHONE_NUMBER'),
    ]

    def __init__(self):
        self.entity_map = {}  # 用于还原

    def scrub(self, text: str, lang: str = "zh") -> tuple[str, dict]:
        """
        脱敏文本，返回 (脱敏后文本, 实体映射表)
        映射表可用于还原模型输出中的占位符
        """
        entity_map = {}
        counter = {}

        # 1. 用 Presidio 检测
        try:
            results = analyzer.analyze(text=text, language=lang)
        except:
            results = []

        # 2. 加上中文正则规则
        for pattern, entity_type in self.CHINESE_PATTERNS:
            for match in re.finditer(pattern, text):
                # 创建 Presidio 格式的结果（简化）
                results.append(type('Result', (), {
                    'entity_type': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'score': 0.9
                })())

        # 3. 按位置排序，从后往前替换（防止位移）
        results.sort(key=lambda x: x.start, reverse=True)
        scrubbed = text
        for result in results:
            entity_type = result.entity_type
            original = text[result.start:result.end]
            counter[entity_type] = counter.get(entity_type, 0) + 1
            placeholder = f"[{entity_type}_{counter[entity_type]}]"
            entity_map[placeholder] = original
            scrubbed = scrubbed[:result.start] + placeholder + scrubbed[result.end:]

        return scrubbed, entity_map

    def restore(self, text: str, entity_map: dict) -> str:
        """还原占位符"""
        for placeholder, original in entity_map.items():
            text = text.replace(placeholder, original)
        return text

# 使用示例
scrubber = PIIScrubber()

user_input = "我叫张三，手机号是13812345678，身份证350123198801011234，请帮我查询"
scrubbed, entity_map = scrubber.scrub(user_input)
print(f"脱敏后：{scrubbed}")
# 输出：我叫[PERSON_1]，手机号是[PHONE_NUMBER_1]，身份证[ID_CARD_1]，请帮我查询

import anthropic
client = anthropic.Anthropic()

# 发给 LLM 的是脱敏版本
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": scrubbed}]
)
reply = response.content[0].text

# 还原模型输出中的占位符（如果模型重复了占位符）
restored_reply = scrubber.restore(reply, entity_map)
print(f"还原后回复：{restored_reply}")
```

**注意事项**：
- Presidio 对中文支持一般，需要结合自定义正则
- 脱敏后语义可能变化，影响模型理解
- 模型可能在输出中引用占位符，需要还原

## 方案二：本地/私有部署

**适用**：数据不能出公司网络（医疗、金融、政务）。

### 方案 2a：Ollama（本地单机）

```bash
# 安装
curl -fsSL https://ollama.ai/install.sh | sh

# 拉取模型
ollama pull llama3.1:8b        # 8B 参数，需要 ~8GB RAM
ollama pull qwen2.5:14b       # 14B，需要 ~16GB RAM

# 启动（默认监听 localhost:11434）
ollama serve
```

```python
import httpx

def local_llm_call(prompt: str, model: str = "llama3.1:8b") -> str:
    """调用本地 Ollama"""
    response = httpx.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120.0
    )
    return response.json()["response"]

# 完全不出网络
result = local_llm_call("分析这份内部合同的主要条款...")
```

### 方案 2b：vLLM（生产级私有部署）

```bash
pip install vllm

# 启动 OpenAI-compatible API server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2 \   # 多 GPU
    --max-model-len 32768
```

```python
from openai import OpenAI

# 使用 OpenAI SDK 但指向私有服务器
private_client = OpenAI(
    base_url="http://your-internal-server:8000/v1",
    api_key="not-needed"
)

response = private_client.chat.completions.create(
    model="Qwen/Qwen2.5-14B-Instruct",
    messages=[{"role": "user", "content": "处理敏感的内部数据..."}]
)
```

**成本参考**：
- 8B 模型：A100 40GB 单卡可跑，价格约 ¥3/小时（国内云）
- 14B 模型：A100 80GB 单卡或两张 40GB
- 70B 模型：4× A100 40GB

## 方案三：敏感字段加密后处理

对于结构化数据，可以只把非敏感字段发给 LLM：

```python
def selective_field_redaction(record: dict, sensitive_fields: list[str]) -> tuple[dict, dict]:
    """
    只把非敏感字段发给 LLM
    返回 (脱敏记录, 原始敏感字段)
    """
    safe_record = {}
    sensitive_data = {}

    for key, value in record.items():
        if key in sensitive_fields:
            sensitive_data[key] = value
            safe_record[key] = f"[REDACTED_{key.upper()}]"
        else:
            safe_record[key] = value

    return safe_record, sensitive_data

# 示例：用户记录分析
user_record = {
    "user_id": "u_12345",       # 非敏感
    "name": "张三",              # 敏感
    "phone": "13812345678",     # 敏感
    "email": "zhang@example.com", # 敏感
    "purchase_count": 15,       # 非敏感
    "last_category": "电子产品",  # 非敏感
    "avg_order_value": 850,     # 非敏感
}

safe_record, sensitive_data = selective_field_redaction(
    user_record,
    sensitive_fields=["name", "phone", "email"]
)

# 只发安全字段给 LLM
prompt = f"根据以下用户行为数据，推荐最合适的商品类别：{safe_record}"
recommendation = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    messages=[{"role": "user", "content": prompt}]
).content[0].text
```

## 方案四：合成数据测试

开发和测试阶段，不要用真实用户数据：

```python
def generate_synthetic_pii(n: int = 100) -> list[dict]:
    """生成合成测试数据（不含真实 PII）"""
    import random
    import string

    fake_names = ["张小明", "李华", "王芳", "陈建国", "刘梅"]
    fake_cities = ["北京", "上海", "广州", "深圳", "杭州"]

    records = []
    for i in range(n):
        # 随机生成（非真实数据）
        phone = "1" + str(random.randint(3, 9)) + \
                ''.join([str(random.randint(0, 9)) for _ in range(9)])
        records.append({
            "name": random.choice(fake_names) + str(i),
            "phone": phone,
            "city": random.choice(fake_cities),
            "age": random.randint(18, 65),
            "test_case_id": f"synthetic_{i}",  # 明确标注是合成数据
        })

    return records

# 或者用 LLM 生成更真实的合成数据
def generate_synthetic_with_llm(schema: str, n: int = 10) -> list[dict]:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""生成 {n} 条符合以下 schema 的虚构测试数据（JSON 数组格式）。
数据必须是完全虚构的，不得包含真实存在的人名、电话、身份证号等。

Schema: {schema}

输出："""
        }]
    ).content[0].text

    import json, re
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return []
```

## 数据留存策略

```python
# 生产环境的 LLM 调用日志应该脱敏
import logging
import hashlib

class PrivacyAwareLogger:
    def __init__(self):
        self.logger = logging.getLogger("llm_calls")

    def log_call(self, user_id: str, prompt: str, response: str, metadata: dict):
        # 不记录原始 prompt（可能含 PII）
        # 只记录：用户哈希、prompt 长度、token 数、延迟
        self.logger.info({
            "user_hash": hashlib.sha256(user_id.encode()).hexdigest()[:8],
            "prompt_length": len(prompt),
            "response_length": len(response),
            "model": metadata.get("model"),
            "input_tokens": metadata.get("input_tokens"),
            "output_tokens": metadata.get("output_tokens"),
            "latency_ms": metadata.get("latency_ms"),
            # 绝不记录 prompt 内容和 response 内容
        })
```

## GDPR 合规检查清单

```
□ 是否向用户说明数据会被发送给第三方 LLM API？
□ API 供应商的数据处理协议（DPA）是否签署？
□ 用户数据在 LLM 供应商侧的保留时间是多久？
□ 日志和缓存中是否有 PII？保留多久？
□ 用户要求删除数据时，LLM 调用日志是否也能删除？
□ 是否有对 LLM 供应商的数据泄露通知机制？
□ 如果使用 fine-tuning，训练数据中是否包含 PII？
□ 跨境数据传输是否符合当地法规（欧盟数据留欧等）？
```

## 一个朴素结论

> 隐私保护的优先级顺序：
> 1. 先搞清楚你自己的日志和存储——这是最高概率的泄露源
> 2. 开发测试阶段用合成数据，别用生产数据
> 3. 高敏感场景用 PII 脱敏或私有部署
>
> 不要把时间浪费在 API 供应商"可能用数据训练"的担忧上，
> 先把自己的日志系统里的 PII 清掉。
