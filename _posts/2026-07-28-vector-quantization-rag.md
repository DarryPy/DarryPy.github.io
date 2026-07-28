---
layout: post
title: 向量量化实战 — 让 RAG 向量库省 32 倍内存的 Scalar / Binary / PQ
date: 2026-07-28
topic: "RAG 与检索"
tags: [RAG, 向量数据库, 量化, 性能优化]
excerpt: '向量库内存爆炸、账单翻倍?Scalar / Binary / Product Quantization 三种量化配合 rescoring,能把 RAG 向量存储砍到 1/32 而召回几乎不掉,带你实战调平衡。'
permalink: /posts/2026-07-28-vector-quantization-rag.html
---

你的向量库刚上线时几百万条,岁月静好;等文档涨到几千万条,内存账单突然翻了十倍,检索延迟也开始抖。float32 的 embedding 每维 4 字节,1536 维一条就是 6KB,一千万条光原始向量就吃掉 60GB 内存,还没算 HNSW 索引结构本身的额外开销。

这时候堆机器不是唯一解。向量量化(quantization)能把这笔开销砍到 1/4 甚至 1/32,而召回几乎不掉。它的核心思路很朴素:embedding 里那些小数点后好几位的精度,对"谁离谁近"这个排序问题其实没那么重要,砍掉大部分反而不影响结果。今天带你把三种主流方式拆开讲透,顺便把生产里真正省钱的那套分层存储讲明白。

## 三种量化,压缩率天差地别

Scalar Quantization(标量量化)最温和:把每维的 float32 压成 int8,4 字节变 1 字节,压缩 4 倍。它先统计每一维的最大最小值,再把这个区间线性映射到 0-255 的整数格子里。信息损失小,是最安全的默认选项,绝大多数团队的第一步都该选它,几乎无脑。

Binary Quantization(二值量化)最激进:每维只保留符号位,正数记 1 负数记 0,直接 32 倍压缩。检索时用 Hamming distance 算异或,CPU 一条指令搞定,快到飞起。但精度损失明显,只适合维度高、且经过良好归一化的 embedding;维度低于几百的向量上它经常翻车,得慎用。

Product Quantization(PQ,乘积量化)最巧妙:把高维向量切成若干子段,每段用 k-means 聚成 256 个质心,原始向量只存每段落到哪个质心的编号。压缩率随子段数可调,是 FAISS 里 IVFPQ 索引的核心,适合上亿级、对精度要求不极端的超大规模场景。

还有个容易被忽略的中间档:float16 或 bfloat16,只压 2 倍、几乎零精度损失,连 rescoring 都省得配。如果你只想把 60GB 压到 30GB、不愿折腾召回评估,half precision 是最省心的起点,很多向量库默认就支持,值得作为第一次尝试。

三者并不互斥。生产里最常见的组合,是拿 Scalar 或 Binary 做内存里的粗筛层,再用磁盘上的 float32 精排;PQ 则更多出现在 FAISS 这类纯库、需要把上亿向量硬塞进单机内存的场景。选型第一步,永远是先看你的数据规模落在哪一档。

## 内存和账单,到底能省多少

拿刚才那一千万条 1536 维向量算笔账:float32 原始要 60GB,压成 int8 后掉到 15GB,上 Binary 直接降到不足 2GB。云上高内存实例每 GB 每月按几块钱算,从 60GB 降到 15GB,一年省下来的钱够多养大半个工程师。

省内存还顺带省延迟。向量小了,单次检索要扫的字节数也小,CPU cache 命中率更高,int8 的距离计算还能吃到 SIMD 指令加速。很多团队上量化本是为了省钱,结果发现 P99 延迟也跟着降了一截,算是白捡的收益。

代价当然有:分层存储把原始向量放磁盘,rescoring 时会多一次随机读。但那只是几百条向量,配合 SSD 和操作系统页缓存,延迟增加通常在个位数毫秒。和省下的几十 GB 内存相比,这笔账怎么算都划算,除非你的场景对尾延迟极度敏感。

还有个隐性收益是索引构建。向量占的字节少了,HNSW 建图时的内存峰值和落盘体积都跟着降,大库重建索引的耗时往往能缩短两三成。对每天增量更新、需要频繁 rebuild 的知识库来说,这比检索期省的那点延迟更实在。

顺带一提,量化让纯 CPU 检索重新变得可行。不少团队为了向量检索上 GPU,其实是被 float32 的内存带宽逼的;换成 int8 或 binary 后,一台大内存 CPU 机器就能扛住原本要一张卡的吞吐,硬件账进一步收窄。

## 精度真的会崩吗?靠 rescoring 救场

单纯量化确实掉召回,尤其 Binary。业界标准解法是两段式:先用压缩向量粗筛出 top-100,再用原始 float32 向量对这 100 条精排(rescoring),取 top-10。粗筛快、精排准,鱼和熊掌兼得,这也是所有主流向量库默认推荐的做法。

为什么粗筛不会漏掉真正的答案?因为量化只是给距离加了点噪声,真正相关的文档即便排名从第 3 掉到第 30,也基本还在 top-100 里。只要粗筛候选池开得够大,精排就有机会把它捞回正确位置,这就是 rescoring 敢激进压缩的底气所在。

关键在于原始向量放哪:全塞内存等于没省,放磁盘或 SSD、只在 rescoring 时读那 100 条,I/O 完全可控。Qdrant、Milvus 都支持"量化向量在内存 + 原始向量在磁盘"的分层存储,内存里只留 1/4 甚至 1/32 的量化数据,这才是省钱的真正关键设计。

| 方式 | 压缩率 | 召回损失(带 rescore) | 检索速度 | 适用场景 |
|------|--------|---------------------|----------|----------|
| 无量化 float32 | 1x | 0 | 基准 | 小规模、精度敏感 |
| Scalar int8 | 4x | <1% | 略快 | 通用默认 |
| Product Quant | 8-64x | 1-5% | 快 | 超大规模 |
| Binary | 32x | 3-8% | 极快 | 高维归一化向量 |

## 代码实战:Qdrant 开启量化

Qdrant 里开量化只是 collection 配置,检索接口不变。这里用 Scalar Quantization 配合磁盘原始向量与 rescoring,一套写法就能覆盖大部分生产需求:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="docs",
    vectors_config=models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE,
        on_disk=True,  # 原始向量落盘,省内存
    ),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            always_ram=True,  # 量化向量常驻内存,检索才快
        )
    ),
)

# 检索时开 rescoring,用原始向量精排
hits = client.query_points(
    collection_name="docs",
    query=query_vec,
    limit=10,
    search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=True,       # 用磁盘原始向量重排
            oversampling=3.0,   # 先取 3 倍候选再精排
        )
    ),
).points
```

oversampling 具体设多少,取决于量化有多激进。Scalar int8 掉得少,2 倍常常够用;Binary 掉得多,往往要 4-5 倍才把召回拉回可接受区间。压缩率越高、召回越脆,就越依赖精排来兜底,这是一对必须一起调的旋钮,只能在你自己的数据上盯着召回率和 P99 两条曲线压测。

如果你用的不是 Qdrant,心法完全一致:Milvus 的 IVF_SQ8 与 SCANN、Elasticsearch 8 的 int8_hnsw、pgvector 0.7+ 的 halfvec 和 bit 类型,思路都是内存放压缩、需要时回原始精排。API 名字各不相同,底层这套"粗筛加精排"的分层套路是通用的。

## 到底该选哪个 + 踩坑清单

先问三个问题:向量多少条?内存预算多少?召回容忍度多大?几百万条别折腾,float32 直接上,量化省的那点内存不值得多担一份复杂度;上千万且预算紧,Scalar int8 是性价比之王;上亿且能接受几个点召回损失,再考虑 Binary 或 PQ。别一上来就追最高压缩率,那是拿召回换面子。

落地路径也别激进。先在离线拿一批标注 query 测出各方案的召回,再灰度切一小部分线上流量对比端到端指标,确认没劣化再全量铺开。量化是可逆的存储层改造,回滚成本很低;但一步到位切最激进的方案、再靠线上救火,那纯属给自己挖坑,冷静分阶段走稳得多。

踩坑清单:

- Binary 量化前务必对 embedding 做 L2 归一化,否则符号位分布失衡,召回直接崩盘。
- rescoring 的 oversampling 别设太小,2 倍以下经常救不回召回,建议从 3 起步再压测。
- 别拿量化召回去死磕 float32 的绝对数值,要看业务端到端的答案质量,量化掉的那 1% 常常无关痛痒。
- PQ 训练需要有代表性的样本,冷启动数据太少时质心不准,先用 Scalar 过渡,数据攒够再切。
- 上线前用真实 query 集跑一遍 A/B,别信公开基准的召回数字,你的领域分布和它天差地别。
- 量化只是存储层的事,不改变 embedding 模型本身;换模型时向量得全量重算,别把这两件事搅在一起。
- 监控盯召回而非距离绝对值,量化会整体拉低相似度数值,凡是写死的相似度阈值都要跟着重新标定。

一句话:量化不是精度和成本二选一,而是用一点点召回换回几十倍的内存和金钱——在生产级 RAG 里,这笔交易几乎永远划算。
