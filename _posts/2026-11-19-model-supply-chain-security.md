---
layout: post
title: 模型供应链安全 — 你下载的权重文件可能是一段木马
date: 2026-11-19
topic: "评估与安全"
tags: [供应链安全, safetensors, 模型安全]
excerpt: 从 pickle 反序列化到 trust_remote_code，拆解模型权重、分词器和依赖包里的攻击面，给出可落地的加载规范与扫描清单。
permalink: /posts/2026-11-19-model-supply-chain-security.html
---

你有没有算过，自己一周从 Hugging Face 上拉了多少个模型？一个 embedding 模型、一个 reranker、一个社区微调的 LoRA，再加一个据说效果更好的量化版本。每次 `from_pretrained` 一行搞定，很少有人会停下来想：这个文件到底是谁传的，里面除了权重还装了什么。

我们花大力气防 Prompt Injection，给输出加护栏，给工具调用加权限，却常常忘了一件事：模型文件本身就是一个外部输入，而且是会被你的进程直接加载、拥有完整机器权限的输入。传统软件早就经历过依赖投毒的教训，npm 和 PyPI 上的恶意包事件每年都有，模型生态现在正走在同一条路上，只是很多团队还没意识到。

## 权重文件为什么能执行代码

问题的根源在 Python 的 pickle。PyTorch 早期的 `.bin`、`.pt`、`.pth` 格式本质上是 pickle 序列化，而 pickle 反序列化时会调用对象的 `__reduce__` 方法，攻击者可以在里面塞任意函数调用。也就是说，你执行 `torch.load` 的那一刻，代码已经跑完了，根本不用等到推理。

我在内网环境复现过一次，构造恶意文件只需要几行：

```python
import pickle
import os

class Payload:
    def __reduce__(self):
        # 加载时执行任意命令，这里只是写个标记文件
        return (os.system, ("touch /tmp/pwned_by_model",))

with open("model.bin", "wb") as f:
    pickle.dump({"weight": Payload()}, f)

# 受害者这边：
# import torch
# torch.load("model.bin", weights_only=False)  # 一加载就中招
```

真实攻击里，这条命令会换成反弹 shell、偷读 `~/.aws/credentials`，或者把 GPU 机器上的环境变量打包外发。更麻烦的是，恶意 payload 可以和正常权重共存，模型加载完照样能推理，效果和原版一模一样，你从输出上完全看不出异常。安全研究团队在公开模型仓库里多次扫出过这类文件，有的伪装成热门模型的「优化版」或「去审查版」，下载量还不低。训练机和推理机往往挂着云账号凭证和内部数据集，一旦失陷，损失远比一个模型大。

## 攻击面不止 pickle

把视角拉宽，一个模型从公开仓库落到你的推理服务，中间至少经过下载、安装依赖、初始化、加载权重、推理这几个环节，每个环节都能被人下手。下面这张表是我做内部安全评审时用的分类：

| 攻击面 | 典型手法 | 触发时机 | 主要防御 |
|---|---|---|---|
| 权重序列化 | pickle `__reduce__` 注入 | 加载权重时 | 只用 safetensors，`weights_only=True` |
| 远程代码 | `trust_remote_code=True` 拉取仓库里的 `modeling_*.py` | 初始化模型时 | 固定 revision，人工审阅代码 |
| 名称混淆 | typosquatting，仿冒组织名或模型名 | 下载时 | 白名单组织，校验来源 |
| 依赖包 | 恶意 pip 包、被投毒的 tokenizer 扩展 | 安装环境时 | 锁版本，哈希校验，私有镜像 |
| 权重后门 | 训练阶段植入触发词行为 | 推理时 | 行为评测，触发词扫描 |

前四项是「加载即中招」，危害直接落到机器上；最后一项更隐蔽，模型日常表现完全正常，只在遇到特定触发词时输出攻击者想要的内容，比如在代码生成里悄悄插入一个弱随机数，或者在分类任务里放过某类请求。你拿到的是别人训练好的成品，看不到训练数据，也就无从审计。

另外要特别说一下 `trust_remote_code`。很多新架构的模型需要它才能加载，因为 Transformers 主干还没合并对应实现。这个参数的含义很直白：从模型仓库下载 Python 文件并在你的进程里执行。仓库作者今天传的代码是干净的，不代表明天不会悄悄改一个 commit，账号被盗的情况也并不罕见。我见过最常见的错误，就是生产配置里写死 `trust_remote_code=True`，同时不指定 `revision`，等于把服务器执行权交给一个随时可变的远程分支。

## 一套能落地的加载规范

规范要写进代码，而不是写在 wiki 里。我们团队最后收敛成一个统一的加载入口，所有服务都必须走它，并在代码审查里禁止业务代码直接调用 `torch.load`：

```python
from pathlib import Path
from safetensors.torch import load_file

ALLOWED_ORGS = {"BAAI", "Qwen", "meta-llama", "intfloat"}

def safe_load(repo_id: str, revision: str, local_dir: Path):
    org = repo_id.split("/")[0]
    if org not in ALLOWED_ORGS:
        raise PermissionError("模型来源不在白名单: " + repo_id)
    if len(revision) != 40:
        raise ValueError("必须固定到完整 commit hash")
    files = list(local_dir.glob("*.safetensors"))
    if not files:
        raise ValueError("拒绝加载非 safetensors 权重")
    state = {}
    for f in files:
        state.update(load_file(str(f)))
    return state
```

safetensors 只存张量和元数据，格式里没有可执行的部分，从根上去掉了反序列化风险，加载速度还更快。白名单和 commit 锁定分别处理名称混淆和远程变更。如果某个模型只提供 `.bin`，就在无网络、无凭证的隔离沙箱里用 `weights_only=True` 加载，转换成 safetensors 后再进入正式环境。

需要远程代码的模型，做法是把仓库里的 Python 文件拉到自己的代码库，像审第三方依赖一样逐行过一遍，重点看有没有网络请求、文件读写和 `subprocess`，然后用本地路径加载。远程代码一旦审过，就应该当成你自己的代码来维护，升级时重新走一遍审查。

## 把扫描接进流水线

光靠加载规范还不够，因为总有人在笔记本上随手试新模型，试完觉得好用就直接塞进镜像。我的建议是在模型进入内部制品库之前，加一道自动化闸门：用 modelscan、picklescan 这类工具扫描序列化文件里的危险 opcode，记录每个文件的 SHA256，把模型卡、许可证、来源 commit 一起存档，形成一份模型版的 SBOM。生产环境只允许从内部制品库拉取，出网访问公开仓库的权限直接关掉。

对于权重后门，静态扫描无能为力，只能靠行为评测。可以准备一组包含常见触发模式的探针 prompt，比如罕见字符组合、特定格式指令、少见语言混写，对比新模型和可信基线模型的输出差异，差异显著的样本单独人工复核。这件事成本不低，所以优先用在会接触敏感数据、会生成代码或直接对外服务的模型上，内部玩具项目可以适当放宽要求。

## 按风险分级，别一刀切

规范太严，研发会绕着走；太松，又等于没有。我的做法是按用途把模型分成三级。第一级是本地实验，允许从公开仓库下载，但必须在没有云凭证、没有内网权限的独立环境里运行，用完即销毁。第二级是内部工具，必须走制品库、通过扫描、锁定版本，远程代码要有人签字审过。第三级是对外服务或接触用户数据的模型，在第二级基础上还要跑行为评测，并留存完整的来源记录。

分级的好处是把安全成本花在刀刃上。大多数模型只是被试一下就丢掉了，没必要为它们走完整流程；真正进入生产链路的往往只有少数几个，对它们做足审查，投入产出比最高。另外要给每个上线模型指定负责人，上游仓库有新版本、出安全公告时，由负责人评估是否升级，而不是等出事了才发现没人知道这个模型是谁引进的。

最后是我自己和团队踩过的坑，供你对照自查：

- 只换了加载函数，没清缓存目录：旧的 `.bin` 文件还躺在 `~/.cache/huggingface` 里，别的脚本照样能加载。
- 以为 GGUF 就安全：格式本身不执行代码，但解析器出过内存越界漏洞，llama.cpp 要跟进版本。
- LoRA adapter 没人管：大家只审基座模型，adapter 照样可能是 pickle 格式。
- Docker 镜像里预装模型：镜像构建时下载的文件同样要过扫描，别让 CI 成为盲区。
- 迷信 `weights_only=True`：它限制了可反序列化的类型，但旧版 PyTorch 出过绕过漏洞，先升级再依赖它。

一句话总结：模型是代码，权重是输入，来源是信任边界，三件事都要像对待第三方依赖一样严肃，别等到机器被人当矿机用了才回头查是哪个模型惹的祸。
