---
layout: post
title: SYSTEM 用户跑 git push 一直卡住？踩坑全记录
date: 2026-05-22
tags: [Git, Windows, 自动化, 踩坑]
excerpt: 一个 Windows 服务以 SYSTEM 身份跑 sync 脚本时 git push 死在 GCM，从 safe.directory 到换 SSH 的完整调试链。
permalink: /posts/2026-05-22-system-user-git-push.html
---

## 起因

把一个 OpenClaw 个人助手部署在 Windows 上做自动化，里面有个**配置同步**的步骤：
改完本地配置 → 跑 `sync.ps1` → 自动 commit + push 到 GitHub。

第一次跑能 push，过了半小时再跑——`git push` **卡死无输出**，timeout 也带不出来。

## 第一层：`safe.directory`

先报的是这个：

```
fatal: detected dubious ownership in repository at 'C:/workspace/openclaw-config'
'C:/workspace/openclaw-config' is owned by:
    EC2AMAZ-N5PPFTF/Administrator
but the current user is:
    NT AUTHORITY/SYSTEM
```

原因很直白：仓库目录是 Administrator 创建的，但脚本被 OpenClaw 网关进程拉起，进程身份是 `NT AUTHORITY/SYSTEM`。
Git 2.35+ 会校验属主，不一致就**拒绝读这个仓库**——本来是防"被别人塞恶意 hook"。

加一个白名单就行：

```bash
git config --global --add safe.directory C:/workspace/openclaw-config
```

`--global` 写在 SYSTEM 用户自己的 `.gitconfig` 里。`git status` 立刻通了。

## 第二层：`git push` 卡死

但 push 还是不动——`git push` 没任何输出，timeout 也只是杀进程。

进程列表里看到的：

```
git
git-credential-manager  ← 这是关键
git-remote-https
```

`git-credential-manager`（GCM）是 Windows 上 Git for Windows 自带的凭据管理器。**它会弹一个 UI 对话框问你 GitHub 账号密码**。
SYSTEM 用户没有桌面会话，对话框弹不出来，于是 GCM 在那儿**无限等待**。
之前能 push 是因为 GCM 内存缓存里还有刚刚某次手动 push 留下的凭据，过期之后就崩了。

加 `GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=Never` 强制非交互，立刻看到真相：

```
fatal: could not read Username for 'https://github.com': 
       terminal prompts disabled
```

## 第三层：解法选型

三个方向：

| 方案 | 实操难度 | 维护成本 |
|---|---|---|
| 给 SYSTEM 配 PAT 写进环境变量，`https://USER:PAT@github.com/...` | 中 | 中（PAT 90 天过期） |
| 改用 SSH key | 中 | 低（SSH key 长期有效） |
| 把脚本改成由 Administrator 任务计划运行 | 低 | 低（但要重组架构） |

我选了 **SSH**。原因：

- 不用维护 PAT 的轮换
- 一次配好之后 SYSTEM、Administrator 都能跑
- 多个仓库统一一个 key，干净

## SSH 切换

1. 在 GitHub Settings → SSH Keys 里加上 SYSTEM 能读到的公钥
2. 私钥放在 `%SystemRoot%\System32\config\systemprofile\.ssh\id_ed25519`（SYSTEM 用户的 home）
3. 把仓库 remote 切换：
   ```bash
   git remote set-url origin git@github.com:USER/repo.git
   ```
4. 第一次 push 加上 `StrictHostKeyChecking=accept-new` 接受 GitHub 主机指纹：
   ```bash
   GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes" git push origin main
   ```

再跑——秒推。`Everything up-to-date` / `xxxx..yyyy main -> main`。

## 总结一下我之前哪里想错了

第一次 push 能成功，让我以为"问题只是 safe.directory"。
其实真正的根因是**进程身份没桌面 + HTTPS 凭据依赖 UI**。
凭据缓存掩盖了底层的不兼容，过期那一刻才暴露。

教训：

- 涉及 SYSTEM / 服务用户的自动化脚本，**默认不要走 HTTPS + GCM 这条路径**
- 凭据要么走 SSH（推荐），要么走环境变量里的 PAT
- 调试 push 卡死时，用 `GIT_TERMINAL_PROMPT=0` 强制非交互，让错误立刻暴露，不要让它静默等待

---

> 自动化最怕的就是这种"有时能跑、有时不能跑"的薛定谔故障。每次都能用，才叫稳定。
