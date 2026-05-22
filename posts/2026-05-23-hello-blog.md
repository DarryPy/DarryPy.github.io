---
layout: post
title: 博客上线 — 给自己留一个安静的写字角
date: 2026-05-23
tags: [杂谈, 博客, 起步]
excerpt: 为什么在 GitHub Pages 起一个个人博客，以及这个页面背后的小心思。
permalink: /posts/2026-05-23-hello-blog.html
---

## 为什么要做这个博客

我一直想有一个**安静的地方**——不被算法干扰、不依赖任何平台、随时可以写、随时可以归档。
朋友圈太碎、知乎太吵、公司内网又没法长期带走。最后还是回到了**最朴素的方案**：一个属于自己的网页。

GitHub Pages 刚好满足条件：

- 免费、稳定、HTTPS 自带
- 内容都在 Git 里，**关掉哪天也能搬走**
- 写一篇 push 一次，没有花哨的 CMS

## 这个站做了什么

这个博客托管在 `gh-pages` 分支，主页是一份纯 HTML（暗黑配色 + 渐变 orb + 玻璃质感卡片）。
文章用 Markdown 写在 `posts/` 目录，Jekyll 自动渲染成 HTML 页面，套上我做的 `_layouts/post.html` 主题。

技术上做了几件**让我自己舒服的小事**：

- **iPhone 刘海屏** safe-area 适配，状态栏颜色与背景一致
- 移动端导航**横向可滚动**，按钮全宽便于点击
- 关掉了双击放大延迟、蓝色高亮等"网页感"
- `prefers-reduced-motion` 用户进来时所有动画自动停
- 暗色模式默认，再也不被白屏闪眼

## 接下来想写些什么

- 后端日常**踩坑实录**：慢 SQL、数据迁移、ClickHouse 调用规范、缓存一致性
- AI Agent 实战：本地 Agent 工作流、Telegram 上跑出来的小玩具
- 工程化思考：日报/周报怎么写得管理者满意，团队节奏怎么稳
- 偶尔的杂谈和读书笔记

不追更新频率，**写的时候才有意思**。

---

> 这是第一篇，欢迎在 [GitHub](https://github.com/DarryPy) 或 [Telegram](https://t.me/asher2050) 找我聊。
