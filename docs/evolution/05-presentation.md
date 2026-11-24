# 05 · 数据呈现优化设计

> 目标：把当前"一张长 markdown + 四张静态 PNG"的仓库，变成**可按维度检索、可分享视图、可交互看趋势**的站点，同时不破坏 README 作为唯一权威清单的地位。
>
> 对应骨架：`site/`（Astro 7 + Pagefind 1.6 + ECharts 6）。实际构建产物：16 页 / 2.2 MB / 0.36 s。

## 1. 信息架构

顶层 7 个页面，与 README 的 `0x00–0x07` 分节**一一对应但不照抄**——README 的编号在站点里一旦变成 URL 就不可变，所以 URL 用语义名，页面标题保留编号做锚点回链。

| 路由（en / zh） | 来源 | 角色 | 一次点击能完成的事 |
| --- | --- | --- | --- |
| `/` `/zh/` | 聚合 | 概览：4 个统计卡 + 3 张趋势图 + 最新年份论文卡片 | 3 秒判断"这个仓库还在更新吗、覆盖哪些年份" |
| `/papers/` `/zh/papers/` | `0x00 Papers`（+ `data/papers/*.json`） | **论文库**：170 条多维筛选表 | 选出"2020 年 AAAI 上的 STGNN 且带代码" |
| `/datasets/` `/zh/datasets/` | `0x02 DataSource` + `datasets/*.md` | 数据集清单（按区域/信号分组） | 找到 METR-LA 的传感器数、采样间隔、本地文档页 |
| `/toolkits/` `/zh/toolkits/` | `0x03 Toolkits` | 工具库（含"是否已实现基线"提示） | 决定用 LibCity 还是 PyG-Temporal |
| `/benchmark/` `/zh/benchmark/` | `0x07 Benchmark & Evaluation` | 基准矩阵 + 汇报检查清单 | 抄一份可信的指标/切分口径 |
| `/trends/` `/zh/trends/` | `updates/hotspots/*.json`、`updates/wordcloud/*.json` | 交互式趋势（替代 4 张 PNG） | 看热点词随年份的迁移 |
| `/lineage/` `/zh/lineage/` | `0x00 Papers` 按年分组 | 研究脉络时间线（**不截断**，`……` 条目以外的全量） | 顺着年份看方法演进 |

设计取舍，逐条说清"为什么不"：

- **不做论文详情页**（`/papers/<id>/`）。170 条中绝大多数只有"标题 + venue + 2~3 个链接"，单页会产出 170 个 400 字的薄页面，属于典型的 thin content，对 SEO 是负数。当某条论文有 `models/*.md` 那样的专门文档时（如 `models/graph-wavenet.md`），再为它生成 `/models/<slug>/` 并提供从论文卡跳转——**按内容量决定是否成页**。
- **不做导航侧边栏**。7 个顶层页面用顶栏就够；侧边栏在移动端收益为负，在桌面端需要维护分组元数据（第三个真相源）——而 README 的 doctoc TOC 已经承担了"全文目录"的角色。
- **`/trends/` 与 `/lineage/` 分开**。前者回答"领域在往哪走"（聚合统计），后者回答"这篇论文站在哪条线上"（单篇定位）。合并会让页面同时需要两种交互（筛选 + 时间轴），复杂度不可控。

双语对应关系由 `README.md` ⇄ `README.zh.md` 的既有约定继承：站点不引入第三套文案真相源，UI 字符串在 `site/src/lib/i18n.ts`，内容字符串从 README/JSON 读。

## 2. 交互设计

### 2.1 四维筛选（year × venue × task × method tag）

状态模型（`site/src/components/PaperList.astro`）：

```ts
interface State {
  q: string;            // 全文检索（标题/venue/短名/标签拼接后的子串匹配）
  years: Set<string>;   // 多选，OR
  venues: Set<string>;  // 多选，OR
  tasks: Set<string>;   // 多选，OR
  tags: Set<string>;    // 多选，AND ← 关键
  sort: 'newest' | 'oldest' | 'venue';
}
```

**组内 OR、组间 AND、标签组内 AND** 是刻意的：

- `year ∈ {2019, 2020}` 的语义是"这两年"，必须 OR，否则点第二个年份就等于取消第一个。
- `tag ∈ {STGNN} AND tag ∈ {Transformer}` 的语义是"同时用了图网络和 Transformer"，必须 AND。如果这里也 OR，选两个标签会得到并集（更多结果），与"筛得越细结果越少"的用户预期相反。
- 实现上靠 DOM 属性匹配，不重建节点：

```html
<article data-year="2020" data-venue="AAAI" data-task="traffic-forecasting"
         data-tag="STGNN|Transformer" data-sort-key="2020|AAAI|gman: a graph...">
```

```js
if (state.tags.size) {
  const own = new Set(el.dataset.tag.split('|').filter(Boolean));
  for (const tag of state.tags) if (!own.has(tag)) return false;   // AND
}
```

性能：170 条 × 4 组比较 ≈ 一次 `requestAnimationFrame` 内完成；2,500 条以内无需虚拟滚动（见 `03-ssg-choice.md` §4 的拐点测算）。

### 2.2 URL query 可分享

筛选状态 **单向** 同步到 URL（`history.replaceState`，不产生历史记录栈）：

```text
/papers/?q=graph&y=2019&y=2020&v=AAAI&t=traffic-forecasting&m=STGNN&m=Transformer&s=venue
```

- 首次进入时 `readUrl()` 从 query 恢复状态 → 粘一个链接给同事，对方看到的是同一视图。
- 页面同时提供"复制筛选链接"按钮（`navigator.clipboard`，HTTP 环境下静默降级——URL 栏本身已是状态）。
- **不**用 hash 路由：hash 不会被发送到服务器，也没有 `<link rel="canonical">` 的意义；query 参数形式对 Pages 完全无副作用（同一份静态 HTML）。

### 2.3 Paper 卡片徽标

三种徽标，颜色即语义（不依赖 JS）：

```html
<span class="pill year">2020</span>        <!-- 蓝：年份 -->
<span class="pill code">code</span>        <!-- 绿：有官方/第三方实现 -->
<span class="pill data">data</span>        <!-- 黄：附带数据集链接 -->
<a class="badge" href="paper">pdf</a>
<a class="badge code" href="code">code</a>
<a class="badge data" href="data">data</a>
<a class="badge" href="review">review</a>
```

`[[code]]` / `[[code-official-tf]]` / `[[code-pytorch]]` / `[[code-not-official]]` 在解析时统一收敛到 `codeUrl`，同时把原始区分保留在 `badges[]` 里（"官方实现"与"第三方复现"对读者是不同信息，但都算"有代码"）。当前实测：**104 条中 26 条带代码链接、2 条带数据集链接**。

## 3. 可视化：把 `updates/*.json` 变成交互图

### 3.1 数据现状（实测，非假设）

| 文件 | 结构 | 体量 | 站点用途 |
| --- | --- | --- | --- |
| `updates/hotspots/yearly-hotspots.json` | `{generated_utc, topk, years: {"2015": [{term, count}], ...}}` | 12 年 × 12 词 | 年份 × 排名热力图 |
| `updates/wordcloud/monthly-paper-wordcloud-latest.json` | `{generated_utc, input, phrase_rules[], top_words: [[term,count],...]}` | 32 词 | Top-20 横向条形图 |
| （派生）论文库本身 | 站点内 `papers` 数组 | 104 条 | 逐年柱状、任务饼图、venue 条形 |

`updates/hotspots/yearly-hotspots.json` 的 `count` 全部为 1（词频按年份切片后没有跨年累加），所以热力图用的是"排名位置"语义而非"热度绝对值"——`visualMap` 的 `min/max` 从数据动态求，不做假设。

### 3.2 替代映射

| README 中的静态 PNG | 站点交互图 | 交互增量 |
| --- | --- | --- |
| `yearly-hotspots-trend.png` | 年份 × rank 热力图（`heatmap` + `visualMap`） | hover 出年份/词/计数；视觉突出"哪年出现新热词" |
| `yearly-theme-trends.png` | 任务分布 + venue 分布 | 可以点图例隐藏类别 |
| `yearly-hotspots.png` | 同上热力图（合并） | — |
| `monthly-paper-wordcloud-latest.png` | Top-20 词频条形图 | 排序可读；词云本身在**信息检索效率上是负资产**（弧形文字难扫读），保留 PNG 作为"月度快照"放回 README，站点只给可排序的版本 |
| `monthly-paper-wordcloud-topk-latest.png` | 同上（合并） | — |

### 3.3 性能与无 JS 降级

四条硬规则（已落地在 `site/src/components/TrendChart.astro`）：

1. **ECharts 按需注册**：只 `use([BarChart, LineChart, PieChart, HeatmapChart, Grid/Tooltip/Legend/Title/DataZoom/VisualMap, CanvasRenderer])`。
2. **动态 `import()`**：ECharts 不进首屏 bundle。产物里是 8 个按需 chunk（`charts.js` 259 KB、`components.js` 259 KB…），**只有在有图表的页面才会被请求**；`/papers/`、`/datasets/` 等页面 0 图表 JS。
3. **`IntersectionObserver` + `rootMargin: 200px`**：进入视口前不初始化。`/trends/` 有 4 张图，首屏只初始化可见的那张。
4. **`<details>` 静态表**：每张图内置同一份数据（`fallback: {headers, rows}`），**无 JS 也能读**；爬虫拿到的也是数字而不是空 `div`。

成本量化：

| 指标 | 数值 |
| --- | --- |
| ECharts 全部 chunk（未压缩） | ~1.1 MB |
| 实际下载（gzip） | ~330 KB，仅图表页 |
| 首屏阻塞 | 0（动态 import + 视口延迟） |
| 无 JS 时 | 静态表可读，图表区为空高度 |
| 未来可选 | `echarts` 换 `echarts/core` + SVG renderer 可再降 20%，但 Canvas 在 4 张图时更省内存，暂不换 |

**不做的事**：不为图表引入图表库的 React/Vue 封装（多一层运行时）；不用 canvas 自绘（维护成本高于收益）；不把 PNG 从仓库删掉（`monthly-paper-search.yml` 仍在生成，且 README 需要它——站点是**补充**，不是替代）。

## 4. 双语路由与 SEO

### 4.1 路由组织

```
/                      → 英文首页（默认 locale，无前缀）
/papers/ /trends/ ...  → 英文子页
/zh/                   → 中文首页
/zh/papers/ /zh/trends/ → 中文子页
```

`astro.config.mjs`：

```js
i18n: {
  defaultLocale: 'en',
  locales: ['en', 'zh'],
  routing: { prefixDefaultLocale: false, redirectToDefaultLocale: false },
}
```

为什么英文不前缀：README 本身以英文为主，顶部链接指向 `README.zh.md`；站点保持一致——**英文是默认，中文是可选项**。反过来（`/en/` + `/zh/`）会让每个英文 URL 多一段前缀，且与 GitHub 上仓库首页的既有读者预期冲突。

为什么中文用 `/zh/` 而不是 `/cn/` 或 `zh-CN`：`zh` 是 BCP-47 主语言标签，Pagefind 的语言检测输出也是 `zh-cn`，两者可对齐；URL 同时短。

### 4.2 hreflang 与 canonical

每页 `BaseLayout.astro` 输出三行：

```html
<link rel="canonical" href="https://coolgiserz.github.io/Awesome-Traffic-Prediction/zh/papers/" />
<link rel="alternate" hreflang="zh-CN" href="https://…/zh/papers/" />
<link rel="alternate" hreflang="en"    href="https://…/papers/" />
<link rel="alternate" hreflang="x-default" href="https://…/papers/" />
```

三个易错点，全部已处理：

1. **canonical 必须含 base 前缀**。`new URL(pathname, site)` 在 Project Pages 下会丢掉 `/Awesome-Traffic-Prediction/`；实现里先 `pathname.replace(BASE, '')` 再 `withBase()` 拼回，保证唯一且带前缀。
2. **`x-default` 指向英文那条**（不是首页）。指向首页会让搜索引擎把语言选择器当成导航而非替代版本。
3. **语言切换链接用 `altLocalePath()` 镜像当前路径**，而不是永远跳首页：

```ts
altLocalePath('/Awesome-Traffic-Prediction/zh/papers/', 'en')
// → '/Awesome-Traffic-Prediction/papers/'
```

### 4.3 主题语言与内容语言

- `<html lang="zh-CN">` / `<html lang="en">`：由 `htmlLang` 映射，别把 `zh` 直接写进 `lang`（部分朗读器会退化到默认语音）。
- `og:locale` = `zh_CN` / `en_US`，`og:locale:alternate` 给对方。
- **不**用 `?lang=zh` 参数式切换：参数化双语会让同一内容有无限 URL 变体，且 Pagefind 只能索引一份。

### 4.4 搜索与双语的配合

Pagefind 对 `dist/` 全量索引，自动按 `<html lang>` 分语言：实测输出 `Discovered 2 languages: en, zh-cn`。搜索结果里中英页面各自成簇，用户不需要手动切语言；`/zh/` 页面的搜索入口仍在同一索引里（跨语言命中是特性，不是 bug）。

## 5. 与现有 Python 流水线的边界

```
scripts/*.py                →  写 updates/*.json + README 的 AUTO 区块
                             ↓（只读）
site/scripts/prepare-data.mjs →  site/src/data/papers.generated.json（生成物，不入库）
                             ↓
astro build → dist/ → pagefind → artifact → Pages
```

三条纪律：

1. **站点从不写回仓库**。`prepare-data.mjs` 只读 `data/`、`updates/`、`README*.md`，唯一写出的两个文件在 `site/` 内且被 `.gitignore` 忽略。
2. **JSON 优先、README 兜底**。`data/papers/*.json` 存在且覆盖率达到 README 条目数的 75% 时用它，否则解析 README。这条守卫让"数据轨迁移到一半"不会把站点打回 3 条论文（实测日志：`REJECT data/papers: 3 papers < expected >=78 (README has 104); falling back to README scrape`）。
3. **站点不改变 `monthly-paper-search.yml` 的输出格式**。未来若要让站点直接消费该流水线产物，加的是读取分支，不是改 Python 侧。

## 6. 阶段二可选增强（明确排序）

| 优先级 | 项 | 增量成本 | 触发条件 |
| --- | --- | --- | --- |
| P1 | `data/papers/*.json` 覆盖完成后切为主数据源（守卫自动生效） | 0（已有） | 数据轨完成 |
| P1 | 论文 → `models/<slug>.md` 的详情页 | 0.5 人日 | 出现 ≥5 篇有专门文档的论文 |
| P2 | Pagefind 过滤器（按 year/venue facet 收窄搜索结果） | 0.5 人日 | 语料 > 500 条 |
| P2 | `/datasets/` 从 `datasets/*.md` 真渲染（content collection） | 1 人日 | 数据集页增至 10+ |
| P3 | 引用量 / 被引趋势（需外部 API，违背零后端） | 3 人日 + 网络依赖 | **暂不做** |
| P3 | 暗色主题手动切换（当前跟随系统 `prefers-color-scheme`） | 0.5 人日 | 有用户反馈 |