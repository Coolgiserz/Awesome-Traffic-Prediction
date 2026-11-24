# 03 · 静态站点生成器选型（SSG Choice）

> 结论先行：**选 Astro 7 + Pagefind，不引入任何 UI 框架岛**。理由不是"生态流行"，而是这个仓库的数据形态（markdown 与 JSON 各占一半、双语文档、170 条量级的多维筛选表、GitHub Project Pages 子路径）恰好落在 Astro 的能力区间内，而其余四个候选各有一处硬伤（见 §2 的"否决理由"列）。
>
> 本文所有结论都用本仓库的真实产物验证过：`site/` 骨架在本机实际 `astro build` 通过，16 个页面、2.2 MB 产物、构建 0.36 s。命令与输出见 §6。

## 1. 约束（先定尺，再选型）

选型必须满足的硬性条件，来自仓库现状（`AGENTS.md` / `CONTRIBUTION.md` / `.github/workflows/*.yml`）：

| # | 约束 | 来源 | 对 SSG 的具体要求 |
| --- | --- | --- | --- |
| C1 | 数据是 **markdown + JSON 混合** | `README.md` 是权威论文清单；`updates/curated/recent-papers.json`、`updates/hotspots/*.json`、`updates/wordcloud/*.json` 是结构化产物 | 必须同时能把 md 当页面渲染、把 json 当数据读取，且**两者可在同一个页面里混用** |
| C2 | **中英双语**，两条线结构必须对齐 | `README.md` ⇄ `README.zh.md`，`scripts/check_readme_sync.py` 强制同步 | i18n 要能在路由层区分（`/` vs `/zh/`），而不是靠运行时切换 |
| C3 | 部署在 **GitHub Project Pages**，路径 `/Awesome-Traffic-Prediction/` | `origin = github.com/Coolgiserz/Awesome-Traffic-Prediction` | 所有 asset/内部链接必须走 base，不能靠事后 sed 替换 |
| C4 | 搜索必须**纯静态零后端** | Pages 只提供静态文件 | 需要预生成索引（Pagefind 类），不能 Algolia Crawler 之外的 SaaS 依赖 |
| C5 | 论文条目需要**多维筛选/排序**（年份 × venue × 任务 × 方法标签） | `docs/evolution` 呈现设计的核心诉求 | 要么客户端过滤表，要么静态预生成页面（N 个组合） |
| C6 | 维护者是**Python 脚本 + 文档**背景，仓库"documentation-first" | `AGENTS.md`：*Avoid introducing application code, build systems* | 站点构建必须能与 `scripts/*.py` 共存，不引入第二个数据真相源 |
| C7 | 构建要能在 CI 里跑，且失败要能回滚 | `.github/workflows/readme-sync-check.yml` 等已有校验 | 构建确定性、产物可复现、无长尾依赖 |

## 2. 候选对比

打分口径：★★★ 优 / ★★ 可用需额外工作 / ★ 明显短板。

| 维度 | **Astro 7** | VitePress | MkDocs Material | Docusaurus | Vite + 纯前端 |
| --- | --- | --- | --- | --- | --- |
| **C1 md+JSON 贴合度** | ★★★ `.astro` 组件里 `import data from '../x.json'` 直接是构建期数据；markdown collection 与 JSON 可同页混用；`src/pages/*.astro` 可只输出 JSON 也可只输出 HTML | ★★ md 优先；JSON 需要自己写 loader/`defineLoader`，页面里混入交互表要写 Vue 组件 | ★ md 优先；JSON 只能靠 `mkdocs-macros` / 自写 plugin（**Python** 插件 API，改一次要重启构建） | ★★ 与 VitePress 类似，MDX 可以，但 JSON 注入要写 React 组件 | ★★★ 完全自由，但要自己实现 md 渲染 + 路由 + i18n |
| **C2 双语 i18n** | ★★★ 原生 `i18n.locales` + `prefixDefaultLocale`，路由自动生成；`hreflang` 自己写但只有一行 | ★★ 需要 i18n 插件 + 目录镜像约定，locale 切换要手接 | ★★ `i18n` plugin（insiders 版功能更全），material 支持，但主题字符串覆盖麻烦 | ★★★ **最强**，官方 i18n 插件 + `translation.json` 全字符串可覆盖 | ★ 全部自建 |
| **C3 Project Pages base** | ★★★ `base` 配置项驱动 `import.meta.env.BASE_URL`，官方 `withBase()` 模式 | ★★★ `base` 配置项；内部链接自动加前缀 | ★★★ `site_url` 一行搞定，Material 处理得最干净 | ★★ `baseUrl` + `trailingSlash` 有已知坑（内部链接要手写 `useBaseUrl()`） | ★★★ 自己控制 |
| **C4 纯静态搜索** | ★★★ Pagefind 官方支持的组合，`pagefind --site dist` 一条命令 | ★★★ **内置** local search（就是 Pagefind 内核），零配置 | ★★★ 内置 search plugin（lunr 或 pagefind） | ★★★ 本地搜索基于 `@easyops-cn/docusaurus-search-local`，社区维护 | ★ 自建 index |
| **C5 多维筛选** | ★★★ 静态 HTML 全量渲染 + DOM 过滤（本仓库 170 条 ≈ 108 KB HTML，见 §4），零框架 | ★★ 需要写 Vue 组件（引入 Vue 运行时 ~35 KB gz） | ★ 客户端过滤要写 `overrides/` 模板 + JS，破坏 Material 的可维护性 | ★★ React 组件，但 Docusaurus 的 MDX 加载路径对"从 JSON 建表"不友好 | ★★★ 自由，但没有现成的东西 |
| **C6 Python 协作成本** | ★★★ **零耦合**：Node 只读 `data/papers/*.json` 与 `updates/*.json`；Python 侧不用改一行 | ★★★ 同上 | ★★ 站点与脚本**同为 Python**，看似更省事——但意味着 `mkdocs` 与 `scripts/` 会共享解释器/依赖，`pip` 冲突会互相拖累；且 `mkdocs-macros` 与 `render_curated_recent.py` 是两套模板语义 | ★★ Node，但插件生态要 TypeScript | ★★★ |
| **C7 构建时长/确定性** | ★★★ 本机 **0.36 s**（16 页），CI 冷缓存预计 < 60 s（含 `pnpm install`） | ★★★ 0.5–2 s | ★★ 1–3 s，但 `pip install mkdocs-material` 常有 docutils/markdown 版本解析摩擦 | ★★ 5–30 s（React SSR + bundling，全量生成快照 JSON） | ★★★ |
| **初始 JS 体积** | ★★★ 0 JS（无 island），图表页才 ~1.1 MB 未压缩（按需 chunk，gzip ≈ 330 KB） | ★★ ~50 KB（Vue 运行时 + 主题） | ★★★ ~30 KB（search + palette 切换） | ★ 150 KB+（React + 路由 + 主题） | ★★★ 自己控制 |
| **否决理由（若有）** | — | 论文表要么退化成 Vue 组件、要么放弃框架换来一堆手写，**双向不讨好** | 客户端筛选与 i18n 都要写 Python 模板插件，与 `scripts/` 抢依赖；**违背 C6 的"单一构建入口"** | 为一个文档站引入 React SSR，产物大、构建慢，**C7 不达标** | 全部自建 = 把维护成本转嫁给未来的贡献者，**C6 不达标** |

补充事实（避免凭空断言，可复核）：

- Pagefind 由 CloudCannon 维护，`pagefind` npm 包当前 **1.6.0**，提供 ESM 客户端 + WASM 索引运行时；VitePress 的本地搜索内核即是它。
- Astro 当前 **7.4.13**（`npm view astro version` 实测）。
- 本仓库的 `site/package.json` 只有 **3 个依赖**：`astro`、`echarts`、`pagefind`。没有 `@astrojs/sitemap`、没有 `@astrojs/mdx`、没有 UI 适配器。

## 3. 为什么 Astro 而不是 VitePress（唯一有竞争力的备选）

VitePress 的强项是"一堆按顺序读的 markdown 文档"，而本仓库的重心是**一张可筛选的表 + 几张图表 + 一段说明**。差别体现在三处具体的工程量上：

1. **论文库页面**。Astro 里它是"构建期读 JSON → 生成 170 个 `<article>` → 一段 40 行 DOM 过滤脚本"（见 `site/src/components/PaperList.astro`）。VitePress 里同样的东西要么写成 `.vue` 单文件组件（引入 Vue），要么放弃框架、用主题 `layout: false` 手搓页面——后者相当于承认选错了工具。
2. **图表页**。ECharts 需要一个挂载点。Astro 用 `data-chart-spec` 属性 + 动态 `import('echarts/core')` 即可，**这点代码在 VitePress 里同样能写**，但需要绕开 SSR（`<ClientOnly>`）。
3. **i18n 路由**。VitePress 的 i18n 是"两套 config 互指"，新增页面要改两处；Astro 是目录即路由（`src/pages/zh/papers.astro`），加页面不需要动配置。

> 这个仓库并不需要"文档站的侧边栏/上一页下一页/编辑链接"——GitHub 上的 README 已经承担了那个角色。站点要做的是**结构化浏览**，Astro 的组件模型比 VitePress 的 markdown-first 模型更贴合。

## 4. 学术条目多维筛选：客户端过滤表 vs 静态预生成页面

结论：**客户端过滤表**（`PaperList.astro`），不做静态预生成组合页。

| 方案 | 页面数 | 产物大小 | 体验 | 判定 |
| --- | --- | --- | --- | --- |
| 静态预生成（每个 `year × venue × task × tag` 组合一页） | 组合数是指数级：本仓库 12 年 × 45 venue × 6 任务 × 10 标签，即使只做单维也 **73 页**，做二维就 >300 页 | 每页都带侧栏/统计，重复内容数十 MB | 首屏快，但**任意组合筛选永远缺页**，"年份 2020 + STGNN + Transformer"这种真实查询得临时加页 | ✗ |
| 客户端过滤（全量渲染 + `hidden` 切换） | 1 页 × 2 语言 | `/papers/index.html` **108 KB**（未压缩，104 条实测），gzip ≈ 22 KB | 首次之后零延迟；筛选状态写 URL | ✓ |
| 客户端过滤 + 虚拟滚动 | 1 页 | 需要 JS 渲染列表 → **牺牲无 JS 可访问性** | 4k 条以上才值得 | 留作后备 |

拐点测算：一条论文卡片约 620 B HTML。**2,500 条**以内全量渲染仍在 ~1.5 MB（gzip ≈ 300 KB）、可接受范围；超过再引入"分年静态页 + 页内过滤"的两级结构。当前 104 条（README 原始行 170 条含 `……` 占位），余量 24 倍。

## 5. 被选方案的主要风险（必须正视）

| # | 风险 | 触发条件 | 缓解（已落地 / 待办） |
| --- | --- | --- | --- |
| R1 | **`data/papers/*.json` 尚在迁移中，只是部分年份的数据** | 数据轨按年份文件逐步产出（当前只有 `2018.json`，3 条） | 已落地**覆盖率守卫**：`prepare-data.mjs` 比较结构化条数与 README 解析条数，低于 75% 直接拒绝并回退 README 解析（`SITE_MIN_STRUCTURED_PAPERS` 可覆盖）。实测输出：`REJECT data/papers: 3 papers < expected >=78 (README has 104); falling back to README scrape` |
| R2 | 站点与 README **两套真相源**漂移 | 有人只改 JSON 不改 README（或反之） | 站点永远**只读**、不写回；`source` 字段把来源渲染到页脚；迁移完成后把 `render_curated_recent.py --check` 扩展为同时校验 JSON 与站点数据（待办，属数据轨） |
| R3 | ECharts chunk 偏大 | 图表页首次加载 ~330 KB（gzip） | 5 类图表按需 `use()`；`IntersectionObserver` 延迟到进入视口才加载；每张图都有 `<details>` 静态表兜底（无 JS 也能读数据） |
| R4 | Astro 主版本升级（6→7 已发生过 breaking） | 每年 1–2 次 major | `package.json` 锁 `^7`；`site/README.md` 记录**升级三件套**：`npx astro check` → `pnpm build` → 手工核对 `/papers/` 的 base 前缀 |
| R5 | Node 侧依赖进入"documentation-first"仓库 | 有人在 CI 里给站点加新插件 | `AGENTS.md` 的"out of scope: build systems"需要在采纳站点时**同步更新**（这是本方案唯一的制度性改动，须由库维护者拍板） |
| R6 | 前端 JS 是可选项，但**筛选/图表依赖 JS** | 关闭 JS 的用户 | 列表与图表全部有静态 HTML 兜底：论文列表默认全量可见（`hidden` 由 JS 添加），图表有 `<details>` 表格 |

## 6. 落地产物与实测数据

```text
site/
├── package.json           # 3 deps: astro / echarts(page) / pagefind(cli)
├── astro.config.mjs       # site + base + i18n + build.format=directory
├── scripts/prepare-data.mjs
├── src/
│   ├── lib/{papers,url,i18n}.ts
│   ├── layouts/BaseLayout.astro
│   ├── components/{PaperList,Paper...}.astro
│   ├── pages/*.astro + pages/zh/*.astro
│   └── styles/global.css
└── public/{favicon.svg,robots.txt,_data.json(generated)}
```

本机实测（Node v26.6.0 / macOS / 冷 `node_modules`）：

```bash
$ cd site && pnpm run build
[build] 16 page(s) built in 357ms
14:58:54 [build] Complete!

$ npx pagefind --site dist
Indexed 2 languages / 16 pages / 2119 words / Finished in 0.025 seconds

$ du -sh dist
2.2M    dist            # 含 748 KB Pagefind 索引与运行时
$ ls -la dist/papers/index.html
108K    dist/papers/index.html
```

两种 base 都验证过：

```bash
# Project Pages（默认）
SITE_BASE=/Awesome-Traffic-Prediction/ npx astro build
grep -o 'href="/Awesome-Traffic-Prediction/papers/"' dist/index.html   # ✓ 命中

# 自定义域名 / 用户页
SITE_BASE=/ SITE_URL=https://atp.example.dev npx astro build
grep -o 'href="/papers/"' dist/index.html                              # ✓ 命中
```

**没有引入的能力（明确不做）**：MDX、React/Vue/Svelte 适配器、Algolia、评论系统、客户端路由。每引入一个都会把 C7（构建确定性）和 C6（维护成本）往下拉一档，而它们对"读论文清单"这件事没有增量价值。