# site/ — Awesome Traffic Prediction 静态站点骨架

Astro 7 + Pagefind 1.6 + ECharts 6 生成的纯静态站点。设计依据见：

- `../docs/evolution/03-ssg-choice.md`（选型）
- `../docs/evolution/04-pages-deploy.md`（部署）
- `../docs/evolution/05-presentation.md`（呈现 / i18n / 可视化）

> **只读消费者**：`site/` 从仓库根读取 `README.md`、`README.zh.md`、`data/papers/*.json`、`updates/**/*.json`，**从不写回**这些路径。生成物（`src/data/papers.generated.json`、`public/_data.json`）已在 `.gitignore` 中忽略。

## 1. 命令

```bash
cd site

# 安装（pnpm 10+；仓库提供 pnpm-lock.yaml，CI 用 --frozen-lockfile）
pnpm install

# 开发服务器：http://localhost:4321/Awesome-Traffic-Prediction/
pnpm dev

# 生产构建（数据桥 → astro build → pagefind 索引）
pnpm build

# 只构建 HTML，不生成搜索索引（快 0.02 s，调试用）
pnpm build:fast

# 预览构建产物（base path 生效，最接近线上）
pnpm preview

# 强制重新解析数据源
pnpm sync-data
```

部署等价命令（与 `.github/workflows/pages.yml` 一致）：

```bash
SITE_URL=https://coolgiserz.github.io \
SITE_BASE=/Awesome-Traffic-Prediction/ \
  pnpm build && touch dist/.nojekyll
# 然后把 dist/ 作为 artifact 上传给 actions/deploy-pages
```

本地/自定义域名（base 变成 `/`）：

```bash
SITE_URL=https://atp.example.dev SITE_BASE=/ pnpm build
```

## 2. 数据源与回退

`scripts/prepare-data.mjs` 按优先级择一：

| 顺序 | 来源 | 判定条件 | 产物 `provenance.source` |
| --- | --- | --- | --- |
| 1 | `data/papers/*.json` | 存在且条数 ≥ max(1, README 条目数 × 75%) | `data/papers` |
| 2 | `README.md`（回退解析） | 上一档不满足 | `README.md`（`degraded: true`） |

覆盖率守卫可用环境变量覆盖：

```bash
SITE_MIN_STRUCTURED_PAPERS=0 pnpm build   # 强制信任 data/papers，哪怕只有 1 条
```

原因：`data/papers/` 由数据轨按年份文件逐步产出。2026-11 时仓库里只有 `data/papers/2018.json`（3 条），而 README 有 104 条 —— 直接信它会让站点从 104 条塌成 3 条。守卫的实测输出：

```text
[prepare-data] REJECT data/papers: 3 papers < expected >=78 (README has 104); falling back to README scrape
[prepare-data] wrote site/src/data/papers.generated.json (104 papers)
```

回退解析的语法约定（与 `CONTRIBUTION.md` 一致）：

```text
[Venue Year] Title [[paper]](url) [[code]](url) [[data]](url) [[review]](url)
```

- 只扫描 `## 0x00 Papers` 小节；`#### YYYY` 标题作为年份兜底；
- `……` / `...` 占位条目跳过；
- `[[code]]`、`[[code-official-*]]`、`[[code-pytorch]]` 统一收敛为 `codeUrl`（原始区分保留在 `badges[]`）；
- 任务标签（`task`）与方法标签（`methodTags`）在无结构化 JSON 时**由标题正则推导**（STGNN / Transformer / Diffusion / …），有 JSON 时直接用 `task_en`、`method_tags`。

## 3. 目录结构

```text
site/
├── package.json               # 3 依赖：astro / echarts / pagefind
├── pnpm-lock.yaml             # CI 用 --frozen-lockfile
├── astro.config.mjs           # site + base + i18n + build.format=directory
├── scripts/
│   └── prepare-data.mjs       # 数据桥（build 第一步）
├── public/
│   ├── favicon.svg
│   ├── robots.txt             # Sitemap 绝对 URL 含 Project Pages 前缀
│   └── _data.json             # 生成物：图表数据（hotspots/wordcloud/统计）
└── src/
    ├── data/papers.generated.json   # 生成物：论文索引（不入库）
    ├── lib/
    │   ├── papers.ts          # 唯一数据访问层（去重 / 分组 / 统计）
    │   ├── url.ts             # 唯一 base 处理点（withBase / altLocalePath）
    │   └── i18n.ts            # en/zh 字典 + 路由表
    ├── layouts/BaseLayout.astro     # head/SEO/hreflang/导航/页脚
    ├── components/
    │   ├── IndexPage.astro          # / 与 /zh/ 共用
    │   ├── PapersPage.astro + PaperList.astro   # 多维筛选 + URL 状态
    │   ├── TrendsPage.astro + TrendChart.astro  # ECharts + 无 JS 兜底表
    │   ├── LineagePage.astro        # 研究脉络时间线
    │   ├── DocsPage.astro           # datasets/toolkits/benchmark 通用容器
    │   └── {datasets,toolkits,benchmark}.content.astro
    ├── pages/
    │   ├── index.astro papers.astro datasets.astro toolkits.astro
    │   ├── benchmark.astro trends.astro lineage.astro search.astro
    │   ├── sitemap.xml.ts           # /sitemap.xml（无额外依赖）
    │   └── zh/{index,papers,datasets,toolkits,benchmark,trends,lineage,search}.astro
    └── styles/global.css
```

## 4. i18n

- 英文在根路径，中文在 `/zh/` 下（`prefixDefaultLocale: false`）。
- 每页输出 `canonical` + `hreflang`（`en` / `zh-CN` / `x-default`），`x-default` 指向英文同路由。
- 新增页面 = 加两个文件（`src/pages/x.astro` + `src/pages/zh/x.astro`），不需要改配置。

## 5. 搜索

- `pagefind --site dist` 在 `astro build` 之后运行，索引写进 `dist/pagefind/`（本仓库 16 页约 748 KB）。
- 客户端资产按 `withBase('/pagefind/')` 解析，Project Pages 子路径下无需额外配置。
- 自动按 `<html lang>` 分英文 / 中文两个索引。

## 6. 构建实测

本机（macOS / Node v26.6.0 / pnpm 10.34.9 / 清空 `node_modules` 后重装）：

```text
pnpm install --frozen-lockfile   →  18.1 s（220 包）
pnpm build
  [prepare-data] 104 papers (README fallback)
  astro build                     →  16 page(s) built in ~300 ms
  pagefind --site dist            →  2 languages / 16 pages / 2119 words / 0.03 s
产物大小                          →  2.2 MB（含 748 KB Pagefind）
最大 HTML                        →  dist/papers/index.html ≈ 108 KB（104 条卡片全量渲染；README 的 170 条是含 …… 占位与重复条目的原始行数）
```

**环境限制记录（非代码问题）**：本机沙箱不允许写 `~/Library/pnpm/**`，因此：

- `pnpm install` 报了 `ERR_PNPM_PRIVATE_INSTALL_CREATE` / `EPERM: .../Library/pnpm/.tools`；
- `package.json` 里原本的 `packageManager: "pnpm@12.16.1"` 会让 corepack/npx 去下载该版本到被禁止的目录，导致连 `pnpm --version` 都失败，已移除该字段（CI 用 `pnpm/action-setup` 指定版本即可）；
- 实际验证用的是 `npx --yes pnpm@10 install --frozen-lockfile` + `npx --yes pnpm@10 run build`，**两条命令均成功**，`pnpm-lock.yaml` 也由 pnpm 10 生成（`lockfileVersion: '9.0'`）。
- 若在正常机器上 `pnpm install` 报同样的 EPERM，是 HOME 权限问题（`sudo chown -R $(id -u):$(id -g) ~/Library/pnpm`），与站点无关。

另一个真实踩过的坑，写在这里防止再犯：`astro build` 会因 `.astro` frontmatter 里把 `/** */` 块注释放在 `interface Props` 之前而报
`[CompilerError] Expected } but found :`。frontmatter 顶部请用 `//` 行注释。

## 7. 部署

见 `../docs/evolution/04-pages-deploy.md`。要点：

- GitHub Project Pages，base 为 `/Awesome-Traffic-Prediction/`；
- `actions/configure-pages` 输出的 `origin` / `base_path` 直接喂给 `SITE_URL` / `SITE_BASE`；
- `dist/.nojekyll` 必须在 `upload-pages-artifact` **之前**写入，否则 `_astro/*` 被 Jekyll 忽略 → 404；
- 产物不进 `main`，走 artifact 部署。
## 3. 论文详情页与搜索分面（Phase 2）

### 详情页

每个 `data/papers/*.json` 记录生成两个页面：

```
/papers/<id>/          英文
/zh/papers/<id>/       中文（同一份数据，中文标签）
```

- 路由：`src/pages/papers/[id].astro` + `src/pages/zh/papers/[id].astro`，`getStaticPaths` 从 `src/lib/papers.ts` 取全部 id。
- 视图：`src/components/PaperDetail.astro`。
- 页面是纯静态 HTML，无客户端 JS；作者、DOI、arXiv、方法标签、核验状态、同方向论文都在首屏。
- 方法标签可点击，回到论文库并带上 `?m=<tag>` 预算好的筛选状态。

### 搜索分面

- `PaperList.astro` 给每张卡片写 `data-pagefind-filter="year:2026, venue:Electronics"`，
  再用一组隐藏的 `<span data-facet-year/data-facet-venue>` 提供分面值来源（`data-pagefind-index-attrs`）。
- `SearchPage.astro`（`/search/` 与 `/zh/search/` 共用）在 Pagefind UI 之上调用
  `pagefind.search(term, { filters })`，用返回的 `data.filters` 渲染侧栏计数。
- 当前索引：`Indexed 2 languages / 224 pages / 2 filters`，`year` 12 个桶、`venue` 45 个桶。

> 注意：Pagefind 的浏览器运行时依赖 WASM + fetch，无法在 Node 里做集成测试。
> 能自动验证的是索引产物（分面桶、fragment URL、页面数）；交互行为请在浏览器里确认。

## 4. 部署

`.github/workflows/pages.yml` 是权威版本（三段式 artifact 部署）。要点：

- `SITE_URL` / `SITE_BASE` 取自 `actions/configure-pages` 的 `origin` / `base_path`，换域名无需改 workflow。
- 构建后 `touch dist/.nojekyll` —— 缺它 `_astro/*` 全部 404。
- PR 只上传 `site-preview-<n>` artifact，不部署生产。
- 仓库侧一次性设置：**Settings → Pages → Source = GitHub Actions**。

本地验证部署形态：

```bash
SITE_BASE=/Awesome-Traffic-Prediction/ \
SITE_URL=https://coolgiserz.github.io \
  pnpm build && touch dist/.nojekyll

# base path 冒烟：必须放在带前缀的目录下服务，否则所有绝对链接 404
mkdir -p /tmp/serve && cp -R dist /tmp/serve/Awesome-Traffic-Prediction
python3 -m http.server 4321 --directory /tmp/serve
```
