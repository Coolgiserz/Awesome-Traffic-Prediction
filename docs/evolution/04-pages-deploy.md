# 04 · GitHub Pages 部署方案

> 适用前提：仓库 `Coolgiserz/Awesome-Traffic-Prediction` 是**普通仓库**，所以站点 URL 是 Project Pages 形式：
>
> ```
> https://coolgiserz.github.io/Awesome-Traffic-Prediction/
> ```
>
> 不是用户页（`coolgiserz.github.io` 根），也**不是** `<org>.github.io` 的根路径。本文所有配置围绕这一点。
>
> 本文里的 workflow 是**文档正文**，不写进 `.github/`。落地时由维护者新建 `.github/workflows/pages.yml`。

## 1. 三段式 workflow（完整可抄）

GitHub 官方推荐的三段式（`configure-pages` → `upload-pages-artifact` → `deploy-pages`）是当前**唯一**不污染仓库分支、不需要 PAT、且被 OIDC 直接支持的 Pages 部署路径。

```yaml
# .github/workflows/pages.yml
name: Deploy Site to GitHub Pages

on:
  push:
    branches: [main]
    paths:
      - "site/**"
      - "README.md"
      - "README.zh.md"
      - "data/papers/**"
      - "updates/**"
      - "datasets/**"
      - ".github/workflows/pages.yml"
  pull_request:
    paths:
      - "site/**"
      - "README.md"
      - "README.zh.md"
      - "data/papers/**"
      - "updates/**"
  workflow_dispatch:

# 只有一段 GitHub Pages 部署能同时进行；cancel-in-progress=false 保证
# "后 push 的构建" 不会取消 "正在发布中的构建"，避免站点停留在半旧状态。
concurrency:
  group: pages
  cancel-in-progress: false

permissions:
  contents: read   # 只读源码
  pages: write     # deploy-pages 需要
  id-token: write  # OIDC 校验

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup pnpm
        uses: pnpm/action-setup@v4
        with:
          version: 9

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: pnpm
          cache-dependency-path: site/pnpm-lock.yaml

      # pages/action 会读取仓库的 Pages 配置，并把 base_path / origin 写进
      # 环境变量，同时注入 astro 需要的 SITE_URL / SITE_BASE（见下一步）。
      - name: Configure Pages
        id: pages
        uses: actions/configure-pages@v5

      - name: Install dependencies
        working-directory: site
        run: pnpm install --frozen-lockfile

      - name: Build site
        working-directory: site
        env:
          # configure-pages 已给出正确的 origin 与 base_path：
          #   base_path = '/Awesome-Traffic-Prediction'（用户页时为空串）
          # 这里用 shell 展开而不是硬编码，换自定义域名时无需改 workflow。
          SITE_URL: ${{ steps.pages.outputs.origin }}
          SITE_BASE: ${{ steps.pages.outputs.base_path }}/
        run: |
          pnpm exec node scripts/prepare-data.mjs
          pnpm exec astro build
          pnpm exec pagefind --site dist
          # .nojekyll 必须在 upload 之前写入 dist（见 §4）
          touch dist/.nojekyll

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site/dist          # 保持默认 tarball + artifact 名 github-pages
          retention-days: 7

  deploy:
    needs: build
    # PR 只构建不部署：拿到"能不能 build"的信号，但不动生产环境
    if: github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

配套的仓库设置（**一次性，人工**）：

```text
Settings → Pages
  Source: GitHub Actions            ← 必须选这个，不是 "Deploy from a branch"
Settings → Environments → github-pages
  Deployment branches: main（可加 protected branches）
```

> 若 Source 仍停留在 "Deploy from a branch"，`deploy-pages` 会报
> `Creating Pages deployment failed — HttpError: Not Found`，这是最常见的第一次失败原因。

## 2. base path 与 site url：唯一真相源在 `astro.config.mjs`

```js
// site/astro.config.mjs（已落地）
const SITE_URL  = process.env.SITE_URL  ?? 'https://coolgiserz.github.io';
const SITE_BASE = process.env.SITE_BASE ?? '/Awesome-Traffic-Prediction/';

export default defineConfig({
  site: SITE_URL,
  base: SITE_BASE,
  trailingSlash: 'always',
  output: 'static',
  build: { format: 'directory', inlineStylesheets: 'auto' },
});
```

| 部署形态 | `SITE_URL` | `SITE_BASE` | 结果 |
| --- | --- | --- | --- |
| Project Pages（默认） | `https://coolgiserz.github.io` | `/Awesome-Traffic-Prediction/` | `https://coolgiserz.github.io/Awesome-Traffic-Prediction/papers/` |
| 自定义域名 | `https://traffic-prediction.dev` | `/` | `https://traffic-prediction.dev/papers/` |
| 用户/组织页（仓库名 `*.github.io`） | `https://coolgiserz.github.io` | `/` | `https://coolgiserz.github.io/papers/` |
| 本地 dev | 任意（不影响） | `/` | `http://localhost:4321/papers/` |

**规则：内部链接一律不许硬编码。** 站点里只有 `site/src/lib/url.ts` 知道 base：

```ts
export const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');
export const withBase = (p = '/') => `${BASE}${p.startsWith('/') ? p : `/${p}`}`;
```

对照检查（真实产物）：

```bash
# Project Pages 构建
SITE_BASE=/Awesome-Traffic-Prediction/ npx astro build
grep -o 'href="/Awesome-Traffic-Prediction/[a-z/]*"' dist/index.html | sort -u
#   href="/Awesome-Traffic-Prediction/"
#   href="/Awesome-Traffic-Prediction/benchmark/"
#   ...
grep -o 'src="/Awesome-Traffic-Prediction/_astro/[^"]*"' dist/index.html
#   src="/Awesome-Traffic-Prediction/_astro/TrendChart...js"    ← JS/CSS 也走 base ✓
```

三处最容易漏的 base 处理：

1. **`sitemap.xml` 与 `robots.txt` 里的绝对 URL**。`robots.txt` 是静态文件（`site/public/robots.txt`），无法读 `import.meta.env`，所以 Project Pages 场景下**必须手写带前缀的完整 URL**：
   ```text
   Sitemap: https://coolgiserz.github.io/Awesome-Traffic-Prediction/sitemap.xml
   ```
   换自定义域名时这一行要一起改（写进 §7 的迁移 checklist）。
2. **`hreflang="x-default"`**。`new URL('/papers/', site)` 会丢掉 `/Awesome-Traffic-Prediction/` 前缀——必须 `new URL(withBase('/papers/'), site)`。已在 `BaseLayout.astro` 修正。
3. **Pagefind 运行时**。`pagefind --site dist` 生成的 `dist/pagefind/` 是相对路径资产；`search.astro` 里用 `withBase('/pagefind/')` 传入 `data-pf-base`，动态 `import()` 才在子路径下正确解析。

## 3. 缓存与构建时长

实测（本机 macOS / Node 26 / 冷 `node_modules`）：

| 阶段 | 耗时 | 说明 |
| --- | --- | --- |
| `pnpm install`（首次） | 35–50 s | 198 包；有 `pnpm-lock.yaml` + `cache: pnpm` 后，CI 热缓存 5–8 s |
| `prepare-data.mjs` | < 100 ms | 纯读 3 个 JSON / 2 个 md，无网络 |
| `astro build` | **0.36 s** | 16 页 |
| `pagefind --site dist` | **0.03 s** | 2 语言 / 2119 词 |
| `upload-pages-artifact` | 10–20 s | 2.2 MB（748 KB 是 Pagefind 索引+WASM） |
| `deploy-pages` | 15–60 s | GitHub 侧排队，不可控 |

CI 端到端（热缓存）：**约 60–90 s**。冷缓存首次：**120–150 s**。

加速要点（按收益排序）：

1. `cache: pnpm` + `cache-dependency-path: site/pnpm-lock.yaml`——**唯一真正重要的**，省下 30 s。
2. `actions/upload-pages-artifact` 的 `retention-days: 7`；artifacts 超过 90 天会占配额。
3. **不要**给这个站点做增量构建缓存（`.astro/`、`dist/`）：`astro build` 只有 0.36 s，缓存恢复 5 s 反而更慢。`build: { format: 'directory' }` 已经让 GitHub Pages 直接命中 `/papers/index.html`，没有 301 跳转。
4. `prepare-data.mjs` 自带内容指纹：数据未变时跳过写文件（`[prepare-data] unchanged (104 papers) -> skip write`），避免无意义的 mtime 抖动。

## 4. `.nojekyll`：要，而且必须在 upload 之前

GitHub Pages 对 artifact 部署**默认仍然会跑一次 Jekyll 处理**（除非 Pages 端检测到 `.nojekyll`）。后果：**任何以 `_` 开头的目录/文件都会 404**——而 Astro 的产物目录正是 `_astro/`。

```bash
# 正确：写在 dist 里，跟着 artifact 一起上传
touch dist/.nojekyll
```

三种写法的差别：

| 写法 | 结果 |
| --- | --- |
| `touch site/public/.nojekyll`（放在 public 源目录） | ✓ 也会被 Astro 复制进 `dist/`，等价且更省一步；**但**容易忘、且 `public/` 在其他场景（如 `astro dev`）会被当静态资源暴露 |
| `touch dist/.nojekyll`（build 之后） | ✓ 显式、紧邻上传步骤，推荐 |
| 不写 | ✗ `_astro/main.js` 404，页面无样式无交互 |

> 生产验证命令（部署后 30 秒内自查）：
> ```bash
> curl -I https://coolgiserz.github.io/Awesome-Traffic-Prediction/_astro/$(ls site/dist/_astro | grep -m1 '\.css$')
> # 期望 200；若 404 → .nojekyll 没生效
> ```

## 5. artifact 部署 vs `gh-pages` 分支

| 维度 | **artifact 部署（推荐）** | `gh-pages` 分支（`peaceiris/actions-gh-pages` 等） |
| --- | --- | --- |
| 产物是否进仓库历史 | **否**，只存在于 Actions artifact（7–90 天可调） | 是，每次部署一个 commit；一年后 `git clone` 多出几百 MB |
| 需要的 token | OIDC（`id-token: write`），无长效 PAT | `GITHUB_TOKEN` 可，但常用 PAT（含 `repo`）→ 泄露面更大 |
| 分支保护 | 不涉及 | `gh-pages` 必须**不在**保护分支列表，否则推送失败 |
| 回滚 | Pages 面板选上一次成功 deployment，或重跑旧 build；**历史 deployment 保留** | `git revert` 到上一个产物 commit 再 push |
| 构建产物可审计 | 只能下载 artifact（有 `retention-days`） | 仓库里随时 `git checkout gh-pages -- .` |
| `main` 分支纯净度 | ✓ 产物永不进 `main` | ✓ 但 `gh-pages` 长期存活于远端（很多工具会把所有分支一并 clone） |
| 首次配置复杂度 | 需要把 Pages Source 改成 "GitHub Actions" | 需要建空分支 + 设置 Source 指向它 |

**结论：artifact 部署。** 唯一的让步是"产物可审计"——用 `retention-days: 7` + 每次 build 都把 `dist/` 作为 artifact 上传（失败也能下载排查）来补。

## 6. 自定义域名 / CNAME

启用自定义域名（例：`traffic-prediction.dev`）需要**同时**改四处，漏一处就白屏：

```text
1) DNS（在域名商）
   CNAME  traffic-prediction  →  coolgiserz.github.io
   # apex 域（traffic-prediction.dev 本身）需 4 条 A 记录：
   # 185.199.108.153 / 185.199.109.153 / 185.199.110.153 / 185.199.111.153

2) 仓库 Settings → Pages → Custom domain → 填 traffic-prediction.dev
   → 勾选 "Enforce HTTPS"（证书签发需 5–60 min，未签发前不要打开强制 HTTPS）

3) 站点配置：SITE_URL=https://traffic-prediction.dev  SITE_BASE=/
   （用 configure-pages 的 origin/base_path 输出则自动正确）

4) 手写的绝对 URL：site/public/robots.txt 的 Sitemap 行
```

**CNAME 文件的坑（artifact 部署特有）**：

- 走 "Deploy from a branch" 时，Pages 会在 `gh-pages` 分支里维护 `CNAME` 文件。
- 走 **artifact 部署**时，`CNAME` **不会**被自动写进 artifact。若仓库里残留一个 `site/public/CNAME`，它会被原样发布并**覆盖**在 Settings 里填的域名——这通常没问题，但如果 Settings 与 `CNAME` 不一致，Pages 会以 `CNAME` 为准并报 `Domain is not verified`。
- 建议：**只用 Settings 配置域名，不在仓库里提交 `CNAME`**。如果确实想要仓库内声明（便于 fork 后自部署），则必须保证 `site/public/CNAME` 与 Settings 逐字符一致，并在 PR 模板里加一条检查。

**换域名后的 SEO 迁移**（一次性）：

```text
1) 旧域 301：GitHub Pages 对 *.github.io 项目页会自动 301 到自定义域名 —— 无需操作
2) 更新 robots.txt 的 Sitemap 绝对 URL（必须）
3) 重新提交 sitemap 到 Search Console / Bing（重新验证站点所有权）
4) 检查 hreflang：换域名后 hreflang 会指向新域 —— 因为走 withBase + Astro.site，自动正确
```

## 7. PR 预览方案

需求：**PR 里能点开一个临时 URL 看效果**，但不能污染生产。

三种方案，按成本排序：

### 方案 A：只 build 不 deploy（零成本，推荐先做）

上面 workflow 里 `deploy` job 已经带 `if: github.event_name != 'pull_request'`。PR 上跑完整构建 → **构建失败即红灯**，等于把"站点能不能编译"变成门禁。

```yaml
      - name: Upload artifact
        if: github.event_name == 'pull_request'
        uses: actions/upload-artifact@v4
        with:
          name: site-preview-${{ github.event.pull_request.number }}
          path: site/dist/
          retention-days: 7
```

审查者下载 artifact（zip）本地打开即可，**不需要任何第三方服务、不需要额外 secret**。

### 方案 B：Netlify / Cloudflare Pages 旁路预览（可选）

把 `site/` 作为第二个项目的 root，接 `netlify.toml`：

```toml
# site/netlify.toml（仅在启用旁路预览时添加）
[build]
  base    = "site"
  command = "node scripts/prepare-data.mjs && astro build && pagefind --site dist"
  publish = "dist"

[build.environment]
  SITE_URL  = "https://<deploy-preview-url>"   # 由 Netlify 注入 DEPLOY_PRIME_URL 更准
  SITE_BASE = "/"
```

代价：多一个平台的账号与配额；Preview URL 里的 base 是 `/`（不是 `/Awesome-Traffic-Prediction/`），所以**预览与生产的 base 语义不同**——这恰好是一个有用的信号（能提前发现硬编码 base 的 bug）。

### 方案 C：Pages 自带的 preview deployment

GitHub Pages 目前**没有**官方的"PR 预览 deployment"（Environments 只对同仓库分支生效）。不要为此引入 `gh-pages` 分支 + 目录式预览（`/pr-123/`），那会让产物分支无限膨胀。

**结论：A 打底，B 按需。** 不做 C。

## 8. 构建失败与回滚

### 失败面（按发生概率）

| 现象 | 根因 | 处置 |
| --- | --- | --- |
| `pnpm install --frozen-lockfile` 失败 `ERR_PNPM_OUTDATED_LOCKFILE` | 改了 `package.json` 没提交 `pnpm-lock.yaml` | 本地 `pnpm install` 后提交 lockfile；**不要**在 CI 里加 `--no-frozen-lockfile`（会掩盖依赖漂移） |
| `prepare-data.mjs` 退出码 1（`FATAL no papers from any source`） | `README.md` 结构被改动，标题解析正则不再命中 | 该 job 失败不会部署，站点保持上一版本；修正则或临时 `SITE_MIN_STRUCTURED_PAPERS=0` |
| `astro build` 报 `Expected } but found :` | `.astro` frontmatter 里注释位置问题（本仓库真踩过：块注释出现在 `interface` 前） | 用 `//` 行注释；本地 `npx astro build` 复现后再推 |
| 部署成功但页面白屏 | `.nojekyll` 缺失 → `_astro/*` 404 | 见 §4；`curl -I .../_astro/xxx.css` 自查 |
| 部署成功但样式全无、链接 404 | base 前缀丢失（硬编码 `/`） | `grep -rn "href=\"/[a-z]" site/src` 应该零命中 |
| `deploy-pages` 报 `Not Found` | Settings → Pages 的 Source 仍是 "Deploy from a branch" | 改成 "GitHub Actions" |
| `deploy-pages` 报 `environment protection rules` | `github-pages` environment 限制了分支/审批 | Settings → Environments → 放开 `main` |

### 回滚

```text
1) 立即回滚（30 秒，保留构建历史）
   Settings → Environments → github-pages → Deployments
   → 选上一个成功 deployment → "Re-deploy"
   （deploy-pages 的 artifact 仍在 retention-days 内，无需重新构建）

2) 代码回滚（需要重新构建）
   git revert <bad-commit> && git push origin main
   → workflow 自动重跑

3) 止血（生产已经坏了而构建又跑不通）
   Settings → Pages → Source 临时改回 "Deploy from a branch" + 一个已知良好的分支
   # 不建议长期使用；仅用于等待修复期间保住可用性
```

前置护栏（建议同时加进 `CONTRIBUTION.md` 的 PR Checklist）：

```bash
# 站点相关 PR 本地必跑
cd site
node scripts/prepare-data.mjs        # 数据桥必须至少能产出 1 条
npx astro build                      # 构建必须绿
npx pagefind --site dist             # 索引必须非空
grep -rn 'href="/[a-z]' src/ || true # 不应有绕过 withBase 的绝对链接
```

## 9. 落地 checklist（按顺序勾）

```text
[ ] 1. site/pnpm-lock.yaml 已被提交（没有它 cache 与 frozen-lockfile 都失效）
[ ] 2. 新建 .github/workflows/pages.yml（抄 §1）
[ ] 3. Settings → Pages → Source = GitHub Actions
[ ] 4. Settings → Environments → github-pages 允许 main
[ ] 5. 首次部署后 curl 自查：首页 200、/_astro/*.css 200、/pagefind/pagefind.js 200
[ ] 6. 若上自定义域名：DNS + Settings + SITE_URL/SITE_BASE + robots.txt（四处一起改）
[ ] 7. AGENTS.md / CONTRIBUTION.md 增补 "site/ 是只读消费者，产物不入库" 的说明
```