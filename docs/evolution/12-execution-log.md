# 执行记录（Phase 1 + Phase 2）

> 本文回填 01/02/11 的设计在真实仓库上的执行结果，Phase 1（数据基座）与 Phase 2（站点）分两节，
> 记录 2026-11-24 实际跑出来的产物、数字与遗留项。设计依据见同目录 01/02/10/11。

## 1. 交付物清单

| 产物 | 说明 |
| --- | --- |
| `data/papers/2015.json` … `data/papers/2026.json` | 12 个年份切片，**104 条**结构化论文记录 |
| `data/schema/paper.schema.json` | 44 字段，`additionalProperties:false`，已按实际数据补丁（见 §5） |
| `scripts/migrate_readme_to_json.py` | README → JSON 单向桥，幂等 + 可验证漂移 |
| `scripts/enrich_openalex.py` | OpenAlex 主源 + Crossref 兜底补元数据 |
| `scripts/README.md` | 两个脚本的用法与坑位说明 |

`README.md` / `README.zh.md` / `.github/` / `updates/` **零改动**：灰度期 JSON 是增量产物，
README 仍是人类编辑面。

## 2. 实测数字

```
records=104   years=2015..2026   unique ids=104   schema errors=0
doi=50 (48%)  authors=49  arxiv_id=29  code_url=26
metadata_source: openalex=35  readme_migration=49  crossref=14  manual=6
```

逐年分布：

| 年份 | 条数 | 年份 | 条数 |
| --- | --- | --- | --- |
| 2015 | 2 | 2021 | 10 |
| 2016 | 1 | 2022 | 4 |
| 2017 | 3 | 2023 | 2 |
| 2018 | 10 | 2024 | 4 |
| 2019 | 22 | 2025 | 4 |
| 2020 | 31 | 2026 | 11 |

对照 README 真实条目数（深度学习小节按年）：1/1/3/10/22/31/10/4/2/4/4/11 = 103，
外加 reviews/statistic/time_series/temporal_network_embedding 小节的 1 条 —— 覆盖率 **100%**。

## 3. 站点侧效果

`site/scripts/prepare-data.mjs` 有一条覆盖率护栏：结构化条数 < README×0.75 时拒绝 JSON、
回退抓 README（`degraded: true`）。Phase 1 之前只有 3 条 → 一直被拒绝。

落地后：

```
[prepare-data] source=data/papers (12 file(s), 104 papers)
[prepare-data] years 2015..2026 | with code 26 | with data 2
[build] 16 page(s) built in 318ms
[pagefind] Indexed 2 languages / 16 pages / 2230 words
dist = 2.2 MB
```

`SITE_BASE=/Awesome-Traffic-Prediction/` 下 `/`、`/papers/`、`/zh/`、`/trends/`、`/lineage/`、
`/search/`、`/pagefind/pagefind.js`、`/_data.json`、`/sitemap.xml` 全部 200。

## 4. 修掉的真实缺陷

| 缺陷 | 处理 |
| --- | --- |
| 2 条记录缺 `paper_url`（违反 schema `required`） | 跨行折断的 2022 条目补 arXiv 链接；`[KDD 2021] [Title](url)` 旧格式改为解包标题链接 |
| 4 处 `code` 链接指向 paperswithcode 落地页（现已跳转/失效） | STGCN→`VeritasYin/STSGCN`，SCINet→`cure-lab/SCINet`，Wavenet 无官方实现→移除链接并保留说明，`sota` 影子链接删除 |
| 2021 条目标题里混进 raw markdown | 新增 `TITLE_LINK_RE`，标题字段只存纯文本 |
| 迁移会覆盖 enrichment 成果（DOI/作者/链接） | 合并逻辑改为「只增不删」，README 原始链接落入 `alt_urls.paper-original` |
| schema 与真实数据不匹配（`alt_years`、`verification: null`、`readme_line: null`） | 补丁 schema 而非删数据，语义写进字段 `description` |

修复后 `python3 scripts/migrate_readme_to_json.py --all --check` 全绿（12 个文件全 `SAME`），
即迁移器对当前数据**完全幂等**。

## 5. Schema 补丁明细

- 新增 `alt_years: integer[]` —— arXiv 首发年 ≠ 正式发表年时记录另一年份（迁移真实需要）。
- `verification.checked_at` / `link_ok` 允许 `null` —— 「未核验」与「核验失败」是两种状态。
- 新增 `verification.note` —— 记录链接被替换/丢弃的原因。
- `provenance.readme_line` 允许 `null` —— curated JSON 与流水线来源没有 README 行号。

## 6. 遗留项（下一阶段）

1. **`link_ok` 未真实探测**：enrichment 把可解析到 DOI 视为 `link_ok: true`，并非 HTTP 探测。
   真正的外链巡检应在 Phase 3 的定时任务里（见 `07-quality-gates.md` L2-1），且必须比对
   `final_url`，否则抓不到 paperswithcode 这类「200 但跳走」的失效。
2. **2023–2026 会议召回仍不足**：2023 仅 2 条、2024 仅 4 条，且顶会占比接近 0。
   这是内容问题不是数据问题，按 `02-paper-scope.md` 的方向配额补，检索必须走 arXiv
   `sortBy=submittedDate` + OpenAlex `filter=title.search:…`（OpenAlex 的 `search=` +
   `sort=cited_by_count` 实测噪声极大）。
3. **`data/venues.json` 未建**：`venue_id` 目前是脚本内映射表推断出来的，35 个 venue 短名
   仍存在变体（`Sensor`/`Sensors`、`WWW`/`WWWC`/`WC`）。建 venue 表后才能启用 E8 断言。
4. **`method_tags` / `direction_tags` 是规则推断**，不是人工确认；站点筛选可用，但进入
   curated digest 前需要人工复核一轮。
5. **双语字段大面积 `null`**：`title_zh` / `note_zh` 仅 curated 19 条有值，其余回落到
   英文标题。这是有意的（宁可空也不机翻），但中文站点的观感会偏英文。

## 7. 验收命令

```bash
# 迁移幂等（必须全 SAME）
python3 scripts/migrate_readme_to_json.py --all --check

# schema 校验（需要 pip install jsonschema）
python3 -c "
import json,glob,jsonschema
v=jsonschema.Draft202012Validator(json.load(open('data/schema/paper.schema.json')))
recs=[r for f in sorted(glob.glob('data/papers/*.json')) for r in json.load(open(f))]
print(len(recs),'records,',sum(1 for r in recs for _ in v.iter_errors(r)),'errors')"

# 元数据覆盖率
python3 scripts/enrich_openalex.py --report

# 现有 CI 三项（不受 Phase 1 影响）
python3 scripts/check_readme_sync.py --en README.md --zh README.zh.md
python3 scripts/render_curated_recent.py --check
python3 scripts/validate_domain_config.py scripts/configs/traffic-monthly.json

# 站点
cd site && ASTRO_TELEMETRY_DISABLED=1 pnpm build
```
---

# Phase 2 执行记录：论文详情页 + 搜索分面 + Pages 部署

## 1. 新增页面

| 路由 | 数量 | 说明 |
| --- | --- | --- |
| `/papers/<id>/` | 104 | 英文论文详情页，静态生成 |
| `/zh/papers/<id>/` | 104 | 中文详情页，同一数据、中文标签 |
| `/search/`、`/zh/search/` | 2 | 重构为共享组件，带分面侧栏 |

构建结果：**16 → 224 页，419 ms**，产物 5.6 MB（其中 Pagefind 1.6 MB、`_astro` 1.1 MB）。

详情页内容（每页首屏即完整信息，零客户端 JS）：

- 标题（zh 页在有 `title_zh` 时用中文，否则回落英文——宁可空也不机翻）
- 作者列表（来自 OpenAlex/Crossref enrichment，49 条有值）
- 事实表：venue + venue_type、year（含 `alt_years`）、DOI、arXiv、task、lineage layer、记录来源
- 方法标签：可点击，跳回论文库并带上 `?m=<tag>` 预置筛选
- 研究方向 / 数据集标签
- 策展备注（`note_zh`/`note_en`，无则显式说明"暂无备注"）
- 核验状态：`Metadata verified: 2026-11-24 (openalex)` 或 `not verified yet`
- 链接区：paper / code / data / review / project，以及 `paper-original`（README 原始链接）
- 同方向论文（`related()`：方向标签重叠数 → 年份接近度）

## 2. 搜索分面

`PaperList.astro` 用 `data-pagefind-filter="year:2026, venue:Electronics"` 给每张卡片打上分面值，
`SearchPage.astro` 用 Pagefind JS API (`pagefind.search(term, { filters })`) 渲染侧栏。

索引实测：

```
Indexed 2 languages / 224 pages / 4970 words / 2 filters
filter index: year = 2015..2026 (12 个桶), venue = 45 个桶
fragment url 示例: /papers/learning-geo-contextual-embeddings-for-commuting-2020/
```

**一处真实限制**：Pagefind 的浏览器 WASM 运行时需要 `fetch` 元数据，无法在 Node 里跑集成测试。
可自动验证的是索引产物本身（分面桶、fragment URL、页面数），运行时行为只能在浏览器里人工确认。
这一点已写进 `09-acceptance.md` 的人工检查项。

## 3. sitemap 扩展

`sitemap.xml` 从 15 条扩到 **223 条**：14 个版块页 × 2 语言 + search + 208 个详情页，
每个 URL 带 `xhtml:link` 三向 alternate（en / zh-CN / x-default），共 666 条 alternate。

## 4. 部署流水线

新建两个 workflow（此前仓库只有 3 个 check，没有任何部署）：

| 文件 | 触发 | 作用 |
| --- | --- | --- |
| `.github/workflows/pages.yml` | push main / PR / 手动 | 三段式 Pages 部署；PR 只出 preview artifact，不部署 |
| `.github/workflows/data-check.yml` | data / README / 迁移脚本变更 | schema 校验 + 迁移幂等 + 覆盖率门禁 |

`pages.yml` 实际落地的要点（相对 `04-pages-deploy.md` 文档版的差异）：

- 补了 `ASTRO_TELEMETRY_DISABLED: "1"` —— Astro 会把遥测状态写 `$HOME`，CI 里是多余的 IO。
- `build` 后追加 `touch dist/.nojekyll` —— 缺它 `_astro/*` 全部 404。
- 新增 `Report build size` 步骤，把页数/体积写进 Step Summary，便于观察数据增长。
- PR 分支走 `upload-artifact`（preview zip），与生产 artifact 完全分开。

## 5. 顺手修掉的内容缺陷

README 里 6 处 paperswithcode 引用（双语各 6 处，共 12 处）全部处理：

| 位置 | 原链接 | 处理 |
| --- | --- | --- |
| L91 STGCN `[[code]]` | pwc paper 页 | → `github.com/VeritasYin/STSGCN` |
| L131 STGSL `[[SOTA]]` | pwc paper 页（302 → HF trending） | 移除链接 |
| L157 Wavenet `[[code]]` | pwc paper 页 | 无官方实现，移除 |
| L158 SCINet `[[code]]` | pwc paper 页 | → `github.com/cure-lab/SCINet` |
| L413 `paper with code` | pwc task 页 | → `huggingface.co/papers/trending` 并改名 |
| L427 `Multivariate Time Series` | pwc task 页 | → `huggingface.co/papers/trending` |

diff 严格 12 行（每个 README 6 行），无任何空白字符污染。
`grep -c paperswithcode README.md README.zh.md` → `0 0`。

## 6. 验收命令

```bash
# 1. 数据层
python3 scripts/migrate_readme_to_json.py --all --check     # 全 SAME
python3 scripts/check_readme_sync.py --en README.md --zh README.zh.md
python3 scripts/render_curated_recent.py --check

# 2. 站点
cd site
ASTRO_TELEMETRY_DISABLED=1 node scripts/prepare-data.mjs    # source=data/papers, 104 papers
SITE_BASE=/Awesome-Traffic-Prediction/ ASTRO_TELEMETRY_DISABLED=1 npx astro build
npx pagefind --site dist                                    # 2 languages / 224 pages / 2 filters
touch dist/.nojekyll

# 3. 路由冒烟（base path 目录结构下）
mkdir -p /tmp/serve && cp -R dist /tmp/serve/Awesome-Traffic-Prediction
python3 -m http.server 4321 --directory /tmp/serve
# 期望 200：/ 、/papers/ 、/papers/dgib-2026/ 、/zh/papers/dgib-2026/ 、
#           /search/ 、/zh/search/ 、/pagefind/pagefind.js 、/sitemap.xml
```

## 7. 仍需人工确认

1. **浏览器里的分面交互**：点选 year/venue chip 后结果数变化、清空按钮生效。
   本机无法在无头环境跑 Pagefind WASM，已记录为人工检查项。
2. **Pages Settings 一次性操作**：`Settings → Pages → Source` 必须切到 **GitHub Actions**，
   否则 `deploy-pages` 报 `Not Found`。这是首次部署最常见的失败点。
3. **首次部署后自查**：`curl -I .../\_astro/<某个css>` 必须 200（验证 `.nojekyll` 生效）。

## 8. CI 首次运行的修正：Node 版本

首次 `pages.yml` 运行失败，报：

```
Node.js v20.20.2 is not supported by Astro!
Please upgrade Node.js to a supported version: ">=22.23.4"
```

根因：`04-pages-deploy.md` 的示例 workflow 写的是 `node-version: 20`，而实际安装的
`astro@7.4.13` 的 `engines.node` 是 `>=22.23.4`（lockfile 里也记录了同一条 engines）。
文档示例是按更早的 Astro 版本写的，照抄即失败。

三处修正：

| 位置 | 改前 | 改后 | 理由 |
| --- | --- | --- | --- |
| `pages.yml` Node | `20` | `lts/*` | Astro 的 engines 是精确下限，`lts/*` 解析到当前 LTS（>=22）且不必随 22.x 补丁版本改动 |
| `pages.yml` pnpm | `9` | `10` | lockfile 实际由 pnpm 12 解析（9.0 格式），10 能读；固定大版本避免 resolution 漂移打破 `--frozen-lockfile` |
| `site/package.json` engines | `>=20.0.0` | `>=22.23.4` | 声明必须与真实依赖一致，否则本地装完能在旧 Node 上"成功"构建到一半 |
| `docs/evolution/04-pages-deploy.md` | 示例 `node 20` / `pnpm 9` | 同步修正 | 文档里的示例会被照抄，是这次失败的源头 |

额外加了一道护栏：`site/scripts/prepare-data.mjs` 在最前面检查 Node 版本，
不满足就立刻 `exit 1`。

修改前的失败模式很坏——`prepare-data` 成功、写了文件，然后 Astro 才崩，
CI 日志里看起来像"数据桥没问题，是别处挂了"。现在两条命令都会在第一步给出
明确原因。可用 `PREPARE_DATA_SKIP_NODE_CHECK=1` 绕过（仅用于调试）。

## 9. 站点的产品化收敛：清理开发向文案 + 中文化字段

站点跑起来之后暴露了一类问题：页面在说**它是怎么做出来的**，而不是在说**读者要看什么**。
这些文字来自设计文档里的原话，被直接抄进了 i18n 字典。

清掉的文本：

| 位置 | 原文 | 问题 |
| --- | --- | --- |
| 论文库 lead | 「筛选状态写入 URL，可分享任意视图」 | 在解释实现，不是读者需要的信息 |
| 趋势页 lead | 「用交互图表替代 README 中的静态 PNG」 | 内部迁移说明 |
| 图表折叠标签 | `Table view (no-JS fallback)` | `fallback` 是工程术语 |
| 图表 hint | `monthly-paper-wordcloud-latest.json · generated 2026-10-01` | 暴露内部文件名与构建时间 |
| 首页 degraded 提示 | `source=README.md · readme entries=104` | 构建细节 |
| 页脚 | `Data source: data/papers · Built 2026-11-24` | 构建细节 |
| 论文详情 | `Record source: data/papers/2026.json` | 内部路径 |
| 论文详情备注 | `Original code release; the paperswithcode landing page now redirects to arXiv.` | 我为了解释链接替换写的备注 |

新增的规则写进了代码注释：**备注是策展文字，不是元数据**，中文页只显示 `note_zh`，
没有就显示中性的空状态；跨语言复用英文备注会让读者看到无法核实的英文内容。

顺带补上的本地化（原来只有 19/104 条有 `task_zh`）：

- `localizeTask()` —— 最长后缀匹配 + 限定词翻译。只需知道核心名词
  （traffic forecasting、traffic flow prediction…），`Large-scale` / `Short-term` /
  `Long-horizon` / `Real-time` / `Multi-step` / `Urban` 作为前缀限定词单独翻译。
  实测 **21/21 个 task 值全部命中**，新增论文通常不需要加词条。
- `localizeLayer()` —— `sota_core` / `classic_baseline` / `survey` / `application`
  在中文页显示为「近年主流 / 经典基线 / 综述 / 应用」。

验证方式：对全部 **224 个页面**做正文提取后扫描 27 个开发向关键词，结果 `0` 命中。
命令留在 `09-acceptance.md` 的人工检查项里。

## 10. Phase 3 起步：2023+ 召回修复

### 10.1 修掉流水线里的一个真 bug

`scripts/paper_search.py` 的 `search_arxiv()` 把查询拼成 `all:<query>`：

```
search_query = "all:" + query        # all:traffic forecasting
sortBy = submittedDate
```

arXiv 的 `all:` 遇到不完整短语时是**单词 OR**，`all:traffic forecasting` 等于
「提到 traffic 或 forecasting 的所有论文」，再按提交日期倒序，返回的是全 arXiv 最新文章。
实测结果：一次运行召回 60 条，去重后与交通预测相关的只有 **1 条**（其余是天体物理、
单细胞、PDE 求解器……）。

修复：多词查询转成**引号短语 + abs:** 字段。

```python
def _arxiv_query(query):
    if re.match(r"^(abs|ti|all|au|cat|co):", q, re.I) or q.startswith('"'):
        return q                      # 调用方可覆盖默认行为
    return f'abs:"{cleaned}"'
```

修复后同一个配置：**40 条全部相关**（Global-Local GAT、ButterMamba、PatchSTG、GAS-PFN…）。

### 10.2 迁移脚本会删数据（已修）

`migrate_readme_to_json.py` 按 README 重写整年文件。一旦 `data/papers/2023.json`
里有不是从 README 来的记录（搜索流水线加的），重跑迁移就会**整批删掉**。
这是数据基座最危险的一类缺陷：不报错、静默、不可逆。

修复：写入前读盘上的现有文件，把「README 不拥有」的记录（按 `(year, 归一化标题)`
判断）追加到记录列表尾部，`order` 顺延。汇总里新增 `carried_over=` 计数。

```
[SUMMARY] years=10 records=89 carried_over=24
```

另外修了策展标签的优先级：curated JSON 里的 `task_en` / `method_tags` 是人工审核过的，
但重跑时会被关键词规则覆盖（UniST 的 `Urban spatio-temporal prediction` 变回
`Traffic forecasting`）。现在 `provenance.curated_labels=true` 的记录，其标签不被规则覆盖。

### 10.3 数据规模

| 年份 | 迁移前 | 迁移后 |
| --- | --- | --- |
| 2023 | 2 | **18** |
| 2024 | 4 | **12** |
| 2025 | 4 | **12** |
| 2026 | 11 | **31** |
| 合计 | 104 | **156** |

按 `provenance.source` 分布：`readme_migration 85 / curated_json 19 / search_pipeline 52`。
元数据覆盖：DOI 103/156，作者 102/156，arXiv id 90/156。

### 10.4 新增配置

`scripts/configs/traffic-arxiv-backfill.json` —— arXiv 补齐配置，与只查 OpenAlex 的
`traffic-monthly.json` 互补。后者对 2022 年之后的会议论文几乎零召回，这是本次数据缺口的
直接原因。

### 10.5 仍在处理中

- 2022 年只有 4 条，同样需要补齐（arXiv 分页需要往回翻数百条，`paper_search.py`
  目前最多翻 4 页 = 400 条，需要加 `start` 偏移参数）。
- `venue_type` 对 arXiv 记录统一是 `preprint`，等 OpenAlex/Crossref 给出正式发表信息后
  应更新为会议/期刊，否则「2023 年后零顶会」的现象在数据层面仍然存在。

## 11. 补齐 2022 + 修复 venue 升级链路

### 11.1 arXiv 分页参数

`search_arxiv()` 原先硬编码 `max_pages = 4`（400 条）。arXiv 按提交日期倒序，
要回捞 2022 年必须翻过 2023–2026 的全部结果，400 条远远不够。新增两个参数：

```
--arxiv-max-pages N        # 走多少页（默认 4）
--arxiv-start-offset M     # 跳过前 M 条，用于续跑
```

`--arxiv-max-pages 12` 跑一次，2022 从 4 条补到 **43 条**，40/40 命中且年份全部正确。

### 11.2 第二个静默 bug：venue 永远升不上去

`enrich_openalex.py` 的 `TYPE_MAP` 缺 `conference-paper` —— 而这正是 OpenAlex 对
绝大多数已发表 CS 论文使用的类型。实测：

```
arXiv:2308.10425 (STAEformer)
  OpenAlex type=conference-paper
  venue=ACM International Conference on Information and Knowledge Management (CIKM)
  doi=10.1145/3583780.3615160
```

数据里明明有 CIKM、ICDM、KDD，但类型没被映射，`venue_type` 保持 `preprint`，
`venue` 保持 `Arxiv`。于是「2023 年后零顶会」这个现象在数据层是假的 —— 筛选条件把它们过滤掉了。

两处修复：

1. `TYPE_MAP` 补齐 OpenAlex 的完整类型枚举（conference-paper / journal-article /
   article / letter / review / report …）。
2. `venue_type == "preprint"` 不再被视为终态。它是 arXiv 导入时的**临时状态**，
   一旦 provider 报出 journal/conference 就应升级。同理 `venue` 从 `Arxiv` 升级为真实 venue。
   `Proceedings` / `Conference` / `Book` 这类无信息值不采纳。

### 11.3 第三个静默 bug：迁移会把 venue 改回去

README 里的 `[Arxiv 2022]` 标签是**当时的标注**，不是当前事实。迁移重跑时会用
README 的 venue 覆盖 enrich 升级后的值，把 CIKM 打回 Arxiv。
现在：`verification.metadata_source` 属于 openalex/crossref 且 `venue_type` 是
journal/conference/workshop 的记录，迁移不再覆盖其 venue。

### 11.4 结果

| 指标 | 之前 | 现在 |
| --- | --- | --- |
| 总记录 | 156 | **195** |
| 2022 | 4 | **43** |
| 2023+ 的 conference+journal | 6 | **37** |
| DOI | 103 | **135** |
| 会议/期刊/预印本 | — | 62 / 58 / 66 |

更新 `02-paper-scope.md` 时用的配额（2022–2023 → 12 条）已经被超过，可据此重新校准。

### 11.5 站点

406 页（195 论文 × 2 locale + 版块页），Pagefind 2 filters，产物 8.7 MB。
新增中文术语表：`venue_type`（期刊 / 会议 / 预印本）与 10 个 `direction_tags`。
