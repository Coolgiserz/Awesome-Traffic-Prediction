# 08 · 三阶段路线图

> 这是演进任务的**主干文档**。目标是让「文档型 Awesome List」在**不违反 AGENTS.md「documentation-first、不引入 build pipeline / runtime service」**的前提下，获得三个能力：结构化数据基座、可浏览的静态站点、可维护的自动化。
>
> 每一阶段都必须满足同一条铁律：**README 在任一时刻都可以被单独使用**——即使所有新东西全部删掉，仓库仍是一份合格的 Awesome List。
>
> 配套细节：[01-data-model](./01-data-model.md)（模型）、[02-paper-scope](./02-paper-scope.md)（收录范围）、[03-ssg-choice](./03-ssg-choice.md)（SSG 选型）、[06-pipeline](./06-pipeline.md)（流水线）、[07-quality-gates](./07-quality-gates.md)（门禁）、[09-acceptance](./09-acceptance.md)（验收命令）。

---

## 0. 起点：现状快照（实测，2026-11-24）

所有估算的基准都来自这张表，后续人日估算都以它为规模参照。

| 指标 | 实测值 | 取证 |
| --- | --- | --- |
| `README.md` / `README.zh.md` 行数 | 483 / 483 | `wc -l` |
| `0x00` 深度学习小节条目 | 94（2015–2026） | 逐小节计数 |
| 全 README `[[paper]]` 条目 | 102（EN 与 ZH 逐条相同） | 正则统计 |
| 条目内**含 DOI** | 24 / 102（23.5%） | 见 02 文 |
| 断崖年份 | 2022=2、2023=2、2024=4、2025=4 | 见 02 文 |
| 跨行折断条目 | 1（L170–171） | `^[0-9]+\. \[` 匹配 170 行，单行 `CANDIDATE_RE` 只命中 102 行 |
| 年份与所在小节不一致 | 2（L89、L156） | 见 01 文 |
| 外部链接总数 / 不同 host | 218 / 64 | 正则统计 |
| `paperswithcode.com` 链接 | 10（README×2 各 6 + `datasets/` 4） | **实测全部 301/302 跳转到 `huggingface.co/papers/trending`** |
| `updates/curated/recent-papers.json` | 19 条（12 `featured`） | `jq length` |
| 结构化产物体积 | `updates/wordcloud` 3.4 MB、`updates/hotspots` 808 KB | `du -sh` |
| `.git` 体积 | 27 MB（`papers/` 16 MB） | `du -sh` |
| 已有 CI | 3 个内容校验 + 1 个 AI 助手 | `.github/workflows/` |
| 已有结构化数据 | **0 条**（`updates/curated/` 只覆盖 2024–2026 的 19 条） | — |

**核心判断**：2015–2023 的 81 条论文仍然**只存在于 markdown 里**，而 2024–2026 的 21 条有两条来源（markdown + `recent-papers.json`）。这个"双体制"是 Phase 1 必须消灭的对象。

**Phase 0 已就绪的部分**（由并行任务产出，本路线图按「已落地」计入）：

- `data/schema/*.schema.json` 6 份（`paper` / `dataset` / `toolkit` / `venue` / `research-group` / `lineage`），draft 2020-12。
- `data/papers/2018.json` 迁移样例（3 条）。
- `site/` Astro 骨架：`astro 7.4.13` + `pagefind 1.6.0` + `echarts 6.1.0`，3 个依赖，无 UI 框架。**本机实测**：`build:fast` 0.47 s 出 15 页；`build`（含 Pagefind）产物 **2.2 MB**（`dist/_astro` 1.1 MB、`dist/pagefind` 748 KB、`papers/index.html` 107 KB）。当前 `papers.generated.json` 的 `provenance.source = "README.md"`、`en/zh = 104/104`、`degraded: true` —— 说明**站点现在是从 README 刮的**，不是从 `data/`。

> 上述 `site/` 与 `data/` 仍在并行任务中演进，`papers.generated.json`、`dist/` 属于构建产物（不入库）；本机实测数据基于 2026-11-24 15:00 的快照。

---

## 1. 总览

| 阶段 | 主题 | 一句话验收 | 人日 |
| --- | --- | --- | ---: |
| **Phase 1** | 数据基座 | README 的每一行都能从 `data/` 重建，且重建结果与今天逐字节可比 | **28** |
| **Phase 2** | 站点上线 | `https://coolgiserz.github.io/Awesome-Traffic-Prediction/` 可访问、双语言、可搜、可筛 | **20** |
| **Phase 3** | 自动化与社区 | 月度流水线零人工合并、打标签有人审闭环、leaderboard 有真实数字 | **31** |
| | | **合计** | **79 人日** |

按单人投入约 0.5 FTE（每周 2.5 天）估算：Phase 1 ≈ 11 周，Phase 2 ≈ 8 周，Phase 3 ≈ 12 周，**合计约 31 周**。允许部分并行（Phase 2 的 2.1/2.2 可以与 Phase 1 的 1.4 并行），压缩后约 **24–26 周**。

**排序理由**（为什么必须先数据后站点）：站点现在能从 README 刮出 104 条并渲染，看起来"站点不依赖数据基座"。但这个 104 条里含 `…` 省略号占位、含跨行折断、含年份错位，且 `degraded: true`。**先用站点固化一个错的解析结果，等于把错误写进 URL 与搜索索引。** 所以 Phase 1 必须在 Phase 2 之前完成到"至少 2015–2026 全部年份都有 JSON"这一步。

---

## 2. Phase 1 · 数据基座（28 人日）

### 2.1 范围

**In**

1. **完整 schema 落地**：6 份 schema 已存在，补 `data/schema/recent-papers.schema.json`（`updates/curated/recent-papers.json` 目前无契约）与顶层 `data/schema/index.json`（文件 → schema 映射）。
2. **全量迁移**：把 README 的 102 条论文 + 0x02 的数据集/工具/团队拆成 `data/` 下的实体 JSON：
   - `data/papers/2015.json` … `2026.json`（12 个文件，`2018.json` 已有）
   - `data/datasets.json`（0x02 的 18 个外链 + 4 个本地 doc）
   - `data/toolkits.json`（0x03 的 6 条）
   - `data/venues.json`（45 个短名的归一，见 01 文）
   - `data/research-groups.json`（0x05 的 15 条编号条目）
   - `data/lineage.json`（`research-lineage.md` 的 7 段）
3. **迁移脚本**：`scripts/migrate_readme_to_json.py`，一条命令把当前 README 解析成待人工确认的 JSON diff，**不直接写权威数据**（落到 `updates/migration/`）。
4. **渲染器**：新增 `scripts/render_readme.py`，从 `data/` 渲染出**与今天等价**的 README 段落；`render_curated_recent.py` 保留但数据源切到 `data/papers/*.json`。
5. **L1 门禁落地**：[07 文](./07-quality-gates.md) L1-1/2/4/5/6/8/10 共 7 条。
6. **灰度机制**（见 §2.5）。

**Out**（明确不做）

- ❌ 不做站点（Phase 2）。
- ❌ 不改论文收录范围（范围是 02 文的事，Phase 1 只搬不选）。
- ❌ 不改数据更新流水线（06 文的多源升级是 Phase 3）。
- ❌ 不动 `README.md` 的**内容**（标题写法、venue 短名、`…` 占位都原样保留；只有渲染顺序由 `order` 字段决定）。
- ❌ 不引入除 `jsonschema`（CI 侧）之外的 Python 依赖。

### 2.2 任务拆解与人日

| # | 任务 | 交付物 | 人日 | 依赖 |
| --- | --- | ---: | ---: | --- |
| 1.1 | 补齐 schema：`recent-papers.schema.json`、`index.json` 映射；review 已有 6 份是否覆盖全部字段 | `data/schema/` | **1** | — |
| 1.2 | 写 `scripts/migrate_readme_to_json.py`（跨行合并、`ALT_README_RE` 兜底、裸 URL、`…` 占位过滤、非 ASCII 连字符归一） | 脚本 + 单测样例 | **4** | 1.1 |
| 1.3 | 跑迁移 → 人工确认 `updates/migration/*.diff.json`（2015–2026 共 102 条） | `data/papers/*.json` | **5** | 1.2 |
| 1.4 | 迁移 0x02 数据集 / 0x03 工具 / 0x05 团队 / `research-lineage` | `data/datasets.json`、`toolkits.json`、`research-groups.json`、`lineage.json` | **3** | 1.2 |
| 1.5 | venue 归一（45 短名 → `venues.json`，含 `WWWC/WC/WWW`、`IF/Information Fusion`、`Sensor/Sensors` 合并） | `data/venues.json` | **2** | 1.3 |
| 1.6 | 写 `scripts/render_readme.py`（按 `section`/`year`/`order` 排序渲染，支持 `--check`） | 脚本 | **4** | 1.3 |
| 1.7 | 度量「数据→README 重建」的等价性，允许 3 类白名单差异（见 §2.3 A1 的 `allow_diff`） | `updates/reports/rebuild-diff.json` | **2** | 1.6 |
| 1.8 | 落地 L1 门禁 7 条（`check_schema` / `check_json_lint` / `check_duplicates` / `check_auto_blocks` / `check_paper_format` / `check_crossrefs` / `check_spelling`） | `scripts/check_*.py` + `pr-gates.yml` | **6** | 1.3、1.4 |
| 1.9 | 灰度切换：README 的 `0x00` 改为渲染产物；更新 `CONTRIBUTION.md`（按 07 文 §4.2 的建议） | PR + 文档 | **1** | 1.7、1.8 |
| | **Phase 1 合计** | | **28** | |

### 2.3 验收标准

**A1 · 重建等价（最核心的一条）**

```bash
# 从 data/ 重新渲染，与仓库里的 README 逐行比对
python3 scripts/render_readme.py --root . --out /tmp/rendered.md --lang en
diff -u README.md /tmp/rendered.md
```

判定：`diff` 的**实质差异行数为 0**。允许进入 `allow_diff` 缓存的只有 3 类：

1. `…` / `……` 占位行（会被渲染成统一的 `…`）；
2. 跨行折断被合并成单行（L170–171）；
3. TOC（`<!-- START doctoc -->` 区块，由外部工具维护，渲染器不产出）。

任何**第 4 类**差异都必须回到 `data/` 去修，而不是在渲染器里打补丁。

**A2 · Schema 全绿**

```bash
python3 scripts/check_schema.py --root .
# => [OK] 12 files validated against 7 schemas, 0 error
```

**A3 · 条目数守恒**

| 计数项 | 迁移前 | 迁移后必须 |
| --- | ---: | ---: |
| `data/papers/*.json` 记录数 | — | **102** |
| 0x00 深度学习小节条目 | 94 | 94 |
| 其他小节 `[[paper]]` 条目 | 8 | 8 |
| `data/datasets.json` | — | 18（0x02 的 18 个外链，含 1 个本地 doc 引用） |
| `data/toolkits.json` | — | 6 |
| `data/research-groups.json` | — | 15（0x05 编号条目：1 个裸 URL + 4 个 company + 10 个 college） |

**A4 · 去重门禁零误报**

```bash
python3 scripts/check_duplicates.py --root .
# => [OK] 102 entries, 0 duplicate doi, 0 duplicate title_norm
# => [WARN] title-collisions: N (escrow, 不算失败)
```

**A5 · 现有 CI 不变红**

```bash
python3 scripts/check_readme_sync.py --en README.md --zh README.zh.md
python3 scripts/render_curated_recent.py --data updates/curated/recent-papers.json \
  --readme-en README.md --readme-zh README.zh.md --check
```
两者必须仍然 `[OK]`。

**A6 · 单条论文的往返（round-trip）**

随机抽 10 条（建议包含：有 DOI 的、只有 arXiv 的、跨行的、带 `[[code-pytorch]]` 的、venue 是 `WWWC` 的），逐条：`JSON → 渲染 → 与 README 原文一致 → 反向解析回 JSON → 与原 JSON 字段级相同`。

### 2.4 风险与回滚

| 风险 | 概率 | 影响 | 缓解 | 回滚 |
| --- | --- | --- | --- | --- |
| 迁移脚本漏解析（跨行、裸 URL、非 ASCII 连字符） | **高** | 条目丢失且无声 | A3 的「条目数守恒」是硬闸门；迁移时**先跑统计再写文件**，统计对不上就不写 | 删掉 `data/papers/*.json` 即可，README 未动 |
| 人工确认 102 条过于枯燥 → 走过场 | 中 | 错误被"确认"进权威数据 | 确认只针对 **diff 摘要**（每条 1 行：`id / year / title / doi`），不做全文 review；正文靠 A1 自动比对 | 同上 |
| venue 归并出错（`WWW` 与 `WWWC` 是否同一个？） | 中 | 站点 venue 统计错 | 归并表进 `data/venues.json` 的 `aliases[]`，保留原短名；渲染时用原短名输出，`venue_id` 只用于聚合 | 改 `aliases` 即可 |
| A1 永远有残余 diff → 无限期追 | **高** | 灰度无法收尾 | 设定**时间盒**：A1 残余 diff ≤ 3 类且 ≤ 5 行即视为通过；超出则把剩余差异写成 `updates/reports/rebuild-known-diff.md` 并接受 | — |
| 渲染器与 `render_curated_recent.py` 职责重叠 → 两套模板 | 中 | 维护成本翻倍 | 明确分工：`render_readme.py` 管 `0x00`/`0x02`/`0x03`/`0x05` 静态段；`render_curated_recent.py` 只管 2 个 auto 块，**不合并** | — |
| `data/` 与 README 同时可写 → 分成两份真相 | **高** | 有人改 README 就丢了 | L1-5 门禁（自动块未被手改）扩到**整个 0x00**：`pr-gates` 检测「README 的 0x00 段 ≠ 渲染结果」，直接 FAIL 并提示改 JSON | 临时关掉该门禁 → 回到「README 优先」的老规则 |

**整体回滚**：Phase 1 的所有产物都在 `data/`、`scripts/`、`docs/` 下，**`git revert` 一个 merge commit 即可**。README 只在 1.9 那一步被改成渲染产物，那一步单独一个 commit，可独立 revert。

### 2.5 与现有 README 并存的灰度方式

这是 Phase 1 最关键的设计。三步走，**任一步都可停**：

**灰度 1（第 1–3 周）· 影子数据，零可见变化**

- `data/` 建起来，`scripts/render_readme.py --check` 在 CI 里跑但 `continue-on-error: true`。
- README **完全不动**。任何人打开仓库，体验与今天一致。
- 唯一的可见变化：CI 多了一个（非阻塞的）job。

**灰度 2（第 4–6 周）· 单小节试点**

- 只把 **`#### 2015`、`#### 2016`、`#### 2017`** 三个小节（共 5 条）改成渲染产物，加上 marker：

  ```markdown
  <!-- START DATA RENDERED: 2015-2017 -->
  #### 2015
  1. [ICLR 2015] ... [[paper]](...)
  ...
  <!-- END DATA RENDERED -->
  ```

- 这三个小节的条目少、无争议、无 auto 块冲突，是理想的试点。
- 验收：连续 2 周无 PR 因这三节报错。

**灰度 3（第 7–11 周）· 全量切换 + 冻结写入**

- 0x00 全段改为渲染产物（`<!-- START DATA RENDERED: 0x00 -->`）。
- `CONTRIBUTION.md` 同步更新：「新增论文请提交 `data/papers/YYYY.json`，不要直接编辑 README」。
- PR 模板加提示（07 文 §4.1）。
- **旧的 `render_curated_recent.py` 与 `updates/curated/recent-papers.json` 暂时保留**，因为 Phase 1 只做"数据源切换"，不删旧机制；删旧是 Phase 3 的事。

**永久保底**：即使灰度 3 完成，`README.md` 仍然可读、可搜索、可在 GitHub 上直接看。渲染只是让它的**权威性**下降，不是让它不可用。

### 2.6 对现有 CI 的影响

| 现有 workflow | Phase 1 影响 | 动作 |
| --- | --- | --- |
| `readme-sync-check.yml` | **升级**：`check_readme_sync.py` 的「计数」语义在 README 变成渲染产物后依然有效（两份 README 都从同一数据渲染 → 计数天然相等）。它从「唯一门禁」退化为「快速兜底」 | 保持不动，不删 |
| `curated-render-check.yml` | **不动**：`recent-papers.json` 与 2 个 auto 块仍在（灰度 3 之后它们的数据源可能切到 `data/`，但脚本行为不变） | 不动 |
| `monthly-paper-search.yml` | **轻微影响**：它写 `updates/monthly-paper-candidates.md` 并用 `dedupe_readme_papers.py` 对比 README。README 变成渲染产物后，去重仍然成立（标题+DOI 是从渲染后的文本里提的） | 不改，但 `scripts/README.md` 需说明去重对象已变 |
| `opencode.yml` | 无影响 | — |
| **新增** `pr-gates.yml` | L1 门禁 7 条，全部 < 60s、零网络 | 新建 |
| **新增** 1 个阻塞风险 | A1 的残余 diff 会让 `pr-gates` 变红 | 用 `allow_diff` 白名单收敛 |

**CI 运行时间预算**：现有 3 个 workflow 合计 < 2 min；新增 `pr-gates.yml` 目标 **< 60 s**（纯 Python stdlib + 零网络）。Phase 1 结束时 PR 反馈总时长应仍 < 3 min。

---

## 3. Phase 2 · 站点上线（20 人日）

### 3.1 范围

**In**

1. **确定 SSG 落地**：沿用 Phase 0 的 Astro 7.4.13 + Pagefind 1.6.0 + ECharts 6.1.0（选型依据见 [03-ssg-choice](./03-ssg-choice.md)）；本机实测产物 2.2 MB、15 页、构建 0.47 s（含 Pagefind 后约 1 s）。
2. **从 `data/` 取数**：把 `site/scripts/prepare-data.mjs` 的 `provenance.source` 从 `README.md` 改成 `data/papers/*.json`，`degraded` 必须变 `false`。
3. **页面集**（15 页已就绪，需要的是"内容正确"而非"页面更多"）：
   - `/`、`/zh/` 概览
   - `/papers/`、`/zh/papers/` 论文库（全量渲染 + 客户端筛选）
   - `/datasets/`、`/benchmark/`、`/trends/`、`/lineage/`、`/toolkits/`、`/search/` 及对应 `zh/`
   - `/sitemap.xml`（已有）
4. **筛选与搜索**：
   - 筛选：年份 × venue × 任务 × method_tags，纯 DOM（`hidden` 切换），筛选状态写进 URL query（可分享）。
   - 搜索：Pagefind 静态索引，`zh-cn` 与 `en` 双语（实测 `build` 已索引 2 语言 / 15 页 / 2116 词）。
5. **GitHub Pages 部署**：`SITE_BASE=/Awesome-Traffic-Prediction/`（`astro.config.mjs` 已支持 `SITE_URL`/`SITE_BASE` 环境变量），`output: 'static'`，用 `actions/deploy-pages`。
6. **站点侧门禁**：L1-9（`build:fast` 必过）+ L2-3（产物体积预算）。

**Out**

- ❌ 不做服务端渲染 / API / 数据库 / 用户系统（见 §5「不做什么」）。
- ❌ 不做实时数据（站点是构建期快照）。
- ❌ 不做评论、点赞、提交表单（贡献走 GitHub PR）。
- ❌ 不做 PDF 在线预览（`papers/` 下 5 个 PDF 共 16 MB，只在站点给链接）。
- ❌ 不做全文检索论文正文（只索引标题/摘要/标签）。

### 3.2 任务拆解与人日

| # | 任务 | 交付物 | 人日 | 依赖 |
| --- | --- | ---: | ---: | --- |
| 2.1 | `prepare-data.mjs` 切换数据源到 `data/papers/*.json`，保留 README 兜底但 `degraded` 必须显式 | 脚本改动 | **2** | Phase 1 |
| 2.2 | 论文库页内容校正：排序（`year`/`order`）、venue 展示用短名、`…` 占位不渲染 | `PaperList.astro` | **3** | 2.1 |
| 2.3 | 客户端筛选（year × venue × task × tag）+ URL query 状态 | 页面内联 JS | **3** | 2.2 |
| 2.4 | Pagefind 搜索调优（中文分词、stub 过滤、`/zh/` 前缀处理） | `astro.config.mjs` + 页面 `data-pagefind-*` | **2** | 2.2 |
| 2.5 | `/benchmark/` 内容：Benchmark Matrix（4 个 dataset × 7 列）+ leaderboard 占位（Phase 3 填数） | 页面 | **2** | 2.1 |
| 2.6 | `/trends/` 内容：把 `updates/hotspots/*` 与 `updates/wordcloud/*.json` 接进 ECharts（图表按需 chunk） | 页面 + 图表组件 | **3** | 2.1 |
| 2.7 | GitHub Pages workflow（`actions/configure-pages` + `upload-pages-artifact` + `deploy-pages`），含 `SITE_BASE` 与 404 处理 | `.github/workflows/pages.yml` | **2** | 2.2 |
| 2.8 | 站点门禁：L1-9 + L2-3（体积预算：`dist/` 12 MB、单页 400 KB、JS chunk 900 KB） | workflow + `check_site_budget.py` | **2** | 2.7 |
| 2.9 | 可访问性/SEO 收尾：`hreflang`、`og:*`、`sitemap` 里 base 前缀、`robots.txt` | 页面 meta | **1** | 2.7 |
| | **Phase 2 合计** | | **20** | |

### 3.3 验收标准

**B1 · 构建成功且产物达标**

```bash
cd site && ASTRO_TELEMETRY_DISABLED=1 pnpm build
```

- 退出码 0；
- `dist/` 存在 `index.html`、`zh/index.html`、`papers/index.html`、`zh/papers/index.html`、`pagefind/pagefind.js`、`sitemap.xml`；
- `dist/` 总体积 ≤ **3 MB**（基线 2.2 MB，留 36% 余量）；
- `dist/papers/index.html` ≤ **200 KB**（基线 107 KB）。

> ⚠️ 实测坑：`astro build` 在受限环境里会因 `@astrojs/telemetry` 写 `~/Library/Preferences/astro` 而 `EPERM` 崩溃。CI 与本地都必须设 `ASTRO_TELEMETRY_DISABLED=1`（或 `telemetry: false`）。这是 2.7 必须带上的环境变量。

**B2 · 数据来自 `data/` 而非 README**

```bash
python3 -c "import json;d=json.load(open('site/src/data/papers.generated.json'));\
print(d['provenance'])"
# => {'source': 'data/papers', 'degraded': False, 'count': 102, ...}
```

当前实测是 `{'source': 'README.md', 'en': 104, 'zh': 104, 'degraded': True}` —— **104 ≠ 102**，多出的 2 条正是 `…` 占位被当成条目的结果。这个数字必须收敛到 102。

**B3 · 线上可访问**

```bash
curl -sSL -o /dev/null -w "%{http_code}\n" https://coolgiserz.github.io/Awesome-Traffic-Prediction/
curl -sSL -o /dev/null -w "%{http_code}\n" https://coolgiserz.github.io/Awesome-Traffic-Prediction/zh/papers/
# => 200 / 200
```

**B4 · 筛选与搜索可用（人工 + 脚本）**

- 访问 `/papers/?year=2020&tag=GNN`，页面 URL 可复制、刷新后筛选保持，结果条数与 `data/papers/2020.json` 里 `method_tags` 含 GNN 的条目数一致；
- 站内搜索 `STGCN`，返回 ≥ 3 条且包含 `Spatio-Temporal Graph Convolutional Networks`；
- 搜索中文「交通流预测」有结果（验证 Pagefind 中文分词）。

**B5 · 内链零 404**

```bash
# 提取 dist 里所有内部链接，逐个查文件存在
python3 - <<'PY'
import re,pathlib
dist=pathlib.Path('site/dist'); miss=[]
for p in dist.rglob('*.html'):
    for href in re.findall(r'href="(/Awesome-Traffic-Prediction/[^"#?]+)', p.read_text()):
        t=dist/href.split('Awesome-Traffic-Prediction/')[1]
        if not (t.exists() or (t/'index.html').exists()):
            miss.append((str(p), href))
print(len(miss), 'missing'); [print(m) for m in miss[:10]]
PY
# => 0 missing
```

**B6 · 站点与 README 数据一致**

同一篇论文（例如 `UniST`）在 README 的年份、站点 `/papers/` 的年份、`data/papers/2024.json` 的 `year` 三者相同。

### 3.4 风险与回滚

| 风险 | 概率 | 影响 | 缓解 | 回滚 |
| --- | --- | --- | --- | --- |
| GitHub Pages 子路径 (`base`) 处理不当 → 全站 404 | **高** | 站点不可用 | `astro.config.mjs` 已把 `base` 交给 `SITE_BASE` 环境变量，所有内部链接走 `src/lib/url.ts` 的 `withBase()`；B5 的内链检查是硬闸门 | 改 `SITE_BASE=/` 或回退到本地 `dist/` |
| Pagefind 中文分词差 | 中 | 中文搜索不可用 | 用 `htmlLang: 'zh-CN'` + Pagefind 的 `--force-language`；实在不行中文搜索降级为"标签筛选" | 移除 `/search/` 导航项 |
| 论文库单页过大（107 KB → 条目增多后线性增长） | 中 | 首屏慢 | 每 100 条分页，或按年份拆 `/papers/2020/`；L2-3 体积预算提前预警 | 分页可独立上线 |
| ECharts 打包进所有页面 | 中 | 首屏体积翻倍 | 只让 `/trends/` 动态 `import('echarts/core')`，`chunkSizeWarningLimit: 900` 已配置 | — |
| 站点建好了没人维护 → 与 README 漂移 | **高** | 两个"真相" | Phase 2 结束时 README 已是渲染产物（Phase 1 成果），两者同源，漂移不可能 | 删 `site/` 即可，README 不受影响 |
| Astro 大版本升级破坏构建 | 低 | 部署挂 | `pnpm-lock.yaml` 锁定；`packageManager: pnpm@12.16.1` | `git revert` lockfile |

**整体回滚**：`site/` 是**完全独立目录**，不写仓库其他位置（`prepare-data.mjs` 只读根目录、只写 `site/` 内）。删除 `site/` 目录 + 停用 `pages.yml` 即可回到纯 README 状态，**零数据损失**。

### 3.5 与现有 README 并存的灰度方式

1. **部署到 `https://coolgiserz.github.io/Awesome-Traffic-Prediction/`，不在 README 顶部加醒目入口**。先在 README 的 `## 0x07 Benchmark & Evaluation` 的 `### Auto Trend Visuals` 小节加一行：
   ```markdown
   - Browsable version: <https://coolgiserz.github.io/Awesome-Traffic-Prediction/>
   ```
   （放在 Phase 2 末期，观察 4 周无负反馈再考虑提到顶部 badge 区。）
2. **站点上线第一周关闭 `/search/`**，只开浏览与筛选——搜索的 flaky 最容易变成"站点坏了"的观感。
3. **README 永远是主入口**：`README.md` 在 GitHub 上的渲染质量优先级高于站点。任何导致 README 变差的设计一律否决。

### 3.6 对现有 CI 的影响

| 项 | 影响 |
| --- | --- |
| 现有 3 个校验 workflow | 无影响 |
| 新增 `pages.yml` | 只在 `push: main` + `paths: [site/**, data/**, updates/**, README*.md]` 触发；`pr-gates.yml` 里另跑 `build:fast` 做 PR 侧校验 |
| CI 时长 | Pages 部署（含 `pnpm install` 冷缓存）预计 60–90 s；PR 侧 `build:fast` 预计 30 s（本机 0.47 s 构建 + install） |
| `data/` 变更触发站点重建 | 需在 `paths` 里显式加 `data/**`，否则改了数据站点不更新——这是最容易漏的一条 |

---

## 4. Phase 3 · 自动化与社区（31 人日）

### 4.1 范围

**In**

1. **多源流水线**：按 [06 文 §2](./06-pipeline.md) 把 `traffic-monthly.json` 的 `sources` 从 `["openalex"]` 扩到 `["openalex","arxiv","crossref"]`，S2 按需（带 key）。
2. **四段式重构**：`fetch` / `normalize` / `render` / `publish` 拆分（06 文 §6），产出 `updates/raw/`、`updates/normalized/`、`updates/reports/`。
3. **LLM 辅助打标签**：按 06 文 §4 落地「规则词典为权威 + LLM 建议进 `tag-suggestions.json` + 人审 accept」的闭环。
4. **Leaderboard**：`data/leaderboards/*.json` + `citations-snapshot.json`，替换 pwc 的 10 处链接（06 文 §5.4）。
5. **贡献者流程**：PR 模板、`CONTRIBUTION.md` 更新、「自动化 PR 的复核清单」、`good first issue` 标签体系。
6. **L2 门禁**：`link-check`（含 `redirect_away` 检测）、`check_dois`、`site-budget`、`repo-report`、`artifact-freshness`。

**Out**

- ❌ 不做 C2C 的"论文提交表单"。
- ❌ 不做自动 merge（任何自动化 PR 仍要人点 merge）。
- ❌ 不做实时排行榜（引用数快照按月，见 §5）。
- ❌ 不做语义搜索 / 向量库。

### 4.2 任务拆解与人日

| # | 任务 | 交付物 | 人日 | 依赖 |
| --- | --- | ---: | ---: | --- |
| 3.1 | 多源配置切换 + `paper_search.py` 加 `authors` 字段（去重键 key_4 需要） | 配置 + 脚本小改 | **3** | Phase 1 |
| 3.2 | 四段拆分：`fetch`（raw 快照）/ `normalize`（`normalize_papers.py`）/ `render` / `publish` | 脚本 + workflow | **5** | 3.1 |
| 3.3 | 稳定排序：整数分 + 全序排序键 + 去掉 candidates 头部时间戳 | 脚本改动 | **2** | 3.2 |
| 3.4 | 失败告警与降级：`$GITHUB_STEP_SUMMARY` + 软降级写 report + 连续 2 次失败开 issue | workflow | **2** | 3.2 |
| 3.5 | LLM 打标签：taxonomy 冻结 + prompt + schema 校验 + 快照落盘 + 人审表格生成 | `scripts/suggest_tags.py` + `updates/reports/tag-suggestions.json` | **5** | 3.2 |
| 3.6 | Leaderboard 数据模型 + 首批 4 个 dataset（METR-LA / PEMS-BAY / NYC-Bike / NYC-Taxi）人工录数 | `data/leaderboards/*.json` | **4** | Phase 2（站点有 `/benchmark/`） |
| 3.7 | 引用数快照（OpenAlex 批量 filter，单次 50 条） | `scripts/snapshot_citations.py` | **2** | 3.6 |
| 3.8 | pwc 链接处置落地（06 文 §5.4 的 5 条规则，含 README×2 与 `datasets/`×4） | PR | **2** | 3.6 |
| 3.9 | L2 门禁 5 条（`check_links` 含 `redirect_away`、`check_dois`、`check_site_budget`、`report_repo_size`、`check_artifact_freshness`）+ baseline 文件 | 脚本 + workflow | **4** | — |
| 3.10 | 贡献者流程：PR 模板、`CONTRIBUTION.md`、issue 模板（新增论文/数据集/报告死链） | 文档 | **1** | 3.9 |
| 3.11 | 收尾：一次性替换 `updates/curated/recent-papers.json` → `data/papers/*.json`（消灭双体制） | 迁移 PR | **1** | 3.2 |
| | **Phase 3 合计** | | **31** | |

### 4.3 验收标准

**C1 · 多源生效且可观测**

```bash
python3 scripts/validate_domain_config.py scripts/configs/traffic-monthly.json --summary
# => sources: ['arxiv','crossref','openalex']
python3 scripts/paper_search.py --config scripts/configs/traffic-monthly.json \
  --max-requests 40 --cache-dir /tmp/ps3 --out /tmp/c.md --report-json /tmp/c.json
python3 -c "import json;r=json.load(open('/tmp/c.json'));\
print({k:r['runtime'][k] for k in ('requests_made','cache_hits')}); print(r['source_errors'])"
# => source_errors: [] (或每条都能指名 source)
```

**C2 · 可复现（同输入同输出）**

```bash
# 用同一份 raw 快照重跑 normalize 两次，产物必须逐字节相同
python3 scripts/normalize_papers.py --raw updates/raw/2026-11/ --out /tmp/n1.json
python3 scripts/normalize_papers.py --raw updates/raw/2026-11/ --out /tmp/n2.json
cmp /tmp/n1.json /tmp/n2.json && echo "reproducible"
```

**C3 · 打标签有人审闭环**

- `updates/reports/tag-suggestions.json` 每条带 `paper_id / task_id / method_ids / confidence / evidence_span / model / prompt_version`；
- 所有 `task_id` / `method_ids` 都在 `data/taxonomy/paper-tags.json` 的白名单内（脚本校验）；
- 未经人审的提案**不出现在** `data/papers/*.json` 的 `task_*` / `method_tags` 字段里（用 `git diff data/papers/` 验证：tag-suggestions 更新不应带 `data/papers/` 改动）。

**C4 · Leaderboard 与引用数**

```bash
python3 -c "import json;print(len(json.load(open('data/leaderboards/metr-la.json'))['rows']))"
# => >= 3（对应 15/30/60 min 三个 horizon）
curl -s "https://api.openalex.org/works?filter=ids.openalex:W4391987728|W4392382537&per-page=50&select=id,cited_by_count"
# => 200，两条记录，用于 snapshot 对账
```

**C5 · pwc 链接清零**

```bash
grep -rn "paperswithcode.com" README.md README.zh.md datasets/ | wc -l
# => 0（或仅剩 docs/ 里的历史说明）
```

**C6 · L2 巡检产出结构化的 `redirect_away`**

```bash
python3 scripts/check_links.py --root . --out updates/reports/link-status.json --budget 300
python3 -c "import json;r=json.load(open('updates/reports/link-status.json'));\
import collections;print(collections.Counter(x['status'] for x in r['results']))"
# => Counter({'ok': N, 'redirect_ok': M, 'blocked': K, 'dead': 0, 'redirect_away': 0})
```

`redirect_away` 必须为 0，因为 pwc 已被处理。**这是验证 06 文 §5 方案落地的唯一硬指标。**

**C7 · 自动化 PR 零人工合并**

连续 2 个月的 `automation/monthly-paper-refresh` PR 满足：
- 正文含 `degraded` 状态标记（有降级时）；
- 正文含 normalize 报告摘要（新增 N / 去重 M / 冲突 K）；
- 若 `未降级 && 无冲突`，reviewer 只需点 approve，不需手改任何文件。

### 4.4 风险与回滚

| 风险 | 概率 | 影响 | 缓解 | 回滚 |
| --- | --- | --- | --- | --- |
| 多源后召回质量下降（Crossref 的 `posted-content` 混入预印本噪声） | **高** | 候选表变脏 | 保留 `post.rules` 的 exclude 规则；Crossref 只做 enrichment 不做独立 query（06 文 §2.2 的三级优先级） | `search.sources` 改回 `["openalex"]` 一行 |
| S2 无 key 时 429 拖慢整条流水线 | 高 | 月度 PR 延迟 | S2 默认不进主链路；要进必须带 key + `source_min_interval_seconds: 3.0` | 从 sources 移除 |
| LLM 打标签不可复现 | 中 | 同论文两次标签不同 | **快照落盘 + 只读快照**，不在 CI 里重调 LLM | 删 `tag-suggestions.json`，回落纯规则 |
| LLM 调用成本失控 | 低 | 账单 | 只对「规则未命中」的条目跑（102 条里约 40–60 条）；单月新增 ≤ 40 条 | 关闭 LLM job |
| Leaderboard 数字录错（论文表格抄错） | **高** | 权威数据错误 | `verified: false` 的行不渲染到 README；人工核对后才置 `true` | 把行改回 `verified: false` |
| 引用数被当成"质量"排序 | 中 | 排行榜沦为"老论文榜"（实测 `UniST` 2024 = 265 次，`HySRD-Net` 2026 = 4 次） | 排行榜**必须标注快照日期**，且按 `horizon_min` 分列，不提供"总榜" | 只展示 `verified` 分数榜，引用数仅作参考列 |
| L2 链接巡检被反爬打成噪音 | 高 | issue 常年开着被静音 | `403/429` 归为 `blocked` 不算失败；baseline 文件吸收历史问题 | 关闭 link-check workflow |

**整体回滚**：Phase 3 的所有改动都在 `scripts/`、`data/`、`.github/` 下，**分 3 个独立 PR（多源+拆分 / 打标签 / leaderboard+巡检）**，可逐块 revert。README 的 pwc 链接替换（3.8）单独一个 commit。

### 4.5 与现有 README 并存的灰度方式

1. **多源**：先在 `workflow_dispatch` 手动跑 3 次，把结果与单源结果做 Jaccard 比对（目标 ≥ 0.8 的重合 + 至少 5 条新召回），再切 cron。
2. **打标签**：建议结果先只写 `tag-suggestions.json`，**连续 2 个月**不进 `data/`；人审 accept 率（被采纳 / 总提案）达到 80% 后才允许"预填 + 人确认"模式。
3. **Leaderboard**：先只上 `/benchmark/` 页面（Phase 2 已有页面），README 保持不动；两处数字对得上、且 `verified` 覆盖 ≥ 80% 后，才改 `datasets/*.md` 的 SOTA 小节。
4. **pwc 链接**：不要一次性全换。按 06 文 §5.4 的顺序——先给 `datasets/` 4 处加"链接已失效"标注（零风险），再换 `README` 的 3 处 `[[code]]`（需实际找 repo），最后换 `[[SOTA]]` 与 2 个 task 入口（需 leaderboard 就绪）。

### 4.6 对现有 CI 的影响

| 项 | 影响 |
| --- | --- |
| `monthly-paper-search.yml` | **被重构**：单 job 拆成 4 段（可用 `needs:` 串联，仍是单 workflow）；词云/热点拆到独立 workflow |
| `readme-sync-check.yml` / `curated-render-check.yml` | Phase 3 末期（3.11）后 `curated-render-check` 可退役（`recent-papers.json` 消失），但**保留 `render_curated_recent.py` 直到确认无引用** |
| 新增 5 个 L2 workflow | 全部 cron，失败不进 PR 阻塞 |
| CI 总时长 | 月度流水线从 ~2 min 涨到 ~5 min（多源 + 3 次重试）；这是可接受的 |

---

## 5. 「不做什么」清单（防过度工程）

这份清单是**硬约束**，不是建议。任何 PR 想越过其中一条，必须先在本文件里删改这一条并说明理由。

| # | 不做 | 为什么 | 替代 |
| --- | --- | --- | --- |
| 1 | **不做后端服务**（API / DB / 服务器） | AGENTS.md 明确「Avoid introducing application code, build systems」；GitHub Pages 只有静态文件 | 静态 JSON + 构建期预生成 |
| 2 | **不做用户系统**（注册 / 登录 / 个人收藏） | 贡献者是研究者，习惯 GitHub PR；用户系统带来账号、隐私、运维三重负担 | GitHub 星标 / fork / PR |
| 3 | **不做实时数据**（实时引用数、实时热点） | 引用数以月为粒度变化，实时化只增加成本；且会把 CI 变成爬虫 | 月度快照 + 标注 `snapshot_at` |
| 4 | **不做全文检索** | 5 个 PDF 16 MB，全文索引体积与构建时间都会爆 | 只索引标题/摘要/标签 |
| 5 | **不做自动 merge** | 论文相关性需要人判断（实测单源召回里已出现"Dhaka 热连通"这类误召回） | 自动化只开到"PR + 摘要 + 建议" |
| 6 | **不做静态组合页**（每个 year×venue×tag 一页） | 12 年 × 45 venue × 6 任务 × 10 标签，单维就 73 页，二维 > 300 页（03 文 §4） | 客户端 DOM 过滤（1 页 107 KB） |
| 7 | **不引入 UI 框架**（React / Vue / Svelte island） | 15 页文档站不需要运行时框架；会把 0 KB 初始 JS 变成 35–150 KB | Astro 组件 + 少量原生 JS |
| 8 | **不做评论 / 点赞 / 阅读计数** | 无服务端即无法可信统计；引入第三方会拖慢首屏 | 无 |
| 9 | **不做多语言扩展到 EN/ZH 之外** | 维护者只有中英双语能力，第三语言必然腐化 | 保持 2 语言，i18n 字典结构已预留 |
| 10 | **不做论文 PDF 自托管** | 版权风险 + 仓库体积（现 16 MB） | 链接到 arXiv/出版社 |
| 11 | **不做"AI 摘要"全量生成** | 摘要已有（OpenAlex/arXiv 都返回），再 LLM 生成只是重写 | 直接用源摘要 |
| 12 | **不做 Cron 频次提升**（周更/日更） | 交通预测领域周增论文个位数；日更只增加 PR 噪音 | 月度（现状）+ 手动 dispatch |
| 13 | **不做 Docker / 打包 / 发布 npm 包** | 没有分发需求 | 纯 git |
| 14 | **不做测试覆盖率门禁** | 脚本是"一次性维护脚本"，覆盖率高但价值密度低 | 关键路径（迁移、归一化、去重键）写单测 |

---

## 6. 关键依赖与并行关系

```
Phase 1 ────────────────────────────► Phase 2 ──────────► Phase 3
 1.1 schema
 1.2 迁移脚本
     └─► 1.3 papers JSON ──┬─► 1.6 渲染器 ─► 1.7 等价度量 ─► 1.9 灰度3
                           ├─► 1.5 venue 归一
                           └─► 1.8 L1 门禁
 1.4 其他实体 ─────────────┘

                     2.1 prepare-data 切源 ─► 2.2 论文库 ─┬─► 2.3 筛选
                                                          ├─► 2.4 搜索
                                                          └─► 2.7 Pages ─► 2.8 门禁 ─► 2.9 SEO
                     2.5 benchmark / 2.6 trends（可并行）

Phase 3: 3.1 多源 ─► 3.2 四段 ─┬─► 3.3 稳定排序
                               ├─► 3.4 告警
                               └─► 3.5 LLM 打标
        3.6 leaderboard ─► 3.7 引用快照 ─► 3.8 pwc 处置
        3.9 L2 门禁（独立，可与 3.1 并行）
        3.10 贡献者流程、3.11 双体制收尾
```

**可压缩的并行点**：1.4（其他实体）与 1.3（papers）可并行；2.5/2.6 与 2.2 可并行；3.9 与 3.1 可并行。3 处并行合计可压缩约 5 人日。

---

## 7. 人日汇总

| 阶段 | 人日 | 主要构成 |
| --- | ---: | --- |
| Phase 1 · 数据基座 | **28** | 迁移 9 + 渲染/等价 6 + 门禁 6 + schema/venue 3 + 灰度 1 + 其他实体 3 |
| Phase 2 · 站点上线 | **20** | 切源 2 + 论文库/筛选/搜索 8 + 部署/门禁 5 + benchmark/trends 5 |
| Phase 3 · 自动化与社区 | **31** | 多源+四段 10 + LLM 打标 5 + leaderboard 8 + L2 门禁 4 + 贡献者流程 3 + 收尾 1 |
| **总计** | **79** | 单人流约 31 周；含并行约 24–26 周 |

**估算口径说明**：

- 1 人日 = 1 名熟悉本仓库的维护者 6 小时有效工作；
- 不含 code review 往返时间（按经验 review 占开发量 15–20%，已隐含在各任务里）；
- 不含论文内容审阅（那是 02 文的范围，且属于持续运营而非一次性迁移）；
- 迁移类任务（1.3、3.6）按「条目数 × 单条 3 分钟」估算：102 条 ≈ 5 人日；leaderboard 4 个 dataset × 3 horizon × 2 metric ≈ 24 个数字，含查原文 ≈ 4 人日。

---

## 8. 一页纸总结

| | Phase 1 | Phase 2 | Phase 3 |
| --- | --- | --- | --- |
| **一句话** | README 变成渲染产物 | 站点可访问可搜索 | 流水线零人工 + 排行榜 |
| **最大交付** | `data/papers/*.json` 102 条 | `pages.yml` + 15 页 | 多源四段流水线 + leaderboard |
| **最大风险** | 重建等价性收敛不了 | GitHub Pages base 路径 | 多源召回质量下降 |
| **硬闸门** | 条目数守恒 102、重建 diff ≤ 5 行 | `dist/` ≤ 3 MB、内链 0 404 | `redirect_away` = 0、tag 白名单全过 |
| **回滚成本** | revert 1 个 commit | 删 `site/` 目录 | 逐块 revert 3 个 PR |
| **人日** | 28 | 20 | 31 |

**如果只能做一件事**：做 Phase 1 的 **1.2 + 1.3**（迁移脚本 + 102 条 JSON）。因为它是后面 51 人日的前提，而且即使站点和自动化全部不做，`data/papers/*.json` 本身也已经是一个可直接消费的开放数据集——比现在只能被人肉 grep 的 markdown 强一个数量级。