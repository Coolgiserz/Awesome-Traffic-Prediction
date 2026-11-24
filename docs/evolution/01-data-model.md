# 01 数据模型：从 README Markdown 到 JSON 单一数据源

> 状态：设计稿（Phase 0）。本文只描述模型与迁移路径，不改动 `README.md` / `README.zh.md` / `scripts/` / `updates/` / `.github/`。
> 本文所有数字均来自本仓库当前快照（2026-11-24 工作区，`README.md` 483 行 / `README.zh.md` 483 行）。

---

## 1. 现状画像：为什么必须迁移

### 1.1 仓库当前的真实数据形态（实测）

用统一正则 `^\s*\d+\.\s+\[([^\]]+)\]\s+(.+?)\s+\[\[paper\]\]\(([^)]+)\)` 扫描两份 README：

| 指标 | 实测值 | 说明 |
| --- | --- | --- |
| `<!-- START AUTO RECENT PAPERS -->` 块内条目 | 15 | 2025×4 + 2026×11，已由脚本渲染 |
| `<!-- START AUTO CURATED DIGEST TABLE -->` 表格数据行 | 12 | 与 `updates/curated/recent-papers.json` 中 `featured=true` 的 12 条一一对应 |
| `updates/curated/recent-papers.json` 条目 | 19 | 年份分布仅 2024(4) / 2025(4) / 2026(11) |
| README 中带 `[[paper]]` 的论文行 | 102 | 全部集中在 `0x00 Papers` |
| 不同论文标题（归一化后） | 102 | 说明**当前无重复条目**，去重仅靠标题+DOI 是“恰好没撞车”，不是“有保障” |
| 带 `[[paper]]` 但 URL 中无 DOI 的条目 | 78 / 102（76.4%） | 去重键的 DOI 分支对四分之三条目失效 |
| 出现的链接标签种类 | 8 | `paper`(103) `code`(24) `review`(2) `code-pytorch`(2) `data`(2) `code-official-tf`(1) `paper-ieee`(1) `code-not-official`(1) |
| 章节年份 vs venue 年份不一致 | 2 | L89 `[TITS 2019]` 位于 `#### 2018`；L156 `[Arxiv 2020]` 位于 `#### 2021` |
| venue 短名变体数 | 45 | 含 `Arxiv`(20) `TITS`(13) `AAAI`(11)，以及 `WWWC`/`WC`/`WWW`、`IF`/`Information Fusion`、`Sensor`/`Sensors` 等同义异写 |
| 跨行折断的条目 | 1 | L170–171 `2. [Arxiv 2022] Pre-training Enhanced ...` 标题折成两行，无缩进 |
| 非 ASCII 标题（被复制成 en-dash/hyphen 变体） | 3 | L104 `Spatial–Temporal`、L157 `Spatial‐temporal`、L183 `spatial–temporal` |
| 解析器边界：`[[code]]` 紧贴前一个链接 | 1 | L115 `...3330884)[[code]](...)` 中间无空格 |
| 裸 URL 条目（不以 `[title](url)` 形式出现） | 2 | L366 `1. http://urban-computing.com/yuzheng`、L415 `2. https://github.com/topics/traffic-prediction` |
| 省略号占位条目 `...` / `……` | 多 | 每节末尾的“待补充”标记，真实条目统计时必须排除 |

### 1.2 痛点归纳（都能在上表找到证据）

1. **双份手工维护**：`README.md` 与 `README.zh.md` 的论文行逐条重复（102 条 × 2），只有章节标题语言不同。当前靠 `scripts/check_readme_sync.py` 事后校验“每年条目数是否相等”，属于**事后告警**而不是**结构上不可能不一致**。
2. **正则解析脆弱**：跨行折断、`)[[code]]` 紧贴、非 ASCII 连字符、`[KDD 2021] [Title](url)` 这种“标题自带链接”的旧格式（L251）、裸 URL（L366/L415）都会让单一正则漏掉条目；`dedupe_readme_papers.py` 里已经出现 `CANDIDATE_RE` + `ALT_README_RE` 双正则兜底，说明这个坑已经在踩。
3. **去重靠标题匹配**：`normalize_title()` 是 `re.sub(r"\W+","",title.lower())`，把标点全部抹掉。后果是 (a) 无法区分同名不同年的作品；(b) 76.4% 条目没有 DOI 可用；(c) 标题一旦被修正（拼写、大小写、连字符），旧条目立刻“复活”为重复。
4. **站点/可视化不可复用**：`scripts/yearly_hotspots.py` 用 `--input README.md` 直接扫 markdown 文本做词频统计——数据层和展示层耦合，任何 markdown 措辞（例如补一句描述）都会污染热点统计。
5. **字段无处安放**：`layers`（SOTA/基线/综述/应用）、`direction_tags`、`datasets`、`affiliations`、`layer` 分级、链接健康状态，这些在 markdown 里没有位置，只能写进 note 的自由文本，无法被查询。
6. **两个“源”并存**：`updates/curated/recent-papers.json`（19 条）已经是一份结构化源，`render_curated_recent.py` 只渲染 `AUTO` 块；而 2015–2024 的 81 条仍然只在 markdown 里。等于**同一份 README 里跑着两套数据体制**。
7. **CI 覆盖面小**：`curated-render-check.yml` 只检查 `AUTO` 块与 JSON 一致；`readme-sync-check.yml` 只比较条目数量。谁都可以在 `#### 2023` 里塞一条 URL 打不开的论文而 CI 全绿。

**结论**：迁移目标不是“把 markdown 换成 JSON”，而是把 **事实源** 从 `README*.md` 挪到 `data/`，让 README 变成**渲染产物**，并在模型层面把语言、实体、去重键、链接健康拆成独立字段。

---

## 2. 目标架构总览

```
data/
├── schema/                 # JSON Schema (draft 2020-12)，机器可读的契约
│   ├── paper.schema.json           ✅ 已落地
│   ├── dataset.schema.json         ✅ 已落地
│   ├── toolkit.schema.json         ✅ 已落地
│   ├── venue.schema.json           ✅ 已落地
│   ├── research-group.schema.json  ✅ 已落地
│   └── lineage.schema.json         ✅ 已落地
├── papers/
│   ├── 2018.json                   ✅ 已落地（3 条真实迁移样例）
│   ├── 2015.json … 2026.json       ⏳ 逐步补齐
│   └── index.json                  ⏳ 由构建脚本生成：id → file、去重键索引
├── datasets.json                   ⏳ dataset 实体集合
├── toolkits.json                   ⏳ toolkit 实体集合
├── venues.json                     ⏳ venue 实体集合（含 45 个短名的归一）
├── research-groups.json            ⏳ research_group 实体集合
└── lineage.json                    ⏳ 研究谱系节点与边（对应 research-lineage.md）
```

读取方向永远是单向的：

```
data/*.json  ──(validate)──►  (render)  ──►  README.md / README.zh.md 的 AUTO 块
     ▲                                             │
     └──────────(check: 渲染结果 == 仓库现状)◄──────┘
```

**硬约束**：渲染器只允许写 `<!-- START ... -->` / `<!-- END ... -->` 之间的内容。这和 `render_curated_recent.py` 现有的 `replace_block()` 完全一致，因此不需要新机制。

---

## 3. 六类实体 Schema

所有 schema 使用 JSON Schema draft 2020-12，文件名 `data/schema/<entity>.schema.json`，顶层都是**记录集合**（`data/papers/YYYY.json` 是数组），单记录校验用 `#` 节点。

通用约定：

| 约定 | 规则 |
| --- | --- |
| 日期字段 | ISO `YYYY-MM-DD` 字符串（不用 epoch，保证 diff 可读） |
| 缺省值 | 一律写 `null` 或 `[]`，**不省略键**（保证 JSON diff 对齐、字段可比对） |
| 枚举 | 小写下划线英文；面向人的英文展示串（如 `TITS`）不进枚举 |
| 自由文本 | 英文 `*_en`、中文 `*_zh` 成对出现，中文缺省回落英文 |
| 溯源 | 任何从 markdown 反解出来的记录必须带 `provenance.readme_line` |

### 3.1 `paper`（核心实体，已落地 `data/schema/paper.schema.json`）

必填：`id`、`year`、`title`、`paper_url`。完整字段语义：

| 字段 | 类型 | 必填 | 枚举 / 约束 | 说明与示例 |
| --- | --- | --- | --- | --- |
| `id` | string | ✅ | `^[a-z0-9]+(-[a-z0-9]+)*-[0-9]{4}$` | 稳定 slug，见 §4：`tgcn-2018` |
| `year` | integer | ✅ | 1990–2100 | 归档 venue 的年份；纯 arXiv 用首次提交年 |
| `order` | integer | ⬜ | ≥1 | 同 `(section, year)` 内的手工排序，缺失时渲染器按 `(year, title)` 排 |
| `section` | string | ⬜ | `reviews` `deep_learning` `statistic` `time_series` `temporal_network_embedding` | 对应 README `0x00` 的 5 个子节 |
| `title` | string | ✅ | ≥5 字符 | 英文规范标题，逐字取自 README / venue |
| `title_zh` | `string` 或 `null` | ⬜ | — | 中文标题，缺省回落 `title` |
| `venue` | `string` 或 `null` | ⬜ | — | 用于 `[Venue Year]` 标签的显示串，如 `TITS`、`Arxiv` |
| `venue_id` | `string` 或 `null` | ⬜ | slug | 外键 → `data/venues.json` |
| `venue_type` | string | ⬜ | `conference` `journal` `preprint` `workshop` `thesis` `unknown` | 用于分层统计（2026 年 11 条里 10 条是期刊） |
| `paper_url` | string | ✅ | `^https?://` | 主链接，优先 DOI / 出版方 / arXiv abs |
| `alt_urls` | object | ⬜ | 值为 URL | 其余链接，键沿用 README 标签：`paper-ieee`、`review` |
| `code_url` | `string` 或 `null` | ⬜ | URL | **兼容字段**：单代码链接时的快捷方式 |
| `code_links` | array | ⬜ | `label ∈ code / code-official-tf / code-pytorch / code-not-official` | 多实现（DCRNN 有官版 TF + PyTorch 复现）走这里 |
| `data_url` / `project_url` | `string` 或 `null` | ⬜ | URL | `[[data]]` 标签、项目主页 |
| `doi` | `string` 或 `null` | ⬜ | `^10\.[0-9]{4,9}/\S+$`（小写、裸 DOI） | 去重主键之一。**注意**：对 `arxiv.org/abs/*` 形式的 URL **不要**抽 DOI——`10.48550/arXiv.xxxx` 这类数据注册 DOI 与正式 DOI 容易混判，arXiv 走 `arxiv_id` 字段单独建键，见 §7.1 第 10 条 |
| `arxiv_id` | `string` 或 `null` | ⬜ | `^[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?$` | 去重主键之一；`/pdf/1811.05320v3.pdf` → `1811.05320v3` |
| `openalex_id` | `string` 或 `null` | ⬜ | `^W[0-9]+$` | 与 `paper_search.py` 的 OpenAlex 结果对接 |
| `short_name` | `string` 或 `null` | ⬜ | — | 模型缩写，digest 表 `[short_name](url)` 用它 |
| `authors` | array<string> | ⬜ | — | 暂留空，由检索流水线回填 |
| `affiliations` | array<slug> | ⬜ | — | 外键 → `research-groups.json` |
| `task_en` / `task_zh` | `string` 或 `null` | ⬜ | — | digest 表 “Task” 列 |
| `method_tags` | array<string> | ⬜ | 受控词表（与 `traffic-monthly.json` 的 post rules 复用同一批关键词：graph / spatio-temporal / transformer / diffusion / uncertainty…） | digest 表 “Method Tags” 列 |
| `note_en` / `note_zh` | `string` 或 `null` | ⬜ | — | digest 表 “Note” 列 |
| `datasets` | array<slug> | ⬜ | — | 外键 → `datasets.json` |
| `layers` | array<enum> | ⬜ | `sota_core` `classic_baseline` `survey` `application` | 见 `02-paper-scope.md` 的分层策略 |
| `direction_tags` | array<enum> | ⬜ | `st_foundation_llm` `diffusion_probabilistic` `dynamic_graph_learning` `long_horizon_transformer` `fewshot_transfer` `privacy_federated` `benchmark_reproducibility` `imputation_data_quality` `trajectory_prediction` `od_demand` | 与 `02-paper-scope.md` 的补充方向一一对应 |
| `featured` | boolean | ⬜ | 默认 `false` | **兼容字段**：digest 表选行依据，沿用 legacy 语义 |
| `in_digest` | boolean | ⬜ | 默认 `false` | 长期字段，替代 `featured`；灰度期两者双写 |
| `status` | string | ⬜ | `active` `dead_link` `superseded` `retracted` | 链接/学术状态，取代现在“删掉或者留着” |
| `verification` | object | ⬜ | `checked_at` `link_ok` `metadata_source ∈ manual/readme_migration/openalex/arxiv/crossref/semantic_scholar` `confidence ∈ high/medium/low` | 迁移记录默认 `confidence=low`，人工核对后可升 |
| `provenance` | object | ⬜ | `source ∈ readme_migration/curated_json/manual/search_pipeline`，`readme_line`，`legacy_id` | **灰度期必填**，用于回滚与“这条从哪来”的追溯 |
| `created_at` / `updated_at` | date | ⬜ | ISO 日期 | — |

已被 `jsonschema.validate()` 实测通过的样例（`data/papers/2018.json` 首条，摘要）：

```json
{
  "id": "tgcn-2018",
  "year": 2018,
  "order": 1,
  "section": "deep_learning",
  "title": "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction",
  "venue": "TITS",
  "venue_id": "tits",
  "venue_type": "journal",
  "paper_url": "https://arxiv.org/pdf/1811.05320v3.pdf",
  "code_url": "https://github.com/lehaifeng/T-GCN",
  "code_links": [{"label": "code", "url": "https://github.com/lehaifeng/T-GCN"}],
  "arxiv_id": "1811.05320v3",
  "layers": ["classic_baseline"],
  "direction_tags": ["dynamic_graph_learning"],
  "provenance": {"source": "readme_migration", "readme_line": 89}
}
```

### 3.2 `dataset`（已落地 `data/schema/dataset.schema.json`）

必填：`id`、`name`、`signal_type`、`region`、`access_type`。

| 字段 | 类型 | 必填 | 枚举 / 约束 | 示例 |
| --- | --- | --- | --- | --- |
| `id` | string | ✅ | slug（无年份后缀） | `metr-la`、`pems-bay` |
| `name` / `name_zh` / `aliases` | string / `string` 或 `null` / array | ✅ / ⬜ / ⬜ | — | `METR-LA`；aliases 承接 README 的 `Los Angeles (METR-LA)` 写法 |
| `region` / `country` / `in_china` | string / `[A-Z]{2}` / bool | ✅ / ⬜ / ⬜ | — | `Los Angeles, US` / `US` / `false` |
| `signal_type` | string | ✅ | `speed` `flow` `occupancy` `demand_od` `demand_zone` `trajectory` `commute_flow` `poi_landuse` `multi` `unknown` | METR-LA = `speed` |
| `source_type` | string | ⬜ | `loop_detector` `camera` `gps_trace` `transit_smartcard` `bike_share` `taxi_trip` `census` `poi` `simulated` `unknown` | — |
| `nodes` | `int` 或 `null` | ⬜ | ≥1 | METR-LA 207，PEMS-BAY 325（与 README Benchmark Matrix 一致） |
| `spatial_unit` | string | ⬜ | `sensor` `zone` `grid` `road_segment` `station` `od_pair` `unknown` | — |
| `interval_minutes` | `number` 或 `null` | ⬜ | ≥0 | `5` |
| `time_range` | object | ⬜ | `start`/`end` 日期 | — |
| `missing_rate` | `number` 或 `null` | ⬜ | 0–1 | — |
| `common_horizons_minutes` | array<int> | ⬜ | — | `[15,30,60]` |
| `metrics` | array<enum> | ⬜ | `MAE` `RMSE` `MAPE` `WAPE` `SMAPE` `CRPS` `PICP` `other` | — |
| `split_policy` | string | ⬜ | `chronological` `random` `varies` `unknown` | README 0x07 明确要求 chronological |
| `access_type` | string | ✅ | `open` `registration` `request` `charged` `unknown` | INRIX 为 `charged` |
| `url` / `local_doc` | `string` 或 `null` / `string` 或 `null` | ⬜ / ⬜ | URL / `^datasets/.*\.md$` | METR-LA 的 `local_doc` = `datasets/METR.md` |
| `paper_ids` | array<slug> | ⬜ | — | 外键 → paper |
| `description_en` / `description_zh` | `string` 或 `null` | ⬜ | — | 承接 README 列表项下的缩进说明段 |
| `license` / `status` | `string` 或 `null` / enum | ⬜ / ⬜ | `active` `deprecated` `mirror_only` | — |
| `provenance` | object | ⬜ | 同 paper | — |

### 3.3 `toolkit`（已落地 `data/schema/toolkit.schema.json`）

必填：`id`、`name`、`url`、`kind`。README 里 `0x03 Toolkits`（6 条）与 `0x06 Related Repositories`（15 条）都归到这里，用 `kind` 区分。

| 字段 | 类型 | 必填 | 枚举 | 示例 |
| --- | --- | --- | --- | --- |
| `kind` | string | ✅ | `library` `benchmark_suite` `routing_engine` `tutorial_site` `related_repo` `awesome_list` | LibCity=`library`，Traffic-Benchmark=`benchmark_suite`，OSRM=`routing_engine` |
| `repo` | `string` 或 `null` | ⬜ | `owner/repo` | `tsinghua-fib-lab/Traffic-Benchmark` |
| `language` / `frameworks` | `string` 或 `null` / array | ⬜ | — | `python` / `["pytorch"]` |
| `supported_tasks` | array<enum> | ⬜ | `traffic_flow` `traffic_speed` `travel_time` `trajectory` `od_demand` `generic_st` | — |
| `bundled_datasets` / `implements_paper_ids` | array<slug> | ⬜ | — | 反向索引，供站点生成“工具→论文→数据集”跳转 |
| `last_release` / `maintained` | `date` 或 `null` / bool | ⬜ | — | 链接巡检流水线回填 |

### 3.4 `venue`（已落地 `data/schema/venue.schema.json`）

必填：`id`、`name`、`short_name`、`type`。

| 字段 | 类型 | 必填 | 枚举 | 说明 |
| --- | --- | --- | --- | --- |
| `short_name` | string | ✅ | — | **必须逐字等于 README 标签里的串**（`TITS`、`Arxiv`、`WWWC`），否则渲染不再一致 |
| `aliases` | array<string> | ⬜ | — | 收录实测到的异写：`WWWC`/`WC`/`WWW`、`IF`/`Information Fusion`、`Sensor`/`Sensors`、`Access`、`IJGIS`、`ICIKM`、`TRPET` |
| `type` | string | ✅ | `conference` `journal` `preprint` `workshop` `course_report` `unknown` | `CS224W` = `course_report` |
| `ccf_rank` / `impact_factor` | enum(A/B/C/null) / `number` 或 `null` | ⬜ | — | **纯人工标注**，禁止让流水线自动填 |
| `publisher` / `url` / `notes` | `string` 或 `null` | ⬜ | — | 45 个短名逐条建档的工作量约 1 人时 |

### 3.5 `research_group`（已落地 `data/schema/research-group.schema.json`）

必填：`id`、`name`、`org_type`。README `0x05` 现有 16 条（Company 3 + College 12 + 1 条裸 URL）。

| 字段 | 类型 | 必填 | 枚举 | 示例 |
| --- | --- | --- | --- | --- |
| `org_type` | string | ✅ | `college` `company` | `college` |
| `institution` / `pi` | `string` 或 `null` | ⬜ | — | `Tsinghua University`；`Huaiyu Wan` |
| `focus_tags` | array<enum> | ⬜ | `urban_science` `spatio_temporal` `its` `traffic_control` `autonomous_driving` `mobility_analytics` `ml_systems` | 由 README 缩进的 `Focus:` 文本归一而来（实测 12 条带说明） |
| `paper_ids` | array<slug> | ⬜ | — | 由 paper.`affiliations` 反查生成 |

### 3.6 `lineage`（已落地 `data/schema/lineage.schema.json`）

把 `research-lineage.md`（43 行，7 个阶段 + Summary）变成有向图。

必填：`id`、`title_en`、`stage`、`era`。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `stage` | integer ≥1 | ✅ | 必须与 `research-lineage.md` 小节序号一致（1..7） |
| `era` | object `{from_year, to_year}` | ✅ | 阶段起止年份（`to_year` 可 null 表示仍在延续） |
| `parent_ids` / `child_ids` | array<slug> | ⬜ | 双向边，写入时两侧都校验存在性 |
| `anchor_paper_ids` | array<slug> | ⬜ | **必须是 `data/papers/` 里真实存在的 id**，禁止指向 `[PLACEHOLDER]` |
| `direction_tags` | array<enum> | ⬜ | 复用 paper 的同一套枚举，实现“方向 ↔ 阶段”互查 |

示例（第 1 阶段，与 `research-lineage.md` “1. Early Statistical and Classical Time-Series Modeling” 对齐）：

```json
{
  "id": "statistical-baseline",
  "stage": 1,
  "era": {"from_year": 1980, "to_year": 2014},
  "title_en": "Early Statistical and Classical Time-Series Modeling",
  "title_zh": null,
  "parent_ids": [],
  "child_ids": ["graph-neural-paradigm"],
  "anchor_paper_ids": [],
  "anchor_dataset_ids": [],
  "direction_tags": []
}
```

---

## 4. id 命名规则与去重键

### 4.1 id 规则

```
<short-name-slug>-<year>
```

- `short-name-slug`：`short_name`（论文自报的模型缩写）→ 小写 → 非字母数字压成单个 `-`。`T-GCN` → `t-gcn`；`STGCN` → `stgcn`；`ASTGCN` → `astgcn`。
- 无 `short_name` 时：取标题前 3 个有效词做 slug（`Missing Value Imputation ...` → `missing-value-imputation-2018`）。
- `year`：取**归档 venue 年份**，不是 README 章节年份。这一条直接暴露了两个不一致项：T-GCN 归档年 2019 → `tgcn-2019`（README 放在 2018 段）；`Spatial-Temporal Fusion GNN` 归档年 2020 → `stfgnn-2020`（README 放在 2021 段）。**迁移时以 venue 年份为准并在 `provenance` 里记 `readme_line`**，章节归属交给 `section` + `year` 字段渲染，避免“id 随章节搬家”。
- 数据集/工具/机构：不带年份，纯 slug（`metr-la`、`libcity`、`fib-tsinghua`）。
- venue：不带年份（`tits`、`ijcai`、`arxiv`）。
- lineage：语义短语（`structure-learning`、`pretrain-transfer`）。

**为什么要 slug 而不是 DOI/哈希**：URL 里要出现（站点路由 `/paper/tgcn-2019`）、人要念、要能在 markdown diff 里一眼识别。哈希 id 满足唯一性但毁掉可读性。

### 4.2 去重键（按优先级短路匹配）

| 优先级 | 键 | 归一化规则 | 覆盖率（实测 102 条） |
| --- | --- | --- | --- |
| 1 | `doi` | 小写、去 `https://doi.org/` 前缀、去尾部 `.,;)` | 24 条（23.5%） |
| 2 | `arxiv_id` | 去 `vN` 版本号后再比 | 30 条 arXiv URL 中可提取 |
| 3 | `title_key` | NFKC 归一 → `casefold` → 所有 `\W+` 删除 → 全角/连字符（`–` `‐` `—`）统一为空 | 102 条（100%） |
| 4 | 人工别名表 | `data/aliases.json`：允许人对“同文不同题”（改名、加副标题）显式声明等价 | 手工 |

规则：

- 1/2 命中即判定为同一作品，**不做标题校验**（DOI 冲突意味着数据本身错了，直接报冲突）。
- 只有 3 命中时，额外比对 `year` 差值 ≤1 才算同一作品（防同名不同年）。
- 4 是逃生舱：任何自动判定的“存疑对”都产出 `alias_suggestion` 记录，交人工审。

**归一化函数必须与现有脚本一致**：`scripts/paper_search.py` 和 `scripts/dedupe_readme_papers.py` 都使用 `re.sub(r"\W+", "", title.lower())`。数据层沿用同一实现（抽到 `data/lib/normalize.py` 供两者 import），可以避免“数据层判重的是 A、CI 判重的是 B”。

### 4.3 双语字段回落策略

```
title_zh  == null  →  展示 title
task_zh   == null  →  展示 task_en
note_zh   == null  →  展示 note_en
name_zh   == null  →  展示 name
description_zh == null → 展示 description_en
```

- 回落发生在**渲染期**，不在数据写入期。绝不把英文回填进 `*_zh` 字段（否则无法区分“真翻译”与“还没翻译”，翻译覆盖率也就统计不出来了）。
- 渲染器必须显式打印覆盖率：`zh coverage: title 0/102, note 0/19`。
- 迁移期 `README.zh.md` 的正文仍是中文，但论文标题本身就是英文——这解释了为什么 zh 版论文列表与 en 版**逐字节相同**（实测两版条目行完全一致）。这也说明：**中文版真正的增量价值只在章节标题、说明文本和 note_zh 上**，结构化之后才有地方承载它。

---

## 5. 与现有 `updates/curated/recent-papers.json` 的合并路径（关键）

### 5.1 现有文件的角色

| 实测数据 | 内容 |
| --- | --- |
| `updates/curated/recent-papers.json` | 19 条；年份分布仅 2024(4) / 2025(4) / 2026(11)；`featured=true` 12 条 |
| 与 README 交叉核对 | 19/19 全部能在 `README.md` 中找到同标题条目；`[Venue Year]` 标签与 `paper_url` 全部一致，0 条漂移 |
| 字段集合 | `id/year/order/venue/title/paper_url/code_url/short_name/task_en/task_zh/method_tags/note_en/note_zh/featured` |
| id 合规性 | 19 个 id 全部满足 `^[a-z0-9]+(-[a-z0-9]+)*-[0-9]{4}$`（已用 jsonschema 实测） |
| 引用方 | `scripts/render_curated_recent.py`、`.github/workflows/curated-render-check.yml`、`CONTRIBUTION.md` |

也就是说：**它已经是新 schema 的一个子集投影**，不存在语义冲突，只有字段缺口。

### 5.2 合并原则

> `updates/curated/recent-papers.json` 不是“另一套数据”，它是 `data/papers/2024.json ∪ 2025.json ∪ 2026.json` 的 **legacy 视图（legacy view）**。

因此走**吸收（absorb）**而不是**并行（parallel）**：

1. **原地吸收**：19 条 legacy 记录按 §5.3 的字段映射生成 2024/2025/2026 三个年切片，写入 `data/papers/`。
2. **保留旧文件不动**（灰度期）：`updates/curated/recent-papers.json` 继续存在，但它变成 **只读镜像**——由 `data/papers/*.json` 经 `export_legacy_curated.py` 反向生成，**禁止人手改**。`curated-render-check.yml` 当前跑在它上面，不改 CI 也能继续绿。
3. **id 零迁移成本**：legacy 的 19 个 id（`unist-2024`、`graphsparsenet-2025`、`pimsct-2026`…）实测**全部满足**新 schema 的 `^[a-z0-9]+(-[a-z0-9]+)*-[0-9]{4}$`（已用 jsonschema 校验，19/19 通过）。所以 id 直接沿用，`provenance.legacy_id` 记录原值，未来切流时不会产生链接断裂。
4. **字段映射表**：

| legacy 字段 | 新字段 | 转换规则 |
| --- | --- | --- |
| `id` | `id` | 原样（已验证合规） |
| `year` | `year` | 原样 |
| `order` | `order` | 原样（决定渲染顺序） |
| `venue` | `venue` / `venue_id` / `venue_type` | `venue` 原样保留显示串；`venue_id` 由 `venues.json` 反查（`Arxiv`→`arxiv`）；`Arxiv` 映射 `venue_type=preprint` |
| `title` | `title` | 原样 |
| `paper_url` | `paper_url` | 原样 |
| — | `doi` / `arxiv_id` | 从 `paper_url` 正则提取：`doi.org/*` → DOI（12 条）；`arxiv.org/abs/*` → arXiv id（4 条）；两者都无 → `null`（3 条） |
| `code_url` | `code_url` **+** `code_links[]` | 双写：`code_url` 保留，同时生成 `[{"label":"code","url":...}]` |
| `short_name` | `short_name` | 原样 |
| `task_en`/`task_zh` | 同名 | 原样 |
| `method_tags` | `method_tags` | 原样，并与受控词表做“未收录标签”告警（不硬拦） |
| `note_en`/`note_zh` | 同名 | 原样 |
| `featured` | `featured` **+** `in_digest` | 双写，`in_digest = featured` |
| — | `section` | 统一 `deep_learning` |
| — | `title_zh`、`layers`、`direction_tags`、`datasets`、`authors`、`affiliations`、`alt_urls`、`data_url`、`project_url`、`openalex_id` | 迁移期一律 `null`/`[]`，后续人工/流水线补 |
| — | `verification` | `{checked_at: 迁移日, link_ok: true, metadata_source: "readme_migration", confidence: "medium"}`（因为 legacy 经人工整理过） |
| — | `provenance` | `{source: "curated_json", legacy_id: <旧 id>, readme_line: <对应行号>}` |

5. **反向导出必须逐字段可还原**：`export_legacy_curated.py` 输出的 JSON 要能与仓库现有文件**逐字节相等**（`--check` 模式断言）。做不到就说明映射有信息丢失，就地修 schema 而不是接受漂移。这是“能吸收”的唯一客观判据。

> 实测：用上述映射把 19 条 legacy 记录转成新 schema 记录后，`jsonschema.validate` 全部通过（`jsonschema` 12 条 DOI、4 条 arXiv id、12 条 featured 全部保真）。反向等价性要在 Phase 2 实现导出脚本时用 `--check` 钉死。

### 5.3 分阶段合并计划

| 阶段 | 动作 | 验收 |
| --- | --- | --- |
| P0（本文档） | 定 schema + 3 条真实样例 + 合并映射表 | `python3 -m json.tool` 全部通过；样例过 `jsonschema` |
| P1 | 一次性反解脚本把 102 条 paper 全部落入 `data/papers/YYYY.json` | 每条都带 `provenance.readme_line`；`--check` 渲染回原文等于原文件 |
| P2 | 吸收 19 条 legacy，实现 `export_legacy_curated.py --check` | 导出结果与 `updates/curated/recent-papers.json` 字节相等 |
| P3 | 渲染器改为读 `data/papers/*.json`；`updates/curated/recent-papers.json` 降级为生成物 | `render_curated_recent.py --check` 在两个源上输出同一 README |
| P4 | 删除 legacy 文件（单独 PR，保留一个 release 周期的 git 历史可达） | CI 全绿；回滚点保留 tag |

---

## 6. 文件组织：按年切片 vs 大单文件

**结论：按年切片 `data/papers/YYYY.json`（采用）；再补一个构建期生成的 `data/papers/index.json` 作为只读索引。不使用单文件 `papers.json`。**

### 6.1 依据（用本仓库的真实数字算）

| 维度 | 单文件 | 按年切片 | 本仓库证据 |
| --- | --- | --- | --- |
| 行数规模 | 一条记录约 30 行 → 102 条 ≈ 3000 行单文件 | 最大年份 2020 的 26 条 ≈ 800 行 | 当前 `updates/curated/recent-papers.json` 306 行 / 19 条已验证 |
| 并行编辑冲突 | 每次加论文都改同一文件，**必冲突** | 加 2026 论文只碰 `2026.json` | 本仓库正在多 agent 并行写作（`docs/evolution/` 03–09 并行），冲突是现实问题 |
| diff 噪声 | 插入一条会重排后续行号 | 只在年份文件内局部变化 | — |
| 增长曲线 | 无界 | 每年新增 1 文件 | 2025 年 4 条、2026 年 11 条（且 2026 只过了部分月份），按年切片可承载 10× 增长仍有约 800 行/年 |
| 加载成本 | 全量解析 | 按年份窗口加载 | `render_curated_recent.py` 默认 `--recent-window-years 2`、`--digest-window-years 3`，**只关心最近 2–3 年**，按年切片天然匹配 |
| 热点统计 | 需全量 | 需全量 → 由 `index.json` 提供 | `yearly_hotspots.py` 需要全量，索引文件解决 |

### 6.2 具体约定

- 文件名 `YYYY.json`，数组，元素为 paper 记录；数组内按 `order` 升序、其次 `title`。
- `data/papers/index.json` 结构：

```json
{
  "generated_at": "2026-11-24",
  "count": 102,
  "by_year": {"2018": 11, "2019": 20, "2020": 26},
  "by_id": {"tgcn-2018": "2018.json"},
  "dedupe": {
    "doi": {"10.1109/tits.2019.2906070": "tgcn-2018"},
    "arxiv": {"1811.05320": "tgcn-2018"},
    "title": {"tgcnatemporalgraphconvolutionalnetworkfortrafficprediction": "tgcn-2018"}
  }
}
```

- `index.json` 是**构建产物**，与 README 同级对待：只由脚本写，PR 里出现手改即拒。
- 切分阈值写死为“按年”；若未来某年超过 1500 行，再按 `YYYY-MM` 或 `YYYY-part1/2` 二次切分，**对外 id 不变**（id 不含文件名）。

---

## 7. 迁移策略

### 7.1 一次性反解脚本（`scripts/migrate_readme_to_json.py`，Phase 1 新建，不改现有脚本）

只读 `README.md`（en 为主源，zh 用于交叉校验），产出 `data/papers/YYYY.json`。

**处理顺序**

```
Step 1  段落状态机：定位 `## 0x00` → 识别 `###` 子节 → 识别 `#### YYYY`
Step 2  逻辑行聚合：把跨行的编号条目合并成一条
Step 3  条目正则解析：标签 / 标题 / 链接组
Step 4  字段派生：venue_id、doi、arxiv_id、code_links、alt_urls
Step 5  id 生成 + 去重键计算 + 冲突检查
Step 6  写入年切片 + 追加 provenance.readme_line
Step 7  回归：把 JSON 渲染回 markdown，与原文逐行 diff
```

| 正则（实测覆盖 102/103 个 `[[paper]]` 条目） | 说明 |
| --- | --- |
| `ITEM = ^\s*(\d+)\.\s+\[([^\]]+)\]\s+(.+?)\s+\[\[paper\]\]\(([^)]+)\)(.*)$` | 100 条单行条目 |
| `ALT  = ^\s*(\d+)\.\s+\[([^\]]+)\]\s+\[(.+?)\]\((https?://[^)\s]+)\)` | L251 旧格式；`dedupe_readme_papers.py` 里已有同名兜底正则，可直接复用命名 |
| `TAG  = ^(?P<venue>.*?)\s*(?P<year>20\d{2})$` | `TITS 2019` → venue=`TITS`、year=`2019` |
| `LINK = \[\[(?P<label>[a-z0-9\-]+)\]\]\((?P<url>https?://[^)\s]+)\)` | 8 种标签全覆盖；`[[code]]` 紧邻前链接时仍能匹配 |

> 计数口径：上面 102 是「能同时被 `ITEM`/`ALT` 匹配的条目数」，103 是 `[[paper]]` 标签出现次数（因为 L251 那条标题自带链接、没有 `[[paper]]` 标签，标签计数里另计）。两者差 1 属于口径差异，不是漏解析。

**边界情况清单（每一条都在本仓库出现过）**

| # | 情况 | 出现位置 | 处理 |
| --- | --- | --- | --- |
| 1 | 标题跨行折断，续行无缩进 | L170–171 | 逻辑行聚合：若行首匹配 `^\s*\d+\.\s+\[\S+\]\s+` 且**不含 `[[paper]]`**，则吞掉后续至出现 `[[paper]]` 的行，用单空格拼接（聚合后得到 1 条，总条目 103→102 中的 1 条即此） |
| 2 | `[[code]]` 与前链接紧邻，无空格 | L115 | `LINK` 正则只要求 `\[\[`，不依赖前置空格；`tail` 用 `.*$` 兜 |
| 3 | 旧格式 `1. [KDD 2021] [Title](url)` | L251（Time Series / Temporal Network Embedding 节） | 兜底正则 `^\s*\d+\.\s+\[([^\]]+)\]\s+\[(.+?)\]\((https?://[^)\s]+)\)`，走 `section=temporal_network_embedding`，`paper_url` 取该 URL |
| 4 | 非 ASCII 连字符（`–` U+2013、`‐` U+2010） | L104、L157、L183 | `title` **原样保留**（不“修正”）；`title_key` 归一化时 NFKC + `\W+` 删除，连字符一律消失 |
| 5 | 省略号占位 `11. ...` / `20. ……` | 各节末 | 识别为占位符，**不计入条目**，单独计数并在报告中列出（实测 4 处） |
| 6 | 裸 URL 条目 | L366、L415 | 单独归类：L366 → research_group（`url` 直填、`name` 从域名派生并标 `needs_review`）；L415 → toolkit（`kind=related_repo`） |
| 7 | 章节年 ≠ venue 年 | L89、L156 | `year` 取 venue 年；`provenance.readme_line` 保留；报告里列为 `year_mismatch` |
| 8 | `venue` 含空格/`&`（12 个短名） | `Information Fusion`、`Complex & Intelligent Systems`、`PLoS ONE`… | `TAG` 用贪婪 `.*?` + 末尾 4 位年份锚定，允许空格与 `&` |
| 9 | 同一列表项下挂着缩进说明段（2–3 空格） | 数据集/机构节，实测 6+12 条 | 归入 `description_en`；`Focus:` 前缀剥离后拆成 `focus_tags` |
| 10 | arXiv `/pdf/` 带版本号 | L89 `1811.05320v3`、L108 `1803.01254v2` | `arxiv_id` 保留版本串，`arxiv_key` 去版本用于去重 |
| 11 | 无 DOI 的 78 条 | IEEE/ACM/MDPI 大多数 | `doi=null`，不猜；靠 `title_key` 去重；后续由 OpenAlex 批量回填 |
| 12 | 标题自带冒号/全大写缩写 | `T-GCN:`、`GMAN:` | `short_name` 取标题冒号前的大写 token，若为纯大写且长度 ≤12 则采用，否则 `null`，交人工 |

**幂等与可重入**

- 脚本不修改 README，只写 `data/`。
- 二次运行必须产出**逐字节相同**的 JSON：所有数组稳定排序、所有键顺序按 schema `properties` 顺序输出（`json.dumps(..., sort_keys=False)` + 有序 dict 构造）、日期取 `--as-of` 参数而非 `now()`。
- 已存在记录以 `provenance.source == "manual"` 为准，**不覆盖**人工修正（`--force` 才覆盖，且写 `.bak`）。

### 7.2 灰度期：三源双写与切流规则

**灰度期的唯一铁律：`data/` 是源，README 是产物，`updates/curated/recent-papers.json` 是 `data/` 的导出快照。**

| 时间段 | 事实源 | 允许的手工编辑对象 | README 产生方式 | CI |
| --- | --- | --- | --- | --- |
| G0（当前） | markdown（AUTO 块除外，其源是 legacy JSON） | `README*.md` 非 AUTO 区、`updates/curated/recent-papers.json` | 手写 + `render_curated_recent.py` 渲染 AUTO 块 | 现状两套 check |
| G1（P1–P2 完成） | **双源**：markdown 仍是事实源，`data/` 是影子副本 | 仅 markdown 与 legacy JSON | 手写；额外跑 `migrate --check`，断言 `data/` 与 markdown **完全一致** | 新增 `data-shadow-check`：任何一处漂移即红 |
| G2（P3 起） | **`data/` 是唯一事实源** | **只能改 `data/papers/*.json`** | 全部由 `render` 从 `data/` 生成 | `render --check` + `jsonschema` + `dedupe-check` + `export_legacy_curated --check` |
| G3 | `data/` | 同 G2 | 同 G2 | 删除 legacy 文件与 `curated-render-check` 对新数据的依赖 |

**G1 → G2 的切流判据（全部满足才切，缺一不切）**

1. `data/papers/*.json` 条目数 == markdown 论文条目数（当前目标 102）。
2. `migrate --check` 连续 2 次全绿（说明反解稳定，不是巧合）。
3. `render` 从 `data/` 渲染出的 markdown 与仓库现状**逐行相同**（不是“数量相同”）。**注意**：当前 `check_readme_sync.py` 只比数量，这正是要升级的点。
4. `export_legacy_curated --check` 与 `updates/curated/recent-papers.json` 字节相等。
5. 至少一个完整的自然月（含一次 `monthly-paper-search.yml` 定时跑）没有出现“读了 markdown 又写回 markdown”的循环副作用。

**G2 的双写规则（仅限 2 处，其余一律不双写）**

- `code_url` 与 `code_links[0].url` 双写：因为 `render_curated_recent.py` 的 `format_recent_item()` 只读 `code_url`。
- `featured` 与 `in_digest` 双写：因为 digest 选行逻辑只读 `featured`。

**AUTO 块的硬约束（三条，任何阶段不变）**

1. 只重写 `<!-- START AUTO RECENT PAPERS -->` … `<!-- END AUTO RECENT PAPERS -->`、`<!-- START AUTO CURATED DIGEST TABLE -->` … `<!-- END AUTO CURATED DIGEST TABLE -->` 之间。
2. `replace_block()` 的 `\n + new_block` 拼接方式保持现状，否则 `curated-render-check.yml` 的 `--check` 会假绿/假红。
3. 渲染器输出必须**按语言**取字段：en 用 `task_en`/`note_en`，zh 用 `task_zh`/`note_zh`（缺省回落），两侧行数永远相同——这是 `check_readme_sync.py` 现状能过、G2 后仍然能过的前提。

### 7.3 回滚方式

回滚必须在 5 分钟内可完成，且不需要“人工回忆改了什么”。

| 层级 | 手段 | 触发条件 |
| --- | --- | --- |
| L1 渲染回滚 | `git checkout -- README.md README.zh.md`（`data/` 保留） | 渲染产出异常、AUTO 块格式坏 |
| L2 数据回滚 | `git revert <commit>`（只回 `data/` 提交） | schema 变更引入批量错误 |
| L3 源回滚 | 把 `data/` 目录整体移回 `data.deprecated/`，恢复「G1 双源」：markdown 重新成为事实源，CI 降级为 `migrate --check` 告警而非失败 | 切流后出现无法在 1 小时内修好的系统性偏差 |
| L4 完全回滚 | 回到 G0：`git checkout <tag-before-migration> -- README.md README.zh.md updates/curated/recent-papers.json scripts/` | 不可抗力 |

支撑手段：

- **打 tag**：切流前打 `pre-data-source-cutover`，P4 结束后打 `data-source-v1`。
- **快照文件**：反解脚本首次运行把 `README.md` / `README.zh.md` / legacy JSON 的 sha256 写进 `data/.migration-manifest.json`，回滚后可用它证明“真的回到了原点”。
- **可逆性证据**：`migrate` 与 `render` 必须互为逆运算（`render(migrate(x)) == x`）。这个等式本身就是回滚能力的证明，应作为 G1 的常规 CI 断言而不是一次性检查。

---

## 8. 落地清单（当前已完成 / 待办）

已完成（本次写入）：

- `data/schema/paper.schema.json`、`dataset.schema.json`、`toolkit.schema.json`、`venue.schema.json`、`research-group.schema.json`、`lineage.schema.json`
- `data/papers/2018.json`（README `#### 2018` 前 3 条真实迁移：`tgcn-2018` / `multi-view-imputation-2018` / `stgcn-2018`，含 paper/code/review 链接）
- 验证：7 个文件全部通过 `python3 -m json.tool`；paper 样例通过 `jsonschema` 校验；19 条 legacy 记录按 §5.2 映射后亦全部通过校验；6 类实体各有一条样例通过各自 schema。

待办（P1 起）：

- `data/papers/2015…2026.json` 全量（含 `index.json`）
- `data/datasets.json`、`data/toolkits.json`、`data/venues.json`（45 个短名）、`data/research-groups.json`、`data/lineage.json`
- `scripts/migrate_readme_to_json.py`、`scripts/export_legacy_curated.py`、`scripts/render_from_data.py`、`scripts/validate_data.py`（**新增文件，不改现有脚本**）
- CI 新增 `data-validate.yml`（jsonschema + 去重 + 交叉引用完整性）