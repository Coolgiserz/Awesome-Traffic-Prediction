# 06 · 数据更新流水线升级方案

> 目标：把「月度爬一批候选 → 人工挑几条贴进 README」的线性脚本，升级成 **fetch → normalize → render → publish** 四段式可复现流水线，并把 SOTA/Leaderboard 这一块从依赖已停服的外部站点迁回可控数据。
>
> 本文是**方案文档**，不修改任何脚本、workflow 或 README。所有数字均来自仓库现状实测（2026-11-24 快照），文末给出取证方式。

---

## 1. 现状（如实描述）

### 1.1 触发与编排

`.github/workflows/monthly-paper-search.yml` 是唯一的数据型 workflow：

| 项 | 现状 |
| --- | --- |
| 触发 | `schedule: cron "0 2 1 * *"`（每月 1 日 02:00 UTC）+ `workflow_dispatch` |
| 权限 | `contents: write`、`pull-requests: write` |
| 运行环境 | `ubuntu-latest` + `actions/setup-python@v5`（`python-version: "3.x"`，非锁定版本） |
| 额外依赖 | `pip install wordcloud matplotlib`（`paper_search.py` 本身零第三方依赖） |
| 产出方式 | `peter-evans/create-pull-request@v6` 开分支 `automation/monthly-paper-refresh` → base `main` → `delete-branch: true` |

单个 job `refresh-paper-candidates` 里串行 6 步：`Validate domain config` → `Run monthly search`（内含 search + dedupe + 手写 heredoc 拼 candidates 文件）→ `Generate word cloud` → `Generate yearly hotspots` → `Detect changes` → `Create pull request`。

**关键点：检索、去重、渲染候选清单、词云、年度热点全部在同一个 job 内完成，任何一步抛错都会让整条流水线失败、当月无产出。**

### 1.2 检索源：实际是单源

配置 `scripts/configs/traffic-monthly.json`:

- `search.sources: ["openalex"]` —— **只有 OpenAlex**。
- 5 条 query、`limit_per_query: 25`、`output_limit: 25`、`year_range.mode: rolling` / `lookback_years: 3`（2026 年运行 → 2024–2026）。
- `post.rules[]` 4 条：`domain`(required, w=1.4) / `intent`(required, w=1.3) / `method-boost`(w=0.35) / `non-road-exclude`(exclude)。
- `post.min_score: 2.0` → 因两条 required 规则权重合计 2.7，**实际几乎只能靠 required 规则过线**，`min_score` 对「不命中的论文」基本无区分力，只对命中 required 的论文做排序门槛。
- `rank.recency_weight: 0.35`、`source_weights: {openalex: 0.2, arxiv: 0.1}`。
- `runtime`: `max_requests: 80`、`min_interval_seconds: 0.8`、`max_retries: 2`、`retry_base_seconds: 2.0`、`cache_dir: .cache/paper-search-http`、`cache_ttl_hours: 48`、`source_min_interval_seconds: {openalex: 0.8}`。

`scripts/paper_search.py` **代码层面已经支持 4 个源**（`SUPPORTED_SOURCES = ("openalex", "arxiv", "semantic_scholar", "crossref")`），也有各自的检索函数，但当前配置只启用了 1 个。所以「多源」的缺口**在配置而不是在代码**——这是本方案里性价比最高的改动点。

### 1.3 已有但未充分利用的工程能力

`paper_search.py` 的 `HttpClient`（约 L129–L245）已经实现了这一套：

- **按源限速**：`_pacing_sleep()` 取 `max(min_interval_seconds, source_min_interval_seconds[source])`，默认 openalex/arxiv 0.8s、semantic_scholar 1.2s、crossref 1.0s。
- **重试+退避**：`RETRYABLE_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}`，尊重 `Retry-After` 头，否则 `retry_base_seconds * 2**attempt`。
- **请求预算**：`max_requests` 硬上限，超限抛 `RuntimeError("request budget exceeded (N)")`。
- **本地响应缓存**：`.cache/paper-search-http/<sha1(url)>.json`，带 `fetched_at_utc` 与 TTL（默认 24h，本仓配置 48h）。
- **统一 UA**：`USER_AGENT = "awesome-traffic-prediction-search/0.4 (mailto:maintainers@awesome-traffic-prediction.invalid)"`，Crossref 请求另外带 `mailto` 参数（符合 polite pool 惯例）。
- **报告**：`--report-json` 输出 config 快照、runtime 快照、`requests_made`、`cache_hits`、`source_errors`、逐条 `score / matched_rules / recency_component / source_component`。

实测（本次运行）：

```bash
python3 scripts/paper_search.py --config scripts/configs/traffic-monthly.json \
  --max-requests 4 --cache-dir /tmp/ps-cache --out /tmp/ps-out.md --report-json /tmp/ps-report.json
# => Wrote 5 entries
# report.runtime: requests_made=1, cache_hits=4
# report.source_errors: 0（预算耗尽被吞进 warning，返回 0）
```

注意两点行为：

1. **预算耗尽不致命**：`run_source_search` 的异常被 catch 进 `source_errors`，只要已有论文就 `return 0`。CI 里这会被 `if: steps.changes.outputs.changed == 'true'` 掩盖成一个「论文变少了但没人发现」的静默降级。
2. **缓存命中也占 `requests_made` 之外**：`cache_hits` 单独计数，不进预算，符合预期。

### 1.4 去重键现状

两处都只用了 **归一化标题 + DOI** 两个键：

- `paper_search.py::dedupe_papers()`（源内去重）：`normalize_title()` = `re.sub(r"\W+","",title.lower())`；`normalize_doi()` 会剥离 `https://doi.org/`、`doi:` 等前缀并正则提取 `10.\d{4,9}/...`。**先匹配标题键就 continue，不看 DOI**。
- `scripts/dedupe_readme_papers.py`（对 README 去重）：同样 title + DOI，理由码 4 种：`existing_doi` / `existing_title` / `candidate_dup_doi` / `candidate_dup_title`。

缺口（实测）：

- **arXiv 版本与正式发表版被判为两篇**：`UniST` 的 `10.48550/arxiv.2402.11838` 与 `COOL` 的 `10.48550/arxiv.2403.01091` 在 OpenAlex 里分别有 `W4391987728`（cited_by=265）与 `W4392382537`（cited_by=86），但 README 里 `COOL` 的 DOI 是 `10.1016/j.inffus.2024.102341`（Information Fusion 版）——标题归一化相同但 DOI 不同，是两个不同 DOI 指向同一工作。只比 DOI 会漏、只比标题会在标题微改（`Trafffic`、副标题增删）时漏。
- **标题归一化过弱**：`re.sub(r"\W+","")` 保留全部字母数字，`Spatial-temporal` 与 `Spatio-Temporal`、`Traffic Flow Prediction` 与 `Traffic-Flow Prediction` 会得到不同 key。
- **跨文件（README.md / README.zh.md）不互相判重**：两文件各自 102 条论文条目，实测两边内部各 0 条 title 重复、0 条 DOI 重复，但脚本没有检查「EN 与 ZH 之间同一条目是否对应」的能力（这属于 07 文门禁范畴）。

### 1.5 候选落 PR 与人工合并

- 产出文件：`updates/monthly-paper-candidates.md`，结构固定为 `# Monthly Paper Candidates` + `Generated on: ... UTC` + 一句免责说明 + 去重后的编号列表。
- 当前内容只有 **4 条**（2026-10-01 生成），且质量明显不稳：第 3 条 `Explainable Machine Learning for Predicting Thermal Discomfort from Multipollutant Exposure on Dhaka Roadways`、第 4 条 `Road freight transport in Spain, 2012–2024 ... SARIMA forecasts` 都属于「标题含 traffic/road + forecast」但**不是交通流预测**的误召回。规则层 `non-road-exclude` 只挡了航空/水运/电力，没挡「交通政策/货运/热舒适」。
- 合并路径：**纯人工**。PR body 只写「Please review candidate relevance before merging.」，没有任何勾选清单、没有「哪些条目进 0x00 的哪一年小节」的映射建议。

### 1.6 渲染链路现状

存在**两条互不相干的渲染路径**：

| 路径 | 数据源 | 产物 | 是否在 CI 校验 |
| --- | --- | --- | --- |
| A. 结构化渲染 | `updates/curated/recent-papers.json`（19 条，12 条 `featured:true`） | README 两个 auto 块（`START/END AUTO RECENT PAPERS`、`START/END AUTO CURATED DIGEST TABLE`） | 是，`curated-render-check.yml` 跑 `--check` |
| B. Markdown 文本生成 | `updates/monthly-paper-candidates.md` | `updates/wordcloud/*`、`updates/hotspots/*` | 否，只在 monthly job 里就地生成 |

- `render_curated_recent.py` 只替换两个 marker 之间的内容；marker 缺失/顺序错会抛 `ValueError("Markers not found or invalid order")`。
- `render_curated_recent.py --check` **只在内容不一致时 `return 1`**，它无法区分「JSON 变了没重渲染」与「有人手改了 auto 块」（见 07 文）。
- 词云/热点读的是 **markdown 文本**（`yearly_hotspots.py::parse_candidates()` 用正则解析 `N. [Venue Year] Title [[paper]](...)`），不是 JSON。也就是说热度图反映的是「README 里写了什么」，而 `monthly-paper-candidates.md` 只有 4 条时，`yearly_hotspots.py --input README.md` 读的是 README（workflow 里 `--input README.md`），两个产物的语义并不对称。
- 词云/热点依赖 `matplotlib` + `wordcloud`，是**导致 CI 需要 pip 安装第三方包**的唯一原因；也是 `updates/wordcloud/` 已经堆到 **3.4 MB**（含按月归档 PNG `monthly-paper-wordcloud-2026-02.png`、`2026-10.png`）与 `updates/hotspots/` 808 KB 的原因。

### 1.7 `opencode.yml` 的定位

`on: issue_comment / pull_request_review_comment`，`if:` 匹配评论里以 `/oc` 或 `/opencode` 开头/包含，跑 `anomalyco/opencode/github@latest`，`model: kimi-for-coding/k2p6`，密钥 `secrets.KIMI_API_KEY`。这是**人触发**的 AI 助手通道，与月度数据流水线无关；它的存在意义是：如果将来引入 LLM 辅助打标签（见 §4），**可以复用这条已有的「API key + Action」骨架**，而不必新开一套密钥管理。

---

## 2. 改进：多源交叉与优先级

### 2.1 每个源的角色定位

不要把四个源当成「四个一样的搜索器」并联，而是按**互补职责**分工：

| 源 | 角色 | 用它拿什么 | 不要用它拿什么 | 速率/配额约束（代码内建） |
| --- | --- | --- | --- | --- |
| **arXiv**（`export.arxiv.org/api/query`） | **最新预印本发现** | 首次提交日期、arXiv id、`arxiv:doi`、summary 摘要 | 期刊卷期、正式发表 venue | `source_min_interval_seconds.arxiv = 0.8`；`per_page ≤ 100`；单次最多 4 页（代码 `max_pages = min(4, ...)`） |
| **OpenAlex**（`api.openalex.org/works`） | **元数据主源 + DOI/引用数** | DOI、`publication_year`、`primary_location.source.display_name`、`abstract_inverted_index`、**`cited_by_count`**、`ids.openalex` | 摘要质量（倒排索引重建后常有断词） | `0.8`；`per-page ≤ 50`；单次最多 4 页 |
| **Crossref**（`api.crossref.org/works`） | **DOI 权威兜底 + 卷期/页码/类型** | `DOI`、`container-title`、`type`（已过滤 `journal-article/proceedings-article/posted-content`）、`is-referenced-by-count` | arXiv 预印本（实测 `10.48550/arxiv.*` 在 Crossref **404**） | `1.0`；`mailto` polite pool；`rows ≤ 100` |
| **Semantic Scholar**（`graph/v1/paper/search`） | **补齐引用/被引与开放获取链接** | `citationCount`、`externalIds.DOI`、`venue`、`openAccessPdf` | 作为每月的稳定主力源 | `1.2`；**无 key 时极易 429，重试也只有 2 次** |

### 2.2 优先级与降级顺序

建议按「**必须** → **可缺** → **可选** → **尽力**」四级跑：

1. **OpenAlex（必须）**：唯一同时给 DOI + venue + 年份 + 引用数的源，也是 dedup 与 leaderboard 的基准。失败 → 本次运行标 `degraded`，仍发布，但 PR 标题加 `[degraded]`。
2. **arXiv（必须）**：唯一能稳定拿到「本月新预印本」的源。失败 → 同上标记，但**不阻断**（因为 arXiv 本月没新论文是常态）。
3. **Crossref（可缺）**：只用于给已有 DOI 补 `container-title` / `type` / 卷期，以及给 OpenAlex 未收录的 DOI 兜底。失败 → 跳过 DOI enrichment，不影响主列表。
4. **Semantic Scholar（可选，默认关闭）**：只在需要 `citationCount` 做排行榜时才开，且必须配 `S2_API_KEY`（`HttpClient` 目前不支持自定义 header，需要小改：给 `fetch_text` 增加 `headers` 参数）。无 key 时保持 `search.sources` 不含它。

### 2.3 Semantic Scholar 限流的具体注意事项

- 现状 `RuntimePolicy.max_retries = 2`，最坏情况对 S2 只重试 2 次，然后整源失败；`score_papers` 不会因缺源而报错，只是该源论文消失。**这意味着 S2 抖一下，排行榜就少一批论文而没人知道。**
- 建议：把 S2 的 `source_min_interval_seconds` 提到 **≥ 3.0**，`max_retries` 单独提到 3，且**只允许它在「引用数刷新」这一条独立作业里跑**，不进月度主检索。
- 若确实要无 key 跑：把 S2 结果当作**加分项而非必需项**——`rank.source_weights.semantic_scholar` 设 0，只在 enrichment 阶段合并，不做独立的 query 搜索。
- 记录 `source_errors` 到 PR body（详见 §6），让「S2 本周全灭」可见。

### 2.4 建议的配置改法（示意，不在本文落地）

```jsonc
"search": {
  "sources": ["openalex", "arxiv", "crossref"],   // 月度主链路
  "limit_per_query": 25,
  "output_limit": 40
},
"runtime": {
  "max_requests": 140,                            // 80 → 140，源多了预算要跟着涨
  "source_min_interval_seconds": {
    "openalex": 0.8, "arxiv": 0.8, "crossref": 1.0, "semantic_scholar": 3.0
  },
  "cache_ttl_hours": 48
}
```

预算估算（实测口径）：5 query × 3 源，每源每 query 最多 4 页，最坏 60 次请求；缓存命中通常能压到 20 次以内。`max_requests: 140` 是留了 2 倍余量的安全值，而不是「需要 140 次」。

### 2.5 跨源合并策略

同一工作可能被 arXiv（预印本）与 OpenAlex（期刊版）各召回一次。合并规则（写进 normalize 段）：

1. 先按 **arXiv id** 合并（arXiv 版 `doi = 10.48550/arxiv.<id>`，或 `Paper.url` 含 `arxiv.org/abs/<id>`）。
2. 再按 **DOI 归一化值** 合并。
3. 再按 **title_fuzzy_key + first_author_family + year** 合并（§3）。
4. 合并后的记录保留：**最早出现的 source 作为 `first_seen_source`**，**OpenAlex 的 DOI/venue/cited_by 作为权威字段**，**arXiv 的 abs 链接作为 `alt_urls.arxiv`**。

---

## 3. 改进：去重键增强 + 稳定排序

### 3.1 五段式 dedup key

替换当前「title 或 DOI 命中即弃」的二选一逻辑，改为**分层判决**：

```text
key_1 = arxiv_id      # 从 doi:10.48550/arxiv.X / arxiv.org/abs/X 提取，去掉版本号 vN
key_2 = doi           # normalize_doi() 现有实现即可（已剥离前缀、正则提取、小写）
key_3 = title_norm    # 归一化标题（见 3.2）
key_4 = author_family +# 首作者姓（小写、去变音符、去 "et al."）
key_5 = year          # 出版年（arXiv 用首次提交年）
```

判决顺序与理由码（对应 `dedupe_readme_papers.py` 现有的 `reason` 字段扩展）：

| 命中 | reason | 处置 |
| --- | --- | --- |
| `key_1` 相同 | `dup_arxiv_id` | 丢弃，保留已存在记录的 `alt_urls.arxiv` |
| `key_2` 相同 | `dup_doi` | 丢弃；若 records 的 venue 不同，把新 venue 记进 `also_published_as` |
| `key_3` 相同 且 `key_5` 相同 | `dup_title_year` | 丢弃，`confidence: high` |
| `key_3` 相同 且 `key_4` 相同 | `dup_title_author` | 丢弃，`confidence: medium`（防标题改字但同作者同工作） |
| `key_3` 相同 但 `key_4`/`key_5` 都不同 | `title_collision` | **保留两条**，写进 `updates/reports/title-collisions.json` 交人工裁决 |

关键改动：**「key_3 相同」不再单独作为丢弃依据**，必须配合 author 或 year 才能自动丢。否则 `COOL`（arXiv 2024 版）与 `COOL`（Information Fusion 版，DOI `10.1016/j.inffus.2024.102341`）这类「同工作多版本」会被错误地当成一个新条目，或反过来两个都进表（重复条目）。实测这两种 DOI 在 README 里只保留了一个，说明现状靠人工兜住了，一旦自动化就会暴露。

### 3.2 title 归一化增强

现状 `re.sub(r"\W+", "", title.lower())` 有三个已知问题：不折叠连字符/复合词、不去停用词、不处理副标题。建议：

```python
def title_norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))  # 去变音符
    t = html.unescape(t).lower()
    t = t.replace("–", "-").replace("—", "-")                 # en/em dash → hyphen
    t = re.sub(r"\b(spatio|spatial)[\s\-]+temporal\b", "spatiotemporal", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)                          # 保留词边界
    t = re.sub(r"\b(a|an|the|of|for|on|in|and|to|with|via)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()
```

对照当前 README 的真实痛点：

- `Spatial‐temporal attention wavenet` (L157) 用的是 **U+2010 连字符**，`\W+` 会把它吃掉变成 `spatialtemporal`，与正常写法的 `spatiotemporal` 不相等；
- `Trafffic Flow Prediction`（`GraphSparseNet` 标题，JSON 里就是 `Trafffic` 三个 f）——归一化**不能也不该**修拼写错误，要在 07 文的「拼写门禁」里报出来，而不是在 dedup 里悄悄抹平。

### 3.3 首作者提取

现有 `Paper` dataclass **没有 authors 字段**（只有 `title/year/venue/url/source/doi/abstract`）。要落地 `key_4` 必须扩字段：

- OpenAlex：`authorships[0].author.display_name`
- arXiv：`a:author/a:name` 第一个
- Crossref：`author[0].family`
- S2：`authors[0].name`

同时 `--report-json` 的 `papers[]` 也要带 `authors: []`（当前不带）。这是**流水线升级里唯一一处必须动 `paper_search.py` 结构的地方**，其余都是配置或新脚本。

### 3.4 稳定排序（可复现的必要条件）

现状 `score_papers()` 的 `cmp_items` 已经比 `functools.cmp_to_key` 逐字段比较，兜底是 `title.lower()` 字典序——**这一步本身是稳定的**。不稳定来自三处：

1. **输入顺序不稳定**：多源并发/多页拉取顺序变了，`dedupe_papers()` 的「先到先留」就变了，保留哪一条（arXiv 版还是期刊版）会跟着变。
2. **`score` 是浮点累加**：`0.1 + 0.2 != 0.3` 级别的差异会让同分论文换位。当前写的是 `round(score, 6)`，刚好够掩盖，但一旦加更多规则权重就会暴露。
3. **`output_limit` 截断**：`out[: cfg.output_limit]` 在边界同分时裁掉谁取决于前两步。

修复方案：

- **source 遍历顺序固定**：`for source in cfg.sources` 之外，把结果按 `(source_priority, arxiv_id or doi or title_norm)` 显式排序后再进 dedup，而不是依赖 append 顺序。
- **分数整数化**：所有 rule weight 与 rank weight 乘以 1000 后转 int 累加，比较时用整数；`recency` 也先量化到 1/1000。
- **最终排序键加唯一兜底**：`(score_int desc, year desc, title_norm asc, doi_or_id asc)` —— 最后一段保证全序，消除「同分同标题不同源」的不确定性。
- **`generated_at_utc` 不进内容**：`updates/monthly-paper-candidates.md` 头部的 `Generated on: <时间>` 会导致**每次运行都产生 diff**。建议把时间戳移到 `updates/reports/<run>.json`，正文保持纯净，这样才能用 `git diff --quiet` 判「有无实质变化」——现有 `Detect changes` 步骤正是因为这个时间戳而永远 `changed=true`。

> 这一条是**低成本高收益**的：不移除时间戳，「有变化才开 PR」这个 intent 永远失效，等于每月无脑开一个 PR。

---

## 4. 自动打标签（task / method_tags）的两条路

当前 `method_tags` 由人工在 `updates/curated/recent-papers.json` 里写，19 条记录产生了 **33 个不同标签**（`Attention` / `Self-Attention` / `GCN` / `STGNN` / `STGCN` / `GCN-LSTM` 并存），已经出现同义分裂的苗头。任务标签 `task_en/task_zh` 同样自由文本（`Traffic forecasting` vs `Traffic flow forecasting` vs `Urban traffic forecasting`）。

两种可行路线，**必须择一作为主路径**：

### 4.1 路线 A：规则词典（推荐作为默认）

**做法**：把标签体系固化成一份 `data/taxonomy/paper-tags.json`（或 `scripts/configs/tag-rules.json`）：

```jsonc
{
  "version": 1,
  "tasks": [
    {"id": "traffic_flow_forecasting", "en": "Traffic flow forecasting", "zh": "交通流预测",
     "match": {"title_any": ["traffic flow", "flow prediction", "flow forecasting"]}},
    {"id": "traffic_speed_forecasting", "en": "Traffic speed forecasting", "zh": "交通速度预测",
     "match": {"title_any": ["traffic speed", "speed prediction"]}},
    {"id": "probabilistic_forecasting", "en": "Probabilistic traffic forecasting", "zh": "概率交通预测",
     "match": {"title_any": ["probabilistic", "uncertainty", "quantile", "diffusion"]}}
  ],
  "methods": [
    {"id": "gnn", "label": "GNN", "match": {"title_any": ["graph neural network", "gnn", "graph convolution"]}},
    {"id": "transformer", "label": "Transformer", "match": {"title_any": ["transformer", "attention"]}},
    {"id": "federated", "label": "Federated", "match": {"title_any": ["federated"]}}
  ],
  "fallback": {"task": "unclassified", "methods": []}
}
```

**优点**

- 完全确定、零成本、零密钥、可审计：同一篇论文今天和明年的标签不会变。
- 可以复用现有 `yearly_hotspots.py::detect_themes()` 的关键词表（它已经定义了 GNN / Transformer / Federated / Transfer / Probabilistic / Physics-Informed 六个主题），**保持两处口径一致**就免了一次返工。
- 天然适合 CI：规则文件本身可以进 07 文的 schema 门禁。

**缺点**

- 覆盖率低：README 里有 24 条带 `[[code]]`、103 条 `[[paper]]`，纯标题匹配对「标题没写 method」的论文（如 `2F-TP: Learning Flexible Spatiotemporal Dependency...`）会落空。
- 需要人工持续维护同义词表；`attention`/`self-attention` 这种包含关系必须显式声明 `implies`（`self-attention → attention`）。

**降级策略**：规则没命中 → 写 `task: "unclassified"` + `method_tags: []` + `needs_review: true`，**不猜**。渲染层遇 `unclassified` 时该行仍显示，但 Task 列写 `Unclassified`（明确可见的缺口，而不是留空）。

### 4.2 路线 B：LLM 辅助 + 人在环审核

**做法**：

1. `normalize` 段之后，对「规则未命中」或「规则命中但 `needs_review`」的条目，调用 LLM（走 `opencode.yml` 已有的 `KIMI_API_KEY` 通道，或独立的 `OPENAI_API_KEY`）。
2. **输入**：`title + abstract`（截断到 1200 字符）+ 冻结的 taxonomy（task id 列表 + method label 列表）。
3. **要求结构化输出**：`{"task_id": "...", "method_ids": ["..."], "confidence": 0.0-1.0, "evidence_span": "..."}`，`task_id`/`method_ids` 必须在 taxonomy 白名单内。
4. **校验**：白名单外的 id 直接丢弃；`confidence < 0.6` 的进人工队列。
5. **人在环**：LLM 结果**永远不直接写进 `recent-papers.json` 的权威字段**，而是写进 `updates/reports/tag-suggestions.json`，由 PR 里的 reviewer 逐条 accept / reject；accept 后才由人（或一个显式的手动命令）合入 JSON。

**优点**

- 覆盖率能上到 90%+，包括标题不含关键词的论文。
- `evidence_span` 可以放进 PR 便于快速审核（比纯标签可信得多）。

**缺点**

- 非确定：同一篇论文两次分类可能不同 → **破坏可复现性**。必须把 LLM 输出**快照进仓库**（`tag-suggestions.json` 带 `model`、`prompt_version`、`created_at`），之后一律读快照，重跑不重调。
- 成本与延迟：按月几十条还好，全量回填 102 条要一次性控制 rate。
- 滥用风险低但存在：LLM 会把 `traffic` 泛化到我不想收的领域（实测当前单源都召回了「Dhaka 热舒适」「西班牙公路货运」两条无关论文），**所以 LLM 只能做「已通过 domain 规则的条目」的二段打标，不能替代第一段 domain 过滤**。

**降级策略**：LLM 超时/额度耗尽/输出不合 schema → **回落路线 A**，并把该批标记 `tag_source: "rules_fallback"`。绝不因为 LLM 挂了就阻塞月度 PR。

### 4.3 取舍结论

| 维度 | A 规则词典 | B LLM + 人审 |
| --- | --- | --- |
| 可复现 | 强 | 弱（需快照） |
| 覆盖率 | 低–中 | 高 |
| 成本 | 0 | 按 token 计 |
| 维护成本 | 同义词表 | prompt + 审核人力 |
| CI 友好 | 是 | 需独立 job + 密钥 |

**建议：A 作为唯一权威路径且先落地；B 作为「建议通道」，输出只到 `tag-suggestions.json`，月度 PR 里附一张待审表。** 等 A 的同义词表稳定、B 的 accept 率稳定在 80% 以上，再考虑把 B 提升为预填值。

---

## 5. SOTA / Leaderboard 数据源改造

### 5.1 问题确认（实测，非推测）

仓库里 `paperswithcode.com` 共出现 **10 处**（README.md 6 + README.zh.md 6，其中 4 处 `…/paper/…` 与 `…/task/…` 在两文件里重复），另有 `datasets/` 下 **4 处 `…/sota/…`**：

| 文件 | 行 | 形态 | 链接 |
| --- | --- | --- | --- |
| `README.md` | 91 | `[[code]]` | `paperswithcode.com/paper/spatio-temporal-graph-convolutional-networks` |
| `README.md` | 131 | `[[SOTA]]` | `paperswithcode.com/paper/spatio-temporal-graph-structure-learning-for` |
| `README.md` | 157 | `[[code]]` | `paperswithcode.com/paper/spatial-temporal-attention-wavenet-a-deep#code` |
| `README.md` | 158 | `[[code]]` | `paperswithcode.com/paper/time-series-is-a-special-sequence-forecasting#code` |
| `README.md` | 413 | 列表项 | `paperswithcode.com/task/traffic-prediction` |
| `README.md` | 427 | 列表项 | `paperswithcode.com/task/multivariate-time-series-forecasting/latest` |
| `datasets/METR.md` | 9 | 标题内联 | `paperswithcode.com/sota/traffic-prediction-on-metr-la` |
| `datasets/NYC-Bike.md` | 9 | 标题内联 | `paperswithcode.com/sota/traffic-prediction-on-nyc-bike-dataset` |
| `datasets/NYC-Taxi.md` | 9 | 标题内联 | `paperswithcode.com/sota/traffic-prediction-on-nyc-taxi-dataset` |
| `datasets/PEMS.md` | 13 | 段落内联 | `paperswithcode.com/sota/traffic-prediction-on-pems-bay` |

（`README.zh.md` 的 6 处与英文版行号完全对应，属于双语复制。）

**实测可用性（2026-11-24）**：

```text
GET https://paperswithcode.com/sota/traffic-prediction-on-metr-la
  → 301 → https://huggingface.co/papers/trending   (最终 200)

GET https://paperswithcode.com/task/traffic-prediction
  → 302 → https://huggingface.co/papers/trending   (最终 200)
```

即：**所有 `paperswithcode.com/*` 链接都会被整体重定向到 Hugging Face 的「trending papers」首页**，不再保留任何原路径语义。也就是说：

- `…/sota/traffic-prediction-on-metr-la` 点进去**看不到 METR-LA 的任何 SOTA 数据**，只看到 HF 的通用热门论文流；
- `…/task/traffic-prediction` 同理，**不是** traffic prediction 任务页；
- 每条链接的**路径信息 100% 丢失**。

这不是「站点慢」，而是「站点已迁走、旧路径做兜底跳转」。因此 4 个 `datasets/*.md` 里的「SOTA」小节当前**实质是空头承诺**——读者点进去得不到任何 leaderboard。

### 5.2 替代方案评估

| 方案 | 能拿到什么 | 代价 | 结论 |
| --- | --- | --- | --- |
| **Hugging Face Papers**（`huggingface.co/papers`） | 每日/每周 trending 论文流、`/papers/<arxiv_id>` 详情页 | 无公开的「按 dataset+metric 排名」接口；只有「热门度」不是「SOTA」 | **不能替代 leaderboard**，只能做「近期热门」入口 |
| **OpenAlex `cited_by_count` 排序** | 每条论文的引用数、可按 `cited_by_count:desc` 排序、支持批量取 | 引用数 ≠ 精度；新论文天然低引用（`HySRD-Net` 2026 只有 4 次，`COOL` 86 次、`UniST` 265 次） | **可以替代「影响力榜」，不能替代「分数榜」** |
| **OpenReview** | ICLR/NeurIPS 等的评审与分数、`forum?id=` 稳定 | 只有投稿版，无统一 metric；覆盖率取决于会议 | **作为 `[[review]]` 链接的权威来源**（README L91 已经在用），不做 leaderboard |
| **人工维护 `data/leaderboards/*.json`** | 完全可控：dataset + horizon + metric + value + paper_id + 出处 | 纯人力，易过期 | **唯一能给出「真 SOTA 数字」的路径** |
| Papers with Code 归档镜像 | 历史数据 | 无维护、随时消失 | 不采用 |

**结论：leaderboard 必须自己维护，外部源只负责「排序辅助」和「发现辅助」。**

### 5.3 建议的 leaderboard 数据结构

```jsonc
// data/leaderboards/metr-la.json
{
  "dataset_id": "metr-la",
  "dataset_doc": "datasets/METR.md",
  "metric": "MAE",
  "split": "chronological",
  "unit": "mph",
  "updated_at": "2026-11-01",
  "snapshot_source": "manual",
  "rows": [
    {"horizon_min": 15, "paper_id": "agc-lstm-2025", "value": 2.91, "source": "paper_table_3", "verified": true},
    {"horizon_min": 30, "paper_id": "hysrd-net-2026", "value": 3.12, "source": "paper_table_3", "verified": false}
  ]
}
```

规则：

- `paper_id` 必须是 `data/papers/*.json` 里存在的 id（可做外键校验，见 07 文）。
- `verified: false` 的行在站点上标灰、README 表格里不渲染，避免把未核对的数字当权威。
- **必须有 `horizon_min`**：README `0x07` 已经强调「report metrics across multiple horizons, not a single step」，把 15/30/60 混在一张榜里就是制造错误比较。
- 每次更新 leaderboard 同时写一条**引用数快照**：

```jsonc
// data/leaderboards/citations-snapshot.json
{"snapshot_at": "2026-11-01", "rows": [{"paper_id": "unist-2024", "cited_by": 265}, ...]}
```

采样方式（实测可用，单次请求可批量取 50 条）：

```text
https://api.openalex.org/works?filter=ids.openalex:W4391987728|W4392382537&per-page=50&select=id,doi,cited_by_count,publication_year
→ 200，返回两条，cited_by_count 分别为 265 / 86
```

按 DOI 检索也通：`filter=doi:10.3390/electronics15030623` → 命中 1 条（`W7126440023`, cited_by=1）。这比 `works/doi:<DOI>` 单条 API 更省请求，**推荐整批用 filter 形式一次取完**。

### 5.4 README 中 pwc 链接的批量处理规则（仅方案，不改 README）

按链接**语义角色**分类处置，不是一刀切删除：

**规则 1 — `[[SOTA]]` 标签（README.md:131）→ 替换**

- 该条是 `AAAI 2020 Spatio-Temporal Graph Structure Learning for Traffic Forecasting`，链接 `…/paper/spatio-temporal-graph-structure-learning-for`。
- 处置：把 `[[SOTA]]` 改为指向**自己的 leaderboard 锚点**（如 `[[SOTA]](./datasets/METR.md#leaderboard)` 或未来站点的 `/leaderboards/metr-la/`）；若该论文暂不在 leaderboard 中，则**降级为 `[[paper]]` 之外的普通无链接**，并加一条待办。
- 理由：`[[SOTA]]` 是**最强语义承诺**，指向一个会跳到 HF trending 的链接是最坏情况。

**规则 2 — `[[code]]` 指向 pwc 论文页（README.md:91/157/158）→ 替换或剔除**

- 现状：这三条的 `[[code]]` 本该指向 **GitHub 仓库**，却指向了 pwc 的论文页（有的还带 `#code` 锚点）。
- 处置优先级：
  1. 能从论文/arXiv 找到官方仓库 → 换成 `github.com/...`；
  2. 找不到 → **删除该 `[[code]]` 标签**（保留其他标签），不要留一个死链；
  3. 如果只想保留「它是开源工作」这一信息，改为 `[[code-listing]](… )` 但**不要指向 pwc**。
- 注意 README 已经有一套更细的 code 角色词表（实测：`[[code]]` 24 处、`[[code-pytorch]]` 2 处、`[[code-official-tf]]` 1 处、`[[code-not-official]]` 1 处），新增/替换时要沿用这套词表，别再发明新词。

**规则 3 — `0x06 Related Repositories` 的 `…/task/…` 两项（README.md:413 / 427）→ 替换**

- `[paper with code](https://paperswithcode.com/task/traffic-prediction)` → 这是**资源入口**，不是论文引用。
- 处置：换成有效入口，二选一（**待人工定夺，本文不替它决策**）：
  - `https://huggingface.co/papers`（承认它就是跳转目标，好歹路径与语义一致）；
  - 或换成仓库内自己的 leaderboard 页面（Phase 2 后）；Phase 1 阶段先换成 `https://github.com/topics/traffic-prediction`（README 第 415 行已经在用这个，避免重复）。
- `[Multivariate Time Series Forecasting](…/task/multivariate-time-series-forecasting/latest)` → 同上；这个 topic 与「交通预测」只有弱相关，可考虑**直接删除**并把第 8 项 `deep-learning-time-series` 提前。

**规则 4 — `datasets/*.md` 的 `…/sota/…` 标题（4 个文件）→ 全部替换为本地 leaderboard**

- 4 处形态：`#### SOTA: [Traffic Prediction on METR-LA](…)`、`## Leaderboard` + `[SOTA Leaderboard](…)`、`#### SOTA: [...]`、`#### SOTA: [...]`。
- 处置：改为指向 `data/leaderboards/<dataset>.json` 渲染出的**本地小节**，例如：

```markdown
#### SOTA / Leaderboard

> 数据源：`data/leaderboards/metr-la.json`（最近更新 2026-11-01，人工核对）
> MAE (mph) @ 15/30/60 min：见 [README Benchmark Matrix](../../README.md#0x07-benchmark--evaluation)

- METR-LA 官方说明：见本页上方 Data Description
- 引用数快照：`data/leaderboards/citations-snapshot.json`
```

- **在 leaderboard 数据落地之前**（Phase 1 早期），这 4 个文件**不要立刻删链接**，而是**加一行显式状态标注**：

```markdown
> ⚠️ 原 SOTA 链接（paperswithcode.com）已于站点迁移后失效，点击会跳转到 Hugging Face trending 页面。
> 本地 leaderboard 正在建设中，见 `docs/evolution/06-pipeline.md` §5。
```

  这样「保留」是有明确标注的保留，不是假装它还好用。

**规则 5 — 统一收尾：给所有外链加可探测标记**

- 不建议手工逐个点。建议后续加一个**只在定时任务**跑的链接巡检（见 07 文 `link-check.yml`），把 `http_code != 200` 或 `final_url != 原文 url` 的条目报进 `updates/reports/link-status.json`。
- `final_url != 原文 url` 这条规则**专门用来抓 pwc 这种「200 但跳走了」的情况**——这是本次实测发现的坑，`curl -I` 只看到 301、`curl -L` 得到 200，靠状态码是抓不出来的。

### 5.5 处置矩阵总表

| 位置 | 现状 | 处置 | 何时 |
| --- | --- | --- | --- |
| `README.md:131` `[[SOTA]]` | pwc paper 页 | 改成 leaderboard 锚点或去链接 | Phase 1 末期 |
| `README.md:91/157/158` `[[code]]` | pwc paper 页 | 换官方 repo，找不到就删标签 | Phase 1 |
| `README.md:413/427` task 页 | pwc task 页 | 换 HF papers 或仓库内页面 | Phase 1 |
| `README.zh.md` 对应 6 处 | 同英文 | 与英文同步（sync 门禁会抓） | 与英文同一 PR |
| `datasets/METR.md:9` | pwc sota 页 | 加失效标注 → 换本地 leaderboard | 标注 Phase 1 / 替换 Phase 2 |
| `datasets/PEMS.md:13` | pwc sota 页 | 同上 | 同上 |
| `datasets/NYC-Bike.md:9` | pwc sota 页 | 同上 | 同上 |
| `datasets/NYC-Taxi.md:9` | pwc sota 页 | 同上 | 同上 |

---

## 6. 流水线结构：拆成四段

### 6.1 为什么拆

现状把 5 件事塞在一个 job 里，导致：

- 词云炸了（`matplotlib` 字体/后端问题是最常见的 CI 失败源）→ 论文也白爬了；
- 论文源全挂 → 词云也白画；
- 无法单独重跑某一段；
- `updates/wordcloud/` 三个月就 3.4 MB，每次 PR 都带 PNG diff，review 噪音极大。

### 6.2 四段的契约

#### 段 1 · `fetch` → 产物：raw JSON 快照

| 项 | 内容 |
| --- | --- |
| 输入 | `scripts/configs/traffic-monthly.json`（源列表、query、year_range、runtime） |
| 命令 | `python3 scripts/paper_search.py --config … --report-json out/fetch-<run>.json` |
| 产物 | `updates/raw/<YYYY-MM>/openalex.json`、`arxiv.json`、`crossref.json`（**每源一份**，保留原始 `Paper` 字段 + `fetched_at_utc` + `requests_made` + `cache_hits` + `source_errors`） |
| 幂等 | 同 run id 重跑覆盖同目录；跨 run 不覆盖 |
| **失败降级** | 任一源失败 → 该源文件写 `{"status":"error","error":"…"}`，**不算整段失败**；全部源失败 → 段失败，不发 PR，只发告警（§7） |

> 现状只有一份合并后的 `--report-json`，且丢掉了未通过 `min_score` 的论文。**保留 raw 快照是「可复现」的唯一前提**——否则下个月想改规则重跑，源已经变了。

#### 段 2 · `normalize` → 产物：去重+打分的规范化 JSON

| 项 | 内容 |
| --- | --- |
| 输入 | 段 1 的 raw JSON + 现有 README（用于对既有条目去重）+ `data/taxonomy/paper-tags.json` |
| 命令（建议脚本） | `python3 scripts/normalize_papers.py --raw updates/raw/<YYYY-MM>/ --readme README.md README.zh.md --taxonomy … --out updates/normalized/<YYYY-MM>/papers.json --report-json updates/reports/normalize-<YYYY-MM>.json` |
| 产物 | `papers.json`（规范化结构：`arxiv_id/doi/title_norm/authors/year/venue/score_int/matched_rules/task_id/method_tags/needs_review/dup_reason`）+ `title-collisions.json` |
| 幂等 | **纯函数**：同输入必同输出（§3.4 整数分 + 全序排序） |
| 失败降级 | schema 校验失败 → 段失败并阻断后续段；`title_collisions` 非空 → 不失败，写报告交人工 |

> 现有 `dedupe_readme_papers.py` 是这条链路的雏形，但它**输出 markdown**而不是 JSON，且只做去重不做归一化。建议**保留它不动**（旧 workflow 还在用），新增 `normalize_papers.py` 作为新链路入口，两者共享 `normalize_title/normalize_doi` 的实现（抽到 `scripts/lib/normalize.py`，避免两份正则漂移）。

#### 段 3 · `render` → 产物：README 自动块 + 站点数据

| 子产物 | 生成者 | 写入目标 |
| --- | --- | --- |
| README recent 块 + digest 表 | `render_curated_recent.py`（**已存在**） | `README.md` / `README.zh.md` 的 4 个 marker 之间 |
| 候选清单 | 建议 `render_candidates.py`（新） | `updates/monthly-paper-candidates.md` |
| 站点数据 | `site/scripts/prepare-data.mjs`（**已存在**，prebuild 阶段） | `site/src/data/papers.generated.json`、`site/public/_data.json` |
| 图表 | `yearly_hotspots.py` / `paper_wordcloud.py`（**已存在**） | `updates/hotspots/`、`updates/wordcloud/` |

关键约束：**README 与站点必须读同一份数据**。现状 `render_curated_recent.py` 读 `updates/curated/recent-papers.json`，站点 `prepare-data.mjs` 优先读 `data/papers/*.json`、回落 README——三者口径不同。Phase 1 统一到 `data/papers/*.json` 后，这条约束才算成立。

失败降级：
- 渲染段失败 → **阻断**（不能发一个半渲染的 README）。
- 图表子任务失败 → **不阻断**，跳过该图并在 PR body 标 `visuals: skipped`（把 `matplotlib` 依赖隔离在这一个子任务里，是拆分最直接的收益）。

#### 段 4 · `publish` → 产物：PR（或直接 commit）

| 项 | 内容 |
| --- | --- |
| 判断 | `git diff --quiet -- <内容文件>`（**不含**时间戳、不含 `updates/raw/`、不含 `updates/wordcloud/` 之外的可再生产物） |
| 有 diff | 开 PR，body 内嵌 normalize 报告摘要（新增 N 条 / 去重 M 条 / 冲突 K 条 / 源错误列表） |
| 无 diff | 不开 PR，写一条 job summary |
| 失败 | 不发 PR；若连续 2 次失败 → 开 issue（§7） |

### 6.3 分段后的 workflow 形态（示意，不落地）

```yaml
# 概念示意：三份 workflow，而非一份巨型 workflow
fetch-normalize-render.yml   # cron 月度；job: fetch → normalize → render → publish（用 needs: 串联，各自可单独 workflow_dispatch）
visuals.yml                  # cron 月度 + 手动；独立 job，失败了不影响上面的 PR 内容
link-check.yml               # cron 每周；只读，产出 report 或 issue（详见 07 文）
```

`needs:` 串联而非单 job 串联的价值：**失败可见性**——GitHub UI 会直接告诉你 `normalize` 挂了，而不是「第 3 个 step 的输出有一堆 WARN」。

---

## 7. 失败告警、dry-run、可复现

### 7.1 失败分类与告警

| 类别 | 触发条件 | 现状表现 | 建议动作 |
| --- | --- | --- | --- |
| **硬失败** | 配置非法（`load_config` 抛错，`return 2`）、全部源失败（`return 1`） | job 失败，无 PR | 开 issue 并 `@maintainer`；PR 不阻塞 |
| **软降级** | 部分源失败、预算耗尽、`title_collisions` 非空 | **静默**（只 stderr WARN，`return 0`） | 写入 `updates/reports/`，并在 PR body 顶部加 `> ⚠️ degraded: <n> source error(s)`；连续 2 个月软降级 → 升级为 issue |
| **静默空窗** | 输出 0 条候选 | 只是 PR 里列表变空，没人注意 | `output_count == 0` 单独告警：**0 条是异常信号，不是正常月份** |
| **产物漂移** | 渲染产物与数据不一致 | `curated-render-check` 抓到（仅 PR 触发） | 已有；保持 |
| **外部依赖失效** | pwc 型「200 但跳走」 | 完全不可见 | 07 文的 `link-check.yml` 专治 |

实现建议：**不引入新服务**（AGENTS.md 明确「不引入 build pipelines 或 runtime services」）。用 GitHub 原生能力：

- `$GITHUB_STEP_SUMMARY` 写摘要；
- 失败时 `gh issue create --label automation-failure`（需要 `issues: write` 权限，当前 workflow 只有 `contents/pull-requests`，**需加一行**）;
- 若不想开 issue，退一步：`peter-evans/create-pull-request` 的 body 加 `<!-- automation:degraded -->`，并在 07 文的 PR 门禁里要求 reviewer 勾选确认。

### 7.2 dry-run

现状**没有任何 dry-run 开关**。`paper_search.py` 有 `--out`（写文件）与不传 `--out`（打印 stdout）的差别，但这不算 dry-run——它照样发起真实 HTTP 请求。

建议的最小可行 dry-run 设计（**不改现有 CLI 语义，用环境变量或新参数**）：

| 层 | 开关 | 行为 |
| --- | --- | --- |
| 网络层 | `PAPER_SEARCH_OFFLINE=1` | `HttpClient.fetch_text()` 命中缓存则用，**未命中直接抛 `OfflineCacheMiss(url)`**。这能验证「用同一份缓存重跑，输出是否一致」 |
| 渲染层 | 现有 `--check` | 已有（`render_curated_recent.py`）；`yearly_hotspots.py` / `paper_wordcloud.py` **没有** `--check`，建议补 `--dry-run`（只算不写） |
| 发布层 | `git diff --quiet` 预演 | 无产物的 dry-run 不写盘即天然安全 |

注意：`--check` 的语义是「不一致就失败」，**不是「不写盘」**——`apply_to_readme()` 在 `check=True` 时确实 `return changed` 不写文件（L119–L123），所以现有 `--check` 是安全的 dry-run。这点要在文档里写清楚，免得有人以为它只校验格式。

### 7.3 可复现性三要素

1. **缓存**：`cache_dir` 目前在 GitHub runner 上是**临时目录**，每次 job 都从零开始，`cache_ttl_hours: 48` 实际只在单次 job 内有效（同 job 多 query 复用）。→ 建议用 `actions/cache` 缓存 `.cache/paper-search-http`，key 带 `hashFiles('scripts/configs/traffic-monthly.json')`。这样同月重跑不重复请求源，也让「上月 raw 快照能否复现」变成可测。
2. **请求预算**：`max_requests` 已存在且实测有效（`--max-requests 4` → 第 5 次抛 `request budget exceeded (4)`，被吞为 WARN）。**但要说清一个坑**：预算耗尽**不返回非零退出码**（实测 `exit=0`），所以「预算耗尽」在 CI 里等于「论文悄悄变少」。这是要修的一处。
3. **种子/顺序**：见 §3.4。整数分 + 全序排序 + 输入显式排序 = 同输入同输出。加上 raw 快照落盘，「用上月的 raw 重跑 normalize」在本地是完全可复现的。

### 7.4 建议的 run 目录布局

```text
updates/
├── raw/2026-11/
│   ├── openalex.json          # 段1产物，按源分文件
│   ├── arxiv.json
│   ├── crossref.json
│   └── manifest.json          # run_id / 起止时间 / config hash / requests_made
├── normalized/2026-11/
│   └── papers.json            # 段2产物
├── reports/
│   ├── normalize-2026-11.json # kept / removed / collisions / reasons
│   ├── title-collisions.json
│   ├── tag-suggestions.json   # 路线 B 的输出，只读快照
│   └── link-status.json       # 07 文 link-check 产物
├── curated/recent-papers.json # 人工权威数据（不动）
├── monthly-paper-candidates.md
├── hotspots/                  # 可再生
└── wordcloud/                 # 可再生
```

`.gitignore` 建议（**仅建议，不改**）：`updates/raw/` 与 `updates/normalized/` 可以选择「只保留最近 3 个月」，避免仓库体积随月增长。当前 `.gitignore` 内容只有一行（实测），加规则是低风险动作。

---

## 8. 与现有流程的兼容（灰度）

本文全部改动都**不要求一次性切换**：

1. 新链路（fetch/normalize/render 三段 + raw 快照）与旧 `monthly-paper-search.yml` **并行跑一个月**，比对两者候选集合差异，差异写进 PR body。
2. 差异稳定后（建议阈值：候选集合 Jaccard ≥ 0.8），把旧 workflow 的 `Run monthly search` 步骤替换为新链路，`Generate word cloud` / `Generate yearly hotspots` 拆到独立 workflow。
3. 拆分后旧 workflow 更名或删除，`scripts/README.md` 的「Monthly GitHub Action」小节同步更新（该文件不在本任务写入范围内，需另开任务）。

---

## 9. 取证方式（可复现本文所有数字）

```bash
# 现状：单源
python3 -c "import json;print(json.load(open('scripts/configs/traffic-monthly.json'))['search']['sources'])"
# => ['openalex']

# 代码支持 4 源
grep -n 'SUPPORTED_SOURCES' scripts/paper_search.py

# 预算耗尽不报错（exit=0）
python3 scripts/paper_search.py --config scripts/configs/traffic-monthly.json \
  --max-requests 4 --cache-dir /tmp/ps-cache --out /tmp/x.md; echo "exit=$?"
# => WARN: ... request budget exceeded (4)   exit=0

# pwc 全部跳转（注意要跟 -L 才能看穿）
curl -sSL -o /dev/null -w "%{http_code} %{url_effective}\n" \
  https://paperswithcode.com/sota/traffic-prediction-on-metr-la
# => 200 https://huggingface.co/papers/trending

# OpenAlex 批量引用数
curl -s "https://api.openalex.org/works?filter=ids.openalex:W4391987728|W4392382537&per-page=50&select=id,cited_by_count"
# => 265 / 86

# arXiv DOI 在 Crossref 不存在
curl -s -o /dev/null -w "%{http_code}\n" https://api.crossref.org/works/10.48550/arxiv.2403.01091
# => 404

# 时间戳导致每次必开 PR
grep -n "Generated on:" .github/workflows/monthly-paper-search.yml

# 产物体积
du -sh updates/wordcloud updates/hotspots
# => 3.4M  808K
```

---

## 10. 与其他文档的关系

- **07-quality-gates.md**：本文产出的 `updates/raw|normalized|reports/` 结构是门禁的检查对象；schema 校验、重复条目检测、外链巡检都在那里给具体规则。
- **08-roadmap.md**：本文的四个段对应 Phase 1（fetch/normalize/render 拆分）与 Phase 3（多源、LLM 打标签、leaderboard）的具体任务与人日。
- **09-acceptance.md**：本文每段的产物都有对应的验收命令。