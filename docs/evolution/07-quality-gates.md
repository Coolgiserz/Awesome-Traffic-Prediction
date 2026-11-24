# 07 · 质量门禁方案

> 目标：把「能跑起来就不管了」的三个 check，扩展成**分层门禁**——PR 必过的「确定性、快、无网络」与定时跑的「非确定性、慢、有网络」严格分离，让 CI 既抓得住真问题，又不会三天两头 flaky 卡 PR。
>
> 本文是**方案文档**。所有「对应脚本文件名」都是**建议**，本文不创建任何脚本、不改 `.github/`。

---

## 1. 现有门禁的能力与边界

### 1.1 清单

仓库当前有 3 个内容校验 workflow（`opencode.yml` 是 `/oc` 评论触发的 AI 助手，不是门禁）：

| Workflow | 触发路径 | 实际执行的命令 | 退出条件 |
| --- | --- | --- | --- |
| `readme-sync-check.yml` | `README.md`、`README.zh.md`、`scripts/check_readme_sync.py`、自身 | `python3 scripts/check_readme_sync.py --en README.md --zh README.zh.md` | 发现差异 `return 1` |
| `curated-render-check.yml` | `README.md`、`README.zh.md`、`scripts/render_curated_recent.py`、`updates/curated/recent-papers.json`、自身 | `python3 scripts/render_curated_recent.py --data updates/curated/recent-papers.json --readme-en README.md --readme-zh README.zh.md --check` | 渲染结果 ≠ 文件内容 `return 1` |
| （月度 workflow 内）`Validate domain config` | 非独立 workflow，是 `monthly-paper-search.yml` 的一个 step | `python3 scripts/validate_domain_config.py scripts/configs/traffic-monthly.json` | 配置非法 `return 2/1` |

当前全部通过（实测）：

```text
$ python3 scripts/check_readme_sync.py --en README.md --zh README.zh.md
[OK] README sync check passed.
- Year sections checked: 12
- Curated digest rows: 12
- Benchmark matrix rows: 4

$ python3 scripts/render_curated_recent.py --data updates/curated/recent-papers.json \
    --readme-en README.md --readme-zh README.zh.md --check
[OK] Curated recent paper sections rendered.
```

### 1.2 `check_readme_sync.py` 的实际能力边界（**只比数量，不比内容**）

这是最需要说清楚的一点。读源码（L36–L86）可以看到它做了三件事，全都是**计数比对**：

1. **年度小节条目数**：`parse_year_counts()` 在 `### Deep Learning Based Traffic Prediction Methods`（ZH 对应 `### 基于深度学习的交通预测方法`）之后，用 `^####\s+(20\d{2})\s*$` 切年份，用 `^\d+\.\s+\[` 数条目，得到 `{年: 条数}`；再逐年在 EN/ZH 间比对。
2. **Curated Digest 表行数**：`count_table_rows()` 取 `### Curated Digest` / `### 重点整理` 之后、下一个 `### ` 之前的、以 `|` 开头的行，**减 2**（表头 + 分隔行）。分隔行必须全是 `-`/`:`，否则直接返回 0。
3. **Benchmark Matrix 表行数**：同上，锚点 `### Benchmark Matrix` / `### 基准矩阵`。

**它明确做不到的事**：

| 场景 | 会不会被抓到 | 为什么 |
| --- | --- | --- |
| EN 某年有 4 条、ZH 有 3 条 | ✅ 抓到 | 计数不等 |
| EN 和 ZH 都是 4 条，但**第 2 条内容完全不同**（一篇讲 METR-LA 一篇讲 PEMS-BAY） | ❌ 漏 | 只数个数 |
| 某条目里 `[[paper]]` 链接被改成了 404 的 URL | ❌ 漏 | 不看链接 |
| 年份标签写错：`[TITS 2020]` 被塞进 `#### 2021` 小节 | ❌ 漏 | 只看小节内的数量 |
| 表格里某一行的 `Method Tags` 列被清空 | ❌ 漏 | 只数行数 |
| EN 删掉一整段（比如 0x05 Research Group 的半个列表） | ❌ 漏 | 只检查 3 个锚点 |
| 条目格式写坏（正文换行导致 `[[paper]]` 掉到下一行） | ❌ 漏 | 解析不到就当没这条 → **两边同时少一条时数量仍然相等** |
| 重复条目 | ❌ 漏 | 完全没做 |

最后一行不是假想：实测 README L170–171 就是**跨行**的条目：

```markdown
2. [Arxiv 2022] Pre-training Enhanced Spatial-temporal Graph Neural Network
   for Multivariate Time Series Forecasting [[paper]](https://arxiv.org/pdf/2206.09113.pdf)
```

`check_readme_sync.py` 的 `^\d+\.\s+\[` 只匹配到第一行、算作 1 条（数量对得上），但任何**按行解析**的工具（`dedupe_readme_papers.py` 的 `CANDIDATE_RE`、`yearly_hotspots.py` 的 `parse_candidates()`）都会解析失败并静默跳过。实测：README 里 `^\d+\.\s+\[` 开头的行共 **170 行**，其中 `CANDIDATE_RE` 能匹配的只有 **102 行**。

### 1.3 `curated-render-check.yml` 的实际能力边界

`render_curated_recent.py::apply_to_readme(path, …, check=True)` 的逻辑是：

```python
original = path.read_text()
rendered = replace_block(original, RECENT_START, RECENT_END, recent_block)
rendered = replace_block(rendered, DIGEST_TABLE_START, DIGEST_TABLE_END, digest_rows)
changed = rendered != original
if check: return changed        # ← 不写盘，只返回是否不一致
```

能力：

- ✅ 抓「JSON 变了但 README 没重渲染」；
- ✅ 抓「有人手改了 auto 块」（因为手改后 `rendered != original` 同样成立）；
- ✅ 抓「marker 丢失/顺序错」（`replace_block` 抛 `ValueError`，job 直接失败）；
- ✅ **是安全的 dry-run**——`check=True` 时确实不写文件。

边界：

| 场景 | 会不会抓到 | 说明 |
| --- | --- | --- |
| 手改了 auto 块 | ✅ 抓到，但**看不出**是手改还是忘渲染 | 报错文案统一是 `Outdated: README.md`，诱导人直接重跑渲染把人的改动覆盖掉 |
| 在 auto 块**外面**手加一条论文 | ❌ 漏 | 只校验 marker 之间 |
| JSON 里 `id` 重复 / 字段缺失 / `year` 是字符串 | ❌ 漏 | `load_entries()` 只检查「是不是 list of dict」，不做 schema 校验；`format_recent_item` 里 `str(e["venue"])` 缺字段会抛 `KeyError` 让 job 崩（不是友好报错） |
| JSON 里 `featured: true` 但 `note_en` 为空 | ❌ 漏 | 渲染出空单元格 |
| `task_zh` 有值但 `task_en` 缺失 | ❌ 漏 | 渲染到 ZH 正常、EN 崩或空 |

### 1.4 `validate_domain_config.py` 的能力边界

- ✅ 复用 `paper_search.load_config()` 做与运行时完全一致的解析（版本号必须为 2、queries 非空、sources 白名单、year_range、runtime 各值 ≥ 0、rule 的 `op` 只能是 `any/all`、keywords 非空）；
- ✅ 额外查 rule id 重复；
- ✅ `--summary` 打印归一化结果；
- ❌ **不是独立 workflow**：只在月度 cron 里跑。有人手改 `traffic-monthly.json` 开 PR，**没有任何门禁会检查这个文件**。

### 1.5 汇总：当前门禁的覆盖缺口

| 缺口 | 影响 |
| --- | --- |
| 无 JSON Schema 校验 | `updates/curated/recent-papers.json` 与将来 `data/papers/*.json` 的任何结构错误（字段缺失、类型错、id 不合法）都会在下游渲染时以 `KeyError`/静默丢条目的形式暴露 |
| 无重复条目检测 | 现状恰好是 0 重复（实测 EN 102 条 title 0 dup / DOI 0 dup，ZH 同），**纯属人工运气**。`dedupe_readme_papers.py` 只对「新候选 vs 已有」去重，不对「已有 vs 已有」查重 |
| 无双语内容对齐 | 只比数量（§1.2） |
| 无链接巡检 | 218 个外链（实测，64 个不同 host）里已有 10 处指向失效的 pwc（见 06 文 §5） |
| 无「auto 块被手改」的辨识能力 | 报错文案会误导人覆盖人工改动 |
| 无拼写检查 | README 里实测存在 `Trafffic`、`intergrating`、`evalution`、`insititution` 各 2 处（EN/ZH 各一份，属于双语复制传播） |
| 无产物体积上限 | `updates/wordcloud/` 已 3.4 MB、`updates/hotspots/` 808 KB、`datasets/resources/PeMS-M.zip` 6.1 MB，`.git` 已 27 MB |

---

## 2. 分层门禁总设计

### 2.1 两层划分

| 层 | 名称 | 触发 | 时间预算 | 网络 | 失败后果 |
| --- | --- | --- | --- | --- | --- |
| **L1** | PR 必过（blocking） | `pull_request`，按 `paths` 过滤 | **< 60s/任务** | **禁止** | 合并被 block |
| **L2** | 定时巡检（advisory） | `schedule`（周/月）+ `workflow_dispatch` | < 20 min | 允许，需限速 | 开 issue / 写 report，**不 block PR** |

**为什么必须分层**：外链可用性天生 flaky——本次实测里 `https://www.mdpi.com/2220-9964/13/6/210` 用 curl 返回 **403**（反爬），但它对浏览器读者完全正常。把这种检查放进 PR 门禁，等于让「碰巧被 MDPI 风控挡了一次」阻断所有人的合并。

### 2.2 L1 门禁清单（PR 必过）

| # | 门禁 | 检查对象 | 建议脚本 | 现状 |
| --- | --- | --- | --- | --- |
| L1-1 | **JSON Schema 校验** | `data/papers/*.json`、`data/venues.json`、`data/datasets/*.json`、`updates/curated/recent-papers.json`、`scripts/configs/*.json` | `scripts/check_schema.py` | ❌ 无 |
| L1-2 | **JSON 可解析 + 稳定序列化** | 所有 `*.json` | `scripts/check_json_lint.py` | ❌ 无 |
| L1-3 | **双语条目内容对齐** | `README.md` ↔ `README.zh.md` 的 0x00 年度小节 | `scripts/check_bilingual_parity.py` | ⚠️ 只有计数（`check_readme_sync.py`） |
| L1-4 | **重复条目检测** | README 两份 + `data/papers/*.json` + `updates/curated/recent-papers.json` | `scripts/check_duplicates.py` | ❌ 无（只有「候选 vs README」去重） |
| L1-5 | **自动块未被手改** | README 的 4 个 marker 区块 | `scripts/check_auto_blocks.py` | ⚠️ 能抓到但不区分原因（`--check`） |
| L1-6 | **论文条目格式规范** | README 所有论文列表项 | `scripts/check_paper_format.py` | ❌ 无 |
| L1-7 | **config 校验进 PR** | `scripts/configs/*.json` | 复用 `scripts/validate_domain_config.py` | ⚠️ 有脚本，但只在 cron 跑 |
| L1-8 | **交叉引用完整性** | `data/papers/*.json` 的 `venue_id`/`datasets[]`/`affiliations[]` 外键 | `scripts/check_crossrefs.py` | ❌ 无 |
| L1-9 | **站点构建可过** | `site/` | `pnpm -C site build:fast` | ❌ 无 |
| L1-10 | **拼写白名单检查** | 已知错词表 | `scripts/check_spelling.py` | ❌ 无 |

### 2.3 L2 定时巡检清单（advisory）

| # | 门禁 | 频率 | 检查对象 | 建议脚本 |
| --- | --- | --- | --- | --- |
| L2-1 | **外链可用性（含「跳走」检测）** | 每周 | README×2、`datasets/*.md`、`models/*.md` 的全部外链 | `scripts/check_links.py` |
| L2-2 | **DOI 解析** | 每周 | 所有 `doi.org/*` 与裸 DOI | `scripts/check_dois.py` |
| L2-3 | **站点构建产物 + 体积** | 每周 | `site/dist/` | `scripts/check_site_budget.py` |
| L2-4 | **仓库体积报告** | 每月 | `updates/`、`datasets/resources/`、`.git` | `scripts/report_repo_size.py` |
| L2-5 | **词云/热点产物新鲜度** | 每月 | `updates/wordcloud/*`、`updates/hotspots/*` 的 `generated_utc` | `scripts/check_artifact_freshness.py` |

---

## 3. 每条门禁的详细规格

> 每条给：**检查逻辑** → **失败文案建议** → **建议脚本文件名**。全部为设计建议，未创建。

### L1-1 · JSON Schema 校验

**检查逻辑**

1. 扫描固定路径集合，按文件名映射到 schema：
   - `data/papers/*.json` → `data/schema/paper.schema.json`
   - `data/datasets/*.json` → `data/schema/dataset.schema.json`
   - `data/venues.json` → `data/schema/venue.schema.json`
   - `data/toolkits.json` → `data/schema/toolkit.schema.json`
   - `data/research-groups.json` → `data/schema/research-group.schema.json`
   - `data/lineage.json` → `data/schema/lineage.schema.json`
   - `updates/curated/recent-papers.json` → 需**新建** `data/schema/recent-papers.schema.json`（现状这份文件无 schema）
2. 用 `jsonschema` 做 `Draft202012Validator.check_schema()`（先校验 schema 自身合法）再 `validate(instance)`。
3. 汇总**全部**错误再退出（不要第一个错就退出），每条给 `json_path`，例如 `papers[3].method_tags[1]`。
4. 依赖策略：`jsonschema` 是第三方包。AGENTS.md 要求「推荐 python3 标准库、不强制引入项目级依赖」→ 建议**在 CI 里单独 `pip install jsonschema`**（不进仓库依赖），或退一步写一个**只覆盖本仓 schema 用到的关键字子集**（`type/required/enum/pattern/minLength/minimum/maximum/items/additionalProperties/format`）的 60 行 validator 放 `scripts/lib/mini_schema.py`，保持零依赖。

**失败文案建议**

```text
[FAIL] Schema validation failed (3 error(s)):

  data/papers/2026.json
    [12].method_tags[1]  expected type array, got string
    [15].venue_type      'journal-article' is not one of ['conference','journal','preprint','workshop','thesis','unknown']
  updates/curated/recent-papers.json
    [7]                  required property 'task_en' is missing

Fix: 见 data/schema/*.schema.json 对应字段说明。
注意：additionalProperties:false —— 新增字段必须先改 schema，再改数据。
```

**建议脚本**：`scripts/check_schema.py`
**建议调用**：`python3 scripts/check_schema.py --root . --strict`

---

### L1-2 · JSON 可解析 + 稳定序列化

**检查逻辑**

1. 遍历仓库内所有 `*.json`（排除 `node_modules/`、`site/dist/`、`.cache/`），`json.loads()` 一遍。任何 `JSONDecodeError` 报出行号列号。
2. **稳定序列化检查**：把内容 `json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=?)` 重新序列化，与磁盘原文比较。不一致 → 提示「文件不是规范化格式」。
   - 注意：现有 JSON **并不统一**——`updates/curated/recent-papers.json` 是 `indent=2`、key 按语义排；`scripts/configs/traffic-monthly.json` 也是 `indent=2`。建议**先不做强制的 `sort_keys`**，只做「`indent=2` + 末尾换行 + 无尾随空格」，避免一次性 reformat 制造巨大 diff。
3. 同时检查 `data/papers/*.json` 的**记录顺序**：按 `(year, order, id)` 排序是否已排好；未排好只 warn（防止每次改一条都产生整文件 diff 的真正解法是排序，但这要在数据落地时定）。

**失败文案建议**

```text
[FAIL] JSON lint failed:

  updates/curated/recent-papers.json:218  Expecting ',' delimiter
    → 第 218 行附近有语法错误（常见原因：上一行 object 少写逗号）

[WARN] Not canonical formatting (1 file):
  data/papers/2018.json  → indent 应为 2、末尾需一个空行
    fix: python3 scripts/check_json_lint.py --fix data/papers/2018.json
```

**建议脚本**：`scripts/check_json_lint.py`

---

### L1-3 · 双语条目内容对齐

**检查逻辑**（这是对 `check_readme_sync.py` 的**增量**，不是替代）

1. 用**跨行感知**的解析器（见 L1-6）把两份 README 的 0x00 年度小节都解析成 `[(year, venue_tag, title, paper_url, extra_links)]`。
2. 对齐方式：
   - **主键用 `paper_url` 归一化值**（`https://doi.org/10.xxxx` 与 `https://dx.doi.org/10.xxxx` 归一化；`arxiv.org/abs/X` 与 `arxiv.org/pdf/Xv2.pdf` 提取同 id）；
   - `paper_url` 缺失时退到 `(year, title_norm)`。
3. 判定：
   - **条数不等** → 沿用现有 `check_readme_sync.py` 的报错（保留它的输出格式，别破坏已习惯的人）；
   - **条数相等但集合不等** → 新增报错，逐条列出 `only in EN` / `only in ZH`；
   - **同一个 `paper_url` 出现在两边的不同年份小节** → warn（年份标签不一致，通常是其中一边写错）；
   - **`venue_tag` 的年份部分**（`[TITS 2020]` 里的 2020）与所在 `####` 小节年份不一致 → warn。实测这种不一致当前不存在，但很容易被新 PR 破坏。
4. 表（Curated Digest / Benchmark Matrix）的对齐：行数沿用现有检查；额外**按 `Paper` 列的 URL** 对齐两边的行，验证「同一组论文、同一顺序」。

**覆盖的现有盲区**：EN 和 ZH 都是 4 条但内容不同（§1.2 表格第 2 行）、链接被改坏。

**失败文案建议**

```text
[FAIL] Bilingual parity check failed (2 error(s)):

  条目集合不一致（README.md vs README.zh.md）:
    only in README.md:
      - [Arxiv 2022] Pre-training Enhanced Spatial-temporal GNN  (https://arxiv.org/pdf/2206.09113.pdf)
    only in README.zh.md:
      - [Arxiv 2022] Pre-training Enhanced Spatio-temporal GNN   (https://arxiv.org/pdf/2206.09113.pdf)
      ↑ 提示：URL 相同但标题不同 —— 若两者是同一篇，只需对齐标题；否则检查是否误改了 ZH 的标题。
  Curated digest 顺序不一致:
    row 5  EN=GraphSparseNet  ZH=ASTAM + Multi-Graph
```

**建议脚本**：`scripts/check_bilingual_parity.py`（保留 `check_readme_sync.py` 不动，新脚本内部可以 import 它的 `count_table_rows`）

---

### L1-4 · 重复条目检测

**检查逻辑**

1. 收集**所有**条目来源：`README.md`、`README.zh.md`、`data/papers/*.json`、`updates/curated/recent-papers.json`（Phase 1 之后前两者变成渲染产物，检查范围收窄到 JSON）。
2. 对条目生成 §06 文档里定义的 5 段 key：`arxiv_id / doi / title_norm / author_family / year`。
3. 三个独立检查：
   - **A. 文件内重复**：同一份 README 内部，`title_norm` 相同或 `doi` 相同 → **FAIL**。实测当前 EN 102 条、ZH 102 条，`title_norm` 重复 0、`doi` 重复 0 —— 这个门禁今天加进来就是绿的，属于「零成本上锁」。
   - **B. 跨文件重复**：`data/papers/*.json` 的 `id` 唯一性 + `doi` 唯一性 → **FAIL**。
   - **C. 跨版本重复（跨来源）**：`title_norm` 相同但 `doi` 不同、且 `author_family` 相同或 `year` 相同 → **WARN** + 写 `updates/reports/title-collisions.json`，交人工裁决（这是 06 文 §3.1 的 `title_collision` 规则在门禁侧的落地）。
4. **`year` 缺失不参与 C 判定**，避免把同一篇论文的 arXiv 预印本与正式版误报。

**为什么 C 只 WARN**：实测就存在真实的多版本情况——`COOL` 的 arXiv 版 DOI 是 `10.48550/arxiv.2403.01091`（OpenAlex `W4392382537`, cited 86），正式版 DOI 是 `10.1016/j.inffus.2024.102341`。两者都在 OpenAlex 里被标题搜索命中。如果 C 判 FAIL，就等于要求仓库只能保留一个版本，而 README 正文与 leaderboard 可能需要引用不同的 URL。

**失败文案建议**

```text
[FAIL] Duplicate entries detected in README.md:
  - doi 10.1016/j.eswa.2026.131214
      L162  [Expert Systems with Applications 2026] HySRD-Net: ...
      L241  [Expert Systems with Applications 2026] HySRD-Net: ...
    为什么重要：同一篇论文两条记录会让 check_readme_sync 的计数通过，但读者看到两个不同小节，
    且 yearly_hotspots 会把它的关键词算两次。
  - title_norm "graphsparsenetanovelmethodforlargescale" ...

[WARN] Possible cross-version duplicates (3) → updates/reports/title-collisions.json
  - "the effect of graph construction on traffic forecasting"
      A: doi=10.48550/arxiv.2401.xxxxx  (arxiv 2024)
      B: doi=10.1016/j.trc.2024.xxxxx   (TRC 2024)
    若确认为同一工作：把 B 的 venue 记入 A 的 also_published_as，并删除 B 条目。
```

**建议脚本**：`scripts/check_duplicates.py`（复用 `dedupe_readme_papers.py` 的 `normalize_title`/`extract_doi`，或抽到 `scripts/lib/normalize.py` 共享）

---

### L1-5 · 自动块未被手改

**检查逻辑**

当前 `render_curated_recent.py --check` **能**抓到「auto 块与渲染结果不一致」，但报错一律是：

```text
[FAIL] README files are not in sync with curated structured data.
- Outdated: README.md
```

这句话把人往「重跑渲染就好」的方向引，**恰好会覆盖掉手改**。所以要加一层**区分**：

1. 取当前 head 的 README 与 base 的 README（`git show origin/main:README.md`）。
2. 对 base 版跑一遍渲染检查：
   - base 已经不一致 → **这是既有的历史漂移**，不归本 PR 的责任 → 报 `[WARN] pre-existing drift`，不 block。
   - base 一致、head 不一致 → **本 PR 引入的** → `[FAIL]`。
3. 进一步区分原因：
   - head 中 marker 之间的内容，**是否存在 base 中也存在、且不等于渲染产物的行**（`diff base_block head_block` 里有非空的新增/删除行，且这些行不是数据里能生成的）→ 判为 **「疑似手改 auto 块」**，给专门的提示文案；
   - 否则判为 **「忘记重渲染」**。
4. 还要检查 marker 成对与唯一：`START AUTO RECENT PAPERS` 在文件里出现**恰好 1 次**、`END` 恰好 1 次、顺序正确；digest 表同理。现在 `replace_block` 用 `text.find()` 只找第一个，**重复 marker 会被静默忽略**。

**额外**：marker 之外**紧邻**的内容（前一后一行）不应该被自动块"吞掉"。实测 README L185 的 `<!-- START AUTO RECENT PAPERS -->` 前面紧贴着 `#### 2024` 小节的最后一条，这是正常的；但要防止有人把 marker 挪到 `#### 2024` 之前，导致渲染器把 2024 的手写条目也重写掉。建议检查：marker 所在行**必须是空行或紧跟渲染产物首行**。

**失败文案建议**

```text
[FAIL] Auto block modified manually — README.md

DEFAULT_DIGEST_TABLE 区块（L205-L231）与 data 的渲染结果不一致，
且以下行在 origin/main 中不存在，判断为本 PR 手工编辑：

  - | 2026 | [Hand-written Row](https://example.com) | ... |
  + | 2026 | [HySRD-Net](https://doi.org/...)         | ... |

处理方式（二选一）：
  1) 若这是有意新增的论文 → 请改 updates/curated/recent-papers.json，然后运行
       python3 scripts/render_curated_recent.py --data updates/curated/recent-papers.json
  2) 若已经手改并确认要保留 → 说明理由，并让 reviewer 手动确认
     （automation 分支合并不受影响，人工 PR 会被 block）

请不要直接运行 render 覆盖，否则手写内容会丢失。
```

**建议脚本**：`scripts/check_auto_blocks.py`
**说明**：可以与 `curated-render-check.yml` 共存——新脚本做「区分原因」，老 workflow 继续做「兜底一致性」。

---

### L1-6 · 论文条目格式规范

**检查逻辑**

1. **跨行合并**：先把「以 `^\d+\.\s+\[` 开头，且到行尾仍未出现 `[[paper]](`」的行和后续缩进行合并成一条逻辑条目。这一步是所有后续检查的前提——现状 170 行列表项里只有 102 行能被单行正则匹配（实测），跨行是主要来源。
2. 检查条目结构（针对 0x00 的论文类列表；`0x01/0x02/0x03/0x06` 的普通资源列表豁免）：
   - 必须匹配 `^\d+\.\s+\[<venue> <year>\]\s+<title>\s+\[\[paper\]\]\(<url>\)(\s+\[\[<role>\]\]\(<url>\))*$`；
   - `<venue> <year>` 中的年必须是 19xx/20xx 四位；
   - `<year>` 必须与所在 `#### <year>` 小节一致（warn）；
   - `[[paper]]` 必须存在且**恰好 1 个**（实测 README 里 `[[paper]]` 共 103 处、列表项 170 行，说明多数资源行没有 `[[paper]]`，所以这条只对 **0x00 论文小节**生效）；
   - 额外链接的 role 必须是白名单：实测已用 `paper/code/code-pytorch/code-official-tf/code-not-official/data/paper-ieee/review/SOTA` 9 种 → 白名单定这 9 个（+ 未来显式新增），其他一律 FAIL。这能直接挡住 `[[sota]]`/`[[SOTA]]` 大小写混用（实测现状是 `[[SOTA]]` 全大写 1 处、`[[SOTA]]` 与小写并存）。
   - 序号：同一小节内编号从 1 连续递增。实测 2015–2026 每年条目数为 `1,1,3,11,20,26,10,3,2,4,4,11`，编号连续。
3. 标题长度 ≥ 5、不含未转义的 `]]`。

**失败文案建议**

```text
[FAIL] Paper entry format (3 issues, README.md):

  L171  条目跨行：标题 "Pre-training Enhanced Spatial-temporal Graph Neural Network" 在 L171 后换行，
        导致 [[paper]] 链接落在 L172 —— 所有按行解析的工具（dedupe / hotspots）都会跳过它。
        fix: 合并为单行。

  L131  [[SOTA]] 不在允许的 role 白名单内：['paper','code','code-pytorch','code-official-tf',
        'code-not-official','data','paper-ieee','review']
        fix: 参考 docs/evolution/06-pipeline.md §5.4 —— [[SOTA]] 应指向本地 leaderboard，而不是外部站点。

  L234  #### 2018 小节内编号不连续：1, 2, 4
```

**建议脚本**：`scripts/check_paper_format.py`

---

### L1-7 · config 校验进 PR

**检查逻辑**

现成能力，只缺触发。新增 workflow（或扩 `readme-sync-check.yml` 的 paths）：

```yaml
on:
  pull_request:
    paths:
      - "scripts/configs/*.json"
      - "scripts/paper_search.py"
      - "scripts/validate_domain_config.py"
run: |
  for f in scripts/configs/*.json; do
    python3 scripts/validate_domain_config.py "$f" || exit 1
  done
```

注意：`validate_domain_config.py` 通过 `import paper_search` 复用解析逻辑，**必须从仓库根目录运行**（否则 `import` 失败）。实测：

```bash
python3 scripts/validate_domain_config.py scripts/configs/traffic-monthly.json
# => [OK] scripts/configs/traffic-monthly.json version=2 years=2024-2026 queries=5 rules=4 output_limit=25
```

**失败文案建议**（沿用脚本现有输出 + 补充上下文）

```text
[FAIL] domain config invalid: search.sources contains unsupported value 'openalex_v2'
Unsupported search source(s): openalex_v2
允许值: openalex, arxiv, semantic_scholar, crossref  (见 scripts/paper_search.py SUPPORTED_SOURCES)
```

**建议脚本**：复用 `scripts/validate_domain_config.py`，**不新建**

---

### L1-8 · 交叉引用完整性

**检查逻辑**（Phase 1 数据落地后启用）

1. 收集 `data/papers/*.json` 全部 `id`；`data/venues.json` 全部 `id`；`data/datasets/*.json` 全部 `id`；`data/research-groups.json` 全部 `id`；`data/toolkits.json` 全部 `id`。
2. 校验外键：
   - `paper.venue_id` ∈ venues；
   - `paper.datasets[]` ⊆ datasets；
   - `paper.affiliations[]` ⊆ research-groups；
   - `dataset.paper_ids[]` ⊆ papers；
   - `paper.code_links[].label` ∈ schema 里定义的 4 个枚举值。
3. 反向校验（孤儿检测，warn 不 fail）：venues/datasets 里没有任何 paper 引用的条目 → 列出来，可能是漏映射。
4. 引用环/自引用：`dataset.paper_ids` 里出现引用了该 dataset 的论文是合法的（论文用数据集），**不检查环**。

**失败文案建议**

```text
[FAIL] Cross-reference check failed (2 error(s)):

  data/papers/2026.json [7].datasets = ["metr-la", "pems08"]
    unknown dataset id 'pems08'  （候选: pems-bay, pems04, pems07, pems08? — 请确认 id 命名）
  data/datasets/pems-bay.json .paper_ids = ["stgcn-2017"]
    unknown paper id 'stgcn-2017'  （data/papers/2017.json 里是 'stgcn-2018'）

[WARN] Orphan entries (1): venues.json 里 'arxiv-preprint' 没有被任何 paper.venue_id 引用
```

**建议脚本**：`scripts/check_crossrefs.py`

> 注意：`data/schema/paper.schema.json` 已用 `pattern` 约束了 id 形态（`^[a-z0-9]+(-[a-z0-9]+)*-[0-9]{4}$`），但**形态合法 ≠ 存在**，所以这个检查仍然必要。

---

### L1-9 · 站点构建可过

**检查逻辑**

1. 只在 PR 触及 `site/**`、`data/**`、`updates/curated/**` 时触发。
2. 跑 `pnpm -C site install --frozen-lockfile` + `pnpm -C site build:fast`（跳过 Pagefind 索引以省时间）。
3. 断言关键产物存在：`site/dist/index.html`、`site/dist/zh/index.html`、`site/src/data/papers.generated.json`。
4. **不在 PR 里跑 `build`（含 Pagefind）**——索引生成慢且体积大，留给 L2-3。

**风险**：`site/package.json` 声明 `"packageManager": "pnpm@12.16.1"` 与 `engines.node >= 20`。CI 需要 `pnpm/action-setup` 装指定版本，否则 lockfile 差异会导致 `--frozen-lockfile` 失败。

**失败文案建议**

```text
[FAIL] Site build failed
  pnpm: command not found → 检查 .github/workflows 是否已加 pnpm/action-setup@v4 (version: 12.16.1)
  -- 或 --
  astro: Cannot find module './src/data/papers.generated.json'
  → prebuild (scripts/prepare-data.mjs) 未产出数据，检查 data/papers/*.json 是否存在可读记录。
```

**建议脚本**：无需新脚本，直接 `run: pnpm -C site build:fast`

---

### L1-10 · 拼写白名单检查

**检查逻辑**

不做全量拼写检查（会淹没有效信息，且大量专有名词会被误报）。只做**已知错词表**：

1. 维护 `scripts/configs/spelling-allowlist.json`：
   ```jsonc
   {
     "banned": ["Trafffic", "intergrating", "evalution", "insititution"],
     "allow": ["NeurIPS", "ICLR", "STGCN", "STGNN", "LibCity", "PyG", "HZJTD", "Apolloscape", "PeMS", "METR-LA", "…"]
   }
   ```
   实测这 4 个词在 `README.md` 与 `README.zh.md` **各出现 1 次**，说明是双语同步复制传播的。
2. 只扫 `README*.md`、`datasets/*.md`、`models/*.md` 的**正文**（跳过代码块、行内 code、URL）。
3. 命中 `banned` → FAIL，并把同一 PR 里另一个文件是否也命中一起报（提示同步修）。

**失败文案建议**

```text
[FAIL] Known misspellings (2):

  README.md:185     "Trafffic"  → "Traffic"     (GraphSparseNet: a Novel Method for Large Scale Trafffic Flow Prediction)
  README.zh.md:185  "Trafffic"  → "Traffic"     ← 同一处双语复制，请一并修

提示：若该拼写是论文原标题（作者笔误），不要改，而是加入 scripts/configs/spelling-allowlist.json 的 allow，
并在条目后加 `<!-- title-as-published -->` 注释。
```

**建议脚本**：`scripts/check_spelling.py`

> 这条必须允许「论文原标题就是错的」豁免——`GraphSparseNet` 标题原文确实写的是 `Trafffic`，我们要保持引用保真。

---

### L2-1 · 外链可用性（含「跳走」检测）

**检查逻辑**

1. 从 `README.md`、`README.zh.md`、`datasets/*.md`、`models/*.md` 提取所有 `](https?://...)`，去重。实测 README 有 **218 个外链、64 个 host**，去重后大约 150 个。
2. **必须跟随重定向**（`curl -L` 语义），并比较**最终 URL**与原始 URL：
   - `final_url != original_url` 且 **路径语义丢失**（例如 `paperswithcode.com/**` → `huggingface.co/papers/trending`）→ 判 **`redirect_away`**，这是本方案最关键的检测能力，**专为 pwc 这类「200 但跳走」设计**；
   - `final_url` 同 host、路径仍在 → 判 `redirect_ok`（如 `doi.org/x` → `linkinghub.elsevier.com/retrieve/pii/...`，实测正常，不应报错）。
3. 状态码分类：
   - `200` → ok
   - `403 / 429` → **`blocked`**，不计为失败（实测 `mdpi.com/2220-9964/13/6/210` 用 curl 返回 403，但浏览器正常）
   - `404 / 410` → `dead`
   - `5xx` / 超时（>15s）→ `unknown`，连续 2 周同一条 `unknown` 才升级为 issue
4. 限速与预算：单 host 并发 1、间隔 ≥ 1s（与 `HttpClient` 的策略一致）；总量上限 300 请求；`User-Agent` 复用 `paper-search` 的 UA 字符串。
5. 产出 `updates/reports/link-status.json`：
   ```jsonc
   {"checked_at":"2026-11-01","total":148,
    "results":[{"url":"https://paperswithcode.com/sota/...","status":"redirect_away",
                "final":"https://huggingface.co/papers/trending","file":"datasets/METR.md","line":9}]}
   ```
6. 期望基线：`scripts/configs/link-check-baseline.json` 记录「已知且已接受的异常」，新出现的 `dead/redirect_away` 才升级为 issue。

**为什么放 L2**：`mdpi` 403、`ieee` 反爬、`sciencedirect` 超时都是常态——实测 218 个链接里 19 个 ieeexplore、6 个 mdpi、5 个 sciencedirect，全部是反爬大户。放进 PR 门禁等于永久性 flaky。

**失败文案建议（issue body）**

```text
[link-check] 2026-11-01: 3 dead / 10 redirect_away

## redirect_away（链接仍返回 200，但已跳转且丢失原路径语义）
- datasets/METR.md:9   paperswithcode.com/sota/traffic-prediction-on-metr-la
                       → huggingface.co/papers/trending
  已规划处置：本地 leaderboard（docs/evolution/06-pipeline.md §5.4 规则 4）
- ...

## dead（404/410）
- README.md:427  paperswithcode.com/task/multivariate-time-series-forecasting/latest

## 本次新增（相对 baseline）
- 无

处理规则见 docs/evolution/06-pipeline.md §5.4。确认后请更新 scripts/configs/link-check-baseline.json 或修链接。
```

**建议脚本**：`scripts/check_links.py`

---

### L2-2 · DOI 解析

**检查逻辑**

1. 提取所有 DOI（`10.\d{4,9}/...`），归一化后去重。
2. 逐个 `HEAD https://doi.org/<doi>` 跟随重定向，`200` 视为可解析；`404` 视为 `unresolved`。
3. **额外做一次 OpenAlex 存在性检查**（批量，效率高，实测一次可取 50 条）：
   ```text
   https://api.openalex.org/works?filter=doi:<doi1>|<doi2>&per-page=50&select=id,doi,cited_by_count
   ```
   - DOI 在 OpenAlex 里不存在但 doi.org 能解析 → warn（通常是刚发表的新论文）；
   - **推荐用这个接口顺带产出 §06 的引用数快照**，一份请求做两件事。

**失败文案建议**

```text
[doi-check] 2026-11-01: unresolved=1
  README.md:152  doi 10.1049/itr2.70144 → HTTP 404 at doi.org
  可能原因：DOI 尚未注册 / 拼写错误 / 出版社撤回。
  若确认无效：把该条 status 设为 "dead_link"，或改指向 arXiv 版。
```

**建议脚本**：`scripts/check_dois.py`

---

### L2-3 · 站点构建产物 + 体积预算

**检查逻辑**

1. 跑完整 `pnpm -C site build`（含 `pagefind --site dist`）。
2. 断言产物：`dist/index.html`、`dist/zh/index.html`、`dist/pagefind/pagefind.js`、`dist/_data.json`。
3. **体积预算**（阈值建议，全部相对当前实测的仓库体量设定）：

| 产物 | 建议上限 | 依据 |
| --- | --- | --- |
| `dist/` 总体积 | 12 MB | 当前 `updates/` 4.2 MB + `datasets/images` 96 KB + 论文页 HTML |
| 单个页面 HTML | 400 KB | 论文列表页随条目增长 |
| JS chunk（ECharts 单独 chunk） | 900 KB | `astro.config.mjs` 里 `chunkSizeWarningLimit: 900`，与配置对齐 |
| `pagefind` 索引 | 3 MB | 文本量级决定 |

4. 超预算 → **warn + 写 report**，不 fail（避免因为加一个视频就 build 挂掉）。

**失败文案建议**

```text
[site-budget] dist/ = 14.2 MB (limit 12 MB)  ⚠️
  largest: dist/papers/2026/index.html  612 KB (limit 400 KB)
           趋势：条目数 2026=11 → 若按当前增速，下季度将超限
  建议：论文列表分页（每页 100 条）或按年份拆路由。
```

**建议脚本**：`scripts/check_site_budget.py`

---

### L2-4 · 仓库体积报告

**检查逻辑**

1. `du -sh` 度量：根目录、`updates/wordcloud`、`updates/hotspots`、`updates/raw`、`datasets/resources`、`papers/`、`.git`。
2. 阈值（现状基线）：`updates/wordcloud` 3.4 MB、`updates/hotspots` 808 KB、`datasets/resources/PeMS-M.zip` 6.1 MB、`papers/` 16 MB、`.git` 27 MB。
3. 规则：
   - 任何 `updates/**` 下的可再生产物 **> 5 MB** → warn；
   - `updates/wordcloud/` 归档 PNG 数量 > 12（即超过一年）→ warn，提示按月轮转；
   - `data/` + `site/src/data/` 合计 > 8 MB → warn（JSON 应该远小于这个）。
4. 产出 `updates/reports/repo-size.json` 并保留历史，便于看趋势。

**失败文案建议**

```text
[repo-size] 2026-11-01
  .git                  27 MB   (papers/ 16 MB 中的 PDF 历史版本占大头)
  papers/               16 MB
  updates/wordcloud      3.4 MB  (2 个归档 PNG + 2 个 latest)
  updates/hotspots       808 KB
  datasets/resources     6.1 MB  (PeMS-M.zip)
  趋势：updates/ 近 3 月 +0 B（未更新）
  ⚠️ 提示：如需长期保留月度归档，考虑把 archives 移到 release assets 而非 git。
```

**建议脚本**：`scripts/report_repo_size.py`

---

### L2-5 · 词云/热点产物新鲜度

**检查逻辑**

1. 读 `updates/wordcloud/monthly-paper-wordcloud-latest.json` 的 `generated_utc` 与 `updates/hotspots/yearly-hotspots.json` 的 `generated_utc`（两个文件都有这个字段，实测）。
2. 与 `updates/monthly-paper-candidates.md` 头部的 `Generated on:` 比较。
3. 若 candidates 比图表新超过 **7 天** → warn「图表落后于候选数据」。
4. 若任一处 `generated_utc` 超过 **60 天** → warn「月度流水线可能已停摆」。

**为什么这条有价值**：`monthly-paper-search.yml` 一旦静默失败（§06 §7.1 的「静默空窗」），仓库里没有任何信号。这条门禁就是那个信号。

**失败文案建议**

```text
[artifact-freshness] ⚠️
  updates/wordcloud/.../latest.json  generated_utc = 2026-10-01  (54 天前)
  updates/monthly-paper-candidates.md Generated on = 2026-10-01
  → 月度流水线可能已停摆 ≥ 1 个周期。检查 Actions → Monthly Paper Search 最近 3 次运行。
```

**建议脚本**：`scripts/check_artifact_freshness.py`

---

## 4. PR 模板改造建议

> **这是建议文本。本任务不创建 `.github/pull_request_template.md`，也不修改现有 `CONTRIBUTION.md`。**

### 4.1 建议的 `.github/pull_request_template.md`

```markdown
## 变更类型

- [ ] 新增论文
- [ ] 新增数据集 / 工具 / 团队
- [ ] 修正已有条目（标题/年份/链接）
- [ ] 修改结构化数据（`data/**`、`updates/curated/**`）
- [ ] 修改自动化或 CI（`scripts/**`、`.github/**`、`site/**`）
- [ ] 纯文档（`docs/**`）

## 自检（提交前请全部勾选）

- [ ] 没有直接编辑自动块：
      `<!-- START AUTO RECENT PAPERS -->` … `<!-- END AUTO RECENT PAPERS -->`
      `<!-- START AUTO CURATED DIGEST TABLE -->` … `<!-- END AUTO CURATED DIGEST TABLE -->`
      新增论文请改 `updates/curated/recent-papers.json`（Phase 1 后为 `data/papers/*.json`）
- [ ] 中英文 README 在同一 PR 内同步修改
- [ ] 已确认不是重复条目（按 DOI **和** 标题都查过）
- [ ] 新增的外部链接我本人在浏览器里点开过，且**落地页面内容与条目描述一致**
- [ ] 论文条目保持单行（`[[paper]]` 链接不要换行到下一行）

## 论文条目的元数据（新增论文时必填）

| 字段 | 值 |
| --- | --- |
| Conference/Journal + Year | 例：`TITS 2023` |
| 标题（与论文原文完全一致，含原文笔误） | |
| DOI 或 arXiv id | |
| 是否官方代码仓库 | 是 / 否 / 无 |
| 建议的 `method_tags`（≤3 个，从现有词表选） | |

## 本地已运行的检查

```bash
python3 scripts/check_readme_sync.py --en README.md --zh README.zh.md
python3 scripts/render_curated_recent.py --data updates/curated/recent-papers.json \
  --readme-en README.md --readme-zh README.zh.md --check
python3 scripts/check_schema.py --root .          # Phase 1 后可用
python3 scripts/check_duplicates.py --root .      # Phase 1 后可用
```

## 给 reviewer 的提示

- 涉及 `scripts/**` 或 `.github/**` 时请至少一位 maintainer 复核
- 若本 PR 含「已知不可达的外链」，请在下面说明原因（会被加入 `scripts/configs/link-check-baseline.json`）

<!-- 说明：本 PR 是否由自动化生成？
     - 月度候选 PR 请保留分支名 automation/monthly-paper-refresh，并在正文标注 degraded 状态。 -->
```

### 4.2 建议的 `CONTRIBUTION.md` 改造要点（**仅建议，不修改原文件**）

| 现有内容（行号） | 建议 |
| --- | --- |
| L13–L18「Before You Open a PR」4 条 | 扩为 6 条：补「不要手改 auto 块」与「新增外链必须人工点开验证落地页」 |
| L22–L26「Paper Entries」 | 补：①条目必须**单行**（跨行会让自动化解析失败）；②`[[...]]` role 只能从白名单选（列出 9 个）；③标题保持论文原样，**包括原文笔误**，不要"顺手修正"（否则 dedup 与引用都对不上） |
| L34–L43「Bilingual Consistency」 | 现状只教跑 `check_readme_sync.py`。补：该脚本**只比数量**，内容对齐需人工保证，并在 Phase 1 后由 `check_bilingual_parity.py` 覆盖 |
| L45–L55「Structured Recent Papers」 | 补：`recent-papers.json` 的字段约束指向 `data/schema/`（Phase 1 落地后）；说明 `featured: true` 才进 digest 表，`task_en/task_zh/note_en/note_zh` 四项缺一不可 |
| L57–L62「Pull Request Checklist」 | 改为引用 PR 模板的复选框，避免两处清单漂移 |
| L64–L67「Merge Policy」 | 补：自动化分支（`automation/*`）合并前必须确认 PR body 里没有 `degraded` 标记，或已明确接受降级 |
| 无 | 新增「链接失效怎么处理」一节：指向 `docs/evolution/06-pipeline.md §5.4` 的处置矩阵，说明 `paperswithcode.com` 系列的现状与替代方案 |
| 无 | 新增「数据与渲染的关系」一节：Phase 1 后 `data/papers/*.json` 是唯一数据源，README 的 auto 块是渲染产物，两者冲突时以数据为准 |

---

## 5. 门禁与现有 CI 的兼容矩阵

| 现有 workflow | 本方案是否替换 | 处理 |
| --- | --- | --- |
| `readme-sync-check.yml` | **保留**，新增 `check_bilingual_parity.py` 并行 | 老脚本的「计数」检查是快速兜底，成本近乎 0，不要删 |
| `curated-render-check.yml` | **保留**，新增 `check_auto_blocks.py` 做原因区分 | 两者职责不同：老的是「一致性兜底」，新的是「归因 + 友好文案」 |
| `monthly-paper-search.yml` | Phase 1 拆分（详见 06 文 §6） | `Validate domain config` step 独立成 L1-7 的 PR 门禁 |
| `opencode.yml` | **不动** | 人工触发的 AI 助手，与门禁无关 |

新增 workflow 建议命名（全部 `workflow_dispatch` 可手动跑）：

```text
.github/workflows/pr-gates.yml        # L1 合集：schema / json / parity / dup / auto-block / format / crossref
.github/workflows/site-build.yml      # L1-9
.github/workflows/link-check.yml      # L2-1 + L2-2，cron 每周一
.github/workflows/site-budget.yml     # L2-3，cron 每周一
.github/workflows/repo-report.yml     # L2-4 + L2-5，cron 每月 1 日
```

合并 L1 到单个 workflow 而不是 10 个 workflow 的理由：GitHub UI 上一个 `PR Gates` 的勾选比 10 个散落的名字更容易读；同时用 `matrix` 或串行 step 保证 60s 内完成。

---

## 6. 门禁的元规则（防止门禁本身变成噪音）

1. **每条 L1 门禁必须 < 15s**，且**零网络**。任何需要网络的检查一律进 L2。
2. **失败文案必须包含三要素**：`文件:行` + `为什么这很重要` + `怎么修`。只写 `FAILED` 的门禁会被无视。
3. **新增门禁先以 `continue-on-error: true` 跑两周**，确认误报率 < 5% 再转为 blocking。
4. **每个 L2 检查必须有 baseline 文件**（`scripts/configs/link-check-baseline.json`），否则历史遗留问题会让 issue 永远开着，最后被静音。
5. **门禁检查的对象要能被单独运行**：所有 `scripts/check_*.py` 必须支持 `--root .` 从仓库根跑，并且 `python3 scripts/check_x.py --help` 有说明。这是「本地能自查」的前提——否则贡献者只能靠 CI 反馈，往返成本高。

---

## 7. 与其他文档的关系

- **06-pipeline.md**：本部门禁的检查对象（`updates/raw|normalized|reports/` 结构、5 段 dedup key、pwc 处置矩阵）全部定义在那里。
- **08-roadmap.md**：L1 门禁集中在 Phase 1（数据基座），L2 门禁分散在 Phase 1 末期与 Phase 3。
- **09-acceptance.md**：给出每条门禁的**可执行验收命令**与「现在就能跑 / 需等落地」的标注。