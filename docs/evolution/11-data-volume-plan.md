# 11 · 数据量迁移计划（Data Volume Plan）

> 本文回答三个问题：**`data/papers/` 分几批迁完、每批多少条、按什么顺序迁**；以及为什么这个顺序不是"从早到晚顺着来"。
>
> 硬约束来自 `site/scripts/prepare-data.mjs` 的完整性护栏（见 [10-addendum-reality-check.md](./10-addendum-reality-check.md) §4.2）：
>
> ```js
> const readmeCount = (await scrapeReadme('README.md', 'en')).length;   // 实测 = 104
> const minExpected = Math.max(1, Math.floor(readmeCount * 0.75));       // 实测 = 78
> if (papers.length < minExpected) return null;                          // 静默回退 README 抓取
> ```
>
> **`data/papers/` 必须 ≥ 78 条，站点才会真正使用结构化数据；低于 78 条时站点 build 依然成功，但用的是 README 刮取结果。**

---

## 0. 口径对齐：为什么「条目数」会数出 102 / 104 / 92 三个数

同一个仓库、同一天，不同工具会报出不同的「论文有多少条」。这些数不是矛盾的，只是**统计边界不同**。迁移计划必须钉死一个口径，否则验收永远吵不清。

| 口径 | 数值 | 定义方式 | 用途 |
| --- | ---: | --- | --- |
| **A · 迁移口径**（`migrate_readme_to_json.py`） | `--all` = **89**（2015–2024）+ curated 路径 **15**（2025–2026）= **104** | 只认 `## 0x00` + `### <SECTION_MAP 命中>` 的层级；**跳过 `AUTO RECENT PAPERS` 块**；`year` 取 `[Venue YYYY]` 标签年，无标签时取 `#### YYYY` 小节年 | **本文的唯一权威口径** |
| **B · 深度学习小节口径**（02 文） | **92** | 只数 `### Deep Learning Based Traffic Prediction Methods` 下的年份小节 | 内容规划（哪些年份缺论文） |
| **C · 站点护栏口径**（`prepare-data.mjs`） | **104** | `scrapeReadme()` → `parseLine()`，**不跳过** auto 块，`section` 匹配 `/0x00|papers/i` | **决定 78 这个阈值** |

**结论**：**护栏输入是 104，阈值是 `floor(104 × 0.75) = 78`。**

- A（104）与 C（104）**数值相同但路径不同**：A 是"脚本 + curated JSON"两段拼出来，C 是"直接刮 README"。两者当下恰好一致（因为 curated 的 19 条里有 15 条与 README auto 块一一对应），但这个一致**不是结构保证**——B4 批次的任务之一就是把这两条路径合并。
- B（92）只是 A 的一个子集，**不用于护栏**，仅用于判断"哪年内容偏薄"。

### 0.1 各年份条目数（迁移口径 A，权威来源 = 运行迁移脚本本身）

**别再自己写正则复刻**——直接问迁移脚本。以下是 `iter_readme_entries()` 的实测输出（2026-11-24）：

```bash
python3 - <<'PY'
import sys, pathlib
sys.path.insert(0, "scripts")
import migrate_readme_to_json as M
from collections import Counter
es = M.iter_readme_entries(pathlib.Path("README.md"))
print("entries:", len(es))
print("by year:", dict(sorted(Counter(e["year"] for e in es).items())))
print("by section:", Counter(e["section"] for e in es))
PY
# 实测:
#   entries: 89
#   by year: {2015:2, 2016:1, 2017:3, 2018:10, 2019:22, 2020:31, 2021:10, 2022:4, 2023:2, 2024:4}
#   by section: {'deep_learning': 78, 'reviews': 6, 'statistic': 2, 'time_series': 2,
#                'temporal_network_embedding': 1}
```

| 年份 | 2024 | 2023 | 2022 | 2021 | 2020 | 2019 | 2018 | 2017 | 2016 | 2015 | `--all` 小计 | 2025 | 2026 | **总计** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 条目数 | 4 | 2 | 4 | 10 | 31 | 22 | 10 | 3 | 1 | 2 | **89** | 4 | 11 | **104** |

**为什么是 89 + 15 而不是一次 104**：`iter_readme_entries()` 在遇到 `<!-- START AUTO RECENT PAPERS -->` 时 `in_auto = True`，**整块跳过**——2025 与 2026 的 15 条全在这个 auto 块里。它们由 `updates/curated/recent-papers.json`（19 条，其中 2025×4、2026×11 与 README 对应）驱动。所以：

- `--all` 覆盖 **10 个年份、89 条**（2015–2024）；
- `data/papers/2025.json`（4 条）与 `2026.json`（11 条）来自 **curated JSON 的另一条路径**；
- 89 + 15 = **104**，与 `data/papers/` 当前实际条数（实测 104）吻合。

> ⚠️ **一个容易踩的坑**：用「`[Venue YYYY]` 标签年 or 小节年」自己复刻正则（我最初的做法）会得到 `{2015:1,…,2018:11, 2019:20, 2020:26, 2021:10, 2022:3, 2023:2, 2024:4, 2025:4, 2026:11, null:8}`——**和迁移脚本的结果不同**。原因是 `iter_readme_entries()` 只认 `## 0x00` 开头（`TOPIC = "0x00"`）+ `### <SECTION_MAP 命中>` 的层级，并且**跳过 auto 块**；自己写的宽松正则会把 `Reviews`/`Temporal Network Embedding` 等副标题下的条目算进不同的 section 与年份。**教训：批次条目数必须从脚本本身取，不要复刻正则。**

**关键事实**：`2020`（31）+ `2019`（22）= **53 条，占全库 51%**。这一条直接决定了「怎么排批次最快过护栏」。

---

## 1. 批次表

批次划分原则：**按内容连贯性切**（同一研究阶段的论文放一批，便于一次性做对 `layers` / `direction_tags`），而不是按"最省事"切。

| 批次 | 年份范围 | 条目数 | 累计 | 过 78 护栏？ | 主要内容 | 预估工时 | 主要工作量 |
| --- | --- | ---: | ---: | :---: | --- | ---: | --- |
| **B0** | 2015–2017 | **6** | 6 | ❌ | 早期深度学习尝试（LSTM/SAE 类），基本冻结 | **1.0** 人日 | 迁移脚本跑通 + 逐条核对 venue 短名 |
| **B1** | 2018–2019 | **32** | 38 | ❌ | STGCN/DCRNN/GraphWaveNet/LC-RNN/GeoMAN/Multi-View 等**经典基线区** | **3.5** 人日 | 反复出现同一模型的多个版本，`layers` 标注最费时 |
| **B2** | 2020–2021 | **41** | **79** | ✅ **首次过线** | 全库峰值年（2020=31）；Transformer 化起点、动态图萌芽 | **4.0** 人日 | 条目最多，`venue` 短名异写最密集（`WWWC/WC/WWW`、`IF/Information Fusion`） |
| **B3** | 2022–2024 | **10** | 89 | ✅ | 内容断层区（README 里这三年共 10 条，明显偏低） | **1.5** 人日 | 条目少，但**需要配合 02 文的补录计划**才有意义 |
| **B4** | 2025–2026 | **15** | **104** | ✅ | `AUTO RECENT PAPERS` 块内的当代论文；与 `updates/curated/recent-papers.json` 有 19 条重叠 | **2.0** 人日 | 需要与 curated JSON 做字段对齐（`short_name`/`task_*`/`method_tags`/`featured`） |
| | **合计** | **104** | | | | **12.0** 人日 | |

**工时口径**：按「条目数 × 单条约 2.5 分钟」+ 每个批次 0.3 人日的固定开销（跑脚本、读 diff、改 `venues.json`）估算。B4 因需与 curated JSON 对齐，单价更高（约 8 分钟/条）。

### 1.1 为什么 B3 只有 10 条、却仍然单独立批

因为 B3 的**问题不在条目数，在内容**。README 里 2022=4、2023=2、2024=4，三年合计 10 条；对比 2020 的 31 条，断层非常明显（02 文已指出「2022 断崖」）。所以 B3 迁完后**必然紧接着一次内容补录**（那是 02 文的范围，不计入本文工时）。

把 B3 单独成批的好处：迁移与补录的边界清晰——**迁移只搬已有的，补录只加新的**，两件事可以在两个 PR 里 review。

---

## 2. 施工顺序：先 B3+B4 还是先 B2？——修正 Lead 的假设

### 2.1 Lead 的假设需要纠正

Lead 提出：*「为什么先 B3+B4：既能立刻修复'2023 后顶会归零'的内容问题，又能用最小条目数把站点救出降级态？」*

**前半句对，后半句不成立。** 核算如下：

| 顺序方案 | 步骤 | 累计条目 | 过护栏（≥78）？ |
| --- | --- | ---: | :---: |
| **方案 X（先 B3+B4）** | B3 | 10 | ❌ |
| | B3+B4 | 25 | ❌ |
| | B3+B4+B2 | 66 | ❌ |
| | B3+B4+B2+B1 | **98** | ✅ |
| | B3+B4+B2+B1+B0 | 104 | ✅ |
| **方案 Y（B0→B4 顺推）** | B0 | 6 | ❌ |
| | B0+B1 | 38 | ❌ |
| | B0+B1+B2 | **79** | ✅ **首次过线** |
| | +B3 | 89 | ✅ |
| | +B4 | 104 | ✅ |

**结论**：

- **「最小条目数把站点救出降级态」是错的** —— 护栏是**绝对条数**门槛（≥78），不是"覆盖了最新年份"就算数。B3+B4 只有 **25** 条，离 78 差 53 条。
- **B2 是唯一能"一批过线"的批次**（41 条，B0+B1+B2 = 79 ≥ 78，恰好过线）。但 B2 单独也不够（41 < 78）。
- **任何"只迁新不迁旧"的方案都必然过不了护栏**，因为 2019–2020 两年就占了 53 条。

### 2.2 推荐顺序：**B2 → B1 → B4 → B3 → B0**（按"过线速度 + 内容价值"双目标）

| 步 | 批次 | 增量 | 累计 | 护栏 | 这一步拿到什么 |
| --- | --- | ---: | ---: | :---: | --- |
| 1 | **B2**（2020–2021，41 条） | +41 | 41 | ❌ | 峰值区一次性做完，建立 `venue` 归并表（31 条 2020 里几乎涵盖所有 venue 异写） |
| 2 | **B1**（2018–2019，32 条） | +32 | **73** | ❌（差 5 条） | 经典基线区；与 B2 相加 73，**只差 5 条过线** |
| 3 | **B4**（2025–2026，15 条） | +15 | **88** | ✅ **过线** | 当代论文 + `curated JSON` 合并；这一步之后站点才真正用结构化数据 |
| 4 | **B3**（2022–2024，10 条） | +10 | 98 | ✅ | 补齐断层区（接着做 02 文的内容补录） |
| 5 | **B0**（2015–2017，6 条） | +6 | 104 | ✅ | 收尾；早期条目最少争议 |

**为什么这个顺序而不是 B0→B4 顺推**：

1. **B2 第一**：41 条 + 涵盖绝大多数 venue 异写 → 先做它，`data/venues.json` 一次建好，后面 63 条都能直接复用，避免每批都改 venue 表。
2. **B1 第二**：B2+B1 = 73，**只差 5 条**。此时做 B4（15 条）就是"用最少的额外工作跨过护栏"，比补 B0（6 条，工程上要单独跑一遍早年数据）更划算。而且 B4 本身有内容价值（修复"最新论文只在 markdown 里"的问题）。
3. **推迟 B0**：2015–2017 只有 6 条，且是"最不容易出错"的一批，放在最后做压力最小。
4. **B3 放在 B4 之后**：B3 迁完就要紧接着补录内容（02 文），两件事连着做上下文成本最低，但它不是过护栏的必要条件。

**若只有 3.5 人日预算**：做 B2 + B1（73 条，差 5 条过线）。此时站点仍然 degraded——**所以要么凑够 78，要么不要宣称"站点已用结构化数据"**。这一点必须诚实。

**若要最快过线（2.5 人日）**：B2 + B1 + B0 = 79 条 ✅（6+32+41）。代价是 B4 的当代论文继续留在 markdown-only 状态，`curated JSON` 的双体制没解决。**不推荐**，但这是一个合法的最小路径。

---

## 3. 每批的验收命令

每批都是同一个五步闭环。命令可直接复制。

### 3.1 通用五步（每批都跑）

```bash
cd "$(git rev-parse --show-toplevel)"
BATCH_YEARS="2020 2021"          # ← 每批改这里

# ① 干跑：看会写什么，不落盘
python3 scripts/migrate_readme_to_json.py --dry-run $(for y in $BATCH_YEARS; do echo --year $y; done)
# 期望: 每个年份一行 [DIFF] / [SAME]，末尾 [SUMMARY] years=N records=M doi=.. arxiv=.. code=.. no_paper_url=..
#       records 应等于该批条目数之和

# ② 真跑
python3 scripts/migrate_readme_to_json.py $(for y in $BATCH_YEARS; do echo --year $y; done)

# ③ 幂等性：立刻再跑一次，必须 [SAME] 全绿、且不产生 diff
python3 scripts/migrate_readme_to_json.py --check $(for y in $BATCH_YEARS; do echo --year $y; done)
# 期望: [OK] data/papers matches README.

# ④ 结构校验（schema 落地后可用；现在用等价断言）
python3 - <<'PY'
import json, glob, sys
YEARS = {2020, 2021}          # ← 每批改这里
total = 0
for y in sorted(YEARS):
    f = f"data/papers/{y}.json"
    rows = json.load(open(f))
    total += len(rows)
    # 必备字段
    for i, r in enumerate(rows):
        for k in ("id", "year", "title", "paper_url"):
            assert k in r, f"{f}[{i}] 缺 {k}"
        assert r["year"] == y, f"{f}[{i}] year={r['year']} ≠ 文件名"
        assert r["id"].endswith(f"-{y}"), f"{f}[{i}] id={r['id']} 不以 -{y} 结尾"
    print(f"{f}: {len(rows)} records OK")
print("batch total:", total)
PY

# ⑤ 条目数守恒：该批的 JSON 条数 == README 该年条目数
python3 - <<'PY'
import json, re, pathlib
YEARS = {2020, 2021}          # ← 每批改这里
# README 侧（口径 A：parseLine 等价实现）
LINK = re.compile(r"\[\[([A-Za-z0-9_-]+)\]\]\((https?://[^)\s]+)\)")
def parse(text):
    out, sec, cur = [], None, None
    for line in text.splitlines():
        if line.startswith("## "): sec = line[3:].strip(); cur = None; continue
        if line.startswith("### "): cur = None; continue
        m = re.match(r"^####\s+(\d{4})\s*$", line)
        if m: cur = int(m.group(1)); continue
        if not re.search("0x00|papers", sec or "", re.I): continue
        m = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if not m: continue
        body = m.group(1).strip()
        links = dict((k.lower(), v) for k, v in LINK.findall(body))
        rest = LINK.sub("", body).strip()
        if not rest or rest in ("……", "..."): continue
        t = re.match(r"^\[([^\]]{2,60}?)\s+(\d{4})\]\s*", rest)
        if t: rest = rest[t.end():].strip()
        title = rest
        tl = re.match(r"^\[([^\]]+)\]\((https?://[^)\s]+)\)", rest)
        if tl: title = tl.group(1).strip()
        title = re.sub(r"\s+", " ", title).strip()
        if not title or len(title) < 8: continue
        y = int(t.group(2)) if t else cur
        if y: out.append(y)
    return out
yrs = parse(pathlib.Path("README.md").read_text())
for y in sorted(YEARS):
    n_json = len(json.load(open(f"data/papers/{y}.json")))
    n_md = yrs.count(y)
    flag = "OK" if n_json == n_md else "MISMATCH"
    print(f"{y}: md={n_md} json={n_json} {flag}")
    assert n_json == n_md, f"{y} 条目数不守恒"
PY
```

### 3.2 每批的「过护栏」判定

```bash
# 累计条目数 vs 阈值
python3 - <<'PY'
import json, glob
total = sum(len(json.load(open(f))) for f in glob.glob("data/papers/*.json"))
readme_count = 104          # 实测，口径 A
threshold = readme_count * 3 // 4
print(f"structured={total}  threshold(floor 0.75×{readme_count})={threshold}  "
      f"{'PASS' if total >= threshold else 'FAIL — 站点将回退 README'}")
PY

# 站点实际采用了哪个源（唯一可信的判定）
cd site && ASTRO_TELEMETRY_DISABLED=1 pnpm build:fast >/dev/null 2>&1 && cd ..
python3 -c "
import json
d = json.load(open('site/src/data/papers.generated.json'))
print('source:', d['provenance'].get('source'), '| count:', d['count'])
assert d['provenance'].get('source') == 'data/papers', '仍在使用 README 抓取！'"
```

### 3.3 批次专属验收

| 批次 | 额外验收 |
| --- | --- |
| **B0**（2015–2017） | 3 个文件各 2/1/3 条；`venue` 是否包含非标准短名（需登记进 `venues.json`） |
| **B1**（2018–2019） | 同一模型的多版本（如 DCRNN 的 ICLR 版 + arXiv 版）确认**没有**重复记录；`alt_urls` 里保留第二版本 URL |
| **B2**（2020–2021） | `venue` 异写归并完成：`WWWC`/`WC`/`WWW` → 同一 `venue_id`；`IF`/`Information Fusion` → 同一 `venue_id`；归并后渲染仍输出**原短名**（不能改变 README 观感） |
| **B3**（2022–2024） | 与 02 文的内容补录批次做接口对齐：补录只 **append**，不重排 `order` |
| **B4**（2025–2026） | 与 `updates/curated/recent-papers.json` 的 19 条对齐：`short_name` / `task_en` / `task_zh` / `method_tags` / `featured` / `code_url` 必须与 curated 一致（curated 是人工权威） |

---

## 4. 迁移反解脚本的边界情况清单（可执行断言）

以下每一条都是**实测存在于本仓库**的真实边界，不是假设。每条给：**现象 → 断言 → 修法**。

### E1 · 跨行折断的条目 【当前 FAIL · 必修】

- **现象**：`README.md:170–171`
  ```markdown
  2. [Arxiv 2022] Pre-training Enhanced Spatial-temporal Graph Neural Network
     for Multivariate Time Series Forecasting [[paper]](https://arxiv.org/pdf/2206.09113.pdf)
  ```
- **实测后果**：迁移后落成 `pre-training-enhanced-spatial-temporal-graph-neu-2022`，**`title` 只有第一行**、**`paper_url: null`**。
  ```
  pre-training-enhanced-spatial-temporal-graph-neu-2022
    title = "Pre-training Enhanced Spatial-temporal Graph Neural Network"   ← 副标题丢了
    paper_url = null                                                        ← 链接丢了
  ```
  且它**违反 schema**（`paper_url` 是 `required`），所以 E11 也一起红。
- **断言**：
  ```python
  recs = json.load(open("data/papers/2022.json"))
  hit = [r for r in recs if "Pre-training Enhanced" in r["title"]]
  assert len(hit) == 1, f"跨行条目丢失（found {len(hit)}）"
  assert "Multivariate Time Series Forecasting" in hit[0]["title"], "标题只保留了第一行"
  assert hit[0]["paper_url"] and hit[0]["paper_url"].endswith("2206.09113.pdf"), hit[0]["paper_url"]
  ```
  **当前实测：两条断言都失败。**
- **修法**：迁移时的条目切分必须**合并续行**。判据：若一行匹配 `^\d+\.\s+` 但**不含** `[[paper]](`，则把后续缩进（或紧邻的非编号）行并入本条，直到出现 `[[paper]](` 或下一条编号。注意不能简单按"缩进"判断——这一条的续行是**无缩进**的。

### E2 · `)[[code]]` 紧贴、无空格

- **现象**：`README.md:115` 类条目，`[[code]]` 直接贴在 `[[paper]](...)` 后面且无空格。
- **影响**：宽松正则会把 `)[[code]]` 当成 paper URL 的一部分。
- **断言**：
  ```python
  for f in glob.glob("data/papers/*.json"):
      for r in json.load(open(f)):
          assert "]](" not in r["paper_url"], f"{r['id']}: paper_url 污染 → {r['paper_url']}"
          assert r["paper_url"].endswith((".pdf", "/", *".0123456789")) or "?" in r["paper_url"] or True
  ```
  更严格的版本：断言所有 `paper_url` 都能通过 `^https?://[^)\s\]]+$`。
- **修法**：先抽 `[[role]](url)` 全部匹配，再把剩余文本当标题；不要用"从 `](` 到行尾"这种贪婪切法。

### E3 · 非 ASCII 连字符

- **现象**：至少 3 处标题用 U+2010/U+2013 而非 ASCII `-`：`README.md:104`（`Spatial–Temporal`）、`:157`（`Spatial‐temporal`，U+2010）、`:183`（`spatial–temporal`）。
- **影响**：`re.sub(r"\W+","")` 会吃掉它们 → `spatialtemporal`，与正常拼写不等；**去重键会失效**。
- **断言**：
  ```python
  import unicodedata
  for f in glob.glob("data/papers/*.json"):
      for r in json.load(open(f)):
          weird = [c for c in r["title"] if c in "\u2010\u2011\u2012\u2013\u2014"]
          assert not weird, f"{r['id']}: 标题含非 ASCII 连字符 {weird}"
  ```
  > ⚠️ 注意矛盾：schema 要求 `title` 是「copied verbatim from README/venue」，而 verbatim 就带非 ASCII 连字符。**正确做法**：`title` 保留原样（不要 normalize），**在 dedup 层做归一化**；断言改为「归一化后的 key 不含非 ASCII 字符」。
- **修法**：
  ```python
  def title_norm(t):                     # 只在去重键里用，不改存储的 title
      t = unicodedata.normalize("NFKD", t)
      t = t.replace("\u2010","-").replace("\u2013","-").replace("\u2014","-")
      return re.sub(r"[^a-z0-9]+", "", t.lower())
  ```

### E4 · 裸 URL 条目（无 `[title](url)` 包裹）

- **现象**：`README.md:366` `1. http://urban-computing.com/yuzheng`、`:415` `2. https://github.com/topics/traffic-prediction`。
- **影响**：`parseLine` 会把它当标题（长度够 8），生成一条 title 为 URL 的记录。
- **断言**：
  ```python
  for f in glob.glob("data/papers/*.json"):
      for r in json.load(open(f)):
          assert not r["title"].startswith("http"), f"{r['id']}: title 是裸 URL"
  ```
  同时断言这两条**不在** `data/papers/`（它们属于 0x05/0x06，不该进论文库）：
  ```python
  assert not any("urban-computing.com/yuzheng" in r["paper_url"]
                 for f in glob.glob("data/papers/*.json") for r in json.load(open(f)))
  ```

### E5 · `…` / `……` 占位条目

- **现象**：每节末尾都有，`README.md` 里多处。
- **影响**：`prepare-data.mjs` 与 `migrate_readme_to_json.py` 都会显式跳过它们；但自己写正则复刻的工具会把 `……` 当成标题的一部分，导致条目数虚高。
- **自测差异对照**：用宽松正则统计 README 得到 102 条，而 `parseLine`（站点护栏用）得到 **104**，`iter_readme_entries`（迁移脚本，跳过 auto 块）得到 **89+15**。**这三个数的差异不全是占位条目造成的**——主要来自"是否跳过 auto 块"和"section 层级判定"。占位条目只是其中一个次要来源，不要把它当成主因。
- **断言**：
  ```python
  for f in glob.glob("data/papers/*.json"):
      for r in json.load(open(f)):
          assert "…" not in r["title"] and r["title"] not in ("...",), r["id"]
  assert not any("……" in r["title"] for f in glob.glob("data/papers/*.json")
                 for r in json.load(open(f)))
  ```

### E6 · 年份与所在小节不一致

- **现象**：`README.md:89` 是 `[TITS 2019]` 但位于 `#### 2018`；`:156` 是 `[Arxiv 2020]` 但在 `#### 2021`。
- **影响**：口径 A 取**标签年**，所以这两条会进 `2019.json` / `2020.json`，而 README 里它们显示在 2018/2021 小节 → **渲染回去会换位置**，`render_readme.py --check` 立刻报 diff。
- **断言**：
  ```python
  # 迁移后，每条记录的 year 必须等于标签年；同时记录它与 README 小节年份的差异
  for f in glob.glob("data/papers/*.json"):
      for r in json.load(open(f)):
          pv = (r.get("provenance") or {}).get("readme_line")
          assert r["year"] == int(f.split("/")[-1][:4]), f"{r['id']} 的 year 与文件名不符"
  # 显式白名单（只有这 2 条允许"标签年 ≠ 小节年"）
  KNOWN_OFFSET = {"stgcn-2018", "stfgcn-2021"}   # ← 实际 id 以落地为准
  ```
- **修法**：`render_readme.py` 必须支持「按 `provenance.readme_line` 的原位置渲染」，或者接受这 2 条被移动并在 `rebuild-allow-diff.md` 里记录。**推荐前者**（保真度优先）。

### E7 · 省略号小节 / 空小节

- **现象**：`### Trajectory` 小节是空的（无条目）；`### ETA` 只有 1 条。
- **影响**：`section` 字段的枚举里没有这些值（`reviews/deep_learning/statistic/time_series/temporal_network_embedding`），强塞会违反 `additionalProperties`/`enum`。
- **断言**：
  ```python
  import json
  schema = json.load(open("data/schema/paper.schema.json"))
  allowed = set(schema["properties"]["section"]["enum"])
  for f in glob.glob("data/papers/*.json"):
      for r in json.load(open(f)):
          assert r["section"] in allowed, f"{r['id']}: section={r['section']} 不在枚举内"
  ```

### E8 · `venue` 短名异写

- **现象**：02 文实测 45 个短名，含 `WWWC`/`WC`/`WWW`、`IF`/`Information Fusion`、`Sensor`/`Sensors`、`Arxiv`/`arXiv` 等。
- **影响**：站点按 venue 聚合会出现"同一会议拆成 3 个"。
- **断言**：
  ```python
  import json, glob, collections
  venues = json.load(open("data/venues.json"))          # 期望是 list 或 {"venues":[...]}
  rows = venues if isinstance(venues, list) else venues["venues"]
  ids = {v["id"] for v in rows}
  aliases = {}
  for v in rows:
      for a in v.get("aliases", []):
          assert a not in aliases, f"alias {a} 重复挂在 {aliases[a]} 与 {v['id']}"
          aliases[a] = v["id"]
  for f in glob.glob("data/papers/*.json"):
      for r in json.load(open(f)):
          vid = r.get("venue_id")
          assert vid is None or vid in ids, f"{r['id']}: venue_id={vid} 不在 venues.json"
  ```

### E9 · `code_url` 指向 pwc 【当前 FAIL · 必修】

- **现象（实测，当前 3 条记录 / 6 处 URL）**：
  ```text
  stgcn-2018                                           code_url = paperswithcode.com/paper/spatio-temporal-graph-convolutional-networks
                                                       code_links[0].url = 同上
  spatial-temporal-attention-wavenet-a-deep-learni-2021  code_url = paperswithcode.com/paper/spatial-temporal-attention-wavenet-a-deep#code
                                                       code_links[0].url = 同上
  time-series-is-a-special-sequence-forecasting-wi-2021  code_url = paperswithcode.com/paper/time-series-is-a-special-sequence-forecasting#code
                                                       code_links[0].url = 同上
  ```
  （复现：`python3 -c "import json,glob;print([(r['id'],u) for f in glob.glob('data/papers/*.json') for r in json.load(open(f)) for u in [r.get('code_url')]+[c['url'] for c in (r.get('code_links') or [])] if u and 'paperswithcode' in u])"`）
- **影响**：`[[code]]` 应指向 GitHub；pwc 论文页即使还剩标题信息，也**不是代码**。其中第 3 条的落地页实测 **404**（见 10 文 §1.2）。迁移脚本会把这个坏链接从 README **复制进 JSON 的两处字段**（`code_url` 与 `code_links[0].url`），等于把债务结构化并翻倍。
- **断言**：
  ```python
  hits = []
  for f in glob.glob("data/papers/*.json"):
      for r in json.load(open(f)):
          for u in [r.get("code_url")] + [c["url"] for c in (r.get("code_links") or [])]:
              if u and "paperswithcode.com" in u:
                  hits.append((r["id"], u))
  print(f"code_url 指向 pwc: {len(hits)}")
  assert len(hits) == 0, hits          # 目标：0；当前实测 6（3 条记录 × 2 处字段）
  ```
  同时断言 `alt_urls` 的 key 在 07 文 L1-6 的白名单内（`alt_urls` 在 schema 里是自由 key，靠门禁管）：
  ```python
  ROLE_OK = {"review", "sota", "paper-ieee", "paper-arxiv", "data", "project", "hf-papers"}
  for f in glob.glob("data/papers/*.json"):
      for r in json.load(open(f)):
          bad = set((r.get("alt_urls") or {})) - ROLE_OK
          assert not bad, f"{r['id']}: 未登记 role {bad}"
  ```
- **修法**：见 10 文 §2.1 档 1——换官方 repo 或置 `null`（**两处字段都要改，不能只改 `code_url`**）。这条必须在迁移过程中一并修掉，否则每多迁一年，pwc 链接就多复制一份。

### E10 · `id` 稳定性（幂等的前提）

- **现象**：`id` 由 `slugify(short_name or title, 32) + "-" + year` 生成（实测 `build_record`）。若人工后来补了 `short_name`，`id` 会**变**，下游（leaderboard 的 `paper_id` 外键、站点的 URL）全部断链。
- **断言**：
  ```python
  # ① 重复迁移后 id 不变
  before = {r["id"] for f in glob.glob("data/papers/*.json") for r in json.load(open(f))}
  # 跑一遍 migrate，再跑 --check
  after = {r["id"] for f in glob.glob("data/papers/*.json") for r in json.load(open(f))}
  assert before == after, f"id 漂移：{before ^ after}"

  # ② id 全局唯一
  import collections
  ids = [r["id"] for f in glob.glob("data/papers/*.json") for r in json.load(open(f))]
  dup = [k for k, v in collections.Counter(ids).items() if v > 1]
  assert not dup, f"id 重复: {dup}"

  # ③ id 符合 schema pattern
  import re
  PAT = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*-[0-9]{4}$")
  bad = [i for i in ids if not PAT.match(i)]
  assert not bad, f"id 不符合 schema pattern: {bad[:5]}"
  ```
- **修法**：**一旦某年批次落地，`id` 冻结**。后续人工改 `short_name` 不得改 `id`；若必须改，走 `provenance.legacy_id` 保留旧值（schema 已定义该字段）。

### E11 · `paper_url` 缺失 【当前 FAIL · 必修】

- **现象**：迁移汇总 `no_paper_url=2`（`[SUMMARY] ... no_paper_url=2`）。实测落盘后仍是 2 条：
  ```
  pre-training-enhanced-spatial-temporal-graph-neu-2022   （= E1，跨行折断导致）
  discrete-time-temporal-network-embedding-via-imp-2021   （= README:251 的 `[KDD 2021] [Title](url)` 旧格式）
  ```
- **影响**：`paper.schema.json` 的 `required: ['id','year','title','paper_url']` —— **缺 `paper_url` 的记录 schema 校验必挂**。也就是说这两条会让 Phase 1 的 L1-1 门禁变红，必须在迁移时解决，不能带着进 Phase 2。
- **两条的具体成因不同**：
  1. `pre-training-...`：跨行折断，链接在第二行（见 E1）。
  2. `discrete-time-...`：`README.md:251` 用的是 **`[KDD 2021] [Title](url)` 旧格式**（标题自己就是 markdown 链接）：
     ```markdown
     1. [KDD 2021] [Discrete-time Temporal Network Embedding via Implicit Hierarchical Learning in Hyperbolic Space](https://dl.acm.org/doi/abs/10.1145/3447548.3467422)
     ```
     迁移脚本 `parse_entry` 没有把这个 `[title](url)` 转成 `title` + `paper_url`。
- **断言**：
  ```python
  missing = []
  for f in glob.glob("data/papers/*.json"):
      for r in json.load(open(f)):
          if not r.get("paper_url"):
              missing.append(r["id"])
          else:
              assert r["paper_url"].startswith(("http://", "https://")), r["id"]
  print("缺 paper_url:", missing)
  assert not missing, missing            # 目标：0；当前实测 2 条
  ```
- **修法**：
  - 为 E1 补跨行合并；
  - 为 `[KDD 2021] [Title](url)` 补一种 entry 形态：`^\[([^\]]+)\]\s+\[([^\]]+)\]\((https?://[^)\s]+)\)` → `venue_year` + `title` + `paper_url`（`dedupe_readme_papers.py` 的 `ALT_README_RE` 已经实现过这个形态，可直接复用）；
  - 若某条确实无任何可用 URL，显式记入 `updates/migration/skipped.json` 并在 `provenance` 里注明，**而不是写 `null`**。

### E12 · `doi` 提取率极低

- **现象**：`--check` 汇总 **`doi=0`**（实测：`[SUMMARY] years=10 records=89 doi=0 arxiv=28 code=26 no_paper_url=2`，即 89 条里 **0 条**抽出 DOI）。而 02 文测得 102 条里有 **24 条 URL 中含 DOI（23.5%）**。
- **实测可补量**：对 `data/papers/` 全部 104 条跑「URL 全文搜 DOI 正则」，**14 条可补**（例如 `10.1145/3308558.3313577`、`10.1145/3292500.3330884`）。也就是 23.5% 里的绝大部分是"URL 里嵌 DOI 但不是 `doi.org/` 形态"。
- **影响**：`extract_doi(paper_url)` 只认 `doi.org/<doi>` 形态，而多数条目是 `ieeexplore.ieee.org/document/<id>`、`dl.acm.org/doi/10.1145/...`、`link.springer.com/article/10.1007/...` 这种形态 → **DOI 字段全空**，进而导致去重的 `key_2` 分支对所有条目失效。
- **断言**（在当前数据上实测可跑通，用 `>= 14` 作为下限防止回退）：
  ```python
  import json, glob, re, urllib.parse
  DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
  recoverable = []
  for f in glob.glob("data/papers/*.json"):
      for r in json.load(open(f)):
          if r.get("doi"): continue
          for u in [r.get("paper_url")] + list((r.get("alt_urls") or {}).values()):
              if not u: continue
              m = DOI.search(urllib.parse.unquote(u))
              if m: recoverable.append((r["id"], m.group(0))); break
  print(f"可补 DOI 的条目: {len(recoverable)}")
  assert len(recoverable) <= 14, "有新的 DOI 可补，说明 extract_doi 没跟上"
  ```
  ⚠️ 上面用的是 `<=`：**目标是这个数字降到 0**。若跑出 `> 14`，说明出现了新的、`extract_doi` 覆盖不到的 URL 形态，需要扩正则。
- **修法**：`extract_doi` 应改为「对 URL 全文跑 DOI 正则」，与 `dedupe_readme_papers.py::extract_doi` 的现有实现对齐（后者已经做了这件事，见其 L32–L52）。**这是两个脚本之间的一处真实不一致，迁移时应统一。** 注意正则会把 `.pdf` 后缀一起吃进去（实测 `10.1007/s10489-021-02587-w.pdf`），需要在提取后 `.rstrip()` 掉 `.pdf`。

### E13 · `venue_type` 判定

- **现象**：schema 枚举 `conference/journal/preprint/workshop/thesis/unknown`。README 里 `Arxiv`(20 条)、`CS224W`(workshop 类课程报告) 等需要显式归类。
- **断言**：
  ```python
  ALLOWED = {"conference","journal","preprint","workshop","thesis","unknown"}
  for f in glob.glob("data/papers/*.json"):
      for r in json.load(open(f)):
          assert r.get("venue_type","unknown") in ALLOWED, r["id"]
  # venue=Arxiv 的必须是 preprint
  for f in glob.glob("data/papers/*.json"):
      for r in json.load(open(f)):
          if (r.get("venue") or "").lower().startswith("arxiv"):
              assert r["venue_type"] == "preprint", f"{r['id']} venue=Arxiv 但 venue_type={r['venue_type']}"
  ```

### E14 · 批次边界处的顺序号

- **现象**：`order` 是「同一年内」的显示序号（`build_record(entry, order=...)`）。跨年不冲突，但**同一年跨批次**时会重排。
- **断言**：
  ```python
  for f in glob.glob("data/papers/*.json"):
      rows = json.load(open(f))
      orders = [r["order"] for r in rows]
      assert orders == list(range(1, len(rows)+1)), f"{f}: order 不连续 {orders}"
  ```

### 4.1 当前状态矩阵（2026-11-24 实测，104 条记录）

把上面 14 条断言一次性跑完的结果。**这张表就是「迁移还剩多少活」的真实刻度**，不要凭印象判断。

| 断言 | 状态 | 实测证据 |
| --- | :---: | --- |
| E1 跨行条目合并 | ❌ **FAIL** | `pre-training-enhanced-spatial-temporal-graph-neu-2022` 的 title 只有第一行、`paper_url: null` |
| E2 `paper_url` 无 `]](` 污染 | ✅ PASS | 0 条 |
| E3 非 ASCII 连字符（归一化键） | ✅ PASS（存储层有意保留原样） | `title` 保留 U+2010/2013，仅在 dedup 层归一 |
| E4 title 非裸 URL | ✅ PASS | 0 条 |
| E5 无 `…` 占位 | ✅ PASS | 0 条 |
| E6 `year` == 文件名 | ✅ PASS | 12 个文件全部一致 |
| E7 `section` 在枚举内 | ✅ PASS | 枚举 = `reviews/deep_learning/statistic/time_series/temporal_network_embedding` |
| E8 venue 归一（`venues.json` 存在性） | ⏳ N/A | `data/venues.json` 尚未落地，断言需等 1.5 |
| E9 `code_url` 无 pwc | ❌ **FAIL** | **6 处**（3 条记录 × `code_url` + `code_links[0].url`） |
| E10 `id` 唯一 + pattern + 幂等 | ✅ PASS | 104 条 id 唯一，全部匹配 `^[a-z0-9]+(-[a-z0-9]+)*-[0-9]{4}$` |
| E11 `paper_url` 必有（schema required） | ❌ **FAIL** | **2 条**：`pre-training-enhanced-...-2022`、`discrete-time-temporal-network-embedding-...-2021` |
| E12 doi 可补量 | ❌ 待修（非阻塞） | 104 条里 **14 条 URL 含可提取 DOI**，但 `doi` 字段全空（`doi=0`） |
| E13 `venue_type` 在枚举内 | ✅ PASS | 全部 `conference/journal/preprint/workshop/...` |
| E14 `order` 连续 1..N | ✅ PASS | 12 个文件全部连续 |

**结论：3 红 + 1 待修 + 1 待建。**

优先级：

1. **E11（2 条缺 `paper_url`）必须最先修**——它让 `paper.schema.json` 的 `required` 直接不满足，会卡住 Phase 1 的 L1-1 门禁，进而卡住 Phase 2 的 L3 部署闸门。
2. **E1 与 E11 是同一个根因**（跨行折断），一起修；另一条 E11 是 `[KDD 2021] [Title](url)` 旧格式。
3. **E9（pwc code_url）**虽然不违反 schema，但它把 10 文 §1 说的坏链接**结构化复制**了——修得越晚，后面每年批次都会再复制一遍。
4. **E12（doi）**不阻塞任何阶段，但它是 `key_2` 去重分支失效的根因，建议在 Phase 1 内顺手修掉（14 条可自动补，成本极低）。

---

## 5. 数据量达标后的门禁改造

### 5.1 现状的问题：护栏是「环境变量 + 隐式比例」，不是门禁

```js
const minExpected = Number(process.env.SITE_MIN_STRUCTURED_PAPERS ?? 0) ||
  Math.max(1, Math.floor(readmeCount * 0.75));
```

三个问题：

1. **`SITE_MIN_STRUCTURED_PAPERS` 存在就完全接管** —— 设成 `1` 就关掉护栏。这意味着**任何人（或任何 CI 配置）都能一行环境变量废掉完整性保护**，而且**没有任何告警**。
2. **`0.75` 是硬编码魔数**，写在 JS 里，Python 侧的迁移脚本、验收脚本、文档都不知道它。
3. **失败是静默的** —— 只 `log()` 一行，`build` 仍然退出 0。CI 看不出来。

### 5.2 改造目标：把「环境变量开关」换成「数据覆盖率硬门禁」

**核心思路：护栏的判定权从 JSON 侧移到 CI 侧的独立脚本，且判定对象是"覆盖率"而不是"条数"。**

#### 改造 A · 参数外置到配置文件

```jsonc
// scripts/configs/site-data-policy.json（建议文件名）
{
  "version": 1,
  "min_coverage_ratio": 0.75,          // floor(readmeCount * ratio) <= structured
  "allow_override_env": false,         // ← 明确禁止 SITE_MIN_STRUCTURED_PAPERS 覆盖
  "require_source": "data/papers",     // provenance.source 必须是这个
  "fail_on_degraded": true
}
```

- `prepare-data.mjs` 读这个文件（它已经是 Node，读 `scripts/configs/*.json` 无额外依赖）；
- **删除** `process.env.SITE_MIN_STRUCTURED_PAPERS` 分支，或至少让它**必须**同时设置 `ALLOW_GUARD_OVERRIDE=1` 才生效（双开关，防误用）；
- 比例从配置读，Python 侧脚本读同一份文件 → **单一事实源**。

#### 改造 B · 门禁脚本独立断言

```bash
# 建议：scripts/check_site_data_coverage.py
python3 scripts/check_site_data_coverage.py --root . --policy scripts/configs/site-data-policy.json
```

检查逻辑（四条，全过才 `[OK]`）：

| # | 检查 | 失败文案 |
| --- | --- | --- |
| 1 | `structured_count >= floor(readme_count × ratio)` | `data/papers 覆盖率不足: 41 / 78 (ratio=0.53 < 0.75)。站点会回退 README 抓取。` |
| 2 | 若构建产物存在：`provenance.source == "data/papers"` | `站点实际使用的是 README 抓取（provenance.source=README.md）。这不是验收通过，是静默降级。` |
| 3 | 若构建产物存在：`provenance.degraded !== true` | `站点处于 degraded 状态。` |
| 4 | 每个 `data/papers/YYYY.json` 的 `year` 与文件名一致、`order` 连续 1..N | `data/papers/2020.json: order 不连续 (1,2,4)` |

#### 改造 C · CI 接线（分层）

| 层 | 何时跑 | 断言的检查 | 失败后果 |
| --- | --- | --- | --- |
| **L1（PR 必过）** | PR 触及 `data/papers/**` | 检查 1 + 4（**纯本地，不构建**） | block 合并 |
| **L2（站点构建后）** | PR 触及 `site/**` 或每周定时 | 检查 1–4（**需先 `build:fast`**） | PR 侧 warn；定时侧开 issue |
| **L3（发布前）** | `pages.yml` 的 `deploy` 前 | 检查 2 + 3（**必须有构建产物**） | **阻断部署** |

**L3 是关键**：它保证「哪怕有人改了环境变量让 build 通过，只要 `provenance.source !== 'data/papers'` 就不许部署」。这是把 5.1 的三个问题一次性关掉的那道闸。

#### 改造 D · 错误文案统一（供 07 文引用）

```text
[FAIL] site data coverage gate

  structured (data/papers/*.json) = 41
  readme entries (parseLine)      = 104
  required (floor 104 × 0.75)     = 78
  coverage                        = 39.4%   ← 未达标

  影响：site/scripts/prepare-data.mjs 会 REJECT data/papers 并回退 README 抓取。
        站点 build 依然成功，但展示的是刮取数据（无 method_tags、无 doi、
        含 `…` 占位、跨行条目丢失），且 provenance.source = "README.md"。

  修法：完成 B2 + B1 两个迁移批次（+41 +32 = 73）后仍需 5 条，
        见 docs/evolution/11-data-volume-plan.md §2.2。
        或在 scripts/configs/site-data-policy.json 中调整 min_coverage_ratio
        —— 但那意味着你接受更低的数据质量，请在 PR 描述中说明理由。
```

---

## 6. 总表：批次 × 护栏 × 工时

| 批次 | 年份 | 条数 | 累计 | 护栏（≥78） | 工时 | 建议顺序 | 完成后的站点状态 |
| --- | --- | ---: | ---: | :---: | ---: | :---: | --- |
| B2 | 2020–2021 | 41 | 41 | ❌ | 4.0 | **1** | 仍 degraded（41 < 78） |
| B1 | 2018–2019 | 32 | 73 | ❌（差 5） | 3.5 | **2** | 仍 degraded（73 < 78） |
| B4 | 2025–2026 | 15 | **88** | ✅ | 2.0 | **3** | **首次真实可用** |
| B3 | 2022–2024 | 10 | 98 | ✅ | 1.5 | 4 | 可用 + 待内容补录 |
| B0 | 2015–2017 | 6 | 104 | ✅ | 1.0 | 5 | 全量完成 |
| | | **104** | | | **12.0** | | |

**最小过线路径**（2.5 人日）：`B2 + B1 + B0` = 41+32+6 = **79 ≥ 78** ✅。代价：B4 的 15 条当代论文继续 markdown-only，`curated JSON` 双体制未解。

**推荐路径**（9.5 人日）：`B2 + B1 + B4` = 41+32+15 = **88 ≥ 78** ✅。这是「过线 + 内容价值」双达标的最短路径。

---

## 7. 复现命令

### 7.1 复核 readmeCount = 104 与阈值 78

**推荐做法（零复刻风险）**：直接看站点构建日志——护栏会自己打印这两个数。

```bash
cd site && ASTRO_TELEMETRY_DISABLED=1 pnpm build:fast 2>&1 | grep -E "prepare-data|REJECT"
cd ..
# 实测输出: [prepare-data] source=data/papers (12 file(s), 104 papers)
# 若被护栏拒绝，则会打印: [prepare-data] REJECT data/papers: N papers < expected >=78 (README has 104); ...
```

若需要独立于站点复算（例如站点还没 build），用下面的 node 片段——它**逐字复刻** `prepare-data.mjs` 的 `parseLine()`：

```bash
cd "$(git rev-parse --show-toplevel)"
node -e '
const fs=require("fs");
const LINK_RE=/\[\[([a-zA-Z0-9_-]+)\]\]\((https?:\/\/[^)\s]+)\)/g;
function parseLine(line){
  const m=/^\d+\.\s+(.*)$/.exec(line.trim()); if(!m) return null;
  let body=m[1].trim();
  if(!body||body.startsWith("……")||body.startsWith("...")) return null;
  const links={}; LINK_RE.lastIndex=0; let lm;
  while((lm=LINK_RE.exec(body))!==null){const k=lm[1].toLowerCase(); if(!links[k]) links[k]=lm[2];}
  body=body.replace(LINK_RE,"").trim();
  const tag=/^\[([^\]]{2,60}?)\s+(\d{4})\]\s*/.exec(body);
  if(tag) body=body.slice(tag[0].length).trim();
  let title=body; const tl=/^\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/.exec(body);
  if(tl) title=tl[1].trim();
  title=title.replace(/\s+/g," ").replace(/[.\s]+$/,"").trim();
  return (!title||title.length<8)?null:true;
}
let section="misc",n=0;
for(const line of fs.readFileSync("README.md","utf8").split(/\r?\n/)){
  const h=/^##\s+(.*)$/.exec(line); if(h){section=h[1].trim();continue;}
  if(!/0x00|papers/i.test(section)) continue;
  if(!parseLine(line)) continue;
  n++;
}
console.log("readmeCount =", n, "| threshold =", Math.floor(n*0.75));'
# 实测: readmeCount = 104 | threshold = 78
```

> ⚠️ 这段只复刻了 `parseLine` + section 过滤，**没有** `###`/`####` 的层级重置逻辑。对 `readmeCount` 无影响（它只数条数），但**如果要数"每个年份多少条"，请改用 §7.2 的迁移脚本调用**，不要在这段上继续加逻辑。

### 7.2 复核各年份条目数

**权威做法**：问迁移脚本（覆盖 2015–2024），再单独读 curated 路径（2025–2026）。

```bash
# ① 2015–2024：迁移脚本视角
python3 - <<'PY'
import sys, pathlib
from collections import Counter
sys.path.insert(0, "scripts")
import migrate_readme_to_json as M
es = M.iter_readme_entries(pathlib.Path("README.md"))
print("entries:", len(es))
print("by year:", dict(sorted(Counter(e["year"] for e in es).items())))
PY
# 实测: entries: 89
#       by year: {2015:2, 2016:1, 2017:3, 2018:10, 2019:22, 2020:31, 2021:10, 2022:4, 2023:2, 2024:4}

# ② 2025–2026：curated JSON 视角（auto 块内，迁移脚本跳过的部分）
python3 -c "
import json
from collections import Counter
d=json.load(open('updates/curated/recent-papers.json'))
print('curated by year:', dict(sorted(Counter(p['year'] for p in d).items())))"
# 实测: {'2024': 4, '2025': 4, '2026': 11}   ← 2024 的 4 条与 ① 重叠，属正常

# ③ 落盘文件的实际分布（结果，不是来源）
python3 -c "
import json,glob
per={f.split('/')[-1][:4]: len(json.load(open(f))) for f in sorted(glob.glob('data/papers/*.json'))}
print(per); print('total', sum(per.values()))"
# 实测: {'2015':2,'2016':1,'2017':3,'2018':10,'2019':22,'2020':31,'2021':10,
#        '2022':4,'2023':2,'2024':4,'2025':4,'2026':11}  total=104
```

### 7.3 复核迁移脚本的当前覆盖状态

```bash
python3 scripts/migrate_readme_to_json.py --all --check
# 实测输出:
#   [SAME ] data/papers/2015.json (2 records)
#   ...
#   [SAME ] data/papers/2024.json (4 records)
#   [SUMMARY] years=10 records=89 doi=0 arxiv=28 code=26 no_paper_url=2
#   [OK] data/papers matches README.
```

> 注意 `--all --check` 只覆盖到 **2024**（10 个年份、89 条），因为 `iter_readme_entries` 跳过了 `<!-- START AUTO RECENT PAPERS -->` 块（那部分由 `updates/curated/recent-papers.json` 管）；而 `data/papers/2025.json`、`2026.json` 是另一条路径（curated JSON）产出的。**这正是 B4 批次要处理的双体制**——迁移脚本对 2025/2026 的 `--check` 是"看不见"的，所以 **B4 的验收必须额外对比 `curated JSON`**（见 §3.3）。

### 7.4 复核站点是否已过护栏

```bash
cd site && ASTRO_TELEMETRY_DISABLED=1 pnpm build:fast 2>&1 | grep -E "prepare-data|REJECT" ; cd ..
python3 -c "
import json
d=json.load(open('site/src/data/papers.generated.json'))
print('source:', d['provenance'].get('source'), '| count:', d['count'])"
# 实测（2026-11-24 15:10 快照）:
#   [prepare-data] source=data/papers (12 file(s), 104 papers)
#   source: data/papers | count: 104
```

> ⚠️ `ASTRO_TELEMETRY_DISABLED=1` 是必须的：不设会因 `@astrojs/permissions`/`@astrojs/telemetry` 写 `~/Library/Preferences/astro` 而 `EPERM` 崩溃（本机实测）。

---

## 8. 与其他文档的关系

- **[10-addendum-reality-check.md](./10-addendum-reality-check.md) §4.2–4.3**：护栏代码与 78 这个数字的来源。
- **[01-data-model.md](./01-data-model.md)**：`paper.schema.json` 的字段定义、`provenance` 的用法（本文 E10 的 `legacy_id` 来自那里）。
- **[02-paper-scope.md](./02-paper-scope.md)**：B3 批次之后的内容补录计划（本文只做"搬运"，不做"选文"）。
- **[08-roadmap.md](./08-roadmap.md) §2**：Phase 1 的 28 人日里，迁移相关部分是 1.2（4 人日）+ 1.3（5 人日）= 9 人日；本文的 12 人日拆得更细，差异在于本文把「venue 归一」（08 文的 1.5，2 人日）计入批次工时。
- **[09-acceptance.md](./09-acceptance.md) §1.1 / §2.3**：条目数守恒与 `provenance.source` 断言（本文给出了它们的**阈值来源**）。