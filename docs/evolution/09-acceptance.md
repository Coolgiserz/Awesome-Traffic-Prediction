# 09 · 验收清单（可执行命令 + 人工检查 + 回滚）

> 本文把 [06-pipeline](./06-pipeline.md)、[07-quality-gates](./07-quality-gates.md)、[08-roadmap](./08-roadmap.md) 的结论落成**可复制粘贴的验收步骤**。
>
> **约定**：每条命令后面都标注「✅ 现在就能跑」或「⏳ 需 Phase N 落地」。所有命令默认从**仓库根目录**执行，Python 3.12.10 / Node 26.6.0 / pnpm 12.16.1 为本机实测版本。

---

## 0. 基线：现在就能跑的全套检查（Phase 0 验收）

这一节的全部命令在**今天的仓库**上就能跑，用来确认「起点没坏」。任何一条失败，说明起点有问题，后续阶段的验收都不可信。

```bash
cd "$(git rev-parse --show-toplevel)"

# 0.1 README 双语结构同步（计数级）
python3 scripts/check_readme_sync.py --en README.md --zh README.zh.md
# 期望:
#   [OK] README sync check passed.
#   - Year sections checked: 12
#   - Curated digest rows: 12
#   - Benchmark matrix rows: 4

# 0.2 自动块与结构化数据一致（check 模式不写盘，是安全的 dry-run）
python3 scripts/render_curated_recent.py \
  --data updates/curated/recent-papers.json \
  --readme-en README.md --readme-zh README.zh.md --check
# 期望: [OK] Curated recent paper sections rendered.

# 0.3 域配置合法性
python3 scripts/validate_domain_config.py scripts/configs/traffic-monthly.json
# 期望: [OK] scripts/configs/traffic-monthly.json version=2 years=2024-2026 queries=5 rules=4 output_limit=25

# 0.4 检索脚本能在预算内跑通（不联网也能跑：全靠缓存）
python3 scripts/paper_search.py \
  --config scripts/configs/traffic-monthly.json \
  --max-requests 4 --cache-dir /tmp/acc-cache \
  --out /tmp/acc-candidates.md --report-json /tmp/acc-report.json
python3 - <<'PY'
import json
r = json.load(open("/tmp/acc-report.json"))
print("requests_made:", r["runtime"]["requests_made"],
      "cache_hits:", r["runtime"]["cache_hits"],
      "papers:", len(r["papers"]),
      "source_errors:", len(r["source_errors"]))
assert r["source_errors"] == [] or r["papers"], "全部源失败且无产出"
PY
# 期望: Wrote 5 entries ... / 退出码 0
# 注意: 预算耗尽只打 WARN 不返回非零（见 06 文 §7.3），所以必须靠 report 里的 source_errors 判断

# 0.5 站点构建（含 Pagefind 索引）
cd site && ASTRO_TELEMETRY_DISABLED=1 pnpm install --frozen-lockfile && ASTRO_TELEMETRY_DISABLED=1 pnpm build && cd ..
# 期望: "15 page(s) built" + "[Building search indexes] Indexed 2 languages"
# ⚠️ 必须设 ASTRO_TELEMETRY_DISABLED=1，否则 @astrojs/telemetry 会尝试写
#    ~/Library/Preferences/astro 并以 EPERM 崩溃（本机实测）

# 0.6 站点产物体积
du -sh site/dist site/dist/pagefind site/dist/papers
# 期望基线: dist 2.2M / pagefind 748K / papers 107K
```

**0.7 · 环境自检（⏳ 无脚本，手工）**

| 检查 | 命令 | 期望 |
| --- | --- | --- |
| Python | `python3 -V` | ≥ 3.10（脚本用 `list[str]`、`X | None` 语法，实测 3.12.10） |
| Node | `node -v` | ≥ 20（`site/package.json` 的 `engines.node`） |
| pnpm | `pnpm -v` | 与 `packageManager: pnpm@12.16.1` 一致 |
| 网络（可选） | `curl -s -o /dev/null -w "%{http_code}\n" https://api.openalex.org/works?search=traffic` | 200 |

---

## Phase 1 验收 · 数据基座

### 1.1 数据完整性（✅ 部分现在能跑 / ⏳ 迁移后全量）

```bash
# ⏳ 需 1.1–1.5 落地
python3 - <<'PY'
import json, pathlib, collections
files = sorted(pathlib.Path("data/papers").glob("*.json"))
total = 0
per_year = {}
for f in files:
    rows = json.loads(f.read_text())
    assert isinstance(rows, list), f"{f} 不是 list"
    total += len(rows)
    per_year[f.stem] = len(rows)
print("files:", len(files), "records:", total)
print(dict(sorted(per_year.items())))
# 期望: files: 12  records: 102
# 期望 per_year: {'2015':1,'2016':1,'2017':3,'2018':11,'2019':20,'2020':26,
#                 '2021':10,'2022':3,'2023':2,'2024':4,'2025':4,'2026':11}
assert total == 102, f"条目数应为 102，实际 {total}"
assert len(files) == 12, "应覆盖 2015–2026 共 12 个年份文件"
PY
```

> 注意 `2021` 的 10 与 `2022` 的 3 是**按 `####` 小节计数**（本机实测），与 02 文按 venue 年份重算的 9/2 不同——两者差异来自 L156 的 `[Arxiv 2020]` 落在 `#### 2021`、L89 的 `[TITS 2019]` 落在 `#### 2018`。迁移时**以小节为准**（保持视觉位置不变），把 venue 年份冲突记进 `verification` 字段。

### 1.2 Schema 全绿（⏳ 需 `check_schema.py`）

```bash
python3 scripts/check_schema.py --root . --strict
# 期望: [OK] 14 files validated against 8 schemas, 0 error
#       （7 份 schema：paper/dataset/toolkit/venue/research-group/lineage/recent-papers +
#         index.json 映射）
# 失败示例（要求）:
#   data/papers/2026.json [12].method_tags[1]  expected type array, got string
```

**人工检查项**

- [ ] 每条 `data/papers/*.json` 记录的 `id` 满足 `^[a-z0-9]+(-[a-z0-9]+)*-[0-9]{4}$`（schema 已约束）；
- [ ] `arxiv_id` 字段去除版本后缀（`2402.11838v2` → `2402.11838`）；
- [ ] `doi` 是裸 DOI（无 `https://doi.org/` 前缀，小写）。**抽查方式**：
  ```bash
  python3 -c "
  import json,glob
  bad=[(f,r['id']) for f in glob.glob('data/papers/*.json')
       for r in json.load(open(f)) if (r.get('doi') or '').startswith('http')]
  print('bad doi:', bad)"   # 期望 []
  ```

### 1.3 JSON 格式规范（⏳）

```bash
python3 scripts/check_json_lint.py --root .
# 期望: [OK] 无语法错误；[WARN] 0 个非规范化文件
python3 scripts/check_json_lint.py --root . --fix   # 只修 indent/尾随换行
```

### 1.4 重复条目零误报（⏳）

```bash
python3 scripts/check_duplicates.py --root .
# 期望: [OK] 102 entries, 0 duplicate doi, 0 duplicate title_norm
#       [WARN] title-collisions: N  → 写入 updates/reports/title-collisions.json
```

**人工检查项**：打开 `updates/reports/title-collisions.json`，逐条回答「这是同一工作的两个版本，还是两篇不同论文」。

已知必查的一条（真实存在）：`COOL` 的 arXiv 版 `10.48550/arxiv.2403.01091`（OpenAlex `W4392382537`）与 Information Fusion 版 `10.1016/j.inffus.2024.102341` 是**同一工作**，仓库当前只保留一个 URL。判定后写进 `also_published_as`。

### 1.5 重建等价（⏳ 需 `render_readme.py`，**Phase 1 的核心闸门**）

```bash
python3 scripts/render_readme.py --root . --lang en --out /tmp/rb-en.md
python3 scripts/render_readme.py --root . --lang zh --out /tmp/rb-zh.md
diff -u README.md /tmp/rb-en.md | grep -cE '^[+-]' ; echo "---"
diff -u README.zh.md /tmp/rb-zh.md | grep -cE '^[+-]'
# 判定: 实质差异 ≤ 5 行，且全部落在 updates/reports/rebuild-allow-diff.md 白名单里
# 允许的 3 类: (1) `…`/`……` 占位 (2) 跨行合并 L170-171 (3) doctoc TOC 区块
```

```bash
# 白名单外差异必须为 0
python3 scripts/render_readme.py --root . --lang en --check --strict
# 期望: [OK] README.md 与 data/ 渲染结果等价（剩余白名单差异 3 处）
```

### 1.6 往返一致性（⏳，人工抽检）

对下列 5 类各抽 2 条，执行 `JSON → 渲染 → 反向解析 → JSON`，字段级相同：

| 类别 | 抽取建议 |
| --- | --- |
| 有 DOI 的期刊论文 | `HySRD-Net`（`10.1016/j.eswa.2026.131214`） |
| 只有 arXiv 的预印本 | `UniST`（`arxiv.org/abs/2402.11838`） |
| 跨行折断条目 | L170–171 `Pre-training Enhanced Spatial-temporal GNN` |
| 多 code 链接 | 带 `[[code-pytorch]]` / `[[code-official-tf]]` / `[[code-not-official]]` 的条目 |
| venue 短名异写 | `WWWC` / `WC` / `WWW` 三者之一 |

### 1.7 L1 门禁接线（⏳）

```bash
# 本地等价执行（与 CI 里 pr-gates.yml 完全一致的命令）
python3 scripts/check_schema.py --root .
python3 scripts/check_json_lint.py --root .
python3 scripts/check_duplicates.py --root .
python3 scripts/check_bilingual_parity.py --root .
python3 scripts/check_auto_blocks.py --root . --base origin/main
python3 scripts/check_paper_format.py --root .
python3 scripts/check_crossrefs.py --root .
python3 scripts/check_spelling.py --root . --allowlist scripts/configs/spelling-allowlist.json
# 全部期望 [OK]；总耗时 < 60s
time (上述 8 条串行)
```

**人工检查项**

- [ ] 在 GitHub 上开一个故意改坏 `data/papers/2026.json` 的草稿 PR，确认 `PR Gates` 变红且报错信息含 `文件:行` + `为什么` + `怎么修` 三要素；
- [ ] 开第二个 PR，只改 README 的 auto 块内容，确认报错文案是「Auto block modified manually」而**不是**「Outdated」，即 07 文 L1-5 的归因生效；
- [ ] 确认所有 L1 门禁**不发起任何网络请求**（断网跑一遍，全部仍应通过）。

### 1.8 现有 CI 不被破坏（✅ 现在就能跑）

```bash
python3 scripts/check_readme_sync.py --en README.md --zh README.zh.md     # 仍 [OK]
python3 scripts/render_curated_recent.py --data updates/curated/recent-papers.json \
  --readme-en README.md --readme-zh README.zh.md --check                  # 仍 [OK]
```

**Phase 1 回滚**

```bash
# 1. 灰度 3 之前：只删数据与环境，README 未动
rm -rf data/papers/*.json            # 或在 git 里只 revert 迁移 commit
# 2. 灰度 3 之后（README 已变渲染产物）：revert 灰度 commit
git revert <灰度3-commit-sha>
# 3. 门禁误报导致的紧急停用：把 pr-gates.yml 里的该 step 改成 continue-on-error: true
# 4. 彻底放弃：git revert <Phase1-merge-sha>，仓库回到纯 markdown 状态
```

> 全部 Phase 1 产物都在 `data/`、`scripts/`、`docs/` 下。**README 只在灰度 3 那一个 commit 里被改**，可单独 revert。

---

## Phase 2 验收 · 站点上线

### 2.1 构建成功（✅ 现在就能跑，产物数字随数据增长变化）

```bash
cd site
export ASTRO_TELEMETRY_DISABLED=1
pnpm install --frozen-lockfile
pnpm build        # astro build && pagefind --site dist
cd ..
```

期望输出（本机实测 2026-11-24）：

```text
15 page(s) built in 468ms
[Building search indexes]
Discovered 2 languages: en, zh-cn
Indexed 2 languages / 15 pages / 2116 words
```

**硬性断言**

```bash
for f in dist/index.html dist/zh/index.html dist/papers/index.html \
         dist/zh/papers/index.html dist/search/index.html dist/sitemap.xml \
         dist/pagefind/pagefind.js; do
  test -f "site/$f" || echo "MISSING: site/$f"
done
# 期望：无输出
```

### 2.2 产物体积预算（⏳ 需 `check_site_budget.py`；✅ 现在可手工量）

| 指标 | 命令 | 基线（实测） | 上限 |
| --- | --- | ---: | ---: |
| `dist/` 总量 | `du -sh site/dist` | 2.2 MB | **3 MB** |
| 单页 HTML | `ls -l site/dist/papers/index.html` | 107 KB | 200 KB |
| JS chunk | `du -sh site/dist/_astro` | 1.1 MB | 1.5 MB |
| Pagefind 索引 | `du -sh site/dist/pagefind` | 748 KB | 1 MB |

```bash
# ⏳ 落地后
python3 scripts/check_site_budget.py --dist site/dist \
  --max-total-mb 3 --max-page-kb 200 --max-js-mb 1.5 --max-index-kb 1024
# 期望: [OK] dist=2.2MB / largest page=108KB (zh/papers) / js=1.1MB / index=748KB
```

### 2.3 数据来源正确（⏳ 需 2.1）

```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("site/src/data/papers.generated.json")
if not p.exists():
    print("SKIP: 构建产物不存在，先跑 pnpm build"); raise SystemExit(0)
d = json.loads(p.read_text())
print("provenance:", d.get("provenance"))
print("count:", d.get("count"))
assert d["provenance"].get("source", "").startswith("data/"), "数据仍来自 README"
assert d["provenance"].get("degraded") is False, "仍处于 degraded 状态"
assert d["count"] == 102, f"条目数应 102，实际 {d['count']}"
PY
# 当前实测（未切源）: {'source': 'README.md', 'en': 104, 'zh': 104, 'degraded': True}
#                      count: 104   ← 多出的 2 条是 `…` 占位被当成条目
# Phase 2 目标:      {'source': 'data/papers', 'degraded': False}  count: 102
```

### 2.4 内链零 404（⏳ 需 2.7；✅ 构建后即可测）

```bash
python3 - <<'PY'
import re, pathlib
dist = pathlib.Path("site/dist")
base = "/Awesome-Traffic-Prediction/"
miss = []
for p in dist.rglob("*.html"):
    html = p.read_text(errors="ignore")
    for href in re.findall(r'href="(' + re.escape(base) + r'[^"#?]*)', html):
        rel = href[len(base):]
        t = dist / rel
        if not (t.exists() or (t / "index.html").exists() or (t.with_suffix(".html")).exists()):
            miss.append((str(p.relative_to(dist)), href))
print(f"{len(miss)} missing internal link(s)")
for m in miss[:15]:
    print("  ", m)
PY
# 期望: 0 missing internal link(s)
```

### 2.5 线上可访问（⏳ 需 2.7 部署）

```bash
for u in "" "papers/" "zh/" "zh/papers/" "datasets/" "benchmark/" "search/"; do
  printf "%-16s " "/$u"
  curl -sSL -o /dev/null -w "%{http_code}\n" -m 20 \
    "https://coolgiserz.github.io/Awesome-Traffic-Prediction/$u"
done
# 期望: 全部 200
```

同时确认 `sitemap.xml` 里的 URL 都带 base 前缀：

```bash
curl -sSL https://coolgiserz.github.io/Awesome-Traffic-Prediction/sitemap.xml \
  | grep -c "/Awesome-Traffic-Prediction/"
# 期望 >= 15
```

### 2.6 筛选与搜索（人工检查项，⏳）

- [ ] `/papers/?year=2020` 条数 = `data/papers/2020.json` 的记录数（**26**）；
- [ ] `/papers/?tag=STGNN` 条数与数据里 `method_tags` 含 `STGNN` 的条目数一致；
- [ ] 刷新页面后筛选条件保持（URL query 生效）；
- [ ] 复制 URL 到新标签页打开，筛选状态一致；
- [ ] 站内搜索 `STGCN` ≥ 3 条结果，首条是 `Spatio-Temporal Graph Convolutional Networks`；
- [ ] 搜索中文「交通流预测」有结果（验证 Pagefind 中文分词）；
- [ ] `/zh/papers/` 的筛选结果数与 `/papers/` **完全相同**（语言只影响文案，不影响数据）；
- [ ] 移动端宽度 375px 下表格可横向滚动，不撑破布局。

### 2.7 数据一致性（人工检查项）

抽查 `UniST`：

| 位置 | 期望值 |
| --- | --- |
| `README.md` | 在 `#### 2024` 小节，`[Arxiv 2024]` |
| `data/papers/2024.json` | `year: 2024`, `paper_url` 含 `2402.11838` |
| `/papers/` 搜索 `UniST` | 年份 2024，venue `Arxiv`，链接同上 |
| `/zh/papers/` | 同一条目，年份相同 |
| `/benchmark/`（若列数据集） | 不出现（该论文不在 leaderboard） |

**Phase 2 回滚**

```bash
# 1. 停部署（最快）：在仓库 Settings → Pages 里把 Source 改回 None，或注释掉 pages.yml 的 on:
# 2. 撤站点目录（完全回滚，零数据损失）
rm -rf site/ && git checkout -- .github/workflows/pages.yml 2>/dev/null || rm -f .github/workflows/pages.yml
# 3. 撤销数据源切换
git revert <2.1-commit-sha>       # prepare-data.mjs 回到 README 兜底
# 4. 清理测试残留（如果本地跑过构建）
rm -rf site/dist site/node_modules site/.astro site/public/_data.json .pnpm-store
```

> `site/` 是**完全独立目录**：`prepare-data.mjs` 只读仓库根、只写 `site/` 内。删掉它不影响 README、`data/`、`scripts/` 的任何内容。

---

## Phase 3 验收 · 自动化与社区

### 3.1 多源配置生效（⏳ 需 3.1）

```bash
python3 scripts/validate_domain_config.py scripts/configs/traffic-monthly.json --summary \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('sources:', sorted(d['sources']))"
# 期望: sources: ['arxiv', 'crossref', 'openalex']
#       （semantic_scholar 仅在配置了 S2_API_KEY 时出现）
```

### 3.2 多源实跑与降级可见（⏳ 需 3.1–3.2）

```bash
python3 scripts/paper_search.py \
  --config scripts/configs/traffic-monthly.json \
  --max-requests 60 --cache-dir /tmp/acc3 \
  --out /tmp/c3.md --report-json /tmp/c3.json

python3 - <<'PY'
import json
r = json.load(open("/tmp/c3.json"))
print("requests:", r["runtime"]["requests_made"], "cache_hits:", r["runtime"]["cache_hits"])
print("papers:", len(r["papers"]))
print("source_errors:", r["source_errors"])
from collections import Counter
print("by source:", Counter(p["source"] for p in r["papers"]))
# 期望: by source 里至少出现 2 个不同 source
PY
```

**硬性断言**：`r["source_errors"]` 里的每一条都能指名 source（格式 `query='...' source='...' error=...`）；若某源全灭，**必须出现在 `source_errors` 里**，不能静默消失。

### 3.3 Raw 快照落盘（⏳ 需 3.2）

```bash
ls -1 updates/raw/$(date +%Y-%m)/ | sort
# 期望: arxiv.json  crossref.json  manifest.json  openalex.json
python3 -c "
import json; m=json.load(open('updates/raw/$(date +%Y-%m)/manifest.json'))
print(m['run_id'], m['config_hash'], m['requests_made'])"
```

### 3.4 归一化可复现（⏳ 需 3.2–3.3，**Phase 3 的硬闸门之一**）

```bash
python3 scripts/normalize_papers.py --raw updates/raw/$(date +%Y-%m)/ --out /tmp/n1.json
python3 scripts/normalize_papers.py --raw updates/raw/$(date +%Y-%m)/ --out /tmp/n2.json
cmp /tmp/n1.json /tmp/n2.json && echo "✅ reproducible"
# 期望: ✅ reproducible（逐字节相同）
```

再验证「换机器也一样」：

```bash
md5sum /tmp/n1.json
PYTHONHASHSEED=0 python3 scripts/normalize_papers.py \
  --raw updates/raw/$(date +%Y-%m)/ --out /tmp/n3.json
cmp /tmp/n1.json /tmp/n3.json && echo "✅ seed-independent"
```

### 3.5 时间戳不再污染候选文件（⏳ 需 3.3）

```bash
grep -c "Generated on:" updates/monthly-paper-candidates.md
# 期望: 0（时间戳已移到 updates/reports/）
# 意义: 只有去掉它，"无实质变化就不开 PR" 才成立
```

### 3.6 打标签闭环（⏳ 需 3.5）

```bash
# a) taxonomy 存在且被 LLM/规则共同引用
python3 -c "
import json; t=json.load(open('data/taxonomy/paper-tags.json'))
print('tasks:', len(t['tasks']), 'methods:', len(t['methods']))"

# b) 所有建议的 id 都在白名单内
python3 - <<'PY'
import json
t = json.load(open("data/taxonomy/paper-tags.json"))
ok_task = {x["id"] for x in t["tasks"]}
ok_m = {x["id"] for x in t["methods"]}
s = json.load(open("updates/reports/tag-suggestions.json"))
bad = [x for x in s if x["task_id"] not in ok_task or not set(x["method_ids"]) <= ok_m]
print("suggestions:", len(s), "out-of-vocabulary:", len(bad))
assert not bad, bad[:3]
PY

# c) 建议未污染权威数据
git diff --stat data/papers/ | tail -1
# 期望: 空（更新 tag-suggestions.json 不应改动 data/papers/）
```

**人工检查项**：从 `tag-suggestions.json` 随机抽 10 条，比对 `evidence_span` 与摘要原文，判断标签是否合理；记录 accept 率（目标 ≥ 80%）。

### 3.7 Leaderboard 完整性（⏳ 需 3.6–3.7）

```bash
for ds in metr-la pems-bay nyc-bike nyc-taxi; do
  python3 -c "
import json;d=json.load(open('data/leaderboards/$ds.json'))
rows=d['rows']; v=[r for r in rows if r.get('verified')]
print('$ds', 'rows=',len(rows), 'verified=',len(v),
      'horizons=', sorted({r['horizon_min'] for r in rows}))"
done
# 期望: 每个 dataset 的 horizons 至少覆盖 15/30/60 中的 2 个；verified 占比 >= 80%
```

外键校验（⏳ 需 `check_crossrefs.py`）：

```bash
python3 scripts/check_crossrefs.py --root .
# 期望: [OK] leaderboard.paper_id 全部命中 data/papers/*.json
# 失败示例: data/leaderboards/metr-la.json rows[2].paper_id = "stgcn-2017"
#           unknown paper id（data/papers/2017.json 里是 'stgcn-2018'）
```

### 3.8 引用数快照（⏳ 需 3.7）

```bash
python3 scripts/snapshot_citations.py --out data/leaderboards/citations-snapshot.json
python3 -c "
import json;d=json.load(open('data/leaderboards/citations-snapshot.json'))
print(d['snapshot_at'], len(d['rows']), 'papers')"

# 人工对账（OpenAlex 批量接口可用，本机实测）
curl -s "https://api.openalex.org/works?filter=ids.openalex:W4391987728|W4392382537&per-page=50&select=id,cited_by_count" \
  | python3 -c "import json,sys; print([(w['id'],w['cited_by_count']) for w in json.load(sys.stdin)['results']])"
# 实测 2026-11-24: [('https://openalex.org/W4391987728', 265), ('https://openalex.org/W4392382537', 86)]
```

**人工检查项**

- [ ] 排行榜页面标注了 `snapshot_at` 日期；
- [ ] 排行榜按 `horizon_min` 分列，**没有**跨 horizon 混排的"总榜"；
- [ ] `verified: false` 的行不出现在 README 的 `0x07` 表格里；
- [ ] 论文数少的新工作（如 2026 年的条目）没有被引用数排序挤到看不见。

### 3.9 pwc 链接清零（⏳ 需 3.8）

```bash
echo "--- 应清零的位置"
grep -rn "paperswithcode.com" README.md README.zh.md datasets/ models/ | wc -l
# 期望: 0

echo "--- 允许保留的位置（历史说明文档）"
grep -rln "paperswithcode.com" docs/ | wc -l
# 期望: >= 1（docs/evolution/06-pipeline.md、07-quality-gates.md 里有说明）

echo "--- datasets 的 SOTA 小节应改为本地引用"
grep -rn "SOTA\|Leaderboard" datasets/*.md
# 期望: 指向 data/leaderboards 或 README#0x07，不再有 pwc URL
```

### 3.10 L2 巡检（⏳ 需 3.9）

```bash
python3 scripts/check_links.py --root . \
  --out updates/reports/link-status.json --budget 300 --timeout 15
python3 - <<'PY'
import json, collections
r = json.load(open("updates/reports/link-status.json"))
c = collections.Counter(x["status"] for x in r["results"])
print("total:", r["total"], dict(c))
# 期望: redirect_away == 0 （pwc 已处理完）
#       dead 允许少量，必须在 scripts/configs/link-check-baseline.json 里
assert c.get("redirect_away", 0) == 0, \
    [x for x in r["results"] if x["status"] == "redirect_away"][:5]
PY
```

> `redirect_away` 的判定口径（06 文 §5.1 实测）：
> `paperswithcode.com/**` → `huggingface.co/papers/trending`，`curl -L` 得 200，**只靠状态码抓不到**。

### 3.11 DOI 解析（⏳ 需 3.9）

```bash
python3 scripts/check_dois.py --root . --out updates/reports/doi-status.json
python3 -c "
import json;r=json.load(open('updates/reports/doi-status.json'))
print('checked:', r['total'], 'unresolved:', len([x for x in r['results'] if x['status']!='ok']))"
# 期望: unresolved 只包含「刚发表、OpenAlex 尚未收录」的条目，且已列入 baseline
```

### 3.12 产物新鲜度（⏳ 需 3.9）

```bash
python3 scripts/check_artifact_freshness.py --root .
# 期望: [OK] wordcloud 与 hotspots 的 generated_utc 落后 candidates 不超过 7 天
# 失败示例（当前真实风险）:
#   [artifact-freshness] ⚠️ latest.json generated_utc = 2026-10-01 (54 天前)
#   → 月度流水线可能已停摆，检查 Actions → Monthly Paper Search 最近 3 次运行
```

### 3.13 自动化 PR 的完整闭环（⏳ 需 3.2–3.4，手动触发一次验证）

```bash
gh workflow run monthly-paper-search.yml
gh run watch
gh pr list --label automation --state open --json number,title,body \
  | python3 -c "
import json,sys
for pr in json.load(sys.stdin):
    print(pr['number'], pr['title'])
    b=pr['body']
    for k in ('degraded','新增','去重','冲突','source_errors'):
        print('  contains', k, ':', k in b)"
# 期望: PR body 里能读到降级状态与 normalize 报告摘要
```

**人工检查项**

- [ ] PR 里没有任何手工编辑的痕迹（只改 `updates/` 与 `data/` 的产物）；
- [ ] 若 `degraded: false` 且冲突为 0，reviewer 只需 approve，不需改文件；
- [ ] 若降级，PR 标题或 body 首行有明显的降级标记，不会被误当正常结果合并。

**Phase 3 回滚**

```bash
# 3.1 多源 → 单源（一行配置，最快）
#   编辑 scripts/configs/traffic-monthly.json: "sources": ["openalex"]
git revert <3.1-commit-sha>

# 3.2 四段拆分 → 回到单 job
git revert <3.2-commit-sha>        # 保留旧 monthly-paper-search.yml 直到新链路稳定

# 3.5 LLM 打标签 → 关闭（建议结果本来就不进权威数据，删掉即可）
rm updates/reports/tag-suggestions.json
#   并把 workflow 里的 LLM job 改为 if: false

# 3.6–3.8 leaderboard / pwc 处置 → 分批回退
git revert <3.8-commit-sha>        # 先把 pwc 链接改回原状（不理想但能立即止血）
git revert <3.6-commit-sha>        # 再撤 leaderboard

# 3.9 L2 巡检噪音 → 停 workflow，不动代码
#   在 link-check.yml 顶部加 if: false
```

> **不要一次性 revert 整个 Phase 3**：3.9 的 L2 巡检与 3.1 的多源互不依赖，逐块回退才能保住已经验证有效的部分。

---

## 附录 A · 命令速查（按「现在能跑 / 需要等待」分类）

### ✅ 现在就能跑（今天已存在的能力）

| 命令 | 用途 |
| --- | --- |
| `python3 scripts/check_readme_sync.py --en README.md --zh README.zh.md` | 双语结构计数同步 |
| `python3 scripts/render_curated_recent.py --data updates/curated/recent-papers.json --readme-en README.md --readme-zh README.zh.md --check` | auto 块一致性（不写盘的 dry-run） |
| `python3 scripts/validate_domain_config.py scripts/configs/traffic-monthly.json` | 域配置校验 |
| `python3 scripts/paper_search.py --config scripts/configs/traffic-monthly.json --out /tmp/x.md --report-json /tmp/x.json` | 检索（真实联网） |
| `python3 scripts/dedupe_readme_papers.py --input /tmp/x.md --readme README.md README.zh.md --out /tmp/y.md --report-json /tmp/y.json` | 候选去重 |
| `python3 scripts/paper_wordcloud.py --input updates/monthly-paper-candidates.md --out-dir /tmp/wc --prefix t` | 词云（需 `pip install wordcloud matplotlib`） |
| `python3 scripts/yearly_hotspots.py --input README.md --out-dir /tmp/hs --topk 12` | 年度热点（同上） |
| `cd site && ASTRO_TELEMETRY_DISABLED=1 pnpm build` | 站点全量构建 |
| `cd site && ASTRO_TELEMETRY_DISABLED=1 pnpm build:fast` | 站点快速构建（跳过 Pagefind） |
| `curl -sSL -o /dev/null -w "%{http_code} %{url_effective}\n" <url>` | 单链接巡检（看 `url_effective` 才抓得到 pwc 式跳转） |

### ⏳ 需对应阶段落地

| 命令 | 阶段 | 前置 |
| --- | --- | --- |
| `python3 scripts/check_schema.py --root .` | P1（1.8） | 1.1 |
| `python3 scripts/check_json_lint.py --root .` | P1（1.8） | — |
| `python3 scripts/check_duplicates.py --root .` | P1（1.8） | 1.3 |
| `python3 scripts/check_bilingual_parity.py --root .` | P1（1.8） | 1.6 |
| `python3 scripts/check_auto_blocks.py --root . --base origin/main` | P1（1.8） | 1.9 |
| `python3 scripts/check_paper_format.py --root .` | P1（1.8） | — |
| `python3 scripts/check_crossrefs.py --root .` | P1（1.8） | 1.4 |
| `python3 scripts/check_spelling.py --root .` | P1（1.8） | — |
| `python3 scripts/migrate_readme_to_json.py --root . --out updates/migration/` | P1（1.2） | — |
| `python3 scripts/render_readme.py --root . --out /tmp/rb.md [--check]` | P1（1.6） | 1.3 |
| `python3 scripts/check_site_budget.py --dist site/dist` | P2（2.8） | 2.7 |
| `python3 scripts/normalize_papers.py --raw updates/raw/<YYYY-MM>/ --out ...` | P3（3.2） | 3.1 |
| `python3 scripts/suggest_tags.py --in updates/normalized/... --out updates/reports/tag-suggestions.json` | P3（3.5） | 3.2 |
| `python3 scripts/snapshot_citations.py --out data/leaderboards/citations-snapshot.json` | P3（3.7） | 3.6 |
| `python3 scripts/check_links.py --root . --out updates/reports/link-status.json` | P3（3.9） | — |
| `python3 scripts/check_dois.py --root .` | P3（3.9） | — |
| `python3 scripts/report_repo_size.py --root . --out updates/reports/repo-size.json` | P3（3.9） | — |
| `python3 scripts/check_artifact_freshness.py --root .` | P3（3.9） | — |
| `pnpm --dir .github/scripts/...`（无此项） | — | — |

---

## 附录 B · 人工检查项汇总（脱离命令的判断）

### Phase 1

- [ ] 随机打开的 10 个 GitHub 上 README 的论文链接，落地页标题与条目标题一致（防"链接能开但指向别的东西"）；
- [ ] README 的 `#### 2015`–`#### 2026` 共 12 个小节，视觉上与原版**完全一致**（灰度 3 后尤其要看这一条）；
- [ ] `data/papers/2020.json` 的 26 条与 README `#### 2020` 的 26 条逐条对应（这是条目最多的年份，最容易漏）；
- [ ] `CONTRIBUTION.md` 已更新为「改 JSON 不改 README」，且 PR 模板里的自检项与 07 文一致；
- [ ] 仓库里**不再存在**「同一条论文在两处维护」的情况（`updates/curated/recent-papers.json` 与 `data/papers/*.json` 的重复是 Phase 3 才收尾，Phase 1 允许共存但必须文档说明）。

### Phase 2

- [ ] 首页在 3G 网络模拟下首屏可读时间 < 3 s（Chrome DevTools Throttling）；
- [ ] `/papers/` 在 102 条时滚动流畅，筛选无卡顿；
- [ ] 中文页面没有出现英文占位文案（i18n 字典漏项的典型症状）；
- [ ] `hreflang` 双向正确：`/papers/` 指向 `/zh/papers/` 且反之；
- [ ] 分享到微信/Twitter 的卡片标题与描述正确（`og:*`）；
- [ ] 站点上所有"Code"链接指向 GitHub，不是 pwc（Phase 2 时可能尚未处理，记入待办）。

### Phase 3

- [ ] 月度 PR 的候选表里，人工判断 10 条中最少 8 条与交通流预测相关（对照现状：4 条里有 2 条不相关）；
- [ ] LLM 打标签的 `evidence_span` 能在摘要里原样找到（不是编造的引文）；
- [ ] leaderboard 的每个数字都能在对应论文的表格中定位到（不许"估计"）；
- [ ] `link-status.json` 里 `blocked` 的条目抽查 3 条，确认浏览器能正常打开（即确实是反爬而非真失效）；
- [ ] 任何自动化产出的文件，人读一遍就能判断"要不要合并"，不需要打开代码。

---

## 附录 C · 三条最重要的验收（如果只做三条）

| # | 命令 | 通过意味着 |
| --- | --- | --- |
| 1 | `python3 scripts/render_readme.py --root . --check --strict` | **README 是渲染产物**——数据基座成立 |
| 2 | `python3 scripts/normalize_papers.py --raw <同目录> --out /tmp/a.json && … --out /tmp/b.json && cmp /tmp/a.json /tmp/b.json` | **流水线可复现**——自动化可信 |
| 3 | `python3 scripts/check_links.py --root .` 里 `redirect_away == 0` | **死链不再伪装成活的**——外部依赖被真正管住 |

这三条分别对应 08 文里 Phase 1 / Phase 3 / Phase 3 的硬闸门，是整条演进链上**一旦失守就会让其余工作白做**的三个点。
## 附录：站点文案回归检查（Phase 2 之后新增）

站点的 i18n 字典是唯一的用户可见文案来源，但它很容易被设计文档的原话污染
（例如「用交互图表替代 README 中的静态 PNG」这类内部迁移说明会直接出现在页面上）。
每次改文案或模板后跑这一段：

```bash
cd site
SITE_BASE=/Awesome-Traffic-Prediction/ ASTRO_TELEMETRY_DISABLED=1 npx astro build

python3 - <<'PY'
import re, glob
BAD = ['README', 'PNG', 'docs/evolution', 'data/papers', 'generated',
       '静态 PNG', '交互图表替代', 'no-JS', 'fallback', 'site/public', '_data.json',
       'prepare-data', 'source=', 'readme entries', 'degraded',
       'Data source', '数据来源', 'Built', '构建于', 'TODO', 'FIXME']
files = glob.glob('dist/**/index.html', recursive=True)
bad = []
for f in files:
    h = open(f).read()
    body = h[h.find('<body'):]
    text = re.sub(r'<script.*?</script>', '', body, flags=re.S)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    hits = [b for b in BAD if b.lower() in text.lower()]
    if hits:
        bad.append((f, hits))
print(f"pages={len(files)} with_dev_text={len(bad)}")
for f, h in bad[:10]:
    print(" ", f, h)
PY
```

期望输出 `with_dev_text=0`。

同时人工确认两件事：

1. **中文页不出现英文策展备注**：`note_zh` 只有 19/104 条有值，
   其余应显示「暂时还没有该论文的策展备注」，而不是英文原文。
2. **中文页的 task / layer 已本地化**：`/zh/papers/` 的任务分面应显示
   「交通预测 / 交通流量预测 / 综述 …」，详情页的脉络层级应显示「近年主流 / 经典基线」。
