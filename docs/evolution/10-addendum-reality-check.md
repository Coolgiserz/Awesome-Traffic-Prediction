# 10 · 实测修正补充（Reality Check Addendum）

> 本文修正 [06-pipeline.md](./06-pipeline.md) 里关于 paperswithcode（下称 pwc）的**判断粒度**，并记录三条实测发现对外部源策略与上线顺序的**强制影响**。
>
> **修正原则**：06 文的结论方向没错（pwc 已不可作为 SOTA 数据源），但把 10 条 pwc 链接**归为同一类失效**是错的。实测显示这些链接落在**三种不同**的状态上，处置策略必须分开。
>
> 全部实测日期：**2026-11-24**（Asia/Shanghai）。所有命令可原样复跑，见 §5。

---

## 1. 修正 06 文 §5.1：pwc 链接不是「全部失效」，是「分三类」

### 1.1 06 文的原始判断

06 文 §5.1 写的是：

> 即：**所有 `paperswithcode.com/*` 链接都会被整体重定向到 Hugging Face 的「trending papers」首页**，不再保留任何原路径语义。

这个判断来自当时只探了 `sota/...` 与 `task/...` 两类路径。**结论对这两类成立，但对 `paper/<slug>` 不成立。**

### 1.2 实测：10 条不同 URL 的逐条结果

测试方法：`curl -sSL` 跟随重定向，带浏览器 UA（`Mozilla/5.0 ... Chrome/131.0`），每条探测 2 次取稳定值。UA 很关键——不带 UA 时 5 条里有 3 条返回 `000`（TLS 层 `SSL_ERROR_SYSCALL`），会误判为"站点挂了"。

| # | 原 URL（`https://paperswithcode.com/` 之后） | 出现位置 | 最终状态 | `url_effective` | 判定 |
| --- | --- | --- | --- | --- | --- |
| 1 | `sota/traffic-prediction-on-metr-la` | `datasets/METR.md:9` | 200 | `huggingface.co/papers/trending` | **信息丢失** |
| 2 | `sota/traffic-prediction-on-pems-bay` | `datasets/PEMS.md:13` | 200 | `huggingface.co/papers/trending` | **信息丢失** |
| 3 | `sota/traffic-prediction-on-nyc-bike-dataset` | `datasets/NYC-Bike.md:9` | 200 | `huggingface.co/papers/trending` | **信息丢失** |
| 4 | `sota/traffic-prediction-on-nyc-taxi-dataset` | `datasets/NYC-Taxi.md:9` | 200 | `huggingface.co/papers/trending` | **信息丢失** |
| 5 | `task/traffic-prediction` | `README.md:413` / `README.zh.md:413` | 200 | `huggingface.co/papers/trending` | **信息丢失** |
| 6 | `task/multivariate-time-series-forecasting/latest` | `README.md:427` / `README.zh.md:427` | 200 | `huggingface.co/papers/trending` | **信息丢失** |
| 7 | `paper/spatio-temporal-graph-convolutional-networks` | `README.md:91` / `README.zh.md:91` | 200/302 | `huggingface.co/papers/1709.04875` | **部分保留** |
| 8 | `paper/spatio-temporal-graph-structure-learning-for` | `README.md:131` / `README.zh.md:131` | 200 | `huggingface.co/papers/trending` | **信息丢失** |
| 9 | `paper/spatial-temporal-attention-wavenet-a-deep#code` | `README.md:157` / `README.zh.md:157` | 200 | `huggingface.co/papers/trending` | **信息丢失** |
| 10 | `paper/time-series-is-a-special-sequence-forecasting#code` | `README.md:158` / `README.zh.md:158` | **404** | `huggingface.co/papers/2106.09305` | **目标缺失** |

**总计数**：仓库内 pwc 引用 **16 处**（`README.md` 6 + `README.zh.md` 6 + `datasets/*.md` 4），去重后 **10 条不同 URL**。

### 1.3 三类状态的本质区别

**A 类 · 信息完全丢失（8 条：上表 #1–6、8、9）**

- HTTP 200，但落地页是 HF 的 **trending 榜单首页**（`<title>Trending Papers - Hugging Face</title>`）。
- 原路径信息（哪个数据集、哪个任务、哪篇论文）**100% 丢失**。
- 危害最隐蔽：状态码是 200，任何"只看状态码"的链接检查都判为健康。
- 其中 #8、#9 尤其糟：这两条在 README 里的标签分别是 `[[SOTA]]` 与 `[[code]]`，属于**最强语义承诺**，却指向一个通用榜单。

**B 类 · 部分保留（1 条：#7）**

- `paper/spatio-temporal-graph-convolutional-networks` → `huggingface.co/papers/1709.04875`。
- **路径里的 slug 被正确映射到了 arXiv id**（STGCN 的 arXiv id 正是 `1709.04875`）。
- 落地页内容核对：`<title>Paper page - Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting</title>`，110 KB，标题正确。
- **但原链接的语义是 `[[code]]`（"这里有代码"），而落地页并没有代码仓库链接**——实测页面里出现的唯一 GitHub 链接是 `github.com/huggingface`（HF 自己的页脚）。
- 判定：**标题/论文信息保留，代码信息丢失**。所以"尚可用"要打个折扣：它能替代 `[[paper]]`，**不能**替代 `[[code]]`。

**C 类 · 404（1 条：#10）**

- `paper/time-series-is-a-special-sequence-forecasting#code` → `huggingface.co/papers/2106.09305`，返回 **404**。
- 同一 arXiv id 的 `https://arxiv.org/abs/2106.09305` 返回 **200**（实测）。
- 判定：**HF 上该论文页不存在**，重定向目标本身是坏的。这是最严重的一类，因为连"跳到别的地方凑合看"都做不到。

### 1.4 所以 06 文的处置矩阵需要改哪几行

| 06 文原表格中的位置 | 原处置 | **修正后** | 理由 |
| --- | --- | --- | --- |
| `README.md:131` `[[SOTA]]` | 改成 leaderboard 锚点 | **不变**（优先级最高） | 实测落点=trending，语义承诺完全落空 |
| `README.md:91` `[[code]]` | 换官方 repo，找不到就删标签 | **不变，但增加一条**：可用 `HF papers/1709.04875` 作为 `[[paper-alt]]` 存在，但仍须换 repo 或删 `[[code]]` | 实测 B 类：论文页在，代码不在 |
| `README.md:157` `[[code]]` | 换 repo / 删标签 | **不变** | 实测落点=trending |
| `README.md:158` `[[code]]` | 换 repo / 删标签 | **升级为 P0** | 实测 404，比前两条更急 |
| `README.md:413/427` `task` 页 | 换 HF papers 或仓库内页面 | **收紧**：不能简单换成 `huggingface.co/papers`——那正是当前错误落点 | 换了等于原地不动 |
| `datasets/*.md` 4 处 | 加失效标注 → 换本地 leaderboard | **不变** | 4 条全是 A 类 |
| （06 文未提到）`README.zh.md` 对应的 6 处 | 与英文同步 | **不变** | 逐条对应 |

**一句话总结修正**：06 文说"10 处全部失效"应改为「**10 条 URL 里 8 条信息完全丢失、1 条只剩标题、1 条直接 404**」。这个精度差异直接决定了处置的优先级排序与能否用 HF 兜底。

---

## 2. 三档链接处理策略（含字段写法）

按「**能不能立刻做**」而不是「重不重要」分三档。字段写法严格对齐 `data/schema/paper.schema.json` 与 `data/schema/dataset.schema.json`（均为 `additionalProperties: false`，所以只能用已定义的字段——**不能自己发明 `link_status`**）。

### 2.1 档 1 · 立即可替换（无需新数据）

适用于：能确定正确目标的、且替代品现成可用的链接。

| 对象 | 替换为 | 字段写法 |
| --- | --- | --- |
| `README.md:158` 的 `[[code]]`（pwc 404） | 删掉该标签；若论文有官方仓库则换 repo | `code_url: null`，`code_links: []` |
| `README.md:91` 的 `[[code]]`（B 类，标题保留） | `code_url` 换成官方 TF repo（`github.com/VeritasYin/STGCN_IJCAI-18` 之类，需人工核实），找不到则 `null` | `code_url: null` / `<repo>`；`alt_urls` 可留 `"hf-papers": "https://huggingface.co/papers/1709.04875"` |
| `README.md:157` 的 `[[code]]`（A 类） | 同上 | 同上 |
| `datasets/METR.md` 等 4 处的 `SOTA` 标题链接 | **先删链接、留标题**，内容改指向 `README.md#0x07-benchmark--evaluation` | dataset 记录：`status: "active"`，不加新字段；doc 侧只改文字 |
| `README.md:413` 的 `[paper with code](…task/traffic-prediction)` | 删除该行（第 415 行已有 `github.com/topics/traffic-prediction`，重复） | — |
| `README.md:427` 的 `[Multivariate Time Series Forecasting](…task/…/latest)` | 删除该行（第 8 项 `deep-learning-time-series` 已覆盖） | — |

**顺序说明**：把「删」排在「换」前面。因为删是**确定性的、不依赖外部核实**的动作，可以当天完成；换需要人工找 repo，可能拖几周。

### 2.2 档 2 · 保留 + status 标记（需要判定，但不需要新数据）

适用于：链接仍有部分价值，或暂时找不到替代，但**必须让读者知道它坏了**。

**schema 已有的两个字段正好够用**：

```jsonc
// data/papers/2021.json —— wavenet 条目（A 类，code 链接失效）
{
  "id": "spatial-temporal-attention-wavenet-2021",
  "status": "active",                    // 论文本身仍有效，不设 dead_link
  "code_url": null,                      // ← 失效的 pwc 链接已清空
  "code_links": [],
  "alt_urls": {},
  "verification": {
    "checked_at": "2026-11-24",
    "link_ok": false,                    // ← 关键：明确记录"这条链过、且坏了"
    "metadata_source": "readme_migration",
    "confidence": "high"                 // ← 已人工确认，所以是 high 而非 low
  },
  "provenance": { "source": "readme_migration", "readme_line": 157, "migrated_at": "2026-11-24" }
}
```

```jsonc
// data/datasets.json —— metr-la（SOTA 链接失效，但数据集本体有效）
{
  "id": "metr-la",
  "status": "active",                    // ← 不要用 deprecated！数据集没废弃，只是外部榜单没了
  "url": "https://...官方数据页面...",
  "local_doc": "datasets/METR.md",
  "provenance": { "source": "readme_migration", "migrated_at": "2026-11-24" }
}
```

**字段选择的理由**：

- `status` 只用来描述**实体本身**。paper 侧的枚举是 `['active','dead_link','superseded','retracted']`，dataset 侧是 `['active','deprecated','mirror_only']`。**pwc 失效不代表论文/数据集失效**，所以 `status` 应保持 `active`，用 `verification.link_ok: false` 表达"链接坏了"。
- `verification.confidence` 的语义是"元数据可信度"：人工逐条核过 → `high`；只做了自动迁移 → `low`（迁移样例 `data/papers/2018.json` 里 3 条是 `null`/`low`，实测）。
- `paper.schema.json` 里 **`verification.link_ok` 是 `boolean`，没有 "unknown"**。所以「没查过」只能写 `null` 或省略字段，不能写 `"unchecked"`。这是 schema 的硬约束，见 §5.6。

**展示层规则（建议，写进 07 文的门禁）**：

- `verification.link_ok === false` 的条目，站点上该链接**置灰 + tooltip「链接已失效（2026-11-24 实测）」**；
- README 侧不改文字（避免与渲染产物打架），但 `docs/evolution/` 记录清单；
- `verification.checked_at` 超过 180 天 → 视为待重检（L2 巡检抓这个）。

### 2.3 档 3 · 移除并注明原因（需要新数据 / 需要人工决策）

适用于：目标本身不存在，或替代需要先建新结构。

| 对象 | 处理 | 前置 |
| --- | --- | --- |
| `README.md:131` 的 `[[SOTA]]` | 移除 `[[SOTA]]` 标签，改为普通论文条目；「SOTA 在哪看」交由 `data/leaderboards/*.json` 渲染 | **依赖 Phase 3 的 leaderboard**。在此之前只做「降级为普通条目」 |
| `datasets/*.md` 4 处的 SOTA 小节内容 | 改成从 `data/leaderboards/<dataset>.json` 渲染的表格 | **依赖 leaderboard + Phase 2 站点** |
| `README.md:413` 的 `[paper with code]` 条目 | 从「相关仓库」列表里移除，并在 `docs/evolution/06-pipeline.md` §5.4 记录原因 | 无前置，但**必须留痕**——否则下一个人会重新加回来 |

**「注明原因」的落点**（三处都写，缺一不可）：

1. 该 PR 的描述里写清「为什么删、实测证据是什么」；
2. `docs/evolution/06-pipeline.md` §5.4 的处置矩阵里标 `✅ 已处理 2026-11-24`；
3. `updates/reports/link-status.json` 里保留这条 URL 的 `redirect_away` 记录，并加 `"resolution": "removed_in_pr#NNN"`。

### 2.4 处置优先级总表

| 优先级 | 对象 | 档位 | 阻塞项 | 建议时间 |
| --- | --- | --- | --- | --- |
| **P0** | `README.md:158` + `README.zh.md:158`（pwc 404） | 1 | 无 | Phase 1 期间即可做 |
| **P0** | `README.md:413`、`:427` + ZH 对应两处（task 页） | 1（删） | 无 | 同上 |
| **P1** | `README.md:91`、`:157` 的 `[[code]]` + ZH | 1（删）/ 2 | 需人工找 repo | Phase 1–3 |
| **P1** | `datasets/*.md` 4 处 SOTA 链接 | 1（去链）+ 2（status） | 无 | Phase 1 |
| **P2** | `README.md:131` 的 `[[SOTA]]` | 3 | leaderboard | Phase 3 |
| **P2** | `datasets/*.md` 的 SOTA 内容重建 | 3 | leaderboard + 站点 | Phase 3 |

---

## 3. 修正 OpenAlex 查询策略

### 3.1 问题：`sort=cited_by_count` + 长自然语言 query = 噪声

06 文 §2.1 只说了「OpenAlex 是元数据主源」，没说**它的 search 对长 query 有多不可靠**。实测把 `traffic-monthly.json` 里那种长 query 拿去做引用排序，前 8 条如下：

**反例 1** —— `search=<长自然语言 query>&sort=cited_by_count:desc`（无 filter）：

```text
query = "traffic flow forecasting with graph neural network and attention mechanism for long term prediction"

   15964  2020  review            A Comprehensive Survey on Graph Neural Networks
   10235  2021  article           Review of deep learning: concepts, CNN architectures, ...
    9386  2020  article           Graph neural networks: A review of methods and applications
    7520  2018  conference-paper  Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework ...
    6809  2021  review            A Survey of Convolutional Neural Networks: ...
    5781  2017  preprint          Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic ...
    4612  2019  conference-paper  Graph WaveNet for Deep Spatial-Temporal Graph Modeling
    4234  2020  article           Digital Twin: Enabling Technologies, Challenges and Open Research
```

**8 条里 4 条与本领域无关**（GNN 综述、CNN 综述、深度学习综述、Digital Twin）。OpenAlex 的相关性打分被 `sort` 完全覆盖，等于"在我粗略匹配到的几万条里，按引用数取前 8 条"。

**反例 2** —— 加上 `filter=type:article,from_publication_date:2024-01-01`：

```text
meta.count = 8358
       0  2026  article  Traffic Flow Forecasting with Dynamic Graph Neural Networks and Incide...
      29  2024  article  A Spatiotemporal Graph Neural Network with Graph Adaptive and Attentio...
     172  2024  article  LSTTN: A Long-Short Term Transformer-based spatiotemporal neural netwo...
       1  2025  article  Traffic Flow Prediction Method Based on Deep Learning and Graph Neural...
```

**领域相关性恢复了**，但 `8358` 这个 count 说明**lenient search 仍在匹配全库**，只是被年份+类型砍掉了一部分。仍然要靠后面的规则过滤兜底。

**反例 3** —— 短 query 也没救回来：

```text
query = "traffic flow prediction" & filter=type:article,from_publication_date:2024-01-01 & sort=cited_by_count:desc
meta.count = 84408        ← 比长 query 还多！

    1818  2024  All One Needs to Know about Metaverse: ...
    1805  2024  Generalized Out-of-Distribution Detection: A Survey
    1729  2024  A review of convolutional neural networks in computer vision
```

**结论**：**只要是 `search=`（全文 lenient 匹配）+ `sort=cited_by_count`，就是不可用的**，跟 query 长短关系不大。Lead 的判断「短 query 才可用」需要修正为「**不是短 query 可用，而是换掉 `search=` 才可用**」。

### 3.2 `title.search` 不是合法 query 参数（实测 400）

Lead 建议用 `title.search`。**实测 `title.search=` 作为顶层 query 参数会 400**：

```text
GET https://api.openalex.org/works?title.search=traffic+flow+prediction
→ HTTP 400
  {"error":"Invalid query parameters error.",
   "message":"title.search is not a valid parameter. Valid parameters are: ..."}
```

正确写法是把它放进 **`filter=`**：

```text
GET https://api.openalex.org/works?filter=title.search:traffic flow prediction,type:article&sort=cited_by_count:desc
→ 200, meta.count = 2059

   3659  2014  article  Traffic Flow Prediction With Big Data: A Deep Learning Approach
   1216  2014  article  Deep Architecture for Traffic Flow Prediction: Deep Belief Networks ...
   1186  2017  article  Deep learning for short-term traffic flow prediction
    969  2015  article  Short-term traffic flow prediction using seasonal ARIMA model ...
    935  2008  article  Online-SVR for short-term traffic flow prediction ...
    894  2018  article  A hybrid deep learning based traffic flow prediction method ...
```

**6/6 全部命中领域**，且 `count=2059` 是一个可信的规模（title 里真的含 "traffic flow prediction" 的文章就这么多）。这是唯一验证有效的排序查询形态。

### 3.3 修正后的 OpenAlex 查询策略（四条硬规则）

| # | 规则 | 依据 |
| --- | --- | --- |
| **O1** | **`title.search:` 只能出现在 `filter=` 里，不能作为顶层参数** | 实测顶层参数 400 |
| **O2** | **禁止 `search=`（全文 lenient）+ `sort=cited_by_count` 组合**；`search=` 只允许配合时间窗 + 后续规则过滤，且**不得用于排行榜** | 反例 1/2/3，噪声率 50%+ |
| **O3** | **必须加 `filter=type:article`**（或显式枚举 `conference-paper`） | 反例 1 里 8 条中 2 条是 `review`、1 条 `preprint` |
| **O4** | **venue 白名单**：交通领域核心 venue 用 `primary_location.source.id` 精确过滤，不做名称模糊匹配 | README 有 45 个 venue 短名，含 `WWWC/WC/WWW`、`IF/Information Fusion` 等异写；按名称匹配必然漏 |

推荐的三个查询形态（按用途分开，不要混）：

```bash
# (a) 月度候选发现：短 query + lenient search + 时间窗，不用 sort
GET /works?search=traffic flow forecasting
          &filter=from_publication_date:2024-01-01,type:article
          &sort=publication_date:desc&per-page=25

# (b) 排行榜按引用排序：title.search 精确匹配 + 时间窗
GET /works?filter=title.search:traffic flow prediction,type:article,from_publication_date:2020-01-01
          &sort=cited_by_count:desc&per-page=50

# (c) 引用数快照 / 元数据补全：按 id 批量取（唯一确定性的用法）
GET /works?filter=ids.openalex:W4391987728|W4392382537&per-page=50&select=id,doi,cited_by_count,publication_year
```

`venue` 白名单的落地形态（建议放进 `scripts/configs/`）：

```jsonc
// scripts/configs/openalex-venues.json（建议文件名）
{
  "note": "按 primary_location.source.id 精确匹配，不按名称",
  "venues": {
    "TITS":  "S4210177424",
    "TRC":   "S2764467640",
    "KDD":   "S4306417261",
    "IJCAI": "S4306417288"
  }
}
```

> ⚠️ 上面的 source id 是**示意值，未经实测核对**。落地时必须逐个 `GET /sources?search=<name>&select=id,display_name` 确认后写入——本文不对具体 id 作断言。

### 3.4 对 06 文 §2 的净影响

| 06 文原文 | 修正 |
| --- | --- |
| 「OpenAlex…`cited_by_count` 可以替代影响力榜」 | **有条件成立**：只在 `filter=title.search:…,type:article` 形态下成立 |
| 「arXiv（必须）…唯一能稳定拿到本月新预印本」 | **方向成立，但"稳定"要打折**——见 §4.1，实测有 429 |
| 「Crossref（可缺）…arXiv DOI 在 Crossref 404」 | **成立，实测复核一致**（`10.48550/arxiv.2403.01091` → 404） |

---

## 4. arXiv 与「稳定性」的两条实测补充

### 4.1 arXiv API 有速率限制，且**同一个 query 会间歇性 429**

这不改变「arXiv 是预印本主源」的结论，但影响 06 文 §2.3 对重试策略的设计。

实测（2026-11-24，同一 IP）：

| 尝试 | 请求 | 结果 |
| --- | --- | --- |
| 1 | `abs:traffic AND abs:forecasting&sortBy=submittedDate&max_results=6`（带自建 UA） | **429**，0 字节 |
| 2 | 间隔 4 s 重试 | 429 |
| 3 | 间隔 45 s 后重试 | 429 |
| 4 | `all:traffic&max_results=1`（**另一个更轻的 query**） | **200**，3133 字节 |
| 5 | 再次 `abs:traffic AND abs:forecasting` | 429 |
| 6 | 间隔 8 s 重试（**最终成功**） | **200**，15603 字节 |

**成功时的内容**（证明"能捞到当日预印本"这个结论是对的）：

```text
2026-11-23 | STMemoLLM: A Spatial-Temporal Memory-Augmented LLM for Robust Traffic Forecasting
2026-11-21 | DeCo: Decoupled Traffic State Modeling with Cross-Variable Coordination
2026-11-20 | Graph Signal Generation via Graph-Aware Diffusion Models
2026-11-20 | Action-Conditioned Spatio-Temporal Forecasting
2026-11-19 | A Sparsity Analysis of Traffic Forecasting Architectures: How Much Missing Data ...
2026-11-19 | From Data Alignment to Parameter Generation: Few-Shot Cross-City Traffic Predict...
```

对照组：`https://arxiv.org/abs/2402.11838`（非 API、普通页面）**始终 200**，无 429。

**可操作结论**（比 06 文 §2.3 更具体）：

- 429 是 **API 侧**的，不是网络问题；`export.arxiv.org` 的 Atom 接口对**复合布尔 query**（`abs:x AND abs:y`）明显更敏感，轻 query（`all:x&max_results=1`）更容易过；
- 06 文建议「arxiv `source_min_interval_seconds: 0.8`」**太激进**——实测间隔 0.8 s 会稳定触发 429。建议 arxiv 单独提到 **≥ 3.0 s**，`max_retries` 提到 **3**，退避基数 ≥ 5 s；
- **重试必须由外层驱动**：`HttpClient` 的 `max_retries: 2` + `retry_base_seconds: 2.0` 意味着最坏只等 2+4=6 s，实测不足以跨过 429 窗口；
- **必须把 arXiv 的失败显式暴露**：现状 `source_errors` 只打 WARN 且 `return 0`（06 文 §7.3）。若 arXiv 连续 429，候选表会**悄悄只剩 OpenAlex**，看不出异常。

### 4.2 「数据量护栏」是硬约束，且它决定了上线顺序

`site/scripts/prepare-data.mjs` 里有一条**硬性完整性护栏**（`loadStructuredPapers()`，实测代码）：

```js
const readmeCount = (await scrapeReadme('README.md', 'en')).length;
const minExpected = Number(process.env.SITE_MIN_STRUCTURED_PAPERS ?? 0) ||
  Math.max(1, Math.floor(readmeCount * 0.75));

if (papers.length < minExpected) {
  log(`REJECT data/papers: ${papers.length} papers < expected >=${minExpected} ` +
      `(README has ${readmeCount}); falling back to README scrape`);
  return null;                                 // ← 直接放弃结构化数据
}
```

**实测数字（按 `parseLine` 逐字复刻得出）**：

```text
readmeCount              = 104      （README §0x00 Papers 内能解析出 title ≥ 8 字符且带 [[…]] 链接的条目）
floor(0.75 × readmeCount) = 78      ← 结构化数据必须 ≥ 78 条才会被采纳
```

当前状态（`data/papers/` 已有 12 个文件、**104 条**）：

```text
$ cd site && ASTRO_TELEMETRY_DISABLED=1 pnpm build:fast
[prepare-data] source=data/papers (12 file(s), 104 papers)

provenance: {'source': 'data/papers', 'files': [...12 files...], 'readmeCount': 104}
count: 104
```

即**护栏已通过**，`degraded` 不再是 `true`。这一点与 06 文 §0 快照里"`degraded: True`、`count: 104 ≠ 102`"的描述不同——那是更早的快照，当时 `data/papers` 尚未覆盖全部年份。

### 4.3 这个护栏对上线顺序的**强制影响**

这是本文最重要的一条结论：

> **Phase 2（站点）想真实可用，Phase 1 必须先把 `data/papers/` 覆盖到 ≥ 78 条。**
> 不是"最好先做"，而是"**做不到就静默降级**"——护栏失败时 `prepare-data.mjs` 会**回退到 README 抓取**，站点照样能 build 成功，但：
>
> - 站点展示的是**刮出来的**数据（含 `…` 占位被当条目、跨行折断丢失、无 `method_tags`、无 `doi`）；
> - `provenance.source` 变成 `README.md`，**没有任何 CI 会报错**；
> - 于是"站点上线"成功，但数据质量停在 Phase 0 水平。

**因此 Phase 2 的验收必须先看 `provenance.source`**，而不是看 `build` 是否退出 0。见 [09-acceptance.md](./09-acceptance.md) §2.3（该处已有这条断言，本文补充了它的**原因**与**阈值来源**）。

**反过来说**：这个护栏是好事。它把"Phase 1 没做完就上线站点"这种情况**从静默错误变成了显式回退**。只要门禁盯着 `provenance.source === 'data/papers'`，就不会被骗。

---

## 5. 复查命令（全部可原样复跑）

### 5.1 pwc 三类状态复现

```bash
cd "$(git rev-parse --show-toplevel)"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"

# ⚠️ 必须带浏览器 UA 且必须 -L（跟重定向）；不带 UA 会出现大量 000 假失效
for u in \
  "https://paperswithcode.com/sota/traffic-prediction-on-metr-la" \
  "https://paperswithcode.com/task/traffic-prediction" \
  "https://paperswithcode.com/paper/spatio-temporal-graph-convolutional-networks" \
  "https://paperswithcode.com/paper/time-series-is-a-special-sequence-forecasting#code" ; do
  printf "%-64s " "${u#https://paperswithcode.com/}"
  curl -sSL -o /dev/null -m 25 -A "$UA" -w "%{http_code} -> %{url_effective}\n" "$u"
done
```

期望（2026-11-24 实测）：

```text
sota/traffic-prediction-on-metr-la                             200 -> https://huggingface.co/papers/trending
task/traffic-prediction                                        200 -> https://huggingface.co/papers/trending
paper/spatio-temporal-graph-convolutional-networks             200 -> https://huggingface.co/papers/1709.04875
paper/time-series-is-a-special-sequence-forecasting#code       404 -> https://huggingface.co/papers/2106.09305
```

**关键判据不是状态码，是 `url_effective`**：`redirect_away`（落点路径语义与原路径无关）= 信息丢失。

### 5.2 B 类「只剩标题、代码没了」的复现

```bash
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
curl -sSL -m 30 -A "$UA" "https://huggingface.co/papers/1709.04875" -o /tmp/hf.html
grep -oiE '<title>[^<]*</title>' /tmp/hf.html        # 标题正确
grep -oE 'https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+' /tmp/hf.html | sort -u | head
# 实测：只出现 https://github.com/huggingface（HF 页脚），没有 STGCN 的代码仓库
```

### 5.3 OpenAlex 查询形态复现

```bash
# (a) 反例：search + sort=cited_by_count → 噪声
curl -s "https://api.openalex.org/works?search=traffic+flow+forecasting+with+graph+neural+network&sort=cited_by_count:desc&per-page=8&select=id,title,cited_by_count,publication_year,type"

# (b) title.search 作为顶层参数 → 400
curl -s -o /dev/null -w "%{http_code}\n" "https://api.openalex.org/works?title.search=traffic+flow+prediction"
# => 400

# (c) 正例：filter=title.search:…,type:article + sort
curl -s "https://api.openalex.org/works?filter=$(python3 -c "import urllib.parse;print(urllib.parse.quote('title.search:traffic flow prediction,type:article'))")&sort=cited_by_count:desc&per-page=6&select=id,title,cited_by_count,publication_year,type"
# => meta.count=2059，前 6 条全部是本领域
```

### 5.4 arXiv 429 复现与最终成功

```bash
UA="awesome-traffic-prediction-search/0.4 (mailto:maintainers@awesome-traffic-prediction.invalid)"
for i in 1 2 3 4 5; do
  code=$(curl -sSL -m 30 -A "$UA" -o /tmp/ax.xml -w "%{http_code}" \
    "https://export.arxiv.org/api/query?search_query=abs:traffic+AND+abs:forecasting&sortBy=submittedDate&sortOrder=descending&max_results=6")
  echo "attempt $i: http=$code bytes=$(wc -c < /tmp/ax.xml)"
  [ "$code" = "200" ] && break
  sleep 8
done
python3 -c "
import xml.etree.ElementTree as ET
ns={'a':'http://www.w3.org/2005/Atom'}
for e in ET.parse('/tmp/ax.xml').getroot().findall('a:entry',ns)[:6]:
    print(e.findtext('a:published',default='',namespaces=ns)[:10],'|',(e.findtext('a:title',default='',namespaces=ns) or '').strip()[:70])"
```

### 5.5 护栏与 `readmeCount` 复现

```bash
# 直接读护栏代码
grep -n "readmeCount\|minExpected\|SITE_MIN_STRUCTURED_PAPERS" site/scripts/prepare-data.mjs

# 实测当前 readmeCount 与阈值（复刻 parseLine 的逻辑，需 node）
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
  if(parseLine(line)) n++;
}
console.log("readmeCount =", n, "| floor(0.75n) =", Math.floor(n*0.75));'
# 实测: readmeCount = 104 | floor(0.75n) = 78

# 结构化数据当前条数
python3 -c "
import json,glob
print('structured =', sum(len(json.load(open(f))) for f in glob.glob('data/papers/*.json')))"
# 实测: structured = 104  → 已过护栏

# 站点实际采用了哪个源
grep -o '"source": "data/papers"\|"source": "README.md"' site/src/data/papers.generated.json 2>/dev/null | head -1
# 期望: "source": "data/papers"
```

### 5.6 对 §2.2 字段写法的 schema 复核

```bash
# 确认 verification 只有这 4 个字段（additionalProperties: false）
python3 -c "
import json; s=json.load(open('data/schema/paper.schema.json'))
print(list(s['properties']['verification']['properties']))
print('link_ok type:', s['properties']['verification']['properties']['link_ok']['type'])
print('status enum:', s['properties']['status']['enum'])"
# 实测: ['checked_at','link_ok','metadata_source','confidence']
#        link_ok type: boolean          ← 所以「未检查」只能省略字段，不能写字符串
#        status enum: ['active','dead_link','superseded','retracted']

python3 -c "
import json; s=json.load(open('data/schema/dataset.schema.json'))
print('dataset has verification:', 'verification' in s['properties'])
print('dataset status enum:', s['properties']['status']['enum'])"
# 实测: dataset has verification: False
#        dataset status enum: ['active','deprecated','mirror_only']
```

> **两条 schema 硬约束，落地时不能绕过**：
>
> 1. **`dataset` 侧没有 `verification` 字段**（实测 `'verification' in properties` → `False`）。所以「数据集 SOTA 链接失效」这件事**无法用字段表达**，只能靠 `docs/evolution/` 的清单与 `updates/reports/link-status.json` 记录。若确实需要，必须**先改 schema**（加 `verification`），而不是往 JSON 里塞未定义字段——`additionalProperties: false` 会让它直接被 schema 校验打回。
> 2. **`dataset.status` 描述的是数据集本体**（`deprecated` = 数据本身不再维护），**不是链接健康**。所以 §2.1 里「datasets 4 处 SOTA 只去链、不改 `status`」的写法是对的；如果改用 `status: "deprecated"` 来表达"链接坏了"，表达的就是**错误事实**（METR-LA 并没有停止维护）。

---

## 6. 一句话汇总

| 项 | 06 文原判断 | 实测修正 |
| --- | --- | --- |
| pwc 链接 | 10 处全部跳 trending，信息全丢 | **8 条全丢、1 条只剩标题（代码没了）、1 条 404** |
| OpenAlex | 可做元数据主源，`cited_by_count` 可排序 | **只有 `filter=title.search:…,type:article` 可用；`search=` + sort 噪声 50%+；`title.search` 作顶层参数 400** |
| arXiv | 预印本主源，0.8 s 间隔足够 | **复合布尔 query 有间歇 429；间隔需 ≥ 3 s、重试需外层驱动** |
| 上线顺序 | 先数据后站点（理由：别固化错误解析） | **更硬的理由：`readmeCount=104` → 护栏阈值 78，数据不足时站点静默回退 README，且 build 仍成功** |