# 02 论文范围规划：2023–2026 数据更新

> 状态：规划稿。本文只定义范围、分层、目标数量、检索式与审核节奏，**不新增任何论文条目**。
> 需要举例而未经验证的地方一律写 `[PLACEHOLDER: 待检索确认]`。所有"现状"数字来自当前工作区 `README.md` / `README.zh.md` 的实测统计（统计口径见 §1.2）。
> 配套 schema 与字段语义见 [`01-data-model.md`](./01-data-model.md)。

---

## 1. 现状盘点

### 1.1 每年条目数（实测）

统计口径：`README.md` 中 `### Deep Learning Based Traffic Prediction Methods` 小节内，每个 `#### YYYY` 标题下、能同时被下面两条正则之一匹配的条目：

```text
ITEM = ^\s*(\d+)\.\s+\[([^\]]+)\]\s+(.+?)\s+\[\[paper\]\]\(([^)]+)\)
ALT  = ^\s*(\d+)\.\s+\[([^\]]+)\]\s+\[(.+?)\]\((https?://[^)\s]+)\)
```

（`ALT` 只用于其他小节，深度学习小节内 100% 由 `ITEM` 命中；`README.zh.md` 结果与之逐条相同。）

| 年份 | 条目数 | 期刊 | 会议 | 预印本 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2015 | 1 | 0 | 1 | 0 | |
| 2016 | 1 | 0 | 1 | 0 | |
| 2017 | 3 | 1 | 1 | 1 | |
| 2018 | 11 | 3 | 8 | 0 | 其中 1 条 `[TITS 2019]` T-GCN 实为 2019 归档 |
| 2019 | 20 | 7 | 11 | 2 | |
| 2020 | 26 | 12 | 10 | 4 | 全仓库峰值 |
| 2021 | 9 | 1 | 5 | 3 | 其中 1 条 `[Arxiv 2020]` STFGNN 实为 2020 |
| 2022 | 2 | 1 | 0 | 1 | **断崖** |
| 2023 | 2 | 0 | 1 | 1 | |
| 2024 | 4 | 2 | 0 | 2 | |
| 2025 | 4 | 2 | 0 | 2 | 全部在 `AUTO RECENT PAPERS` 块内 |
| 2026 | 11 | 10 | 0 | 1 | 同上；10 条期刊里大量为 MDPI/Elsevier 常规期刊 |
| **合计** | **94** | 39 | 40 | 15 | |

按发布类型汇总：期刊 39（41.5%）、会议 40（42.6%）、预印本 15（15.9%）。
（`venue_type` 的判定表见 §1.4；45 个短名全部已归类，无 `unknown`。）

### 1.2 其它关键数字

| 指标 | 实测值 | 对规划的含义 |
| --- | ---: | --- |
| 深度学习小节条目合计 | 94 | 本规划的主战场 |
| 全 README 带 `[[paper]]` 的条目 | 102 | = 94 + Reviews 6 + Time Series 2（另有 Statistic 2 条走 `ITEM` 但不在 DL 小节） |
| 其中能抽到 DOI | 24（23.5%） | 去重不能只靠 DOI |
| 其中 URL 指向 arXiv | 30 | 预印本占比高，版本漂移风险大 |
| `#### 2015–2018` 条目 | 16（1+1+3+11） | 经典基线区，基本冻结 |
| `#### 2019–2021` 条目 | 55（20+26+9） | 主力基线区，**重复率高**（同一模型多版本） |
| `#### 2022–2026` 条目 | 23（2+2+4+4+11） | 近五年只有 23 条，其中 2022 仅 2 条 |
| 2022–2026 占总数比 | 24.5% | 一个"近年论文合集"的仓库，近五年不到四分之一 |
| 2023–2026 期刊/预印本比 | 期刊 14 : 预印本 6 | 近年靠 MDPI/Frontiers 类期刊填充，**顶会（KDD/NeurIPS/ICLR/ICML/WWW/AAAI/IJCAI）在 2023–2026 段为 0 条** |

> **最刺眼的一条结论**：2023 年之后，`README` 里**没有任何 KDD / NeurIPS / ICLR / ICML / WWW / AAAI / IJCAI 的条目**。2019–2020 两年有 12 个会议来源，2021 还有 5 个，2022 起归零。这不是领域没产出，而是采集方式（月更 OpenAlex 关键词检索 + 人工过筛）对顶会的召回不足——OpenAlex 对 NeurIPS/ICLR 的 venue 归一化很差，且会议论文常先以 arXiv 出现、后被 dedupe 当作重复丢弃。

### 1.3 结构性问题（规划必须解决）

1. **2022 断崖**：2 条。时间序列上 2021(9) → 2022(2) → 2023(2) 的凹陷不像真实分布，更像"某次重构丢了数据"。
2. **近年偏向低影响期刊**：2026 年 11 条中 10 条期刊，`Electronics`(3)、`Systems`、`PLoS ONE`、`Scientific Reports` 等；缺少同年的 KDD/NeurIPS 前沿工作。
3. **重复与近重复**：2020 年 26 条里存在同一主题多篇（`Spatio-Temporal Graph Structure Learning`、`GMAN`、`STSGCN`、`AGCRN`、`ST-GRAT`…），但**缺少"这是基线还是 SOTA"的标记**，读者无法判断该引谁。
4. **预印本与正式版并存风险**：15 条预印本里，`STFGNN`（L156，`[Arxiv 2020]` 放在 2021 段）、`STTN`（L126，`code-not-official`）都已正式发表，但仓库仍链 arXiv。
5. **综述过少**：仅 6 条，且最新的是 `Applied Intelligence 2022`。2023 之后没有综述覆盖 LLM4ST / 基础模型这一波。
6. **无 benchmark 复现条目**：`0x07 Benchmark & Evaluation` 只有 5 条文字说明 + 1 张矩阵表（4 行），没有任何"可复现实验/排行榜"的论文条目（如 Traffic-Benchmark / LibCity / OpenSTL 相关论文只在 Toolkits 里出现）。

### 1.4 `venue_type` 判定表（供统计与审核复用）

| 类型 | 短名清单（README 实测出现过的写法） |
| --- | --- |
| `journal` | TITS, TKDE, IJGIS, TRPET, TGIS, Information Fusion, IF, Applied Intelligence, Neural Networks, Expert Systems with Applications, PLoS ONE, IET Intelligent Transport Systems, ISPRS IJGI, ISPRS, Sensors, Sensor, Electronics, Systems, Scientific Reports, Neural Computing and Applications, Lecture Notes in Computer Science, Complex & Intelligent Systems, Access, Mobile Computing, ICTON |
| `conference` | NIPS, AAAI, IJCAI, KDD, ICLR, CVPR, ICCV, WWW, WWWC, WC, NeurIPS, ICDE, ICDM, PAKDD, IJCNN, ICIKM, IWPHM, Sigspatial |
| `preprint` | Arxiv |
| `course_report` | CS224W |
| `unknown` | 无（45 个短名全部已归类） |

注意 `WWWC`/`WC`/`WWW`、`IF`/`Information Fusion`、`Sensor`/`Sensors` 是同义异写，`01-data-model.md` §3.4 用 `venue.aliases` 归一。

---

## 2. 目标与分层策略

### 2.1 目标条目数（分年份段）

设计原则：**近年密、早年疏**；经典基线不追求数量而追求"每个范式至少 2 个可运行基线"；综述与 benchmark 单独成层，避免被期刊灌水淹没。

| 年份段 | 现状 | 最低目标 | 理想目标 | 增补重点 | 增补倍率 |
| --- | ---: | ---: | ---: | --- | ---: |
| 2015–2018 | 16 | 16 | 20 | 基本冻结；仅补漏（如 ConvLSTM 生态、早期 GNN 变体） | 1.0–1.25× |
| 2019–2021 | 55 | 55 | 60 | 不加量，改结构：补 `layers` 标记；把预印本升级为正式版链接 | 1.0–1.1× |
| 2022–2023 | 4 | 12 | 18 | **重点补齐断崖**：顶会 STGNN / 图结构学习 / 概率预测 | 3–4.5× |
| 2024–2025 | 8 | 16 | 24 | LLM4ST / 基础模型 / 长时域 / 扩散 | 2–3× |
| 2026 | 11 | 16 | 24 | 保留高质量期刊，**替换**纯灌水条目；补顶会与 arXiv 前沿 | 1.5–2.2× |
| **合计（DL 小节）** | **94** | **115** | **146** | | 1.2–1.55× |

分层的目标配额（按理想目标 146 条计）：

| 层 | 定义 | 现状 | 目标条数 | 占比目标 |
| --- | --- | ---: | ---: | ---: |
| `sota_core` | 近 3 年（2024–2026）在该方向被反复对比、有公开实现或进排行榜的工作 | 0（未标记） | 45 | 31% |
| `classic_baseline` | 2015–2021 被后续论文当基线对比的方法（DCRNN / STGCN / Graph WaveNet / ASTGCN / STSGCN / AGCRN / GMAN / ST-MetaNet / MTGNN …） | 0（未标记） | 50 | 34% |
| `survey` | 综述与开放问题梳理 | 6 | 15 | 10% |
| `application` | 面向落地的任务：估算 ETA、事件/事故检测、拥堵治理、信号控制联动、数据补全 | ~4 | 36 | 25% |

> 分层字段即 `paper.schema.json` 的 `layers[]`，可多层。**同一条论文允许同时是 `classic_baseline` + `application`**，但 `sota_core` 与 `classic_baseline` 互斥（否则"谁是基线"会被搞混）。

### 2.2 分层判定准则（人工审核时按此打标签）

| 层 | 硬条件 | 反例（不该进） |
| --- | --- | --- |
| `sota_core` | ① 2024 年后发表；② 至少在 1 个公开 benchmark（METR-LA / PEMS-BAY / PEMS03/04/07/08）上报告多步指标；③ 有公开代码 **或** 被 ≥2 篇独立论文作为对比方法引用 | 只在自建数据上实验、无对比表的期刊短文 |
| `classic_baseline` | ① 2021 年前发表；② 被近 3 年论文作为 baseline 的频次高（可用 `paper_search.py` 的引用数排序辅助判断） | 同一模型的第 3 个复现版本 |
| `survey` | 标题或摘要明确 survey/review/overview/open issues；或明确给出 taxonomies | 带 "review" 字样的单方法实验论文 |
| `application` | 任务不是"预测未来流量/速度"，而是补全、事件检测、ETA、控制联动、仿真校准等 | 纯理论推导 |

### 2.3 分层与 README 呈现的关系

`layers` 不改变 README 的年份分组（保持向后兼容），只在**渲染时**追加可见标记，例如：

```text
3. [IJCAI 2018] Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting [[paper]](...) [[code]](...) `baseline`
```

标记只在 `02-paper-scope.md` 约定的灰度期内以**可选渲染开关**形式提供（`--emit-layer-badge`），默认关闭，避免破坏 `check_readme_sync.py` 的条目计数逻辑。

---

## 3. 需要补充的方向清单（含可直接跑的检索式）

### 3.1 检索式书写规范

- **OpenAlex**：走 `scripts/paper_search.py` 的配置格式（`scripts/configs/traffic-monthly.json` v2），`search.queries[]` + `post.rules[]`。查询串是**自然语言式关键词**，脚本会拼成 `?search=<query>&filter=from_publication_date:...`。
  - 直接手跑 URL 形态：`https://api.openalex.org/works?search=<query>&filter=from_publication_date:2023-01-01,to_publication_date:2026-12-31&sort=publication_date:desc&per-page=50&page=1`
- **arXiv**：脚本用 `search_query=all:<query>`；需要字段限定、布尔组合、日期区间时必须手写 `search_query`：
  - `https://export.arxiv.org/api/query?search_query=<urlencoded>&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending`
  - 字段前缀：`ti:` `abs:` `all:` `cat:`；布尔：`AND` `OR` `ANDNOT`；短语：`"..."`；日期：`submittedDate:[202301010000 TO 202612312359]`
- 所有查询都必须带排除规则（沿用 `traffic-monthly.json` 的 `non-road-exclude`）：

```json
{
  "id": "non-road-exclude",
  "field": "title",
  "op": "any",
  "keywords": ["airspace","air traffic","aviation","flight","uav","drone","maritime","shipping","vessel","power load","ev load","microgrid"],
  "required": false,
  "exclude": true,
  "weight": 0.0
}
```

- 每条方向给的是**检索式**，不是论文清单。§4 会把它们整理成可直接落到 `scripts/configs/` 的配置片段（新增配置文件，不改现有 `traffic-monthly.json`）。

### 3.2 方向 A：时空基础模型 / LLM4ST

**为什么补**：`research-lineage.md` 第 7 节已点名"unified spatio-temporal models and prompt-based frameworks"，但仓库只有 1 条（`UniST`，2024），且 2023–2026 段没有同方向的第二条可对照工作。

**OpenAlex queries**

```
"urban spatio-temporal prediction foundation model"
"large language model spatio-temporal forecasting"
"prompt learning urban spatio-temporal"
"spatio-temporal pretraining cross-city generalization"
"universal spatio-temporal model zero-shot"
"urban computing LLM agent prediction"
```

**arXiv `search_query`**

```
(all:"spatio-temporal" OR all:"spatial-temporal") AND (all:"large language model" OR all:"LLM") AND (all:"forecasting" OR all:"prediction")
(abs:"foundation model" OR ti:"foundation model") AND abs:"spatio-temporal"
ti:"prompt" AND abs:"spatio-temporal" AND abs:"forecasting"
abs:"zero-shot" AND abs:"traffic" AND abs:"forecasting"
```

**post 规则增补（软加权）**

```json
{"id":"fm-llm-boost","field":"title","op":"any",
 "keywords":["foundation model","large language model","llm","prompt","zero-shot","pretrain"],
 "required":false,"exclude":false,"weight":0.6}
```

**预期产出**：6–10 条。代表方向举例（**待检索确认，不得直接写入 README**）：
`[PLACEHOLDER: 待检索确认 —— 时空基础模型 / 城市 LLM 方向 2024–2026 代表作 1]`、`[PLACEHOLDER: 待检索确认 —— 同上代表作 2]`。

### 3.3 方向 B：扩散与概率预测（不确定性）

**为什么补**：仓库现有 `probabilistic traffic forecasting` 只是 `traffic-monthly.json` 的一个 query，DL 小节里对应条目仅 2026 年的 `HySRD-Net` 1 条（`method_tags` 含 `Probabilistic Forecasting`），缺少 CRPS/分位数/扩散模型的对照集合。

**OpenAlex queries**

```
"probabilistic traffic forecasting uncertainty quantification"
"diffusion model spatio-temporal forecasting"
"denoising diffusion traffic prediction"
"quantile regression traffic forecasting"
"conformal prediction spatio-temporal"
"generative model traffic flow distribution"
```

**arXiv `search_query`**

```
(abs:"diffusion model" OR abs:"diffusion probabilistic") AND (abs:"traffic" OR abs:"spatio-temporal") AND abs:"forecasting"
abs:"uncertainty quantification" AND abs:"traffic prediction"
abs:"conformal prediction" AND (abs:"time series" OR abs:"spatio-temporal")
abs:"quantile" AND abs:"traffic forecasting"
```

**post 规则增补**

```json
{"id":"prob-boost","field":"title","op":"any",
 "keywords":["diffusion","probabilistic","uncertainty","quantile","conformal","generative","crps"],
 "required":false,"exclude":false,"weight":0.55}
```

**预期产出**：6–8 条。示例占位：`[PLACEHOLDER: 待检索确认 —— 时空扩散模型基线]`、`[PLACEHOLDER: 待检索确认 —— 概率预测 benchmark 论文]`。

**审核要点**：必须有校准指标（PICP/CRPS/分位数损失），只报 MAE 的"概率"论文一律拒。

### 3.4 方向 C：动态图结构学习

**为什么补**：这是仓库**已有优势**方向（`Spatio-Temporal Graph Structure Learning` 2020、`Dynamic Hypergraph ...` 2023、`Adaptive Graph Convolution Networks` 2023、`AD-STGCN` 2026），但缺 2022–2024 中间链条，且没有 `classic_baseline` 标记。

**OpenAlex queries**

```
"dynamic graph structure learning traffic forecasting"
"adaptive adjacency learning spatio-temporal graph"
"graph structure learning road network"
"hypergraph traffic forecasting"
"causal graph discovery spatio-temporal"
"latent graph learning multivariate time series"
```

**arXiv `search_query`**

```
abs:"graph structure learning" AND (abs:"traffic" OR abs:"spatio-temporal")
abs:"adaptive" AND abs:"graph" AND abs:"traffic forecasting"
(abs:"hypergraph" OR abs:"multi-graph") AND abs:"traffic"
abs:"causal" AND abs:"graph" AND abs:"spatio-temporal forecasting"
```

**post 规则增补**

```json
{"id":"graph-structure-boost","field":"title","op":"any",
 "keywords":["structure learning","adaptive graph","dynamic graph","hypergraph","causal graph","latent graph"],
 "required":false,"exclude":false,"weight":0.5}
```

**预期产出**：8–12 条（补 2022–2024 中间链条）。

### 3.5 方向 D：长时序 Transformer 与 iTransformer / TimeMixer 系

**为什么补**：仓库 `Time Series Forecasting` 只有 2 条（Informer 2020 + 1 条 ICLR 2019），2026 年才出现 `Feature-enhanced iTransformer`、`STVMamba`、`TSAformer` 三条各自独立的长时域工作，**缺少主干方法（iTransformer / PatchTST / TimeMixer / DLinear 等通用时序 backbone）的锚点**，导致读者无法判断这些工作基于什么。

**OpenAlex queries**

```
"inverted transformer multivariate time series forecasting"
"patch time series transformer long-term forecasting"
"time mixer multiscale decomposition forecasting"
"linear model time series forecasting long horizon"
"state space model time series forecasting mamba"
"long-term traffic flow forecasting transformer"
```

**arXiv `search_query`**

```
ti:"iTransformer" OR ti:"PatchTST" OR ti:"TimeMixer" OR ti:"DLinear"
abs:"long-term" AND abs:"forecasting" AND (abs:"transformer" OR abs:"state space")
abs:"mamba" AND (abs:"time series" OR abs:"traffic")
abs:"long-horizon" AND abs:"traffic" AND abs:"forecasting"
```

**post 规则增补**

```json
{"id":"long-horizon-boost","field":"title","op":"any",
 "keywords":["itransformer","patchtst","timemixer","dlinear","mamba","long-term","long-horizon","state space"],
 "required":false,"exclude":false,"weight":0.6}
```

**预期产出**：6–8 条，其中 3–4 条应放进 `### Time Series Forecasting` 小节（`section=time_series`）而不是 DL 小节。

**审核要点**：这类方法与交通专用模型的对比口径不同（长时域 vs 15/30/60 分钟），必须在 `note_en` 里注明评测设置，否则读者会误读 SOTA。

### 3.6 方向 E：Few-shot 与跨城市迁移

**为什么补**：仓库有历史积累（`Learning from Multiple Cities` 2019、`Transfer Learning with GNNs` 2020、`What is the Human Mobility in a New City` 2020、`Contrastive Pre-training` 2025、`PIMCST` 2026），但 2021–2024 空白，且没有统一的 `direction_tags=fewshot_transfer` 归类。

**OpenAlex queries**

```
"few-shot spatio-temporal forecasting"
"cross-city transfer traffic prediction"
"meta learning traffic forecasting"
"domain adaptation spatio-temporal graph"
"zero-shot traffic prediction new city"
"spatial transfer learning road network"
```

**arXiv `search_query`**

```
abs:"few-shot" AND (abs:"traffic" OR abs:"spatio-temporal") AND abs:"forecasting"
(abs:"cross-city" OR abs:"cross-city transfer" OR abs:"new city") AND abs:"traffic"
abs:"meta-learning" AND abs:"traffic prediction"
abs:"domain adaptation" AND abs:"graph" AND abs:"traffic"
```

**post 规则增补**

```json
{"id":"transfer-boost","field":"title","op":"any",
 "keywords":["few-shot","transfer","cross-city","meta-learning","domain adaptation","zero-shot","generalization"],
 "required":false,"exclude":false,"weight":0.55}
```

**预期产出**：6–8 条。

### 3.7 方向 F：隐私与联邦学习

**为什么补**：仓库仅 1 条（`FL + STGNN` 2024，ISPRS IJGI）。`research-lineage.md` 第 6 节点名该方向为"桥接学术与运营约束"的关键，实际条目数与谱系权重严重不匹配。

**OpenAlex queries**

```
"federated learning traffic forecasting"
"privacy-preserving spatio-temporal prediction"
"differential privacy trajectory data"
"decentralized learning urban mobility"
"secure aggregation transportation data"
```

**arXiv `search_query`**

```
abs:"federated learning" AND (abs:"traffic" OR abs:"spatio-temporal" OR abs:"mobility")
abs:"differential privacy" AND (abs:"trajectory" OR abs:"traffic")
abs:"privacy" AND abs:"spatio-temporal" AND abs:"forecasting"
```

**post 规则增补**

```json
{"id":"privacy-boost","field":"title","op":"any",
 "keywords":["federated","privacy","differential privacy","decentralized","secure aggregation"],
 "required":false,"exclude":false,"weight":0.5}
```

**预期产出**：4–6 条（该方向文献绝对量小，宁缺毋滥）。

### 3.8 方向 G：Benchmark 与可复现

**为什么补**：`0x07 Benchmark & Evaluation` 有矩阵表但没有论文条目；`0x03 Toolkits` 里的 `Traffic-Benchmark`、`LibCity`、`OpenSTL` 都没在 DL 小节出现。**这是 `sota_core` 判定所依赖的基础设施**：没有可复现的排行榜，"SOTA" 就是自说自话。

**OpenAlex queries**

```
"traffic forecasting benchmark reproducible evaluation"
"unified benchmark spatio-temporal forecasting"
"library toolkit traffic prediction experiments"
"evaluation protocol spatio-temporal forecasting metrics"
"reproducibility urban spatio-temporal models"
```

**arXiv `search_query`**

```
abs:"benchmark" AND (abs:"traffic forecasting" OR abs:"spatio-temporal forecasting")
abs:"reproducib" AND abs:"traffic" AND abs:"forecasting"
abs:"open-source" AND abs:"traffic prediction" AND abs:"library"
```

**post 规则增补**

```json
{"id":"benchmark-boost","field":"title","op":"any",
 "keywords":["benchmark","reproducib","evaluation","toolkit","library","leaderboard"],
 "required":false,"exclude":false,"weight":0.6}
```

**预期产出**：3–5 条，落 `section=deep_learning` 但 `layers=["sota_core","benchmark_reproducibility"]`，并在 `0x07` 的 Benchmark Matrix 里交叉引用（矩阵表由 `dataset` 实体渲染，见 `01-data-model.md` §3.2）。

### 3.9 方向 H：数据质量与补全（已被现有条目踩到但无归类）

仓库 2018 年就有 `Missing Value Imputation ... Multi-View Learning`，2020 年有 `Traffic Flow Imputation Using Parallel Data and GANs`，但没有 `imputation_data_quality` 的方向聚合，也没有把 METR-LA 的高缺失率（公开常识，需在 `dataset.missing_rate` 中登记确认）与模型选择关联起来。检索式：

```
OpenAlex: "traffic data imputation spatio-temporal"
          "missing data traffic sensor network recovery"
arXiv:    abs:"imputation" AND (abs:"traffic" OR abs:"spatio-temporal")
```

### 3.10 方向总览与优先级

| 优先级 | 方向 | `direction_tags` | OpenAlex queries | 预期新增 | 依赖 |
| --- | --- | --- | ---: | ---: | --- |
| P0 | 断崖修复 2022–2023 顶会 | （跨方向） | — | 8–14 | §3.4 §3.5 |
| P0 | 时空基础模型 / LLM4ST | `st_foundation_llm` | 6 | 6–10 | — |
| P1 | 动态图结构学习 | `dynamic_graph_learning` | 6 | 8–12 | — |
| P1 | 长时序 Transformer 系 | `long_horizon_transformer` | 6 | 6–8 | — |
| P1 | 扩散与概率预测 | `diffusion_probabilistic` | 6 | 6–8 | — |
| P2 | Few-shot / 跨城市迁移 | `fewshot_transfer` | 6 | 6–8 | — |
| P2 | Benchmark 与复现 | `benchmark_reproducibility` | 5 | 3–5 | dataset 实体先落地 |
| P3 | 隐私联邦 | `privacy_federated` | 5 | 4–6 | — |
| P3 | 数据补全 | `imputation_data_quality` | 2 | 2–4 | — |

合计预期新增 49–75 条，与 §2.1 的「+21 ~ +52」目标相容（上限略高，因为存在被淘汰的候选）。

---

## 4. 落成配置：`scripts/configs/traffic-scope-2023.json`（规划，Phase 1 新增）

**不改 `traffic-monthly.json`**（它服务于月更流水线，语义是"最近 3 年滚动"）。新增一份存量补库配置，`year_range.mode=fixed`：

```json
{
  "version": 2,
  "name": "traffic-scope-2023",
  "search": {
    "queries": [
      "urban spatio-temporal prediction foundation model",
      "large language model spatio-temporal forecasting",
      "prompt learning urban spatio-temporal",
      "probabilistic traffic forecasting uncertainty quantification",
      "diffusion model spatio-temporal forecasting",
      "dynamic graph structure learning traffic forecasting",
      "adaptive adjacency learning spatio-temporal graph",
      "hypergraph traffic forecasting",
      "inverted transformer multivariate time series forecasting",
      "patch time series transformer long-term forecasting",
      "few-shot spatio-temporal forecasting",
      "cross-city transfer traffic prediction",
      "federated learning traffic forecasting",
      "traffic forecasting benchmark reproducible evaluation"
    ],
    "sources": ["openalex", "arxiv"],
    "year_range": {"mode": "fixed", "from_year": 2022, "to_year": 2026},
    "limit_per_query": 30,
    "output_limit": 120
  },
  "post": {
    "rules": [
      {"id": "domain", "field": "title", "op": "any",
       "keywords": ["traffic", "transport", "road", "mobility", "flow", "spatio-temporal", "spatial-temporal"],
       "required": true, "exclude": false, "weight": 1.4},
      {"id": "intent", "field": "title", "op": "any",
       "keywords": ["forecast", "prediction", "predicting", "imputation", "benchmark"],
       "required": true, "exclude": false, "weight": 1.3},
      {"id": "fm-llm-boost", "field": "title", "op": "any",
       "keywords": ["foundation model", "large language model", "llm", "prompt", "zero-shot", "pretrain"],
       "required": false, "exclude": false, "weight": 0.6},
      {"id": "prob-boost", "field": "title", "op": "any",
       "keywords": ["diffusion", "probabilistic", "uncertainty", "quantile", "conformal", "generative"],
       "required": false, "exclude": false, "weight": 0.55},
      {"id": "graph-structure-boost", "field": "title", "op": "any",
       "keywords": ["structure learning", "adaptive graph", "dynamic graph", "hypergraph", "causal graph", "latent graph"],
       "required": false, "exclude": false, "weight": 0.5},
      {"id": "long-horizon-boost", "field": "title", "op": "any",
       "keywords": ["itransformer", "patchtst", "timemixer", "dlinear", "mamba", "long-term", "long-horizon", "state space"],
       "required": false, "exclude": false, "weight": 0.6},
      {"id": "transfer-boost", "field": "title", "op": "any",
       "keywords": ["few-shot", "transfer", "cross-city", "meta-learning", "domain adaptation", "generalization"],
       "required": false, "exclude": false, "weight": 0.55},
      {"id": "privacy-boost", "field": "title", "op": "any",
       "keywords": ["federated", "privacy", "differential privacy", "decentralized"],
       "required": false, "exclude": false, "weight": 0.5},
      {"id": "non-road-exclude", "field": "title", "op": "any",
       "keywords": ["airspace", "air traffic", "aviation", "flight", "uav", "drone", "maritime", "shipping", "vessel", "power load", "ev load", "microgrid"],
       "required": false, "exclude": true, "weight": 0.0}
    ],
    "min_score": 2.0
  },
  "rank": {
    "recency_weight": 0.3,
    "source_weights": {"openalex": 0.2, "arxiv": 0.1},
    "tie_breakers": ["year_desc", "title_asc"]
  },
  "runtime": {
    "max_requests": 120,
    "min_interval_seconds": 0.8,
    "max_retries": 2,
    "retry_base_seconds": 2.0,
    "cache_dir": ".cache/paper-search-http",
    "cache_ttl_hours": 48,
    "source_min_interval_seconds": {"openalex": 0.8, "arxiv": 0.8}
  }
}
```

**必须补的人工检索通道**（OpenAlex 对顶会归一化差，脚本覆盖不到）：

| 会议 | 检索方式 | 目标 |
| --- | --- | --- |
| KDD | `https://dblp.org/search/publ/api?q=traffic%20forecasting%20venue%3AKDD%3A&format=json&h=100` | 2022–2026 |
| NeurIPS | `venue:NeurIPS:` 同上 | 2022–2026 |
| ICLR | `venue:ICLR:` 同上（OpenReview 另跑 `https://api2.openreview.net/notes/search?query=traffic+forecasting`） | 2022–2026 |
| ICML | `venue:ICML:` 同上 | 2022–2026 |
| WWW / AAAI / IJCAI | `venue:WWW:` / `venue:AAAI:` / `venue:IJCAI:` | 2022–2026 |

这些结果**不进自动流水线**，由人在 `data/papers/<year>.json` 里直接新增记录（`provenance.source = "manual"`），因为顶会论文的 venue 字段需要人工确认（DBLP 的 venue 写法、年份、页码）。检索串本身仍然可作为交付文本直接使用。

---

## 5. 维护节奏

### 5.1 例行节奏

| 频率 | 动作 | 产出 | 负责 |
| --- | --- | --- | --- |
| 每月 1 日 | `monthly-paper-search.yml` 自动跑 `traffic-monthly.json`，产出 `updates/monthly-paper-candidates.md`（当前实测每次仅 4 条候选，见该文件 11 行） | 候选清单 | CI |
| 每月上旬 | 人工过筛候选：按 §5.2 清单逐条判定 | `data/papers/<year>.json` 新增/拒绝记录 | 维护者 |
| 每季度 | 跑一次 `traffic-scope-2023.json`（§4）做存量补库；同步跑 DBLP 顶会通道 | 季度增量 PR | 维护者 |
| 每半年 | 复核 `layers` 分层：上一季度的 `sota_core` 若已被超越则降为 `classic_baseline` | 分层变更 PR | 维护者 |
| 每年 1 月 | 关闭上一年年份段；生成新 `data/papers/<new>.json`；更新 `02-paper-scope.md` 的现状盘点表 | 年报 + 本文档修订 | 维护者 |
| 每次 PR | `jsonschema` 校验 + 去重键校验 + `render --check` | CI 绿灯 | CI |

### 5.2 人工审核清单（每条候选都过一遍）

**A. 相关性（任一不过即拒）**

1. 预测对象是**道路交通**（流量/速度/密度/OD/行程时间）或其时空建模方法。航空、水运、电力负荷、UAV 一律拒（对应 `non-road-exclude`）。
2. 任务确为"预测/补全/评估"，不是纯检测/纯控制/纯分类——除非归入 `application` 且说明了与预测的关系。
3. 时间范围符合目标年份段（候选元数据年份常与正式发表年差 1，**以归档 venue 年为准**，见 `01-data-model.md` §4.1）。

**B. 质量（`sota_core` 需全过；`classic_baseline` 只看 2、3）**

1. 至少 1 个公开 benchmark + 多步（15/30/60 min）指标表。
2. 有 ≥2 个强基线对比（DCRNN/STGCN/Graph WaveNet 之类），不是只比 ARIMA/LSTM。
3. 有代码链接，或方法描述足以复现（超参、划分、种子）。
4. **不采信**"我们达到 SOTA"的自述；以是否被后续论文对比为准。

**C. 元数据（逐字段核对）**

| 字段 | 核对方式 |
| --- | --- |
| `title` | 与出版方页面逐字一致（注意 `–` vs `-`、大小写、副标题） |
| `year` | 取归档年，非 arXiv 首次提交年（两者不同时在 `note_en` 注明） |
| `venue` / `venue_id` | 在 `data/venues.json` 中存在；新短名先加 venue 再引用 |
| `doi` / `arxiv_id` | 二者至少一个；DOI 优先，arXiv 版本记录 `vN` |
| `paper_url` | 必须可打开（`verification.link_ok=true`），优先 DOI 解析地址 |
| `code_url` / `code_links` | 只收官方仓库或明确标注 `code-not-official` 的复现 |
| `method_tags` | 复用 `traffic-monthly.json` 的受控词表；新标签需在同 PR 更新词表 |
| `direction_tags` | 只能在 schema 枚举内 |
| `layers` | 按 §2.2 准则 |
| `note_en`/`note_zh` | 一句话说清"这篇相对已有条目新增了什么"，禁止写"提出了一个新方法"这类空话 |

**D. 去重（三条命中去重即拒）**

1. `doi` 已存在 → 拒（若为同一论文的正式版替换 arXiv 版，走"升级链接"而非新增）。
2. `arxiv_id`（去版本号）已存在 → 拒。
3. `title_key`（NFKC + `casefold` + 删 `\W+`）已存在且年份差 ≤1 → 拒；年份差 >1 或标题近似 → 进 `alias_suggestion` 人工裁决。

**E. 淘汰（从 README 移除或降级）**

满足任一即进入淘汰评审，**不直接删**，先改 `status`：

| 信号 | 处置 |
| --- | --- |
| `link_ok=false` 连续 2 次巡检 | `status=dead_link`；补镜像链接，仍无则移入 `data/papers/archive/` |
| 撤稿/勘误 | `status=retracted`，README 中加标注 |
| 同模型被更完整版本取代（如 arXiv 早期版 vs 正式扩展版） | 保留正式版，旧记录 `status=superseded` |
| `sota_core` 连续 2 年未被任何论文对比 | 降为 `classic_baseline` |
| 主题偏离（如被确认为非道路交通） | 移出 DL 小节，`status` 保持 `active` 但移入 `application` 或删除 |

**淘汰的节奏**：每季度一次批量评审，**单次淘汰不超过当年条目数的 15%**，且必须在 PR 描述里逐条给出理由——这是防止"算法判重误删"的最后一道闸。

### 5.3 灰度期与本文档的联动

- 本文档 §1 的统计表是**基线快照**（2026-11-24）。切换数据源后（`01-data-model.md` §7.2 的 G2 阶段），这张表应改为由 `data/papers/*.json` 自动生成的 `docs/evolution/stats.md`，本文档只保留"目标值"列。
- 每次年份段变更（新年份开启、旧年份冻结）都要在本文档追加一行修订记录，格式：

```text
| 修订日 | 变更 | 影响范围 | 数据基线 |
| 2026-11-24 | 初稿：现状盘点 + 目标 + 8 个方向检索式 | README 94 条 DL 条目 | 2026-11-24 快照 |
```

---

## 6. 与其它文档的边界

| 文档 | 职责 | 与本文关系 |
| --- | --- | --- |
| `01-data-model.md` | 实体 schema、id/去重/双语规则、迁移与回滚 | 本文的分层字段（`layers`）与方向枚举（`direction_tags`）由它定义 |
| `02-paper-scope.md`（本文） | 数量目标、分层配额、补什么方向、怎么审 | — |
| `research-lineage.md` | 研究谱系叙述 | 本文 §3 的方向需能挂到谱系阶段上（`lineage.schema.json` 的 `direction_tags` 复用同一枚举） |
| `CONTRIBUTION.md` | 贡献流程 | 本文 §5.2 的审核清单应最终并入其 "Content Conventions" |

> 本文不改动 `README.md` / `README.zh.md` / `scripts/` / `.github/` / `updates/` / `site/`。所有规划项均落在 `data/` 与新增脚本，符合仓库"documentation-first + 轻量维护脚本"的定位（见 `AGENTS.md`）。