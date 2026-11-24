# Awesome-Traffic-Prediction 现状评审

评审日期：2026-11-24　评审对象：README.md / README.zh.md + data/papers（195 条）+ site/

> 评审立场：**信息量不等于价值**。195 条里如果有低质量条目混排、代表性论文隐身，
> 这个库的可信度反而低于一个 80 条但条条站得住的库。本报告按「可信度 > 覆盖面」排序。

---

## 结论摘要

当前数据层是**结构正确的，但价值排序是错的**。

三个具体症状：

1. **代表性论文完全隐身**。Graph WaveNet、ST-MetaNet、AGCRN、DCRNN、STGCN 这些被引用数千次、
   定义了整个领域技术路线的论文，在库里和一篇 PLoS ONE 的新文章拥有完全相同的呈现权重。
2. **`featured` 标记选错了对象**。12 条 featured 全部来自 curated 子集，
   **没有一条来自顶会**——其中 3 条是 MDPI/PLoS ONE 这类快速出版期刊。
   与此同时，UniST 已经升级为 KDD 2024，COOL 升级为 Information Fusion，标记却没跟上。
3. **venue 与标签两套词表都是碎的**。同一个 KDD 有 `KDD` 和
   `ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)` 两种写法；
   `GCN` 和 `Graph Convolution`、`RNN / LSTM` 和 `Recurrent` 是同一个概念的两个标签。

---

## 一、数据质量核查

### 1.1 覆盖情况（195 条）

| 维度 | 数值 |
| --- | --- |
| 顶会/顶刊论文（按 venue 判定） | **67 条** |
| 会议 / 期刊 / 预印本 / 研讨会 | 62 / 58 / 66 / 2 |
| 有 DOI | 135 |
| 有代码链接 | 26 |
| 有策展备注（可读的一句话） | 19 |
| 有空方法标签 | **52** |

### 1.2 代表性论文核查

按领域公认的技术路线检查（DCRNN → STGCN → Graph WaveNet → AGCRN → STSGCN → GMAN → STAEformer → UniST）：

| 论文 | 在库 | 有代码 | 有标签 | featured |
| --- | --- | --- | --- | --- |
| DCRNN (ICLR'18) | ✓ | ✓ | 1 | ✗ |
| STGCN (IJCAI'18) | ✓ | ✓ | 2 | ✗ |
| ASTGCN (AAAI'19) | ✓ | ✓ | 3 | ✗ |
| **Graph WaveNet (IJCAI'19)** | ✓ | ✓ | **0** | ✗ |
| **ST-MetaNet (KDD'19)** | ✓ | ✓ | **0** | ✗ |
| STSGCN (AAAI'20) | ✓ | ✓ | 1 | ✗ |
| GMAN (AAAI'20) | ✓ | ✓ | 1 | ✗ |
| AGCRN (NeurIPS'20) | ✓ | ✓ | 2 | ✗ |
| STFGNN (2020) | ✓ | ✓ | 1 | ✗ |
| STAEformer (CIKM'23) | ✓ | ✗ | 2 | ✗ |
| UniST (KDD'24) | ✓ | ✗ | 2 | ✓（但记录里仍是 Arxiv） |

**Graph WaveNet 的方法标签是空的**——这是 Wikipedia 级别的基础工作，标签为空意味着
按「Graph Convolution」筛选时会漏掉它。

### 1.3 Featured 标记的错位

| 年份 | 原 venue | 升级后 venue | 期刊等级 |
| --- | --- | --- | --- |
| 2024 | Arxiv | **KDD** | 顶会 |
| 2024 | Arxiv | **Information Fusion** | 顶刊 |
| 2024 | Complex & Intelligent Systems | 同 | 中 |
| 2024 | ISPRS IJGI | 同 | 中 |
| 2025 | Arxiv | 同 | 预印本 |
| 2025 | Arxiv | 同 | 预印本 |
| 2025 | Sensors | 同 | MDPI |
| 2026 | Neural Networks | 同 | 中 |
| 2026 | Electronics | 同 | MDPI |
| 2026 | Arxiv | 同 | 预印本 |
| 2026 | Expert Systems with Applications | 同 | 中 |
| 2026 | PLoS ONE | 同 | 中低 |

12 条 featured 中，2 条已升级为顶会/顶刊（但记录没更新），
其余多数是快速出版期刊或预印本——而 67 条真正的顶会论文一条都没被标。

---

## 二、README 评审

| 问题 | 证据 | 影响 |
| --- | --- | --- |
| 2023–2026 内容极少 | README 里 2023:2 / 2024:4 / 2025:4 / 2026:11，而 data 里对应 18/12/12/31 | README 与数据层已经分叉，读者看到的仍是旧内容 |
| 经典论文与普通条目同级 | 0x00 按年份平铺，无任何层级 | 读者无法判断哪篇该先读 |
| 2018 段落年份错位 | T-GCN 标 `[TITS 2019]` 但在 `#### 2018` 下 | 年份筛选与引用会错 |
| 跨行折断条目 | L170–171 一条被拆成两行 | 正则解析、后续自动化为它写过特例 |
| venue 短名不统一 | `WWWC`/`WC`/`WWW`、`Sensor`/`Sensors` | 按 venue 聚合会分裂 |
| 空章节 | `### Trajectory` 无内容 | 观感差 |

---

## 三、站点呈现评审

**已有的**：多维筛选（年/venue/任务/方法）、URL 可分享、Pagefind 搜索 + 2 个分面、
论文详情页（作者/DOI/arXiv/标签/相关论文）、趋势图、研究脉络时间轴。

**没有的（按重要性排序）**：

1. **没有「代表性论文」这一层**。首页只显示最新 8 篇，全是新文章。
   新人进入这个站，得不到"先读这 10 篇"的指引。
2. **没有质量信号**。没有引用量、没有 venue 等级徽标（顶会/顶刊/预印本视觉上无差别）。
3. **没有经典/近年之分**。`layers` 字段已经在数据里（sota_core 59 / classic_baseline 34 /
   survey 6 / application 5），但站点完全没有用到。
4. **趋势图在放大噪声**。词频云以标题词频为指标，而标题词频与"领域重要性"无关——
   它反映的是命名习惯。用它替代 README 的静态图是进步，但它本身不是洞察。
5. **论文详情页缺少"这篇为什么重要"**。只有 19/195 有策展备注。

---

## 四、优化方案（按性价比排序）

### P0 · 让代表性论文站到前排（1–2 天）

1. **引入 `impact` 分层**（数据层）：新增字段
   `impact: {tier: landmark|notable|standard, citations?, reason?}`。
   landmark = 定义技术路线或高被引基线（DCRNN/STGCN/Graph WaveNet/AGCRN/STSGCN/GMAN/STAEformer/UniST…）。
2. **修正 featured**：以「venue 等级 + 是否基线」而不是「近期」为准。
3. **站点新增 `/landmarks/` 页面**：按技术路线分组（图卷积基础 → 自适应图 → Transformer →
   基础模型），每篇一句话说明它解决了什么。这是新人唯一的入口路径。
4. **卡片加 venue 等级徽标**：顶会 / 顶刊 / 预印本，视觉可分辨。

### P1 · 修词表一致性（0.5–1 天）

1. **venue 规范化**：`data/venues.json` 存 `{id, canonical_name, short_name, tier, type}`，
   `venue_id` 成为唯一键；`venue` 降级为显示名。解决 KDD 两种写法、`WWWC` 之类。
2. **方法标签受控词表**：合并同义（`GCN` → `Graph Convolution`，`Recurrent` → `RNN / LSTM`，
   `Dynamic Graph` → `Adaptive Graph`，`STGCN` 不算方法标签），并写进 schema 的 enum。
3. 补 Graph WaveNet / ST-MetaNet 等空标签。

### P2 · 呈现优化（1–2 天）

1. 论文卡分组显示：`landmark` 置顶并加「奠基」徽标；同分组的按年份排。
2. 详情页增加「为什么重要」区块（landmark 必填，standard 可空）。
3. 趋势页：词频图降级为辅助，主视图改为「按 venue 等级的逐年产出」——
   这才是"领域怎么走的"。
4. README 与站点同源：用 `render_curated_recent.py` 的方式把 data 渲染回 README，
   消除 README 停留在 2023–2026 共 21 条而数据层已有 73 条的分叉。

### P3 · 不建议做

- **不要**继续扩量到 300+。覆盖率已经不是瓶颈，质量分层才是。
- **不要**引入引用量实时抓取。OpenAlex 的 `cited_by_count` 对 2025–2026 论文严重失真，
  且会让每次构建依赖外网。用人工判定的 `tier` 更稳。
- **不要**做用户系统、评论、点赞——这是个资料库，不是社区。

---

## 五、验收标准

优化完成后应能回答这三个问题：

1. 一个刚进领域的研究生，能否在站上找到「按顺序读这 8 篇」的路径？
2. 一个已经在做的人，能否按「顶会 + 某方法标签」筛出真正值得参考的论文，
   且不会因为 venue 写法不同而漏掉？
3. 随便点开一篇 landmark 论文，能否知道它为什么重要、有没有官方代码、在哪个数据集上评测？
---

## 六、P0 落地结果（2026-11-24）

### 6.1 新增 `impact` 分层

schema 新增 `impact: {tier, role, role_zh, track}`，与 `featured` 彻底分开：

- `impact` 回答「这篇重要吗」，由人判断，写一句话说明它建立/解决了什么。
- `featured` 降级为「进过策展精读榜」的遗留字段，不再当作质量信号使用。

分级结果：**landmark 11 / notable 4 / standard 180**。

11 篇 landmark，按研究轨道分组（这也是 `/landmarks/` 页面的结构）：

| 轨道 | 论文 |
| --- | --- |
| early_deep | ST-ResNet (AAAI'17) |
| graph_basics | DCRNN (ICLR'18)、STGCN (IJCAI'18)、Graph WaveNet (IJCAI'19)、ASTGCN (AAAI'19) |
| adaptive_graph | ST-MetaNet (KDD'19)、AGCRN (NeurIPS'20) |
| transformer | GMAN (AAAI'20)、STAEformer (CIKM'23) |
| foundation_model | STEP (KDD'22)、UniST (KDD'24) |

每篇都有中英双语的一句话说明，例如：

> DCRNN：首次用有向图上的扩散过程建模交通，并确立了 METR-LA / PEMS-BAY 评测协议。
> STAEformer：证明配好时空自适应嵌入的朴素 Transformer 就能超过复杂 STGNN。

### 6.2 新增 `data/venues.json`

venue 从自由文本变成受控表：`{id, short_name, tier, type, aliases}`，38 → 40 个条目。

修复的碎片化实例：

| 之前 | 之后 |
| --- | --- |
| `KDD` / `ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)` | `kdd` |
| `AAAI` / `Proceedings of the AAAI Conference on Artificial Intelligence` | `aaai` |
| `WWWC` / `WC` / `WWW` | `www` |
| `Sensor` / `Sensors` | `sensors` |
| `TRPET` / `TR-C` | `trc` |
| `CS224W` 与 `IWPHM` 被错误合并成同一个 `ijcai_ws` | 拆成 `cs224w` / `iwphm` |

venue 等级分布（195 条）：**top 81 / mid 31 / preprint 73 / low 10**。

### 6.3 方法标签受控词表

44 个碎片值收敛为 **25 个受控值**：

- `GCN` → `Graph Convolution`，`Recurrent` → `RNN / LSTM`，`Dynamic Graph` → `Adaptive Graph`
- `Graph Modeling` → `STGNN`，`Spatio-Temporal CNN` → `Temporal Convolution`
- `Deep Clustering` → `Clustering`
- 删除非方法类标签：`STGCN`（模型名）、`Long-Horizon`（属于 direction 语义）、
  `Real-time` / `Short-Term` / `Efficiency` 等描述词

补齐 3 篇空标签的经典论文（Graph WaveNet、ST-MetaNet、ST-ResNet）——
Graph WaveNet 原本一个标签都没有，按「Graph Convolution」筛选会直接漏掉它。

### 6.4 站点呈现

| 改动 | 效果 |
| --- | --- |
| 新增 `/landmarks/` 与 `/zh/landmarks/` | 11 篇奠基论文按 5 条研究轨道排列，每篇一句「为什么重要」 |
| 首页首屏 | 从「最近 8 篇」改为「6 篇奠基 + 4 篇最新」 |
| 论文卡片徽标 | 顶会/顶刊 81 个、预印本 83 个、奠基 11 个，视觉可分辨 |
| 默认排序 | 从「最新优先」改为「重要优先」（landmark → notable → standard，组内按年份） |
| 导航 | 新增 Landmarks 入口，排在 Papers 之前 |

### 6.5 迁移链路的连带修复

`impact`、规范化后的 venue、受控标签都是「人/工具产出的真相」，
迁移脚本必须认得它们，否则下一次重跑会把工作全部覆盖。补了三处：

1. `build_record` 通过 `data/venues.json` 解析 venue_id/short_name/type，
   与 enrichment 用同一张表，避免两套规则打架。
2. `impact` 加入「只增不删」的保留字段。
3. `method_tags` 的规则输出也走 `TAG_CANON` 映射，并内置三条 `MANUAL_TAGS`。

修完后 `migrate_readme_to_json.py --all --check` 重新全绿（每年 SAME）。

### 6.6 验收自查

- 新生能否找到「按顺序读这 8 篇」：**能**，`/landmarks/` 就是这条路径。
- 能否按「顶会 + 方法标签」筛出可信论文：**能**，venue 已归一，徽标可见。
- 点开一篇 landmark 能否知道为什么重要：**能**，`role` / `role_zh` 是必填项。

---

## 七、尚未处理

1. ~~49 条记录方法标签仍为空~~ —— **已清零**，见第十节。
2. **`featured` 与新 `impact` 并存**。README 的 curated digest 表仍由 `featured` 驱动，
   两者语义重叠但不一致；建议下一步让 digest 表按 `impact.tier` 选取。
3. ~~README 与数据层未同源~~ —— **已完成**，见第八节。

---

## 八、README 与数据层同源（已完成）

### 8.1 问题

README 里 2023–2026 只有 21 条手写条目，`data/papers` 同期有 73 条。
读者在仓库页看到的和站点上看到的不是同一批内容，且手写区块还在逐年阻塞新数据。

### 8.2 改造

`render_curated_recent.py` 从「读 curated JSON」改为「读 `data/papers` + `data/venues.json`」，
成为真正意义上的渲染器：

| 块 | 规则 |
| --- | --- |
| `AUTO RECENT PAPERS` | 最近 5 年，每年最多 12 条 |
| `AUTO CURATED DIGEST TABLE` | landmark 按年份升序（讲技术演进），其后是 notable |

**排序键：significance → venue tier → 年份 → 标题。**

中间那一层是关键。只用「重要性 + 年份」时，2026 的 12 个名额被 arXiv 预印本占满
（2026 有 17 条 preprint、3 条顶会）。加上 venue tier 后，2026 段落开头变成
AAAI / IJCAI / ICDE。

同时把 README 里手写的 `#### 2022` / `#### 2023` / `#### 2024` 三段并入自动块，
消除「手写区 vs 自动区」两套内容的割裂。

### 8.3 发现并修掉的数据错误

渲染后逐条核对 2022 段落，发现一条标注为 **WWW 2022**、DOI 却是
`10.1109/iwcmc55113.2022.9825059` 的论文。Crossref 确认：

```
container: 2022 International Wireless Communications and Mobile Computing (IWCMC)
```

成因是 venue 推断做过无边界的子串匹配：`"iwcmc"` 里包含 `"www"`。

新增 `scripts/check_venue_consistency.py` 作为门禁，规则是：如果 DOI slug 里出现
已知 venue 标识（如 `3219819` = KDD、`iwcmc` = IWCMC），而记录标注的是另一个 venue，
就报错。已接入 `data-check.yml`。

校验器做过反向验证：把该记录改回 `WWW`，`--strict` 确实退出码 1 并指出具体冲突。

### 8.4 结果

README 与站点现在渲染同一份数据：

| 指标 | 值 |
| --- | --- |
| data 记录 | 195 |
| README 渲染年份 | 2022–2026（5 年） |
| README 每年条目 | 12 |
| digest 行 | 15（11 landmark + 4 notable） |
| 站点页面 | 408 |

`check_readme_sync`、`render_curated_recent --check`、`migrate --check`、
`check_venue_consistency --strict`、schema 校验、站点文案审计全部通过。

### 8.5 仍未处理

- ~~`featured` 字段与 curated JSON~~ —— **已完成退休**，见第九节。
- ~~2015–2021 仍是手写区块~~ —— **已全部纳入自动渲染**，见第九节。

---

## 九、收尾：退休 curated JSON，README 全量同源

### 9.1 迁移人工文案

`updates/curated/recent-papers.json` 的 19 条记录里唯一不可替代的是人工写的
**中文备注**（如「偏通用模型与提示式框架」「两阶段长预测框架」）与 `task_zh`。
这些已迁入 `data/papers`：

- 只填空字段，不覆盖已有内容（`note_zh` 19 条、`task_zh` 19 条、`short_name` 补到 43 条）。
- 与 `impact.role_zh` 做过逐条比对，**无一条重复**——role 讲「为什么重要」，
  note 讲「这篇做了什么」，两者互补。

文件本身移到 `docs/evolution/legacy-curated-snapshot.json` 并加了说明头，仅作审计留档。

### 9.2 迁移脚本与 README 的关系重新定义

移走手写区块后暴露一个设计问题：`migrate_readme_to_json.py` 认为「README 里的条目
= 数据的全部」，于是把 2018/2019/2020/2021 的 `order` 全部重排为 1、2……，
`--check` 每次都报 DIFF。

现在脚本会先比较「README 提供的条数」与「磁盘上该年份的记录数」，
当 README 只贡献少数几条时判定该年份**已由数据层拥有**，直接跳过并打印：

```
[INFO] 5 year(s) now have no manually written entries; their records are
       owned by data/papers and rendered back into the README, so they are
       skipped here.
```

`--curated` 参数默认值改为空串，缺文件不再是错误。

### 9.3 README 全量同源

`render_curated_recent.py --recent-window-years 0 --per-year 0` 渲染全部 12 个年份：

| 年份 | 条目 | 年份 | 条目 |
| --- | --- | --- | --- |
| 2015 | 2 | 2021 | 10 |
| 2016 | 1 | 2022 | 43 |
| 2017 | 3 | 2023 | 18 |
| 2018 | 10 | 2024 | 12 |
| 2019 | 22 | 2025 | 12 |
| 2020 | 31 | 2026 | 31 |

合计 **195 条**，与 `data/papers` 一一对应。`--per-year 0` 表示不限条数。

现在 README 的深度学习小节不再是手写内容，而是纯渲染产物；
`CONTRIBUTION.md` 的「Source of truth」也已改为 `data/papers/*.json`。

### 9.4 修改 README 的唯一路径

```
编辑 data/papers/YYYY.json
  → python3 scripts/render_curated_recent.py --recent-window-years 0 --per-year 0
  → python3 scripts/check_readme_sync.py
```

CI 的 `curated-render-check.yml` 使用同一组参数，手改 README 自动块会直接失败。

### 9.5 最终状态

| 检查 | 结果 |
| --- | --- |
| readme-sync | OK（12 年小节 / 15 digest 行） |
| render --check | OK（195 条 / 2015–2026 / 15 行） |
| migrate --check | OK（数据层拥有的年份自动跳过） |
| venue consistency | OK |
| schema | 195 条，0 错误 |
| 站点 | 408 页，0 处开发向文本 |

仓库只有一个编辑入口：改 JSON，渲染 README，站点构建时读取同一份数据。

---

## 十、标签补齐与降噪

### 10.1 先剔除跑偏的记录

49 条空标签里混着 4 条**不属于本领域**的论文，是早期检索管线放进来的：

| 记录 | 为什么不属于 |
| --- | --- |
| An Empirical Study on Internet Traffic Prediction | 互联网/WWW 流量，不是道路交通 |
| Deep Sequence Modeling for Anomalous ISP Traffic | ISP 网络流量 |
| Energy-Efficient Wake-Up Signalling for Machine-Type Devices | M2M 唤醒信令 |
| Federated Learning for 5G Base Station Traffic | 5G 基站负载 |

已删除，`order` 重新编号。**195 → 191 条。** 这一条比补标签更重要：
一个交通预测的库里有 ISP 流量论文，比少几十条标签更伤可信度。

### 10.2 扩规则 + 人工判定

剩下 45 条分两步：

- **规则扩展**：`METHOD_RULES` 从 17 条扩到 38 条，新增 Graph Pooling、Self-Supervised、
  Multi-View、Latent Graph Learning、U-Net、MLP、Autoencoder、Statistical Model、
  Test-Time Adaptation、Adversarial Robustness 等族。补齐 21 条。
- **人工判定**：剩余 24 条标题里确实不含方法词（如 "DNN-Based Prediction Model"、
  "Road2Vec"、"2F-TP"），逐条读题判定，写进 `MANUAL_TAGS`（现在共 27 条条目）。

关键设计：**`MANUAL_TAGS` 里的记录，规则不再叠加。**

原因是复查时发现迁移会多打标签——`"Probabilistic Data Fusion ..."` 因为标题里有 "fusion"
被打上 `Multi-Modal Fusion`，两篇综述因为标题里有 "Survey" 被同时打上 `Survey` 和 `STGNN`。
人工标签是**结论**不是提示，规则不该在其上再叠。改完后迁移对全库 method_tags 的复现差异为 **0**。

### 10.3 标签降噪

补齐后出现 9 个只出现 1 次的标签，筛选时点击它们只能得到单行结果，没有意义。
按「上位类别」合并：

| 细粒度 | 合并到 |
| --- | --- |
| Residual Network / U-Net | Temporal Convolution |
| Spatial-Temporal Similarity | Clustering |
| Multi-Task | Matrix Factorization |
| Multi-Path | Multi-View |
| Test-Time Adaptation | Transfer Learning |
| Latent Graph Learning | Adaptive Graph |

**词表 40 → 33 个**，保留两个单例（`Statistical Model`、`Physics-Informed`），
因为它们是独立范式而非某个类别的细分。

最终分面分布：Transformer 50 / STGNN 41 / Graph Convolution 36 / Adaptive Graph 30 /
RNN·LSTM 13 / Temporal Convolution 12 / Graph Attention 12 …

### 10.4 中文标签

33 个方法标签补了中文映射（`Graph Convolution` → 图卷积、`STGNN` → 时空图网络 …），
`data-value` 仍是英文，筛选逻辑与 URL 不受影响。

### 10.5 最终状态

| 指标 | 值 |
| --- | --- |
| 记录 | 191（删除 4 条跑偏） |
| 空方法标签 | **0** |
| 方法标签词表 | 33 |
| 站点页面 | 400 |

所有门禁通过：readme-sync、render --check、migrate --check（含 method_tags 复现差异 0）、
venue consistency、schema（191 条 0 错误）、站点文案审计（400 页 0 处开发向文本）。
