# Domain Config v2 (for `paper_search.py`)

This directory contains extensible search configs used by `scripts/paper_search.py --config ...`.

## Files

- `domain-config-v2.schema.json`: schema reference
- `traffic-monthly.json`: monthly config used by GitHub Action

## Core Logic

`paper_search.py` applies 3 steps:

1. Collect candidates from `search.queries` and `search.sources`.
2. Apply `post.rules[]`:
   - `required=true`: paper must match this rule.
   - `exclude=true`: if matched, paper is removed.
   - `weight`: score bonus for matched positive rules.
3. Rank by score and keep top `search.output_limit`.

Score formula:

```text
score = sum(matched rule weights)
      + rank.recency_weight * normalized_recency
      + rank.source_weights[source]
```

Threshold:

- `post.min_score` is a hard threshold on final score.
- `rank` does not filter papers by itself; it only orders/scales results.

## Why this is more extensible than a flat `filters` field

- Each rule has its own `id`, `field`, `op`, `required`, `exclude`, and `weight`.
- New domain behavior can be added by appending rules instead of changing script code.
- One config can express:
  - hard include constraints,
  - hard exclude constraints,
  - soft relevance boosts.

## Run

```bash
python3 scripts/validate_domain_config.py scripts/configs/traffic-monthly.json
python3 scripts/paper_search.py --config scripts/configs/traffic-monthly.json --out /tmp/papers.md
```

Supported `search.sources`:

- `openalex`
- `arxiv`
- `semantic_scholar`
- `crossref`

Polite runtime controls (`runtime` object):

- `max_requests`: hard budget for one run
- `min_interval_seconds`: global pacing floor
- `source_min_interval_seconds`: source-specific pacing overrides
- `max_retries` + `retry_base_seconds`: backoff policy for rate-limit/transient failures
- `cache_dir` + `cache_ttl_hours`: local response cache to reduce repeated requests

## 中文说明

- `post.min_score`：最终得分阈值，低于该值直接丢弃。
- `rank`：排序与打分增益配置，不是“缺少字段”，而是“如何排序”的规则集合。
- `rules[]`：可扩展规则单元，后续新增子领域时通常只需改配置，不用改脚本。
