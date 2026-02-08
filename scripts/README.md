# Paper Search Tool

This folder contains lightweight maintenance scripts for documentation updates.

## `paper_search.py`

Searches recent papers from free sources and outputs markdown list items usable in README sections.

- Sources: OpenAlex + arXiv
- Extended sources: Semantic Scholar + Crossref
- Dependency: `python3` only (standard library)
- Supports two modes:
  - Legacy CLI parameters
  - Domain config v2 (`--config`)

### Example

```bash
python3 scripts/paper_search.py \
  "traffic flow forecasting graph neural network" \
  --extra-queries "traffic speed prediction;spatio-temporal traffic forecasting;incident traffic forecasting" \
  --year-from 2024 \
  --year-to 2026 \
  --limit 20 \
  --out /tmp/traffic_papers.md
```

### Config-driven example (recommended)

```bash
python3 scripts/paper_search.py \
  --config scripts/configs/traffic-monthly.json \
  --out /tmp/traffic_papers.md \
  --report-json /tmp/traffic_papers_report.json
```

### Notes

- Output is candidate-only; manual curation is still required.
- The HTTP layer is polite by default:
  - request pacing per source
  - retry + backoff for `429`/`5xx`
  - request budget cap (`runtime.max_requests`)
  - local response cache (`runtime.cache_*`)
- Supported `search.sources` values:
  - `openalex`
  - `arxiv`
  - `semantic_scholar`
  - `crossref`
- `post.rules[]` in config is extensible:
  - `required=true`: mandatory match rule
  - `exclude=true`: hard rejection rule
  - `weight`: contributes to ranking score when matched
- `post.min_score` is the final score threshold after rule match + ranking components.
- `rank` controls ranking only:
  - `recency_weight`: favors recent years
  - `source_weights`: source-level preference
  - `tie_breakers`: deterministic ordering when scores tie
- Optional runtime knobs:
  - in config: `runtime` object
  - in CLI override: `--max-requests`, `--min-interval-seconds`, `--max-retries`, `--retry-base-seconds`, `--cache-dir`, `--cache-ttl-hours`

## Domain Config Files

- Schema reference: `scripts/configs/domain-config-v2.schema.json`
- Logic guide: `scripts/configs/README.md`
- Traffic monthly workflow config: `scripts/configs/traffic-monthly.json`

## `validate_domain_config.py`

Validates domain config v2 before runtime.

### Example

```bash
python3 scripts/validate_domain_config.py scripts/configs/traffic-monthly.json
```

## `dedupe_readme_papers.py`

Deduplicates candidate paper lists against existing README entries using dual keys:

- Key 1: normalized title
- Key 2: DOI

A candidate is removed if either key already exists in README entries.

### Example

```bash
python3 scripts/dedupe_readme_papers.py \
  --input /tmp/monthly_papers.md \
  --readme README.md README.zh.md \
  --out /tmp/monthly_papers_dedup.md \
  --report-json /tmp/monthly_papers_dedup_report.json
```

## `paper_wordcloud.py`

Generates a keyword word cloud from markdown paper titles with stopword/noise filtering.
It also outputs a Top-K term bar chart for easier trend reading.

### Example

```bash
python3 scripts/paper_wordcloud.py \
  --input updates/monthly-paper-candidates.md \
  --out-dir updates/wordcloud \
  --prefix monthly-paper-wordcloud \
  --extra-noise "paper,traffic,forecasting,prediction"
```

## `check_readme_sync.py`

Checks structural consistency between `README.md` and `README.zh.md`:

- Yearly paper-list counts in deep-learning sections
- Curated digest table row counts
- Benchmark matrix row counts

### Example

```bash
python3 scripts/check_readme_sync.py --en README.md --zh README.zh.md
```

## `render_curated_recent.py`

Renders recent-paper year blocks and curated digest rows in both README files from structured JSON.

- Input: `updates/curated/recent-papers.json`
- Output targets:
  - `README.md`
  - `README.zh.md`
- Controlled blocks:
  - `<!-- START AUTO RECENT PAPERS --> ...`
  - `<!-- START AUTO CURATED DIGEST TABLE --> ...`
- Default behavior:
  - Recent year blocks: latest 2 years in JSON (`--recent-window-years 2`)
  - Digest rows: featured papers from latest 3 years (`--digest-window-years 3`)

### Example

```bash
python3 scripts/render_curated_recent.py \
  --data updates/curated/recent-papers.json \
  --readme-en README.md \
  --readme-zh README.zh.md
```

### Check-only mode (CI)

```bash
python3 scripts/render_curated_recent.py \
  --data updates/curated/recent-papers.json \
  --readme-en README.md \
  --readme-zh README.zh.md \
  --check
```

## Monthly GitHub Action

- Workflow file: `.github/workflows/monthly-paper-search.yml`
- Schedule: monthly at `02:00 UTC` on day 1 (`cron: 0 2 1 * *`)
- Pipeline:
  1. Validate domain config (`scripts/validate_domain_config.py`)
  2. Search candidate papers (`scripts/paper_search.py --config scripts/configs/traffic-monthly.json`)
  3. Deduplicate against README (`scripts/dedupe_readme_papers.py`, title + DOI)
  4. Write `updates/monthly-paper-candidates.md`
  5. Generate visuals (`wordcloud` + `yearly hotspots`)
- Word cloud outputs: `updates/wordcloud/monthly-paper-wordcloud-latest.png` and monthly archive png.
- Top-K terms chart: `updates/wordcloud/monthly-paper-wordcloud-topk-latest.png`
- Yearly hotspots: `updates/hotspots/yearly-hotspots-trend.png` (single trend heatmap), `updates/hotspots/yearly-theme-trends.png` (theme line chart), `updates/hotspots/yearly-hotspots.png`, `updates/hotspots/yearly-hotspots.md`, `updates/hotspots/yearly-hotspots.json`
- The workflow opens a pull request when outputs change.

## README Sync GitHub Action

- Workflow file: `.github/workflows/readme-sync-check.yml`
- Trigger: pull requests (and pushes to `main`) that touch README sync-related files.
- Check command: `python3 scripts/check_readme_sync.py --en README.md --zh README.zh.md`

## Curated Render Check Action

- Workflow file: `.github/workflows/curated-render-check.yml`
- Trigger: pull requests (and pushes to `main`) that touch curated JSON, renderer script, or README files.
- Check command: `python3 scripts/render_curated_recent.py --data updates/curated/recent-papers.json --check`
