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

Renders the README auto-blocks from `data/papers/*.json` — the structured data is
now the source of truth, and the README is a rendered view of it.

Two blocks are controlled:

- `AUTO RECENT PAPERS`: the latest N years, capped per year.
- `AUTO CURATED DIGEST TABLE`: landmark papers in chronological order (they tell
  the story of the field), then notable recent work.

Ordering is significance first, then venue tier, then recency. Without the venue
tier step a freshly-posted arXiv preprint outranks an AAAI paper of the same
significance, which is exactly the failure this ordering exists to prevent.

```bash
# render 5 recent years, up to 12 entries each
python3 scripts/render_curated_recent.py --recent-window-years 5

# CI gate
python3 scripts/render_curated_recent.py --recent-window-years 5 --check
```

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

## `migrate_readme_to_json.py`

One-way bridge from the README paper lists to the structured data source under
`data/papers/YYYY.json` (Phase 1 of `docs/evolution/01-data-model.md`).

- Source of truth during the grey release: README stays human-editable, JSON is generated.
- Idempotent: re-running over migrated years keeps ids, human-filled fields, and every
  value a later stage (enrichment) established. A field is only *added*, never erased.
- Reversible: `--check` exits non-zero when `data/papers` drifts from the README, which
  is what the CI gate should run.
- `URL_FIXES` in the script encodes verified link corrections (for example: papers with
  code pages that redirect away are replaced by the real repository or dropped).

```bash
# dry run: show what would change
python3 scripts/migrate_readme_to_json.py --all --dry-run

# migrate or refresh every year found in the README
python3 scripts/migrate_readme_to_json.py --all

# CI gate
python3 scripts/migrate_readme_to_json.py --all --check
```

Notes:

- Entries inside the auto-rendered blocks (`AUTO RECENT PAPERS`) are skipped: they come
  from `updates/curated/recent-papers.json` and are merged by provenance, not by line.
- Legacy `[Venue Year] [Title](url)` entries (title in markdown-link form) are unwrapped,
  and titles broken across two README lines are joined by the item regex.
- `provenance.readme_line` records the source line so drift can be audited line by line.

## `enrich_openalex.py`

Fills metadata the README never carried: DOI, OpenAlex work id, author list, venue type,
and verification stamps. OpenAlex is queried first; Crossref is the fallback when
OpenAlex rate-limits (HTTP 429), which happens often on shared CI IPs.

- Only metadata fields are written. Notes, tags, layers and featured flags are untouched.
- `verification.metadata_source` records which provider answered, so a later run can
  skip already-verified records.
- The original README link is preserved as `alt_urls.paper-original` when the primary
  link is upgraded to a DOI.

```bash
# coverage report, no network
python3 scripts/enrich_openalex.py --report

# enrich (dry run first, then --apply)
python3 scripts/enrich_openalex.py --limit 40
python3 scripts/enrich_openalex.py --apply

# re-verify records that were already stamped
python3 scripts/enrich_openalex.py --apply --refresh --year 2026
```

Both scripts are standard-library only, except the optional `jsonschema` package used for
schema validation (`pip install jsonschema`).

## `check_venue_consistency.py`

Catches venue labels that contradict the link they point at. A venue tag is an
inference; when the primary link is a DOI, the DOI slug usually names the real
venue, so a mismatch means either the label or the link is wrong.

This exists because of a real bug: a paper labelled `WWW` had the DOI
`10.1109/iwcmc55113.2022.9825059` (an IWCMC proceedings paper). The rule that
produced the label matched the substring "www" inside "iwcmc".

```bash
python3 scripts/check_venue_consistency.py           # report
python3 scripts/check_venue_consistency.py --strict  # CI: non-zero on findings
```

## `snapshot_citations.py`

Takes a **dated** citation snapshot for the papers the collection highlights
(`impact.tier` = landmark or notable) and writes `data/citations.json`.

Deliberately a snapshot, not a live lookup:

- A build that calls OpenAlex makes the site depend on a third party being up,
  and silently changes its content whenever the index is refreshed.
- A citation count without an "as of" date is not evidence. The file records
  `retrieved_on`, and the site prints it next to every number.
- Only 15 records are queried. Citations on a 2026 preprint are noise, and
  fetching all 191 would spend the request budget on numbers nobody would act on.

```bash
python3 scripts/snapshot_citations.py            # fetch and write
python3 scripts/snapshot_citations.py --dry-run  # print, do not write
```

Run it by hand when the numbers matter (e.g. before publishing a release). The
output is reviewed like any other data file; nothing in CI fetches it.
