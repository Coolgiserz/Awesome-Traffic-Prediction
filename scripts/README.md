# Paper Search Tool

This folder contains lightweight maintenance scripts for documentation updates.

## `paper_search.py`

Searches recent papers from free sources and outputs markdown list items usable in README sections.

- Sources: OpenAlex + arXiv
- Dependency: `python3` only (standard library)

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

### Notes

- Output is candidate-only; manual curation is still required.
- Use `--must-keywords` to constrain domain relevance in titles.
- Use `--pred-keywords` and `--exclude-keywords` to improve precision for traffic forecasting papers.

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

## Monthly GitHub Action

- Workflow file: `.github/workflows/monthly-paper-search.yml`
- Schedule: monthly at `02:00 UTC` on day 1 (`cron: 0 2 1 * *`)
- Output file: `updates/monthly-paper-candidates.md`
- Word cloud outputs: `updates/wordcloud/monthly-paper-wordcloud-latest.png` and monthly archive png.
- Top-K terms chart: `updates/wordcloud/monthly-paper-wordcloud-topk-latest.png`
- Yearly hotspots: `updates/hotspots/yearly-hotspots-trend.png` (single trend heatmap), `updates/hotspots/yearly-theme-trends.png` (theme line chart), `updates/hotspots/yearly-hotspots.png`, `updates/hotspots/yearly-hotspots.md`, `updates/hotspots/yearly-hotspots.json`
- The workflow commits directly to `main` when outputs change.
