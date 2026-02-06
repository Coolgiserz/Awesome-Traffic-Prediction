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
