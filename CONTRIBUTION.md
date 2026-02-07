# Contributing

Thanks for helping improve this awesome list.

## Scope

This is a documentation-first repository for traffic prediction resources.

- Main content: `README.md`, `README.zh.md`, dataset/model markdown pages.
- Automation helpers: lightweight scripts under `scripts/`.
- Out of scope: application services, deployment stacks, complex build systems.

## Before You Open a PR

1. For major reorganization, open an issue first to discuss structure and scope.
2. Check for duplicates before adding papers/datasets/tools.
3. Verify links and citation metadata (venue, year, title).
4. If you update `README.md`, update `README.zh.md` in the same PR.

## Content Conventions

### Paper Entries

- Recommended format: `[Conference/Journal Year] Title [[paper]](https://example.com/paper) [[code]](https://github.com/org/repo)`
- Keep yearly sections in chronological order.
- Prefer official paper links (publisher/arXiv/DOI) and official code repos.

### Dataset Pages

- Put dataset docs in `datasets/`.
- Include a brief English description and Chinese explanation when applicable.
- Keep image/resource links valid.

### Bilingual Consistency

- Keep section structure aligned between `README.md` and `README.zh.md`.
- Run:

```bash
python3 scripts/check_readme_sync.py --en README.md --zh README.zh.md
```

The script checks yearly paper-count alignment, digest row-count alignment, and benchmark matrix row alignment.

### Structured Recent Papers

- Source of truth: `updates/curated/recent-papers.json`
- Do not manually edit auto blocks in README files:
  - `<!-- START AUTO RECENT PAPERS --> ... <!-- END AUTO RECENT PAPERS -->`
  - `<!-- START AUTO CURATED DIGEST TABLE --> ... <!-- END AUTO CURATED DIGEST TABLE -->`
- After JSON edits, render with:

```bash
python3 scripts/render_curated_recent.py --data updates/curated/recent-papers.json
```

## Pull Request Checklist

1. Update relevant markdown files and keep formatting consistent.
2. Run the README sync check script.
3. If you touched automation output logic, update `scripts/README.md`.
4. Include a concise PR description of what changed and why.

## Merge Policy

- Do not push directly to `main`.
- Use pull requests and at least one reviewer for content-affecting updates.
