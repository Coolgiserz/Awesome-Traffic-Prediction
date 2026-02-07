#!/usr/bin/env python3
"""Render README sections from structured curated paper JSON.

The script renders:
- Recent paper year blocks (auto-selected by year window)
- Curated digest table rows (auto-selected by year window + featured flag)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RECENT_START = "<!-- START AUTO RECENT PAPERS -->"
RECENT_END = "<!-- END AUTO RECENT PAPERS -->"
DIGEST_START = "<!-- START AUTO CURATED DIGEST ROWS -->"
DIGEST_END = "<!-- END AUTO CURATED DIGEST ROWS -->"


def load_entries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Curated paper JSON must be a list.")
    entries: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        entries.append(item)
    return entries


def sort_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda x: (int(x.get("year", 0)), int(x.get("order", 9999))))


def format_recent_item(index: int, e: dict[str, Any]) -> str:
    venue = str(e["venue"])
    year = int(e["year"])
    title = str(e["title"])
    paper_url = str(e["paper_url"])
    code_url = e.get("code_url")
    suffix = f" [[code]]({code_url})" if code_url else ""
    return f"{index}. [{venue} {year}] {title} [[paper]]({paper_url}){suffix}"


def pick_years(entries: list[dict[str, Any]], window_years: int) -> list[int]:
    years = sorted({int(e.get("year", 0)) for e in entries if int(e.get("year", 0)) > 0})
    if not years:
        return []
    if window_years <= 0:
        return years
    max_year = years[-1]
    min_year = max_year - window_years + 1
    return [y for y in years if y >= min_year]


def render_recent_block(entries: list[dict[str, Any]], recent_window_years: int) -> str:
    selected_years = pick_years(entries, recent_window_years)
    out: list[str] = []
    for year in selected_years:
        year_entries = [e for e in entries if int(e.get("year", 0)) == year]
        if not year_entries:
            continue
        out.append(f"#### {year}")
        out.append("")
        for i, e in enumerate(year_entries, 1):
            out.append(format_recent_item(i, e))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_digest_rows(entries: list[dict[str, Any]], lang: str, digest_window_years: int) -> str:
    rows: list[str] = []
    featured_all = [e for e in entries if bool(e.get("featured", False))]
    years = pick_years(featured_all, digest_window_years)
    featured = [e for e in featured_all if int(e.get("year", 0)) in set(years)]
    for e in featured:
        year = int(e["year"])
        short_name = str(e["short_name"])
        paper_url = str(e["paper_url"])
        task = str(e["task_en"] if lang == "en" else e["task_zh"])
        tags = ", ".join(str(x) for x in e.get("method_tags", []))
        code_url = e.get("code_url")
        code_cell = f"[code]({code_url})" if code_url else "N/A"
        note = str(e["note_en"] if lang == "en" else e["note_zh"])
        rows.append(f"| {year} | [{short_name}]({paper_url}) | {task} | {tags} | {code_cell} | {note} |")
    return "\n".join(rows).rstrip() + "\n"


def replace_block(text: str, start_marker: str, end_marker: str, new_block: str) -> str:
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0 or end < 0 or start > end:
        raise ValueError(f"Markers not found or invalid order: {start_marker} ... {end_marker}")
    prefix = text[: start + len(start_marker)]
    suffix = text[end:]
    return f"{prefix}\n{new_block}{suffix}"


def apply_to_readme(path: Path, recent_block: str, digest_rows: str, check: bool) -> bool:
    original = path.read_text(encoding="utf-8")
    rendered = replace_block(original, RECENT_START, RECENT_END, recent_block)
    rendered = replace_block(rendered, DIGEST_START, DIGEST_END, digest_rows)
    changed = rendered != original
    if check:
        return changed
    if changed:
        path.write_text(rendered, encoding="utf-8")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="Render README sections from curated JSON.")
    ap.add_argument("--data", default="updates/curated/recent-papers.json")
    ap.add_argument("--readme-en", default="README.md")
    ap.add_argument("--readme-zh", default="README.zh.md")
    ap.add_argument(
        "--recent-window-years",
        type=int,
        default=2,
        help="Recent list uses the latest N years in structured data. Use <=0 for all years.",
    )
    ap.add_argument(
        "--digest-window-years",
        type=int,
        default=3,
        help="Digest rows use featured items from latest N years. Use <=0 for all years.",
    )
    ap.add_argument("--check", action="store_true", help="Fail if rendered output differs from files.")
    args = ap.parse_args()

    entries = sort_entries(load_entries(Path(args.data)))
    recent_block = render_recent_block(entries, args.recent_window_years)
    digest_en = render_digest_rows(entries, "en", args.digest_window_years)
    digest_zh = render_digest_rows(entries, "zh", args.digest_window_years)

    changed_en = apply_to_readme(Path(args.readme_en), recent_block, digest_en, args.check)
    changed_zh = apply_to_readme(Path(args.readme_zh), recent_block, digest_zh, args.check)

    if args.check and (changed_en or changed_zh):
        print("[FAIL] README files are not in sync with curated structured data.")
        if changed_en:
            print(f"- Outdated: {args.readme_en}")
        if changed_zh:
            print(f"- Outdated: {args.readme_zh}")
        return 1

    print("[OK] Curated recent paper sections rendered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
