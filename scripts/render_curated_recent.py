#!/usr/bin/env python3
"""Render README auto-blocks from the structured data source.

Why this exists
---------------
The README used to be the source of truth and the JSON was derived from it. That
direction has flipped: `data/papers/*.json` now carries everything the collection
knows (2023-2026 alone went from 21 README entries to 73 structured records), so a
README that is not rendered from it silently goes stale.

This script renders two blocks in both READMEs:

  <!-- START AUTO RECENT PAPERS --> ... <!-- END AUTO RECENT PAPERS -->
  <!-- START AUTO CURATED DIGEST TABLE --> ... <!-- END AUTO CURATED DIGEST TABLE -->

Selection rules
---------------
Recent block  : the latest N years present in the data, chronological inside a
                year, capped so the README does not turn into a data dump.
Digest table  : landmark papers first (they are what a reader should see), then
                the most significant recent work, one row per paper.

Reading order is decided by `impact.tier`, not by recency alone. A 2026 PLoS ONE
article does not outrank DCRNN.

Usage
-----
    python3 scripts/render_curated_recent.py            # write
    python3 scripts/render_curated_recent.py --check    # CI gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RECENT_START = "<!-- START AUTO RECENT PAPERS -->"
RECENT_END = "<!-- END AUTO RECENT PAPERS -->"
DIGEST_TABLE_START = "<!-- START AUTO CURATED DIGEST TABLE -->"
DIGEST_TABLE_END = "<!-- END AUTO CURATED DIGEST TABLE -->"

IMPACT_ORDER = {"landmark": 0, "notable": 1, "standard": 2}
# A venue-published paper outranks a preprint of the same significance: the
# README is read by people looking for work they can cite.
VENUE_TIER_ORDER = {"top": 0, "mid": 1, "low": 2, "preprint": 3}


# --------------------------------------------------------------------- io


def load_papers(data_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            records.extend(r for r in data if isinstance(r, dict))
    return records


def load_venues(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {v["id"]: v for v in data if isinstance(v, dict) and v.get("id")}


# ---------------------------------------------------------------- helpers


def sort_key(p: dict[str, Any], venues: dict[str, dict[str, Any]]) -> tuple:
    """Reading order inside a section.

    Significance first, then how hard the venue was to get into, then recency.
    Without the venue tier a freshly-posted arXiv preprint outranks an AAAI paper
    of the same tier, which is the exact failure this ordering exists to avoid.
    """
    tier = IMPACT_ORDER.get((p.get("impact") or {}).get("tier", "standard"), 2)
    vinfo = venues.get(p.get("venue_id") or "", {})
    vtier = VENUE_TIER_ORDER.get(vinfo.get("tier", "preprint"), 3)
    return (tier, vtier, -(int(p.get("year") or 0)), str(p.get("title") or "").lower())


def recent_sort_key(p: dict[str, Any]) -> tuple:
    return (int(p.get("year") or 0), int(p.get("order") or 9999), str(p.get("title") or ""))


def display_name(p: dict[str, Any]) -> str:
    """Human-readable label for the digest table.

    `short_name` is an acronym and is often absent; truncating the title at a
    fixed width produces things like "Pre-training Enhanced Spatial-te", which
    reads worse than no label at all.
    """
    short = (p.get("short_name") or "").strip()
    if short:
        return short
    title = str(p.get("title") or "")
    head = title.split(":")[0].strip()
    if 3 <= len(head) <= 42:
        return head
    words = title.split()
    out: list[str] = []
    for w in words:
        if len(" ".join(out + [w])) > 38:
            break
        out.append(w)
    return " ".join(out) or title[:38]


def display_venue(p: dict[str, Any], venues: dict[str, dict[str, Any]]) -> str:
    """Prefer the README-conventional short tag; fall back to the record's label."""
    vid = p.get("venue_id")
    if vid and vid in venues:
        short = venues[vid].get("short_name")
        if short:
            return short
    return str(p.get("venue") or "Arxiv")


def links_for(p: dict[str, Any]) -> str:
    """`[[paper]](url)` plus every code/data link, README link syntax."""
    parts: list[str] = []
    if p.get("paper_url"):
        parts.append(f"[[paper]]({p['paper_url']})")
    for link in p.get("code_links") or []:
        label = link.get("label") or "code"
        parts.append(f"[[{label}]]({link['url']})")
    if p.get("code_url") and not any(l.get("url") == p["code_url"] for l in p.get("code_links") or []):
        parts.append(f"[[code]]({p['code_url']})")
    if p.get("data_url"):
        parts.append(f"[[data]]({p['data_url']})")
    return " ".join(parts)


def format_recent_item(index: int, p: dict[str, Any], venues: dict[str, dict[str, Any]]) -> str:
    venue = display_venue(p, venues)
    year = int(p.get("year") or 0)
    links = links_for(p)
    line = f"{index}. [{venue} {year}] {p['title']}"
    return f"{line} {links}".rstrip() if links else line


def pick_recent_years(papers: list[dict[str, Any]], window: int) -> list[int]:
    years = sorted({int(p["year"]) for p in papers if p.get("year")})
    if not years:
        return []
    if window <= 0:
        return years
    return years[-(window):]


def render_recent_block(
    papers: list[dict[str, Any]], venues: dict[str, dict[str, Any]], window: int, per_year: int
) -> str:
    years = pick_recent_years(papers, window)
    out: list[str] = []
    for year in years:
        bucket = sorted(
            [p for p in papers if int(p.get("year") or 0) == year],
            key=recent_sort_key,
        )
        if not bucket:
            continue
        # Significance first inside a year: the README is a reading list, not a log.
        bucket.sort(key=lambda x: sort_key(x, venues))
        out.append(f"#### {year}")
        out.append("")
        limit = len(bucket) if per_year <= 0 else per_year
        for i, p in enumerate(bucket[:limit], 1):
            out.append(format_recent_item(i, p, venues))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def digest_selection(
    papers: list[dict[str, Any]], venues: dict[str, dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Landmarks first, then recent significant work, one row each."""
    # Landmarks read chronologically: the table tells the story of the field.
    landmarks = sorted(
        [p for p in papers if (p.get("impact") or {}).get("tier") == "landmark"],
        key=lambda x: (int(x.get("year") or 0), str(x.get("title") or "").lower()),
    )
    others = sorted(
        [
            p
            for p in papers
            if (p.get("impact") or {}).get("tier") != "landmark"
            and (p.get("impact") or {}).get("tier") == "notable"
        ],
        key=lambda x: sort_key(x, venues),
    )
    return (landmarks + others)[:limit]


def render_digest_rows(
    selection: list[dict[str, Any]], lang: str, venues: dict[str, dict[str, Any]]
) -> str:
    rows: list[str] = []
    for p in selection:
        year = int(p.get("year") or 0)
        short = display_name(p)
        url = p.get("paper_url") or ""
        task = p.get("task_zh") if lang == "zh" else p.get("task_en")
        task = task or p.get("task_en") or ""
        tags = ", ".join(p.get("method_tags") or []) or "—"
        code_url = p.get("code_url")
        code_cell = f"[code]({code_url})" if code_url else "N/A"
        # impact.role is the curator's "why it matters"; note_* is the older,
        # sparser field. Prefer the former and fall back to the latter.
        impact = p.get("impact") or {}
        note = (impact.get("role_zh") if lang == "zh" else impact.get("role")) or (
            p.get("note_zh") if lang == "zh" else p.get("note_en")
        ) or "—"
        venue = display_venue(p, venues)
        rows.append(f"| {year} | [{short}]({url}) | {task} | {tags} | {code_cell} | {venue} · {note} |")
    return "\n".join(rows).rstrip() + "\n"


def render_digest_table(
    selection: list[dict[str, Any]], lang: str, venues: dict[str, dict[str, Any]]
) -> str:
    header = (
        "| Year | Paper | Task | Method Tags | Code | Note |\n| --- | --- | --- | --- | --- | --- |\n"
        if lang == "en"
        else "| 年份 | 论文 | 任务 | 方法标签 | 代码 | 备注 |\n| --- | --- | --- | --- | --- | --- |\n"
    )
    return header + render_digest_rows(selection, lang, venues)


# ------------------------------------------------------------------ write


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
    rendered = replace_block(rendered, DIGEST_TABLE_START, DIGEST_TABLE_END, digest_rows)

    changed = rendered != original
    if check:
        return changed
    if changed:
        path.write_text(rendered, encoding="utf-8")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="Render README sections from structured paper data.")
    ap.add_argument("--data-dir", default="data/papers")
    ap.add_argument("--venues", default="data/venues.json")
    ap.add_argument("--readme-en", default="README.md")
    ap.add_argument("--readme-zh", default="README.zh.md")
    ap.add_argument(
        "--recent-window-years",
        type=int,
        default=3,
        help="How many of the latest years get a section. Use <=0 for all years.",
    )
    ap.add_argument(
        "--per-year",
        type=int,
        default=12,
        help="Maximum entries per year section, significance-ordered.",
    )
    ap.add_argument("--digest-limit", type=int, default=20, help="Rows in the curated digest table.")
    ap.add_argument("--check", action="store_true", help="Fail if rendered output differs from files.")
    args = ap.parse_args()

    papers = load_papers(Path(args.data_dir))
    if not papers:
        print(f"[FAIL] no records found under {args.data_dir}")
        return 1
    venues = load_venues(Path(args.venues))

    recent_block = render_recent_block(papers, venues, args.recent_window_years, args.per_year)
    selection = digest_selection(papers, venues, args.digest_limit)
    digest_en = render_digest_table(selection, "en", venues)
    digest_zh = render_digest_table(selection, "zh", venues)

    changed_en = apply_to_readme(Path(args.readme_en), recent_block, digest_en, args.check)
    changed_zh = apply_to_readme(Path(args.readme_zh), recent_block, digest_zh, args.check)

    if args.check and (changed_en or changed_zh):
        print("[FAIL] README files are not in sync with structured data.")
        if changed_en:
            print(f"- Outdated: {args.readme_en}")
        if changed_zh:
            print(f"- Outdated: {args.readme_zh}")
        return 1

    print("[OK] Curated recent paper sections rendered.")
    print(f"- records: {len(papers)}")
    print(f"- recent years: {pick_recent_years(papers, args.recent_window_years)}")
    print(f"- digest rows: {len(selection)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())