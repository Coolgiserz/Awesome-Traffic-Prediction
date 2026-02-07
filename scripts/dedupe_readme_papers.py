#!/usr/bin/env python3
"""Deduplicate candidate papers against existing README papers.

Matching keys:
- normalized title
- DOI extracted from paper URL/text

Input candidate format:
- 1. [Venue Year] Title [[paper]](url) [[code]](url)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote

CANDIDATE_RE = re.compile(r"^\s*\d+\.\s+\[([^\]]+)\]\s+(.+?)\s+\[\[paper\]\]\(([^)]+)\)(.*)$")
# Some existing README items are in the form: 1. [KDD 2021] [Title](url)
ALT_README_RE = re.compile(r"^\s*\d+\.\s+\[([^\]]+)\]\s+\[(.+?)\]\((https?://[^)\s]+)\)")
LINK_RE = re.compile(r"\]\((https?://[^)\s]+)\)")
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def normalize_title(title: str) -> str:
    return re.sub(r"\W+", "", title.lower())


def extract_doi(text: str) -> str:
    s = unquote(text).strip()
    s = s.strip("<>()[]{} \t\r\n\"'")
    low = s.lower()

    for pfx in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ):
        if low.startswith(pfx):
            return s[len(pfx) :].strip().strip(".,;)").lower()

    if low.startswith("doi:"):
        return s[4:].strip().strip(".,;)").lower()

    m = DOI_RE.search(s)
    if not m:
        return ""
    return m.group(0).strip().strip(".,;)").lower()


def parse_candidate_line(line: str) -> dict[str, str] | None:
    m = CANDIDATE_RE.match(line)
    if not m:
        return None
    return {
        "venue_year": m.group(1).strip(),
        "title": m.group(2).strip(),
        "paper_url": m.group(3).strip(),
        "tail": m.group(4).rstrip(),
    }


def extract_existing_keys(readme_paths: list[Path]) -> tuple[set[str], set[str]]:
    title_keys: set[str] = set()
    doi_keys: set[str] = set()

    for path in readme_paths:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            parsed = parse_candidate_line(line)
            if parsed:
                title_keys.add(normalize_title(parsed["title"]))
                doi = extract_doi(parsed["paper_url"])
                if doi:
                    doi_keys.add(doi)

            alt = ALT_README_RE.match(line)
            if alt:
                title_keys.add(normalize_title(alt.group(2).strip()))
                doi = extract_doi(alt.group(3))
                if doi:
                    doi_keys.add(doi)

            for url in LINK_RE.findall(line):
                doi = extract_doi(url)
                if doi:
                    doi_keys.add(doi)

        for doi in DOI_RE.findall(text):
            doi_keys.add(doi.lower())

    return title_keys, doi_keys


def format_candidate(index: int, item: dict[str, str]) -> str:
    return (
        f"{index}. [{item['venue_year']}] {item['title']} "
        f"[[paper]]({item['paper_url']}){item['tail']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Deduplicate candidate papers against README existing entries.")
    ap.add_argument("--input", required=True, help="Input markdown list file")
    ap.add_argument("--readme", required=True, nargs="+", help="README markdown files to compare against")
    ap.add_argument("--out", required=True, help="Output markdown list file")
    ap.add_argument("--report-json", default="", help="Optional JSON report path with kept/removed stats")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    readme_paths = [Path(p) for p in args.readme]

    existing_title_keys, existing_doi_keys = extract_existing_keys(readme_paths)

    candidates: list[dict[str, str]] = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_candidate_line(line)
        if parsed:
            candidates.append(parsed)

    kept: list[dict[str, str]] = []
    removed: list[dict[str, Any]] = []
    seen_title_keys: set[str] = set()
    seen_doi_keys: set[str] = set()

    for c in candidates:
        title_key = normalize_title(c["title"])
        doi_key = extract_doi(c["paper_url"])

        reason = ""
        if doi_key and doi_key in existing_doi_keys:
            reason = "existing_doi"
        elif title_key and title_key in existing_title_keys:
            reason = "existing_title"
        elif doi_key and doi_key in seen_doi_keys:
            reason = "candidate_dup_doi"
        elif title_key and title_key in seen_title_keys:
            reason = "candidate_dup_title"

        if reason:
            removed.append(
                {
                    "reason": reason,
                    "title": c["title"],
                    "paper_url": c["paper_url"],
                    "doi": doi_key,
                }
            )
            continue

        kept.append(c)
        if title_key:
            seen_title_keys.add(title_key)
        if doi_key:
            seen_doi_keys.add(doi_key)

    out_path.write_text(
        ("\n".join(format_candidate(i, item) for i, item in enumerate(kept, 1)) + "\n") if kept else "",
        encoding="utf-8",
    )

    reason_counts: dict[str, int] = {}
    for r in removed:
        k = str(r["reason"])
        reason_counts[k] = reason_counts.get(k, 0) + 1

    print(
        "[dedupe] readmes=%d existing_titles=%d existing_dois=%d input=%d kept=%d removed=%d"
        % (
            len(readme_paths),
            len(existing_title_keys),
            len(existing_doi_keys),
            len(candidates),
            len(kept),
            len(removed),
        )
    )
    if reason_counts:
        details = ", ".join(f"{k}={v}" for k, v in sorted(reason_counts.items()))
        print(f"[dedupe] reasons: {details}")

    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(
                {
                    "input_count": len(candidates),
                    "kept_count": len(kept),
                    "removed_count": len(removed),
                    "reason_counts": reason_counts,
                    "removed": removed,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
