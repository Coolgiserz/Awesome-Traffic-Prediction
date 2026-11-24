#!/usr/bin/env python3
"""Take a dated citation snapshot for the papers the collection highlights.

Why a snapshot and not a live lookup
------------------------------------
Citation counts are evidence, not truth. A number fetched at build time would
(a) make every build depend on a third party being up, and (b) silently change
the site whenever OpenAlex re-indexes. What a reader needs is "as of when", so
this writes a dated file that is committed and reviewed like any other data.

It deliberately covers only `impact.tier == landmark` and `notable`: a citation
count on a 2026 preprint is noise, and fetching all 191 records would burn the
OpenAlex budget for numbers nobody will act on.

    python3 scripts/snapshot_citations.py            # fetch and write
    python3 scripts/snapshot_citations.py --dry-run  # print only

Output: data/citations.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

MAILTO = "awesome-traffic-prediction@example.org"
UA = f"awesome-traffic-prediction/1.0 (mailto:{MAILTO})"
TIERS = ("landmark", "notable")


def load_targets(data_dir: Path) -> list[dict]:
    out = []
    for path in sorted(glob.glob(f"{data_dir}/*.json")):
        for rec in json.load(open(path, encoding="utf-8")):
            tier = (rec.get("impact") or {}).get("tier")
            if tier in TIERS:
                out.append(rec)
    return out


def fetch(url: str, timeout: float = 20.0, retries: int = 3) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 403):
                return None
            time.sleep(2.0 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(2.0 * (attempt + 1))
    return None


def lookup(rec: dict) -> dict | None:
    """Resolve one record to an OpenAlex work, preferring an exact DOI."""
    doi = rec.get("doi")
    if doi and not doi.startswith("10.48550/"):
        work = fetch(f"https://api.openalex.org/works/https://doi.org/{urllib.parse.quote(doi)}")
        if work:
            return work
    arxiv = (rec.get("arxiv_id") or "").split("v")[0]
    if arxiv:
        work = fetch(f"https://api.openalex.org/works/https://doi.org/10.48550/arxiv.{arxiv}")
        if work:
            return work
    title = rec.get("title") or ""
    if len(title) >= 12:
        q = urllib.parse.urlencode(
            {"filter": f"title.search:{title[:170]}", "per-page": "3",
             "select": "id,doi,title,cited_by_count,publication_year"}
        )
        page = fetch(f"https://api.openalex.org/works?{q}")
        norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
        for work in (page or {}).get("results", []):
            if norm(work.get("title") or "") == norm(title):
                return work
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/papers")
    ap.add_argument("--out", default="data/citations.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--interval", type=float, default=0.6)
    args = ap.parse_args()

    targets = load_targets(Path(args.data_dir))
    print(f"targets: {len(targets)} records (impact tier landmark/notable)")

    entries = []
    missing = []
    for rec in targets:
        work = lookup(rec)
        time.sleep(args.interval)
        if not work:
            missing.append(rec)
            continue
        entries.append({
            "id": rec["id"],
            "title": rec["title"],
            "year": rec.get("year"),
            "tier": rec["impact"]["tier"],
            "cited_by_count": work.get("cited_by_count"),
            "openalex_id": (work.get("id") or "").rsplit("/", 1)[-1] or None,
            "doi": (work.get("doi") or "").replace("https://doi.org/", "") or None,
        })

    # Second pass for anything the first sweep could not resolve: OpenAlex
    # intermittently 404s on a valid work, and a snapshot with holes is worse
    # than a slightly slower one.
    still_missing = []
    for rec in missing:
        time.sleep(args.interval * 2)
        work = lookup(rec)
        if not work:
            still_missing.append(rec["id"])
            continue
        entries.append({
            "id": rec["id"],
            "title": rec["title"],
            "year": rec.get("year"),
            "tier": rec["impact"]["tier"],
            "cited_by_count": work.get("cited_by_count"),
            "openalex_id": (work.get("id") or "").rsplit("/", 1)[-1] or None,
            "doi": (work.get("doi") or "").replace("https://doi.org/", "") or None,
        })
    missing = still_missing

    entries.sort(key=lambda e: (-(e["cited_by_count"] or 0), e["year"] or 0))

    payload = {
        "_comment": "Dated citation snapshot. Not used at build time; committed so a "
                    "number always carries the date it was true.",
        "source": "OpenAlex",
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "retrieved_on": date.today().isoformat(),
        "count": len(entries),
        "unresolved": missing,
        "entries": entries,
    }

    if args.dry_run:
        for e in entries[:15]:
            print(f"  {e['cited_by_count']:>6}  {e['year']}  {e['title'][:60]}")
        print(f"resolved {len(entries)} / {len(targets)}, unresolved {len(missing)}")
        return 0

    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(entries)} entries, {len(missing)} unresolved")
    for e in entries[:12]:
        print(f"  {e['cited_by_count']:>6}  {e['year']}  {e['title'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())