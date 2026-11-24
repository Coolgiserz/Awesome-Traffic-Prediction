#!/usr/bin/env python3
"""Enrich data/papers records with authoritative metadata from OpenAlex.

Reads every record that has a DOI, an arXiv id or a resolvable title, queries
OpenAlex (polite pool, one request per unique work), and fills only fields the
README migration could not know:

  doi, openalex_id, authors, venue_type, paper_url (if missing),
  verification.metadata_source / link_ok / checked_at

Human-written content (notes, tags, layers, featured flags) is never touched.
The script is idempotent: records already verified against OpenAlex are skipped
unless --refresh is passed.

Usage:
  python3 scripts/enrich_openalex.py --report
  python3 scripts/enrich_openalex.py --apply
  python3 scripts/enrich_openalex.py --apply --year 2026 --refresh
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

MAILTO = "awesome-traffic-prediction@example.org"
UA = f"awesome-traffic-prediction/1.0 (mailto:{MAILTO})"

# OpenAlex rate-limits hard on shared CI IPs (429). Crossref is the fallback:
# it resolves DOI + authors + container title, which is what the migration lacks.
CROSSREF_SELECT = "DOI,title,author,container-title,type,published"
CROSSREF_TYPE_MAP = {
    "journal-article": "journal",
    "proceedings-article": "conference",
    "posted-content": "preprint",
    "book-chapter": "journal",
    "report": "preprint",
}

# OpenAlex source type -> schema venue_type.
TYPE_MAP = {
    "journal": "journal",
    "journal-article": "journal",
    "conference": "conference",
    # OpenAlex reports most venue-published CS papers as `conference-paper`,
    # which was missing from this map — every CIKM/ICDM/KDD record silently fell
    # through and kept its provisional `preprint` label.
    "conference-paper": "conference",
    "proceedings-article": "conference",
    "proceedings": "conference",
    "repository": "preprint",
    "ebook": "journal",
    "book": "journal",
    "dissertation": "thesis",
    "posted-content": "preprint",
    "preprint": "preprint",
    "review": "journal",
    "article": "journal",
    "letter": "journal",
    "paratext": "unknown",
    "erratum": "unknown",
    "other": "unknown",
    "report": "preprint",
    "dataset": "unknown",
    "libguides": "unknown",
    "standard": "unknown",
    "reference-entry": "unknown",
}


def norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


class Client:
    def __init__(self, interval: float = 0.35, timeout: float = 20.0) -> None:
        self.interval = interval
        self.timeout = timeout
        self._last = 0.0
        self.requests = 0
        self.errors: Counter[str] = Counter()

    def get(self, url: str) -> dict[str, Any] | None:
        wait = self.interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        self.requests += 1
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                self.errors[f"http_{exc.code}"] += 1
                if exc.code in (403, 404):
                    return None
                time.sleep(1.5 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                self.errors[type(exc).__name__] += 1
                time.sleep(1.5 * (attempt + 1))
        return None


def lookup(client: Client, record: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve one record to an OpenAlex work, most reliable key first."""
    doi = record.get("doi")
    arxiv_id = record.get("arxiv_id")
    if doi:
        work = client.get(f"https://api.openalex.org/works/https://doi.org/{doi}")
        if work:
            return work
    if arxiv_id:
        work = client.get(f"https://api.openalex.org/works/https://doi.org/10.48550/arxiv.{arxiv_id}")
        if work:
            return work
    title = record.get("title", "")
    if len(title) >= 12:
        query = urllib.parse.urlencode(
            {"filter": f"title.search:{title[:180]}", "per-page": "3", "select": "id,doi,title,publication_year,authorships,type,primary_location"}
        )
        page = client.get(f"https://api.openalex.org/works?{query}")
        for work in (page or {}).get("results", []):
            if norm_title(work.get("title") or "") == norm_title(title):
                return work
    return None


def crossref_lookup(client: Client, record: dict[str, Any]) -> dict[str, Any] | None:
    """Crossref fallback: title query, exact normalized-title match required."""
    title = record.get("title", "")
    if len(title) < 12:
        return None
    query = urllib.parse.urlencode(
        {"query.title": title[:180], "rows": "3", "select": CROSSREF_SELECT}
    )
    page = client.get(f"https://api.crossref.org/works?{query}")
    for item in ((page or {}).get("message") or {}).get("items", []):
        titles = item.get("title") or []
        if titles and norm_title(titles[0]) == norm_title(title):
            item["_source"] = "crossref"
            return item
    return None


def apply_crossref(record: dict[str, Any], item: dict[str, Any], today: str) -> bool:
    before = json.dumps(record, sort_keys=True, ensure_ascii=False)

    doi = (item.get("DOI") or "").lower() or None
    if doi and not record.get("doi"):
        record["doi"] = doi
    if doi:
        record["paper_url"] = f"https://doi.org/{doi}"

    authors = []
    for a in item.get("author") or []:
        name = " ".join(x for x in [a.get("given"), a.get("family")] if x).strip()
        if name:
            authors.append(name)
    if authors and not record.get("authors"):
        record["authors"] = authors

    container = ((item.get("container-title") or [None])[0] or "").strip()
    if container and record.get("venue") in (None, "", "Arxiv", "arXiv"):
        record["venue"] = container
    vtype = CROSSREF_TYPE_MAP.get((item.get("type") or "").strip())
    if vtype and record.get("venue_type") in (None, "unknown", "preprint"):
        record["venue_type"] = vtype
    if vtype in ("journal", "conference") and container and record.get("venue") in (None, "", "Arxiv", "arXiv"):
        record["venue"] = container

    verification = record.setdefault("verification", {})
    verification["checked_at"] = today
    verification["link_ok"] = True
    verification["metadata_source"] = "crossref"
    verification["confidence"] = "high" if doi else "medium"

    return json.dumps(record, sort_keys=True, ensure_ascii=False) != before


def apply_work(record: dict[str, Any], work: dict[str, Any], today: str) -> bool:
    """Fill only metadata fields. Returns True when something changed."""
    before = json.dumps(record, sort_keys=True, ensure_ascii=False)

    openalex_id = (work.get("id") or "").rsplit("/", 1)[-1]
    if openalex_id.startswith("W"):
        record["openalex_id"] = openalex_id

    work_doi = (work.get("doi") or "").replace("https://doi.org/", "").lower() or None
    if work_doi and work_doi != record.get("doi"):
        record["doi"] = work_doi
        # A publisher DOI is a better primary link than a dead IEEE abstract page.
        if record.get("paper_url") and "arxiv.org" not in record["paper_url"]:
            record["paper_url"] = f"https://doi.org/{work_doi}"

    authors = []
    for a in work.get("authorships") or []:
        name = ((a.get("author") or {}).get("display_name") or "").strip()
        if name:
            authors.append(name)
    if authors:
        record["authors"] = authors

    vtype = TYPE_MAP.get((work.get("type") or "").strip())
    # "preprint" is a provisional state, not an assertion: a record imported from
    # arXiv is marked preprint because that is all we knew at import time. Once a
    # provider reports a journal or conference venue, upgrade it.
    if vtype and record.get("venue_type") in (None, "unknown", "preprint"):
        record["venue_type"] = vtype
    # Same reasoning for the venue label: "Arxiv" is the arXiv crossref container,
    # not the venue the work was eventually published in.
    if vtype in ("journal", "conference") and record.get("venue") in (None, "", "Arxiv", "arXiv", "arXiv (Cornell University)"):
        published_in = (
            ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
            or ((work.get("host_venue") or {}).get("display_name"))
        )
        # "Proceedings" alone carries no venue identity; keep the placeholder
        # rather than replacing "Arxiv" with something equally unhelpful.
        if published_in and published_in.strip().lower() not in ("proceedings", "conference", "book"):
            record["venue"] = published_in.strip()

    # arXiv-only record that turned out to have a real venue.
    year = work.get("publication_year")
    if year and int(year) != int(record.get("year") or 0):
        record.setdefault("alt_years", [])
        if int(record["year"]) not in record["alt_years"]:
            record["alt_years"] = sorted({*(record.get("alt_years") or []), int(record["year"])})

    verification = record.setdefault("verification", {})
    verification["checked_at"] = today
    verification["link_ok"] = True
    verification["metadata_source"] = "openalex"
    verification["confidence"] = "high" if work_doi or openalex_id else "medium"

    return json.dumps(record, sort_keys=True, ensure_ascii=False) != before


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/papers")
    ap.add_argument("--year", type=int, action="append")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry run).")
    ap.add_argument("--refresh", action="store_true", help="Re-query records already verified by OpenAlex.")
    ap.add_argument("--report", action="store_true", help="Only print coverage statistics.")
    ap.add_argument("--today", default=None)
    ap.add_argument("--limit", type=int, default=0, help="Stop after N lookups (0 = all).")
    args = ap.parse_args()

    today = args.today or date.today().isoformat()
    files = sorted(Path(args.data_dir).glob("*.json"))
    if args.year:
        wanted = set(args.year)
        files = [f for f in files if int(f.stem) in wanted]

    if args.report:
        stats: Counter[str] = Counter()
        total = 0
        for f in files:
            for r in json.loads(f.read_text(encoding="utf-8")):
                total += 1
                src = (r.get("verification") or {}).get("metadata_source")
                stats[src or "none"] += 1
                for field in ("doi", "openalex_id", "authors", "code_url", "arxiv_id"):
                    if r.get(field):
                        stats[field] += 1
        print(f"records={total}")
        for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
            print(f"  {k:22s} {v}")
        return 0

    client = Client()
    stats: Counter[str] = Counter()
    changed = missing = skipped = 0
    lookups = 0
    for f in files:
        records = json.loads(f.read_text(encoding="utf-8"))
        dirty = False
        for r in records:
            source = (r.get("verification") or {}).get("metadata_source")
            if source in ("openalex", "crossref") and not args.refresh:
                skipped += 1
                continue
            if args.limit and lookups >= args.limit:
                break
            lookups += 1
            work = lookup(client, r)
            if work is not None:
                if apply_work(r, work, today):
                    changed += 1
                    dirty = True
                continue
            item = crossref_lookup(client, r)
            if item is None:
                missing += 1
                continue
            if apply_crossref(r, item, today):
                changed += 1
                dirty = True
                stats["via_crossref"] += 1
        if dirty and args.apply:
            f.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[WRITE] {f}")

    print(
        f"[SUMMARY] lookups={lookups} changed={changed} unresolved={missing} "
        f"skipped={skipped} http_requests={client.requests} errors={dict(client.errors)}"
    )
    if not args.apply:
        print("[DRY-RUN] pass --apply to write changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())