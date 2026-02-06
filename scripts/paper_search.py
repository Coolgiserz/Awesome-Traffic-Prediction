#!/usr/bin/env python3
"""Simple free academic paper search tool for traffic forecasting updates.

Sources:
- OpenAlex (works API)
- arXiv (Atom API)

Outputs markdown list lines suitable for README updates.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List

USER_AGENT = "awesome-traffic-prediction-search/0.1 (mailto:2811159909@qq.com)"


@dataclass
class Paper:
    title: str
    year: int
    venue: str
    url: str
    source: str


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def search_openalex(query: str, year_from: int, year_to: int, limit: int) -> List[Paper]:
    filt = f"from_publication_date:{year_from}-01-01,to_publication_date:{year_to}-12-31"
    params = {
        "search": query,
        "filter": filt,
        "sort": "publication_date:desc",
        "per-page": str(min(max(limit, 1), 50)),
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    data = fetch_json(url)
    out: List[Paper] = []
    for w in data.get("results", []):
        title = (w.get("title") or "").strip()
        if not title:
            continue
        year = int(w.get("publication_year") or 0)
        if year < year_from or year > year_to:
            continue
        venue = (
            ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
            or "OpenAlex"
        )
        url = w.get("doi") or w.get("id") or ""
        if url.startswith("https://doi.org/"):
            pass
        elif url.startswith("10."):
            url = "https://doi.org/" + url
        elif w.get("id"):
            url = w["id"]
        out.append(Paper(title=title, year=year, venue=venue, url=url, source="openalex"))
        if len(out) >= limit:
            break
    return out


def search_arxiv(query: str, year_from: int, year_to: int, limit: int) -> List[Paper]:
    q = "all:" + query
    params = {
        "search_query": q,
        "start": "0",
        "max_results": str(min(max(limit, 1), 50)),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    xml_text = fetch_text(url)
    root = ET.fromstring(xml_text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: List[Paper] = []
    for entry in root.findall("a:entry", ns):
        title = re.sub(r"\s+", " ", (entry.findtext("a:title", default="", namespaces=ns) or "").strip())
        published = entry.findtext("a:published", default="", namespaces=ns)
        try:
            year = int(published[:4])
        except Exception:
            continue
        if year < year_from or year > year_to:
            continue
        link = entry.findtext("a:id", default="", namespaces=ns)
        if not title or not link:
            continue
        out.append(Paper(title=title, year=year, venue="arXiv", url=link, source="arxiv"))
        if len(out) >= limit:
            break
    return out


def build_queries(base_query: str, extra_queries: str) -> List[str]:
    queries = [base_query.strip()]
    if extra_queries.strip():
        queries.extend([q.strip() for q in extra_queries.split(";") if q.strip()])
    # Keep order and drop duplicates.
    seen = set()
    out: List[str] = []
    for q in queries:
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append(q)
    return out


def dedupe(papers: Iterable[Paper]) -> List[Paper]:
    seen = set()
    out: List[Paper] = []
    for p in papers:
        key = re.sub(r"\W+", "", p.title.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def keep_domain_relevant(papers: Iterable[Paper], must_keywords: List[str]) -> List[Paper]:
    kws = [k.strip().lower() for k in must_keywords if k.strip()]
    if not kws:
        return list(papers)
    out: List[Paper] = []
    for p in papers:
        t = p.title.lower()
        if any(k in t for k in kws):
            out.append(p)
    return out


def keep_prediction_focused(papers: Iterable[Paper], pred_keywords: List[str]) -> List[Paper]:
    kws = [k.strip().lower() for k in pred_keywords if k.strip()]
    if not kws:
        return list(papers)
    out: List[Paper] = []
    for p in papers:
        t = p.title.lower()
        if any(k in t for k in kws):
            out.append(p)
    return out


def remove_excluded(papers: Iterable[Paper], exclude_keywords: List[str]) -> List[Paper]:
    kws = [k.strip().lower() for k in exclude_keywords if k.strip()]
    if not kws:
        return list(papers)
    out: List[Paper] = []
    for p in papers:
        t = p.title.lower()
        if any(k in t for k in kws):
            continue
        out.append(p)
    return out


def to_markdown(papers: List[Paper]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. [{p.venue} {p.year}] {p.title} [[paper]]({p.url})")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search recent traffic forecasting papers from free sources.")
    ap.add_argument("query", help="Search query, e.g. 'traffic flow forecasting graph neural network'")
    ap.add_argument(
        "--extra-queries",
        default="",
        help="Semicolon-separated query variants, e.g. 'traffic speed prediction;spatio-temporal traffic forecasting'",
    )
    ap.add_argument("--year-from", type=int, default=2024)
    ap.add_argument("--year-to", type=int, default=datetime.now(timezone.utc).year)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--sources", default="openalex,arxiv", help="Comma list: openalex,arxiv")
    ap.add_argument(
        "--must-keywords",
        default="traffic,transport,road,mobility,flow",
        help="Comma list; keep papers whose title contains at least one keyword.",
    )
    ap.add_argument(
        "--pred-keywords",
        default="forecast,prediction,predicting",
        help="Comma list; keep papers whose title contains prediction-related keywords.",
    )
    ap.add_argument(
        "--exclude-keywords",
        default="air quality,dengue,maritime,shipping,anomaly,accident,risk,aqi",
        help="Comma list; drop papers whose title contains these keywords.",
    )
    ap.add_argument("--out", default="", help="Optional output markdown file path")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    sources = {x.strip().lower() for x in args.sources.split(",") if x.strip()}
    queries = build_queries(args.query, args.extra_queries)
    papers: List[Paper] = []

    try:
        for q in queries:
            if "openalex" in sources:
                papers.extend(search_openalex(q, args.year_from, args.year_to, args.limit))
            if "arxiv" in sources:
                papers.extend(search_arxiv(q, args.year_from, args.year_to, args.limit))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    papers = dedupe(papers)
    papers = keep_domain_relevant(papers, args.must_keywords.split(","))
    papers = keep_prediction_focused(papers, args.pred_keywords.split(","))
    papers = remove_excluded(papers, args.exclude_keywords.split(","))
    papers.sort(key=lambda p: p.year, reverse=True)
    papers = papers[: args.limit]

    md = to_markdown(papers)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md + ("\n" if md else ""))
        print(f"Wrote {len(papers)} entries to {args.out}")
    else:
        print(md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
