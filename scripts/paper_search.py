#!/usr/bin/env python3
"""Search recent papers from free sources with optional domain config v2.

Sources:
- OpenAlex (works API)
- arXiv (Atom API)
- Semantic Scholar (Graph API)
- Crossref (works API)

This script uses a polite HTTP strategy by default:
- source-level pacing
- retry with backoff for rate-limit/transient errors
- request budget cap
- local response cache
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

USER_AGENT = "awesome-traffic-prediction-search/0.4 (mailto:maintainers@awesome-traffic-prediction.invalid)"
HTTP_TIMEOUT_SECONDS = 12

SUPPORTED_SOURCES = ("openalex", "arxiv", "semantic_scholar", "crossref")
RETRYABLE_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}

# Built-in safety guard to reduce common topic leakage.
DEFAULT_EXCLUDE_HINTS = (
    "airspace",
    "air traffic",
    "aviation",
    "flight",
    "uav",
    "drone",
    "maritime",
    "shipping",
    "vessel",
    "power load",
    "ev load",
    "microgrid",
)


@dataclass
class Paper:
    title: str
    year: int
    venue: str
    url: str
    source: str
    doi: str
    abstract: str


@dataclass
class Rule:
    rule_id: str
    field: str
    op: str
    keywords: list[str]
    required: bool
    exclude: bool
    weight: float


@dataclass
class RankConfig:
    recency_weight: float
    source_weights: dict[str, float]
    tie_breakers: list[str]


@dataclass
class RuntimePolicy:
    max_requests: int
    min_interval_seconds: float
    max_retries: int
    retry_base_seconds: float
    cache_dir: str
    cache_ttl_hours: int
    source_min_interval_seconds: dict[str, float]


@dataclass
class SearchConfig:
    queries: list[str]
    sources: list[str]
    year_from: int
    year_to: int
    limit_per_query: int
    output_limit: int
    rules: list[Rule]
    min_score: float
    rank: RankConfig
    runtime: RuntimePolicy


DEFAULT_RUNTIME_POLICY = RuntimePolicy(
    max_requests=120,
    min_interval_seconds=0.6,
    max_retries=2,
    retry_base_seconds=2.0,
    cache_dir=".cache/paper-search-http",
    cache_ttl_hours=24,
    source_min_interval_seconds={
        "openalex": 0.8,
        "arxiv": 0.8,
        "semantic_scholar": 1.2,
        "crossref": 1.0,
    },
)


class HttpClient:
    def __init__(self, policy: RuntimePolicy) -> None:
        self.policy = policy
        self.requests_made = 0
        self.cache_hits = 0
        self._last_request_ts: dict[str, float] = {}
        self._cache_dir = Path(policy.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url: str) -> Path:
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, url: str) -> str | None:
        ttl_seconds = max(0, self.policy.cache_ttl_hours) * 3600
        if ttl_seconds <= 0:
            return None
        p = self._cache_path(url)
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > ttl_seconds:
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            payload = obj.get("payload")
            if isinstance(payload, str):
                self.cache_hits += 1
                return payload
        except Exception:
            return None
        return None

    def _save_cache(self, url: str, payload: str) -> None:
        ttl_seconds = max(0, self.policy.cache_ttl_hours) * 3600
        if ttl_seconds <= 0:
            return
        p = self._cache_path(url)
        data = {
            "url": url,
            "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "payload": payload,
        }
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _pacing_sleep(self, source: str) -> None:
        interval = max(
            self.policy.min_interval_seconds,
            self.policy.source_min_interval_seconds.get(source, self.policy.min_interval_seconds),
        )
        now = time.time()
        last = self._last_request_ts.get(source, 0.0)
        wait = interval - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts[source] = time.time()

    def _retry_delay_seconds(self, attempt: int, err: Exception) -> float:
        if isinstance(err, urllib.error.HTTPError):
            retry_after = err.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), self.policy.retry_base_seconds)
                except Exception:
                    pass
        return self.policy.retry_base_seconds * (2**attempt)

    def fetch_text(self, url: str, *, source: str, accept_json: bool = False) -> str:
        cached = self._load_cache(url)
        if cached is not None:
            return cached

        attempts = max(0, self.policy.max_retries) + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            if self.requests_made >= self.policy.max_requests:
                raise RuntimeError(f"request budget exceeded ({self.policy.max_requests})")

            self._pacing_sleep(source)
            self.requests_made += 1

            headers = {"User-Agent": USER_AGENT}
            if accept_json:
                headers["Accept"] = "application/json"
            req = urllib.request.Request(url, headers=headers)

            try:
                with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                self._save_cache(url, text)
                return text
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in RETRYABLE_HTTP_CODES or attempt == attempts - 1:
                    break
                delay = self._retry_delay_seconds(attempt, exc)
                print(
                    f"WARN: transient HTTP {exc.code} from {source}; retry in {delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                delay = self._retry_delay_seconds(attempt, exc)
                print(f"WARN: transient URL error from {source}; retry in {delay:.1f}s", file=sys.stderr)
                time.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError("unexpected fetch failure")

    def fetch_json(self, url: str, *, source: str) -> dict[str, Any]:
        text = self.fetch_text(url, source=source, accept_json=True)
        return json.loads(text)


def normalize_title(title: str) -> str:
    return re.sub(r"\W+", "", title.lower())


def normalize_doi(value: str) -> str:
    if not value:
        return ""
    s = urllib.parse.unquote(value).strip().strip("<>()[]{} \t\r\n\"'")
    low = s.lower()
    for pfx in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if low.startswith(pfx):
            return s[len(pfx) :].strip().strip(".,;)").lower()
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", s, flags=re.IGNORECASE)
    if not m:
        return ""
    return m.group(0).strip().strip(".,;)").lower()


def normalize_abstract_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def reconstruct_openalex_abstract(inv: dict[str, Any] | None) -> str:
    if not isinstance(inv, dict) or not inv:
        return ""
    pos_to_token: dict[int, str] = {}
    for token, positions in inv.items():
        if not token:
            continue
        if not isinstance(positions, list):
            continue
        for p in positions:
            if isinstance(p, int) and p >= 0:
                pos_to_token[p] = str(token)
    if not pos_to_token:
        return ""
    ordered = [pos_to_token[i] for i in sorted(pos_to_token.keys())]
    return normalize_abstract_text(" ".join(ordered))


def parse_sources_csv(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in (x.strip().lower() for x in raw.split(",")):
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def build_queries(query: str, extra_queries: str) -> list[str]:
    queries = [query.strip()] if query.strip() else []
    if extra_queries.strip():
        queries.extend([q.strip() for q in extra_queries.split(";") if q.strip()])
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def resolve_paper_url(doi: str, fallback: str) -> str:
    if doi:
        return f"https://doi.org/{doi}"
    return fallback


def search_openalex(
    query: str,
    year_from: int,
    year_to: int,
    limit: int,
    client: HttpClient,
) -> list[Paper]:
    filt = f"from_publication_date:{year_from}-01-01,to_publication_date:{year_to}-12-31"
    per_page = min(max(limit, 1), 50)
    out: list[Paper] = []
    page = 1
    max_pages = max(1, min(4, (limit + per_page - 1) // per_page))

    while len(out) < limit and page <= max_pages:
        params = {
            "search": query,
            "filter": filt,
            "sort": "publication_date:desc",
            "per-page": str(per_page),
            "page": str(page),
        }
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        data = client.fetch_json(url, source="openalex")
        results = data.get("results", [])
        if not results:
            break
        for w in results:
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
            doi = normalize_doi(str(w.get("doi") or ""))
            link = resolve_paper_url(doi, str(w.get("id") or ""))
            abstract = reconstruct_openalex_abstract(w.get("abstract_inverted_index"))
            if not link:
                continue
            out.append(
                Paper(
                    title=title,
                    year=year,
                    venue=venue,
                    url=link,
                    source="openalex",
                    doi=doi,
                    abstract=abstract,
                )
            )
            if len(out) >= limit:
                break
        page += 1
    return out


def search_arxiv(
    query: str,
    year_from: int,
    year_to: int,
    limit: int,
    client: HttpClient,
) -> list[Paper]:
    batch = min(max(limit, 1), 100)
    start = 0
    out: list[Paper] = []
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    max_pages = max(1, min(4, (limit + batch - 1) // batch))
    page = 0
    while len(out) < limit and page < max_pages:
        params = {
            "search_query": "all:" + query,
            "start": str(start),
            "max_results": str(batch),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
        xml_text = client.fetch_text(url, source="arxiv")
        root = ET.fromstring(xml_text)
        entries = root.findall("a:entry", ns)
        if not entries:
            break
        for entry in entries:
            title = re.sub(
                r"\s+",
                " ",
                (entry.findtext("a:title", default="", namespaces=ns) or "").strip(),
            )
            abstract = normalize_abstract_text(entry.findtext("a:summary", default="", namespaces=ns) or "")
            published = entry.findtext("a:published", default="", namespaces=ns)
            doi = normalize_doi(entry.findtext("arxiv:doi", default="", namespaces=ns) or "")
            try:
                year = int(published[:4])
            except Exception:
                continue
            if year < year_from or year > year_to:
                continue
            link = entry.findtext("a:id", default="", namespaces=ns).strip()
            if not title or not link:
                continue
            out.append(
                Paper(
                    title=title,
                    year=year,
                    venue="arXiv",
                    url=link,
                    source="arxiv",
                    doi=doi,
                    abstract=abstract,
                )
            )
            if len(out) >= limit:
                break
        start += batch
        page += 1
    return out


def search_semantic_scholar(
    query: str,
    year_from: int,
    year_to: int,
    limit: int,
    client: HttpClient,
) -> list[Paper]:
    per_page = min(max(limit, 1), 100)
    offset = 0
    out: list[Paper] = []
    max_pages = max(1, min(4, (limit + per_page - 1) // per_page))
    page = 0
    while len(out) < limit and page < max_pages:
        params = {
            "query": query,
            "limit": str(per_page),
            "offset": str(offset),
            "fields": "title,abstract,year,venue,url,externalIds",
        }
        url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)
        data = client.fetch_json(url, source="semantic_scholar")
        rows = data.get("data", [])
        if not rows:
            break
        for row in rows:
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            year = int(row.get("year") or 0)
            if year < year_from or year > year_to:
                continue
            venue = str(row.get("venue") or "Semantic Scholar").strip() or "Semantic Scholar"
            external_ids = row.get("externalIds") or {}
            doi = normalize_doi(str(external_ids.get("DOI") or ""))
            fallback = str(row.get("url") or "").strip()
            abstract = normalize_abstract_text(str(row.get("abstract") or ""))
            link = resolve_paper_url(doi, fallback)
            if not link:
                continue
            out.append(
                Paper(
                    title=title,
                    year=year,
                    venue=venue,
                    url=link,
                    source="semantic_scholar",
                    doi=doi,
                    abstract=abstract,
                )
            )
            if len(out) >= limit:
                break
        offset += per_page
        page += 1
    return out


def extract_crossref_year(item: dict[str, Any]) -> int:
    for k in ("published-print", "published-online", "issued", "created"):
        parts = (((item.get(k) or {}).get("date-parts")) or [])
        if parts and parts[0] and isinstance(parts[0][0], int):
            return int(parts[0][0])
    return 0


def search_crossref(
    query: str,
    year_from: int,
    year_to: int,
    limit: int,
    client: HttpClient,
) -> list[Paper]:
    per_page = min(max(limit, 1), 100)
    offset = 0
    out: list[Paper] = []
    max_pages = max(1, min(4, (limit + per_page - 1) // per_page))
    page = 0
    allowed_types = {"journal-article", "proceedings-article", "posted-content"}

    while len(out) < limit and page < max_pages:
        params = {
            "query.bibliographic": query,
            "filter": f"from-pub-date:{year_from}-01-01,until-pub-date:{year_to}-12-31",
            "sort": "published",
            "order": "desc",
            "rows": str(per_page),
            "offset": str(offset),
            "mailto": "maintainers@awesome-traffic-prediction.invalid",
        }
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
        data = client.fetch_json(url, source="crossref")
        rows = ((data.get("message") or {}).get("items")) or []
        if not rows:
            break
        for row in rows:
            if str(row.get("type") or "").strip() not in allowed_types:
                continue
            titles = row.get("title") or []
            title = str(titles[0] if titles else "").strip()
            if not title:
                continue
            year = extract_crossref_year(row)
            if year < year_from or year > year_to:
                continue
            container = row.get("container-title") or []
            venue = str(container[0] if container else "Crossref").strip() or "Crossref"
            doi = normalize_doi(str(row.get("DOI") or ""))
            fallback = str(row.get("URL") or "").strip()
            abstract = normalize_abstract_text(str(row.get("abstract") or ""))
            link = resolve_paper_url(doi, fallback)
            if not link:
                continue
            out.append(
                Paper(
                    title=title,
                    year=year,
                    venue=venue,
                    url=link,
                    source="crossref",
                    doi=doi,
                    abstract=abstract,
                )
            )
            if len(out) >= limit:
                break
        offset += per_page
        page += 1
    return out


def dedupe_papers(papers: Iterable[Paper]) -> list[Paper]:
    seen_title: set[str] = set()
    seen_doi: set[str] = set()
    out: list[Paper] = []
    for p in papers:
        title_key = normalize_title(p.title)
        doi_key = p.doi.strip().lower()
        if title_key and title_key in seen_title:
            continue
        if doi_key and doi_key in seen_doi:
            continue
        out.append(p)
        if title_key:
            seen_title.add(title_key)
        if doi_key:
            seen_doi.add(doi_key)
    return out


def rule_match(text: str, rule: Rule) -> bool:
    kws = [k.lower() for k in rule.keywords if k.strip()]
    if not kws:
        return False
    if rule.op == "all":
        return all(k in text for k in kws)
    return any(k in text for k in kws)


def score_papers(papers: Iterable[Paper], cfg: SearchConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    year_span = max(cfg.year_to - cfg.year_from + 1, 1)

    for p in papers:
        title_text = p.title.lower()
        source_text = p.source.lower()
        score = 0.0
        matched_rules: list[str] = []
        rejected_reason = ""

        for rule in cfg.rules:
            field_text = title_text
            if rule.field == "source":
                field_text = source_text
            elif rule.field == "venue":
                field_text = p.venue.lower()

            matched = rule_match(field_text, rule)
            if rule.exclude and matched:
                rejected_reason = f"excluded:{rule.rule_id}"
                break
            if rule.required and not matched:
                rejected_reason = f"missing_required:{rule.rule_id}"
                break
            if matched and not rule.exclude:
                score += rule.weight
                matched_rules.append(rule.rule_id)

        if rejected_reason:
            continue

        recency = (p.year - cfg.year_from) / year_span
        recency = max(0.0, min(1.0, recency))
        score += cfg.rank.recency_weight * recency
        score += cfg.rank.source_weights.get(p.source.lower(), 0.0)

        if score < cfg.min_score:
            continue

        out.append(
            {
                "paper": p,
                "score": round(score, 6),
                "matched_rules": matched_rules,
                "recency_component": round(cfg.rank.recency_weight * recency, 6),
                "source_component": round(cfg.rank.source_weights.get(p.source.lower(), 0.0), 6),
            }
        )

    def cmp_items(a: dict[str, Any], b: dict[str, Any]) -> int:
        sa = float(a["score"])
        sb = float(b["score"])
        if sa != sb:
            return -1 if sa > sb else 1

        pa: Paper = a["paper"]
        pb: Paper = b["paper"]
        for tb in cfg.rank.tie_breakers:
            if tb == "year_desc" and pa.year != pb.year:
                return -1 if pa.year > pb.year else 1
            if tb == "year_asc" and pa.year != pb.year:
                return -1 if pa.year < pb.year else 1
            if tb == "title_desc":
                ta = pa.title.lower()
                tbv = pb.title.lower()
                if ta != tbv:
                    return -1 if ta > tbv else 1
            if tb == "title_asc":
                ta = pa.title.lower()
                tbv = pb.title.lower()
                if ta != tbv:
                    return -1 if ta < tbv else 1
            if tb == "source_desc":
                sa2 = pa.source.lower()
                sb2 = pb.source.lower()
                if sa2 != sb2:
                    return -1 if sa2 > sb2 else 1
            if tb == "source_asc":
                sa2 = pa.source.lower()
                sb2 = pb.source.lower()
                if sa2 != sb2:
                    return -1 if sa2 < sb2 else 1

        ta = pa.title.lower()
        tbv = pb.title.lower()
        if ta == tbv:
            return 0
        return -1 if ta < tbv else 1

    out.sort(key=functools.cmp_to_key(cmp_items))
    return out[: cfg.output_limit]


def to_markdown(scored: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, row in enumerate(scored, 1):
        p: Paper = row["paper"]
        lines.append(f"{i}. [{p.venue} {p.year}] {p.title} [[paper]]({p.url})")
    return "\n".join(lines)


def parse_rule(raw: dict[str, Any], idx: int) -> Rule:
    rid = str(raw.get("id") or f"rule-{idx}")
    field = str(raw.get("field") or "title").strip().lower()
    op = str(raw.get("op") or "any").strip().lower()
    if op not in {"any", "all"}:
        raise ValueError(f"rule '{rid}' has unsupported op: {op}")
    keywords = [str(x).strip() for x in (raw.get("keywords") or []) if str(x).strip()]
    if not keywords:
        raise ValueError(f"rule '{rid}' must provide non-empty keywords")
    return Rule(
        rule_id=rid,
        field=field,
        op=op,
        keywords=keywords,
        required=bool(raw.get("required", False)),
        exclude=bool(raw.get("exclude", False)),
        weight=float(raw.get("weight", 1.0)),
    )


def resolve_year_range(raw: dict[str, Any], now_year: int) -> tuple[int, int]:
    mode = str(raw.get("mode") or "rolling").lower()
    if mode == "rolling":
        lookback_years = int(raw.get("lookback_years", 3))
        if lookback_years < 1:
            raise ValueError("year_range.lookback_years must be >= 1")
        year_to = int(raw.get("to_year", now_year))
        year_from = year_to - lookback_years + 1
        return year_from, year_to
    if mode == "fixed":
        year_from = int(raw.get("from_year"))
        year_to = int(raw.get("to_year"))
        if year_from > year_to:
            raise ValueError("year_range.from_year cannot be larger than to_year")
        return year_from, year_to
    raise ValueError(f"Unsupported year_range.mode: {mode}")


def parse_runtime(raw: dict[str, Any] | None) -> RuntimePolicy:
    r = raw or {}
    source_min_intervals = dict(DEFAULT_RUNTIME_POLICY.source_min_interval_seconds)
    source_min_intervals.update(
        {
            str(k).strip().lower(): float(v)
            for k, v in (r.get("source_min_interval_seconds") or {}).items()
            if str(k).strip()
        }
    )
    runtime = RuntimePolicy(
        max_requests=int(r.get("max_requests", DEFAULT_RUNTIME_POLICY.max_requests)),
        min_interval_seconds=float(
            r.get("min_interval_seconds", DEFAULT_RUNTIME_POLICY.min_interval_seconds)
        ),
        max_retries=int(r.get("max_retries", DEFAULT_RUNTIME_POLICY.max_retries)),
        retry_base_seconds=float(
            r.get("retry_base_seconds", DEFAULT_RUNTIME_POLICY.retry_base_seconds)
        ),
        cache_dir=str(r.get("cache_dir", DEFAULT_RUNTIME_POLICY.cache_dir)),
        cache_ttl_hours=int(r.get("cache_ttl_hours", DEFAULT_RUNTIME_POLICY.cache_ttl_hours)),
        source_min_interval_seconds=source_min_intervals,
    )
    if runtime.max_requests < 1:
        raise ValueError("runtime.max_requests must be >= 1")
    if runtime.min_interval_seconds < 0:
        raise ValueError("runtime.min_interval_seconds must be >= 0")
    if runtime.max_retries < 0:
        raise ValueError("runtime.max_retries must be >= 0")
    if runtime.retry_base_seconds < 0:
        raise ValueError("runtime.retry_base_seconds must be >= 0")
    if runtime.cache_ttl_hours < 0:
        raise ValueError("runtime.cache_ttl_hours must be >= 0")
    for source, interval in runtime.source_min_interval_seconds.items():
        if interval < 0:
            raise ValueError(f"runtime.source_min_interval_seconds['{source}'] must be >= 0")
    return runtime


def apply_runtime_overrides(cfg: SearchConfig, args: argparse.Namespace) -> SearchConfig:
    runtime = cfg.runtime
    if args.max_requests is not None:
        runtime.max_requests = int(args.max_requests)
    if args.min_interval_seconds is not None:
        runtime.min_interval_seconds = float(args.min_interval_seconds)
    if args.max_retries is not None:
        runtime.max_retries = int(args.max_retries)
    if args.retry_base_seconds is not None:
        runtime.retry_base_seconds = float(args.retry_base_seconds)
    if args.cache_dir is not None:
        runtime.cache_dir = str(args.cache_dir)
    if args.cache_ttl_hours is not None:
        runtime.cache_ttl_hours = int(args.cache_ttl_hours)
    return cfg


def load_config(path: str, now_year: int) -> SearchConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(data.get("version", 1))
    if version != 2:
        raise ValueError(f"Unsupported config version: {version}, expected 2")

    search = data.get("search") or {}
    queries = [str(x).strip() for x in (search.get("queries") or []) if str(x).strip()]
    if not queries:
        raise ValueError("search.queries cannot be empty")

    sources_raw = [str(x).strip().lower() for x in (search.get("sources") or []) if str(x).strip()]
    sources = parse_sources_csv(",".join(sources_raw)) if sources_raw else ["openalex", "arxiv"]
    invalid_sources = [s for s in sources if s not in SUPPORTED_SOURCES]
    if invalid_sources:
        raise ValueError(f"Unsupported search source(s): {', '.join(invalid_sources)}")

    year_from, year_to = resolve_year_range(search.get("year_range") or {}, now_year)
    limit_per_query = int(search.get("limit_per_query", 25))
    output_limit = int(search.get("output_limit", limit_per_query))
    if limit_per_query < 1 or output_limit < 1:
        raise ValueError("limit_per_query and output_limit must be >= 1")

    post = data.get("post") or {}
    rules = [parse_rule(r, i + 1) for i, r in enumerate(post.get("rules") or [])]
    min_score = float(post.get("min_score", 0.0))

    rank = data.get("rank") or {}
    source_weights = {
        str(k).strip().lower(): float(v)
        for k, v in (rank.get("source_weights") or {}).items()
        if str(k).strip()
    }
    tie_breakers = [str(x).strip() for x in (rank.get("tie_breakers") or ["year_desc", "title_asc"])]
    rank_cfg = RankConfig(
        recency_weight=float(rank.get("recency_weight", 0.0)),
        source_weights=source_weights,
        tie_breakers=tie_breakers,
    )

    runtime_cfg = parse_runtime(data.get("runtime"))

    return SearchConfig(
        queries=queries,
        sources=sources,
        year_from=year_from,
        year_to=year_to,
        limit_per_query=limit_per_query,
        output_limit=output_limit,
        rules=rules,
        min_score=min_score,
        rank=rank_cfg,
        runtime=runtime_cfg,
    )


def build_legacy_config(args: argparse.Namespace, now_year: int) -> SearchConfig:
    if not args.query.strip():
        raise ValueError("query is required when --config is not provided")
    year_to = int(args.year_to or now_year)
    year_from = int(args.year_from)
    if year_from > year_to:
        raise ValueError("--year-from cannot be larger than --year-to")

    sources = parse_sources_csv(args.sources) or ["openalex", "arxiv"]
    invalid_sources = [s for s in sources if s not in SUPPORTED_SOURCES]
    if invalid_sources:
        raise ValueError(f"Unsupported --sources values: {', '.join(invalid_sources)}")

    rules: list[Rule] = []
    must_kws = [x.strip() for x in args.must_keywords.split(",") if x.strip()]
    pred_kws = [x.strip() for x in args.pred_keywords.split(",") if x.strip()]
    exc_kws = [x.strip() for x in args.exclude_keywords.split(",") if x.strip()]

    if must_kws:
        rules.append(
            Rule(
                rule_id="legacy-domain",
                field="title",
                op="any",
                keywords=must_kws,
                required=True,
                exclude=False,
                weight=1.0,
            )
        )
    if pred_kws:
        rules.append(
            Rule(
                rule_id="legacy-intent",
                field="title",
                op="any",
                keywords=pred_kws,
                required=True,
                exclude=False,
                weight=1.0,
            )
        )
    if exc_kws:
        rules.append(
            Rule(
                rule_id="legacy-exclude",
                field="title",
                op="any",
                keywords=exc_kws,
                required=False,
                exclude=True,
                weight=0.0,
            )
        )
    rules.append(
        Rule(
            rule_id="builtin-non-road",
            field="title",
            op="any",
            keywords=list(DEFAULT_EXCLUDE_HINTS),
            required=False,
            exclude=True,
            weight=0.0,
        )
    )

    queries = build_queries(args.query, args.extra_queries)
    return SearchConfig(
        queries=queries,
        sources=sources,
        year_from=year_from,
        year_to=year_to,
        limit_per_query=max(args.limit, 1),
        output_limit=max(args.limit, 1),
        rules=rules,
        min_score=0.0,
        rank=RankConfig(recency_weight=0.0, source_weights={}, tie_breakers=["year_desc", "title_asc"]),
        runtime=parse_runtime(None),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Search recent papers from free sources.")
    ap.add_argument(
        "query",
        nargs="?",
        default="",
        help="Search query, e.g. 'traffic flow forecasting graph neural network'",
    )
    ap.add_argument("--config", default="", help="Optional config JSON (domain config v2).")
    ap.add_argument("--extra-queries", default="", help="Semicolon-separated query variants.")
    ap.add_argument("--year-from", type=int, default=2024)
    ap.add_argument("--year-to", type=int, default=datetime.now(timezone.utc).year)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument(
        "--sources",
        default="openalex,arxiv",
        help="Comma list: openalex,arxiv,semantic_scholar,crossref",
    )
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
        default=(
            "air quality,dengue,maritime,shipping,anomaly,accident,risk,aqi,"
            "airspace,aviation,flight,uav,drone,vessel,power load,ev load,microgrid"
        ),
        help="Comma list; drop papers whose title contains these keywords.",
    )
    ap.add_argument("--max-requests", type=int, default=None, help="Override runtime request budget")
    ap.add_argument(
        "--min-interval-seconds",
        type=float,
        default=None,
        help="Override global minimum interval between requests to each source",
    )
    ap.add_argument("--max-retries", type=int, default=None, help="Override retry count")
    ap.add_argument("--retry-base-seconds", type=float, default=None, help="Override retry backoff base")
    ap.add_argument("--cache-dir", default=None, help="Override HTTP cache directory")
    ap.add_argument("--cache-ttl-hours", type=int, default=None, help="Override HTTP cache TTL")
    ap.add_argument("--out", default="", help="Optional output markdown file path")
    ap.add_argument("--report-json", default="", help="Optional JSON report with scores and rule hits")
    return ap.parse_args()


def run_source_search(
    source: str,
    query: str,
    year_from: int,
    year_to: int,
    limit: int,
    client: HttpClient,
) -> list[Paper]:
    if source == "openalex":
        return search_openalex(query, year_from, year_to, limit, client)
    if source == "arxiv":
        return search_arxiv(query, year_from, year_to, limit, client)
    if source == "semantic_scholar":
        return search_semantic_scholar(query, year_from, year_to, limit, client)
    if source == "crossref":
        return search_crossref(query, year_from, year_to, limit, client)
    raise ValueError(f"Unsupported source: {source}")


def main() -> int:
    args = parse_args()
    now_year = datetime.now(timezone.utc).year

    try:
        cfg = load_config(args.config, now_year) if args.config else build_legacy_config(args, now_year)
        cfg = apply_runtime_overrides(cfg, args)
    except Exception as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
        return 2

    client = HttpClient(cfg.runtime)

    papers: list[Paper] = []
    source_errors: list[str] = []
    for query in cfg.queries:
        for source in cfg.sources:
            try:
                papers.extend(
                    run_source_search(
                        source=source,
                        query=query,
                        year_from=cfg.year_from,
                        year_to=cfg.year_to,
                        limit=cfg.limit_per_query,
                        client=client,
                    )
                )
            except Exception as exc:
                source_errors.append(f"query='{query}' source='{source}' error={exc}")

    for e in source_errors:
        print(f"WARN: {e}", file=sys.stderr)
    if not papers and source_errors:
        print("ERROR: all source requests failed", file=sys.stderr)
        return 1

    papers = dedupe_papers(papers)
    scored = score_papers(papers, cfg)
    md = to_markdown(scored)

    if args.out:
        Path(args.out).write_text(md + ("\n" if md else ""), encoding="utf-8")
        print(f"Wrote {len(scored)} entries to {args.out}")
    else:
        print(md)

    if args.report_json:
        report = {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": {
                "year_from": cfg.year_from,
                "year_to": cfg.year_to,
                "queries": cfg.queries,
                "sources": cfg.sources,
                "output_limit": cfg.output_limit,
                "min_score": cfg.min_score,
            },
            "runtime": {
                "max_requests": cfg.runtime.max_requests,
                "min_interval_seconds": cfg.runtime.min_interval_seconds,
                "max_retries": cfg.runtime.max_retries,
                "retry_base_seconds": cfg.runtime.retry_base_seconds,
                "cache_dir": cfg.runtime.cache_dir,
                "cache_ttl_hours": cfg.runtime.cache_ttl_hours,
                "source_min_interval_seconds": cfg.runtime.source_min_interval_seconds,
                "requests_made": client.requests_made,
                "cache_hits": client.cache_hits,
            },
            "source_errors": source_errors,
            "papers": [
                {
                    "title": row["paper"].title,
                    "year": row["paper"].year,
                    "venue": row["paper"].venue,
                    "url": row["paper"].url,
                    "source": row["paper"].source,
                    "doi": row["paper"].doi,
                    "abstract": row["paper"].abstract,
                    "score": row["score"],
                    "matched_rules": row["matched_rules"],
                    "recency_component": row["recency_component"],
                    "source_component": row["source_component"],
                }
                for row in scored
            ],
        }
        Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote score report to {args.report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
