#!/usr/bin/env python3
"""Migrate README.md entries into structured data/papers/YYYY.json records.

Phase 1 of the repository evolution described in docs/evolution/01-data-model.md.
The README stays the human-editable surface during the grey release; this script
is the one-way bridge that has to be idempotent and reversible:

  * idempotent: re-running it over an already migrated year keeps stable ids,
    keeps human-written notes, and does not reorder records;
  * reversible: `--report` prints a coverage table so `--check` can assert that
    every README line is either represented in JSON or explicitly ignored.

Usage:
  python3 scripts/migrate_readme_to_json.py --year 2019 --dry-run
  python3 scripts/migrate_readme_to_json.py --all
  python3 scripts/migrate_readme_to_json.py --all --check
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

START_RECENT = "<!-- START AUTO RECENT PAPERS -->"
END_RECENT = "<!-- END AUTO RECENT PAPERS -->"

# README section headings -> schema `section` enum.
SECTION_MAP: list[tuple[str, str]] = [
    ("reviews", "reviews"),
    ("deep learning based", "deep_learning"),
    ("基于深度学习", "deep_learning"),
    ("statistic based", "statistic"),
    ("基于统计", "statistic"),
    ("time series forecasting", "time_series"),
    ("时间序列预测", "time_series"),
    ("temporal network embedding", "temporal_network_embedding"),
    ("时序网络嵌入", "temporal_network_embedding"),
]

# Venue short name -> (slug, type). Only the venues that actually occur in the
# README are listed; unknown venues fall back to a slugified name + "unknown".
VENUE_TABLE: dict[str, tuple[str, str]] = {
    "TITS": ("tits", "journal"),
    "TRPET": ("trpart-c", "journal"),
    "TR-C": ("trpart-c", "journal"),
    "TKDE": ("tkde", "journal"),
    "Access": ("access", "journal"),
    "Sensor": ("sensors", "journal"),
    "Sensors": ("sensors", "journal"),
    "IJGIS": ("ijgis", "journal"),
    "Information Fusion": ("information-fusion", "journal"),
    "Applied Intelligence": ("applied-intelligence", "journal"),
    "Complex & Intelligent Systems": ("complex-intelligent-systems", "journal"),
    "Neural Computing and Applications": ("neural-computing-and-applications", "journal"),
    "Neural Networks": ("neural-networks", "journal"),
    "Expert Systems with Applications": ("expert-systems-with-applications", "journal"),
    "Scientific Reports": ("scientific-reports", "journal"),
    "PLoS ONE": ("plos-one", "journal"),
    "PLOS ONE": ("plos-one", "journal"),
    "Electronics": ("electronics", "journal"),
    "Systems": ("systems", "journal"),
    "IET Intelligent Transport Systems": ("iet-its", "journal"),
    "ISPRS IJGI": ("isprs-ijgi", "journal"),
    "ISPRS": ("isprs-ijgi", "journal"),
    "Lecture Notes in Computer Science": ("lncs", "conference"),
    "Neurocomputing": ("neurocomputing", "journal"),
    "Arxiv": ("arxiv", "preprint"),
    "arXiv": ("arxiv", "preprint"),
    "AAAI": ("aaai", "conference"),
    "IJCAI": ("ijcai", "conference"),
    "KDD": ("kdd", "conference"),
    "SIGKDD": ("kdd", "conference"),
    "NIPS": ("neurips", "conference"),
    "NeurIPS": ("neurips", "conference"),
    "ICLR": ("iclr", "conference"),
    "ICML": ("icml", "conference"),
    "ICDE": ("icde", "conference"),
    "ICDM": ("icdm", "conference"),
    "CVPR": ("cvpr", "conference"),
    "ICCV": ("iccv", "conference"),
    "ECCV": ("eccv", "conference"),
    "SIGSPATIAL": ("sigspatial", "conference"),
    "Sigspatial": ("sigspatial", "conference"),
    "WWW": ("www", "conference"),
    "WWWC": ("www", "conference"),
    "WC": ("www", "conference"),
    "WSDM": ("wsdm", "conference"),
    "CIKM": ("cikm", "conference"),
    "ICIKM": ("cikm", "conference"),
    "IWPHM": ("iwphm", "workshop"),
    "CS224W": ("cs224w", "workshop"),
    "CIKM/ICDM": ("cikm", "conference"),
}

TASK_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("Missing value imputation", re.compile(r"imputation|missing value", re.I)),
    ("Trajectory prediction", re.compile(r"trajector", re.I)),
    ("Travel time estimation", re.compile(r"travel time|\beta\b", re.I)),
    ("Demand prediction", re.compile(r"demand|order|\bod\b|bike|taxi", re.I)),
    ("Accident / incident prediction", re.compile(r"accident|incident|crash", re.I)),
    ("Nowcasting", re.compile(r"nowcast|precipitation", re.I)),
    ("Traffic speed prediction", re.compile(r"\bspeed\b", re.I)),
    ("Traffic flow prediction", re.compile(r"\bflow\b", re.I)),
    ("Survey", re.compile(r"survey|review|overview|future directions", re.I)),
]

METHOD_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("Graph Convolution", re.compile(r"gcn\b|graph convolution|graph convolutional", re.I)),
    ("STGNN", re.compile(r"spatio-?temporal graph|stgnn|\bgnn\b|graph neural", re.I)),
    ("Graph Attention", re.compile(r"graph attention|\bgat\b|attention based spatial", re.I)),
    ("Adaptive Graph", re.compile(r"adaptive (adjacency|graph)|dynamic graph|structure learning", re.I)),
    ("Transformer", re.compile(r"transformer|attention|informer|itransformer|mamba", re.I)),
    ("Diffusion", re.compile(r"diffusion", re.I)),
    ("GAN", re.compile(r"\bgan\b|generative adversarial", re.I)),
    ("Foundation Model", re.compile(r"foundation model|pre-?train|universal model|prompt|large language", re.I)),
    ("Meta-Learning", re.compile(r"meta-?learning", re.I)),
    ("Few-Shot", re.compile(r"few-?shot", re.I)),
    ("Transfer Learning", re.compile(r"transfer|multi-?city|cross-?city", re.I)),
    ("Federated Learning", re.compile(r"federated", re.I)),
    ("Probabilistic", re.compile(r"probabilistic|uncertainty|bayesian|quantile", re.I)),
    ("Spatio-Temporal CNN", re.compile(r"conv-?lstm|convlstm|3d convolution|\bcnn\b", re.I)),
    ("RNN / LSTM", re.compile(r"\blstm\b|\brnn\b|\bgru\b", re.I)),
    ("Embedding", re.compile(r"embedding|representation learning", re.I)),
    ("Clustering", re.compile(r"cluster", re.I)),
    ("Matrix Factorization", re.compile(r"matrix factorization|factorization", re.I)),
    ("Deep Clustering", re.compile(r"deep embedded clustering", re.I)),
]

DIRECTION_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("st_foundation_llm", re.compile(r"foundation model|universal model|prompt|large language|pre-?train", re.I)),
    ("diffusion_probabilistic", re.compile(r"diffusion|probabilistic|uncertainty|generative adversarial", re.I)),
    ("dynamic_graph_learning", re.compile(r"adaptive graph|dynamic graph|structure learning|hypergraph", re.I)),
    ("long_horizon_transformer", re.compile(r"transformer|long[- ]?horizon|informer|mamba", re.I)),
    ("fewshot_transfer", re.compile(r"few-?shot|meta-?learning|transfer|multi-?city|cross-?city", re.I)),
    ("privacy_federated", re.compile(r"federated|privacy", re.I)),
    ("benchmark_reproducibility", re.compile(r"benchmark|reproduc|evaluation", re.I)),
    ("imputation_data_quality", re.compile(r"imputation|missing value|sparsity", re.I)),
    ("trajectory_prediction", re.compile(r"trajector", re.I)),
    ("od_demand", re.compile(r"origin-destination|\bod\b|demand", re.I)),
]

# Curated URL corrections applied on every migration run. These encode the
# verification results from docs/evolution/10-addendum-reality-check.md:
# paperswithcode landing pages now redirect away, so a `code` link pointing there
# is either replaced with the real repository or dropped altogether.
URL_FIXES: dict[str, dict[str, Any]] = {
    "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting": {
        "code_url": "https://github.com/VeritasYin/STSGCN",
    },
    "Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction": {
        "code_url": "https://github.com/cure-lab/SCINet",
    },
    # No official release exists; dropping the broken pointer is the honest fix.
    "Spatial\u2010temporal attention wavenet: A deep learning framework for traffic prediction considering spatial\u2010temporal dependencies": {
        "code_url": None,
    },
    "Spatio-Temporal Graph Structure Learning for Traffic Forecasting": {
        "drop_alt": ["sota"],
    },
}

LINK_RE = re.compile(r"\[\[([A-Za-z0-9_-]+)\]\]\((https?://[^)\s]+)\)")
TITLE_LINK_RE = re.compile(r"^\[([^\]]{8,})\]\((https?://[^)\s]+)\)")
TAG_RE = re.compile(r"^\[(.{2,80}?)\s+((?:19|20)\d{2})\]\s*")
ITEM_RE = re.compile(r"^(\d+)\.\s+(.*)$")
YEAR_HEAD_RE = re.compile(r"^####\s+((?:19|20)\d{2})\s*$")
TOPIC = "0x00"


def slugify(text: str, max_len: int = 48) -> str:
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "paper"


def load_venues(path: Path) -> dict[str, dict[str, Any]]:
    """venue_id -> {short_name, tier, type, aliases}."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {v["id"]: v for v in data if isinstance(v, dict) and v.get("id")}


def canonical_venue(venue: str | None, venues: dict[str, dict[str, Any]]) -> tuple[str | None, str | None, str]:
    """Resolve a README venue tag through data/venues.json.

    The alias table is the single place that decides "KDD" and "ACM SIGKDD
    Conference..." are the same venue, so the migration and the enrichment pass
    cannot disagree.
    """
    if not venue:
        return None, None, "unknown"
    key = venue.strip().lower()
    for vid, entry in venues.items():
        names = [vid.lower(), entry.get("short_name", "").lower(), *[a.lower() for a in entry.get("aliases", [])]]
        if key in names and key:
            return vid, entry.get("short_name") or venue, entry.get("type", "unknown")
    # Fall back to the built-in table for anything not yet listed.
    return venue_info(venue)


def venue_info(venue: str | None) -> tuple[str | None, str | None, str]:
    if not venue:
        return None, None, "unknown"
    key = venue.strip()
    hit = VENUE_TABLE.get(key)
    if hit is None:
        # case-insensitive fallback
        for k, v in VENUE_TABLE.items():
            if k.lower() == key.lower():
                hit = v
                break
    if hit is None:
        return slugify(key, 32), slugify(key, 32), "unknown"
    return hit[0], hit[0], hit[1]


def pick_task(title: str) -> str:
    for name, rx in TASK_RULES:
        if rx.search(title):
            return name
    return "Traffic forecasting"


# Controlled tag vocabulary. Keep in sync with the one-off normalisation pass:
# a tag that is not in this list should not be produced by the rules at all.
TAG_CANON = {
    "GCN": "Graph Convolution",
    "Recurrent": "RNN / LSTM",
    "Dynamic Graph": "Adaptive Graph",
    "Graph Modeling": "STGNN",
    "Spatio-Temporal CNN": "Temporal Convolution",
    "Deep Clustering": "Clustering",
    "STGCN": None,
    "Long-Horizon": None,
    "Prompting": "Foundation Model",
}

# Titles that carry no method keyword but whose method is not in doubt.
MANUAL_TAGS: dict[str, list[str]] = {
    "Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction":
        ["Temporal Convolution", "Residual Network"],
    "Graph WaveNet for Deep Spatial-Temporal Graph Modeling":
        ["Graph Convolution", "Temporal Convolution", "Adaptive Graph", "STGNN"],
    "Urban Traffic Prediction from Spatio-Temporal Data Using Deep Meta Learning":
        ["Meta-Learning", "STGNN", "Adaptive Graph"],
}


def pick_tags(title: str) -> list[str]:
    out: list[str] = []
    for name, rx in METHOD_RULES:
        if not rx.search(title):
            continue
        mapped = TAG_CANON.get(name, name)
        if mapped and mapped not in out:
            out.append(mapped)
    if not out and title in MANUAL_TAGS:
        return list(MANUAL_TAGS[title])
    return out


def pick_short_name(title: str) -> str | None:
    """Derive a display acronym from an explicit `NAME:` prefix only.

    Free-form words are not acronyms; returning None keeps the record honest and
    lets the site fall back to the truncated title.
    """
    m = re.match(r"^([A-Za-z][A-Za-z0-9\-+]{1,23})\s*[:：]", title)
    if m:
        return m.group(1).strip()
    return None


def pick_directions(title: str) -> list[str]:
    return [name for name, rx in DIRECTION_RULES if rx.search(title)]


def extract_doi(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"doi\.org/(10\.[0-9]{4,9}/\S+)", url, re.I)
    if m:
        return m.group(1).rstrip(".").lower()
    return None


def extract_arxiv(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", url, re.I)
    if m:
        return m.group(1)
    m = re.search(r"10\.48550/arxiv\.([0-9]{4}\.[0-9]{4,5})", url, re.I)
    if m:
        return m.group(1)
    return None


def section_for(heading: str) -> str | None:
    low = heading.lower()
    for needle, section in SECTION_MAP:
        if needle in low:
            return section
    return None


def parse_entry(body: str) -> dict[str, Any] | None:
    """Parse one `[Venue Year] Title [[paper]](url) ...` line body."""
    links: dict[str, str] = {}
    for key, url in LINK_RE.findall(body):
        links.setdefault(key.lower(), url)
    stripped = LINK_RE.sub("", body).strip()

    venue = year = None
    m = TAG_RE.match(stripped)
    if m:
        venue = m.group(1).strip()
        year = int(m.group(2))
        stripped = stripped[m.end():].strip()

    # Legacy shape: `[Venue Year] [Title](url)` puts the title itself in a markdown
    # link; unwrap it so the title field never contains raw markdown.
    title_link = TITLE_LINK_RE.match(stripped)
    if title_link:
        links.setdefault("paper", title_link.group(2))
        stripped = title_link.group(1)

    title = re.sub(r"\s+", " ", stripped).strip().rstrip(".").strip()
    if len(title) < 8:
        return None
    return {"venue": venue, "year": year, "title": title, "links": links}


def iter_readme_entries(path: Path) -> list[dict[str, Any]]:
    """Yield every paper entry of the 0x00 section, skipping auto-rendered blocks."""
    text = path.read_text(encoding="utf-8")
    in_auto = False
    section: str | None = None
    sub = ""
    current_year: int | None = None
    out: list[dict[str, Any]] = []

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if line == START_RECENT:
            in_auto = True
            continue
        if line == END_RECENT:
            in_auto = False
            continue
        if line.startswith("## "):
            heading = line[3:].strip()
            section = section_for(heading) if heading.startswith(TOPIC) else None
            sub = heading
            current_year = None
            continue
        if line.startswith("### "):
            heading = line[4:].strip()
            # Topic headings ("### Deep Learning Based ...") set the section, and
            # sub-topic headings inside a topic must NOT overwrite it.
            if section_for(heading):
                section = section_for(heading)
            sub = heading
            current_year = None
            continue
        if line.startswith("#### "):
            heading = line[5:].strip()
            m_year = YEAR_HEAD_RE.match(line)
            if m_year:
                current_year = int(m_year.group(1))
            continue
        if section is None or in_auto:
            continue
        m = ITEM_RE.match(line)
        if not m:
            continue
        if m.group(2).strip().startswith(("……", "...")):
            continue
        parsed = parse_entry(m.group(2))
        if not parsed:
            continue
        parsed["section"] = section
        parsed["readme_line"] = lineno
        parsed["heading"] = sub
        parsed["year"] = parsed["year"] or current_year
        if parsed["year"] is None:
            continue
        out.append(parsed)
    return out


def load_curated(path: Path) -> dict[str, dict[str, Any]]:
    """Index updates/curated/recent-papers.json by normalized title."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    index: dict[str, dict[str, Any]] = {}
    for item in data if isinstance(data, list) else []:
        key = re.sub(r"[^a-z0-9]+", "", str(item.get("title", "")).lower())
        if key:
            index[key] = item
    return index


def existing_index(papers_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    """Index already migrated records by (year, normalized title) so ids stay stable."""
    index: dict[tuple[int, str], dict[str, Any]] = {}
    if not papers_dir.exists():
        return index
    for f in sorted(papers_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for item in data if isinstance(data, list) else []:
            key = re.sub(r"[^a-z0-9]+", "", str(item.get("title", "")).lower())
            index[(int(item.get("year", 0)), key)] = item
    return index


def build_record(
    entry: dict[str, Any],
    order: int,
    curated: dict[str, Any] | None,
    prev: dict[str, Any] | None,
    venues: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    title = entry["title"]
    year = int(entry["year"])
    links = entry["links"]
    venue = entry["venue"]

    # Apply curated URL corrections *before* deriving paper_url / code_links.
    fix = URL_FIXES.get(title)
    if fix:
        links = dict(links)
        if "code_url" in fix:
            if fix["code_url"]:
                links["code"] = fix["code_url"]
            else:
                links.pop("code", None)
        for key in fix.get("drop_alt", []):
            links.pop(key, None)

    venue_id, venue_short, venue_type = canonical_venue(venue, venues or {})
    paper_url = links.get("paper") or links.get("paper-ieee") or links.get("paper-arxiv")
    code_links = []
    for label in ("code", "code-official", "code-official-tf", "code-pytorch", "code-not-official"):
        if links.get(label):
            normalized = {
                "code": "code",
                "code-official": "code",
                "code-official-tf": "code-official-tf",
                "code-pytorch": "code-pytorch",
                "code-not-official": "code-not-official",
            }[label]
            code_links.append({"label": normalized, "url": links[label]})
    code_url = code_links[0]["url"] if code_links else None

    alt_urls = {}
    for key, url in links.items():
        if key in {"paper", "code", "code-official", "code-official-tf", "code-pytorch",
                   "code-not-official", "data"}:
            continue
        alt_urls[key] = url

    short_name = (curated.get("short_name") if curated else None) or pick_short_name(title)

    record = {
        "id": f"{slugify(short_name or title, 32)}-{year}",
        "year": year,
        "order": order,
        "section": entry["section"],
        "title": title,
        "title_zh": None,
        "venue": venue_short or venue,
        "venue_id": venue_id,
        # A venue's type comes from the venue table, not from guessing per record.
        "venue_type": (venues.get(venue_id, {}).get("type") or venue_type) if venue_id else venue_type,
        "paper_url": paper_url,
        "alt_urls": alt_urls,
        "code_url": code_url,
        "code_links": code_links,
        "data_url": links.get("data"),
        "project_url": None,
        "doi": extract_doi(paper_url),
        "arxiv_id": extract_arxiv(paper_url),
        "openalex_id": None,
        "short_name": short_name,
        "authors": [],
        "affiliations": [],
        "task_en": pick_task(title),
        "task_zh": None,
        "method_tags": pick_tags(title),
        "note_en": None,
        "note_zh": None,
        "datasets": [],
        "layers": [],
        "direction_tags": pick_directions(title),
        "featured": bool(curated.get("featured")) if curated else False,
        "in_digest": bool(curated.get("featured")) if curated else False,
        "status": "active" if paper_url else "dead_link",
        "verification": {
            "checked_at": None,
            "link_ok": None,
            "metadata_source": "readme_migration",
            "confidence": "low" if not curated else "medium",
        },
        "provenance": {
            "source": "readme_migration",
            "readme_line": entry["readme_line"],
            "legacy_id": curated.get("id") if curated else None,
            "migrated_at": None,
        },
        "created_at": None,
        "updated_at": None,
    }

    # Section-level defaults.
    if entry["section"] == "reviews":
        record["layers"] = ["survey"]
    elif entry["section"] == "deep_learning":
        record["layers"] = ["classic_baseline"] if year <= 2019 else ["sota_core"]
    else:
        record["layers"] = ["application"]

    # Curated JSON is the richer source for the fields it owns.
    if curated:
        record["task_en"] = curated.get("task_en") or record["task_en"]
        # Curator labels win over rule inference; recorded so a future run can
        # tell which tags were reviewed by a human.
        record.setdefault("provenance", {})["curated_labels"] = True
        record["task_zh"] = curated.get("task_zh")
        record["note_en"] = curated.get("note_en")
        record["note_zh"] = curated.get("note_zh")
        tags = curated.get("method_tags")
        if tags:
            record["method_tags"] = list(tags)
        if curated.get("code_url") and not record["code_url"]:
            record["code_url"] = curated["code_url"]
            record["code_links"] = [{"label": "code", "url": curated["code_url"]}]

    # Preserve everything a human or an enrichment pass already filled in.
    # Rule: the README can only *gain* a value; it never erases one that a later
    # stage (enrichment / manual curation) established. Without this the migrator
    # would silently drop DOIs, authors and verified links on every re-run.
    if prev:
        record["id"] = prev.get("id", record["id"])
        for field in ("title_zh", "note_en", "note_zh", "task_zh", "authors", "datasets",
                      "affiliations", "layers", "direction_tags", "status", "featured",
                      "in_digest", "short_name", "doi", "arxiv_id", "openalex_id",
                      "alt_years", "project_url", "data_url", "impact"):
            if prev.get(field) not in (None, [], False, {}):
                record[field] = prev[field]
        # Enrichment upgrades a provisional "Arxiv"/"preprint" to the venue the
        # work actually appeared in. The README tag is the *original* label, not
        # the current truth, so it must not undo that upgrade.
        prev_verified = (prev.get("verification") or {}).get("metadata_source") in ("openalex", "crossref")
        if prev_verified and prev.get("venue_type") in ("journal", "conference", "workshop"):
            record["venue"] = prev.get("venue") or record.get("venue")
            record["venue_id"] = prev.get("venue_id") or record.get("venue_id")
            record["venue_type"] = prev["venue_type"]
        # Labels that a curator already reviewed must not be replaced by the
        # keyword rules on a re-run.
        if (prev.get("provenance") or {}).get("curated_labels"):
            for field in ("method_tags", "task_en"):
                if prev.get(field):
                    record[field] = prev[field]
        if prev.get("alt_urls"):
            merged = dict(record.get("alt_urls") or {})
            merged.update(prev["alt_urls"])
            record["alt_urls"] = merged
        if prev.get("code_links"):
            record["code_links"] = prev["code_links"]
            record["code_url"] = prev.get("code_url") or record.get("code_url")
        for field in ("verification", "provenance", "created_at", "updated_at"):
            if prev.get(field):
                record[field] = prev[field]
        # Enriched records already point at the DOI: keep that as the primary link
        # and remember the raw README URL as the original source.
        if prev.get("paper_url") and prev["paper_url"] != record.get("paper_url"):
            record["alt_urls"] = {
                **(record.get("alt_urls") or {}),
                "paper-original": record.get("paper_url") or prev["paper_url"],
            }
            record["paper_url"] = prev["paper_url"]

    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--curated", default="updates/curated/recent-papers.json")
    ap.add_argument("--venues", default="data/venues.json")
    ap.add_argument("--out-dir", default="data/papers")
    ap.add_argument("--year", type=int, action="append", help="Only migrate this year (repeatable).")
    ap.add_argument("--all", action="store_true", help="Migrate every year found in the README.")
    ap.add_argument("--dry-run", action="store_true", help="Report without writing files.")
    ap.add_argument("--check", action="store_true",
                    help="Exit 1 if data/papers is out of sync with the README.")
    ap.add_argument("--today", default=None, help="Override the date stamped into records.")
    args = ap.parse_args()

    from datetime import date
    today = args.today or date.today().isoformat()

    entries = iter_readme_entries(Path(args.readme))
    curated = load_curated(Path(args.curated))
    venues = load_venues(Path(args.venues))
    out_dir = Path(args.out_dir)
    prev_index = existing_index(out_dir)

    per_year: dict[int, list[dict[str, Any]]] = {}
    for e in entries:
        per_year.setdefault(int(e["year"]), []).append(e)
    for year in per_year:
        per_year[year].sort(key=lambda x: x["readme_line"])

    years = sorted(per_year)
    if args.year:
        years = [y for y in years if y in set(args.year)]
    elif not args.all:
        print("[INFO] nothing selected; pass --year YYYY or --all")
        return 2

    drifted = False
    stats: Counter[str] = Counter()
    for year in years:
        records = []
        for order, entry in enumerate(per_year[year], 1):
            key = re.sub(r"[^a-z0-9]+", "", entry["title"].lower())
            c = curated.get(key)
            prev = prev_index.get((year, key))
            record = build_record(entry, order, c, prev, venues)
            if not record["provenance"].get("migrated_at"):
                record["provenance"]["migrated_at"] = today
            if not record["created_at"]:
                record["created_at"] = today
            record["updated_at"] = today
            records.append(record)
            stats["records"] += 1
            stats["with_doi"] += 1 if record["doi"] else 0
            stats["with_arxiv"] += 1 if record["arxiv_id"] else 0
            stats["with_code"] += 1 if record["code_url"] else 0
            stats["no_paper_url"] += 1 if not record["paper_url"] else 0

        target = out_dir / f"{year}.json"
        # Records the README does not own (added by the search pipeline or by
        # hand) must survive: overwriting the whole year file would silently
        # delete every paper that was never in the README.
        if target.exists():
            try:
                on_disk = json.loads(target.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                on_disk = []
            if isinstance(on_disk, list):
                owned = {
                    (int(r.get("year", 0)), re.sub(r"[^a-z0-9]+", "", str(r.get("title", "")).lower()))
                    for r in records
                }
                extras = [
                    r for r in on_disk
                    if (int(r.get("year", 0)), re.sub(r"[^a-z0-9]+", "", str(r.get("title", "")).lower()))
                    not in owned
                ]
                if extras:
                    # Keep their relative order after the README-owned block.
                    next_order = max([int(r.get("order", 0)) for r in records], default=0) + 1
                    for extra in extras:
                        extra = dict(extra)
                        if int(extra.get("order", 0)) <= next_order:
                            extra["order"] = next_order
                            next_order += 1
                        records.append(extra)
                    stats["carried_over"] += len(extras)
        rendered = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
        current = target.read_text(encoding="utf-8") if target.exists() else None
        if current != rendered:
            drifted = True
            if not args.dry_run and not args.check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(rendered, encoding="utf-8")
                print(f"[WRITE] {target} ({len(records)} records)")
            else:
                print(f"[DIFF ] {target} ({len(records)} records)")
        else:
            print(f"[SAME ] {target} ({len(records)} records)")

    print(
        f"[SUMMARY] years={len(years)} records={stats['records']} "
        f"carried_over={stats['carried_over']} "
        f"doi={stats['with_doi']} arxiv={stats['with_arxiv']} "
        f"code={stats['with_code']} no_paper_url={stats['no_paper_url']}"
    )

    if args.check:
        if drifted:
            print("[FAIL] data/papers is out of sync with README; re-run without --check.")
            return 1
        print("[OK] data/papers matches README.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())