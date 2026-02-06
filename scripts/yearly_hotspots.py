#!/usr/bin/env python3
"""Extract yearly hotspots from markdown paper lists.

Outputs:
- yearly-hotspots.json
- yearly-hotspots.md
- yearly-hotspots.png
- yearly-hotspots-trend.png
- yearly-theme-trends.png
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "its",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "via",
    "with",
}

DEFAULT_NOISE = {
    "approach",
    "approaches",
    "method",
    "methods",
    "model",
    "models",
    "framework",
    "frameworks",
    "study",
    "analysis",
    "data",
    "learning",
    "deep",
    "neural",
    "network",
    "networks",
    "system",
    "systems",
    "novel",
    "new",
    "enhanced",
    "adaptive",
    "dynamic",
    "real",
    "time",
    "short",
    "term",
    "long",
    "paper",
    "traffic",
    "forecasting",
    "prediction",
}

DEFAULT_PHRASES = [
    "traffic flow",
    "traffic speed",
    "spatio temporal",
    "short term",
    "long horizon",
    "graph neural network",
    "graph convolutional network",
    "graph transformer",
    "federated learning",
    "contrastive learning",
    "probabilistic forecasting",
    "information bottleneck",
    "few shot",
    "transfer learning",
]

PHRASE_MAP = {
    "spatio temporal": "spatiotemporal",
    "graph neural network": "gnn",
    "graph convolutional network": "gcn",
    "few shot": "few-shot",
}


def parse_candidates(md_path: Path) -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    text = md_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        # Works for:
        # 1. [Venue 2024] Title [[paper]](...)
        # 1. [Venue 2024] [Title](...)  (rare in README)
        m = re.match(r"^\s*\d+\.\s+\[([^\]]+)\]\s+(.+?)\s+\[\[paper\]\]\(", line)
        if not m:
            m = re.match(r"^\s*\d+\.\s+\[([^\]]+)\]\s+\[(.+?)\]\(", line)
        if not m:
            continue
        venue_year = m.group(1)
        title = m.group(2).strip()
        y = re.search(r"(19|20)\d{2}", venue_year)
        if not y:
            continue
        items.append((int(y.group(0)), title))
    return items


def tokenize(title: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", title.lower())
    out: list[str] = []
    for t in tokens:
        t = t.strip("-")
        if len(t) <= 2:
            continue
        out.append(t)
    return out


def extract_phrases(title: str) -> list[str]:
    s = re.sub(r"[^a-zA-Z0-9\s\-]", " ", title.lower())
    s = re.sub(r"\s+", " ", s).strip()
    out: list[str] = []
    for p in DEFAULT_PHRASES:
        hits = len(re.findall(rf"\b{re.escape(p)}\b", s))
        if hits:
            mapped = PHRASE_MAP.get(p, p)
            out.extend([mapped] * hits)
    return out


def build_year_counters(
    items: list[tuple[int, str]],
    extra_noise: set[str],
) -> dict[int, Counter]:
    stop = DEFAULT_STOPWORDS | DEFAULT_NOISE | extra_noise
    by_year: dict[int, Counter] = defaultdict(Counter)
    for year, title in items:
        for tk in tokenize(title):
            if tk in stop:
                continue
            by_year[year][tk] += 1
        for ph in extract_phrases(title):
            if ph in stop:
                continue
            by_year[year][ph] += 1
    return dict(by_year)


def write_markdown(year_counters: dict[int, Counter], out_md: Path, topk: int) -> None:
    lines = [
        "# Yearly Paper Hotspots",
        "",
        f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]
    for year in sorted(year_counters):
        lines.append(f"## {year}")
        lines.append("")
        top_terms = year_counters[year].most_common(topk)
        if not top_terms:
            lines.append("- No terms")
            lines.append("")
            continue
        for i, (term, cnt) in enumerate(top_terms, 1):
            lines.append(f"{i}. `{term}` ({cnt})")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def write_json(year_counters: dict[int, Counter], out_json: Path, topk: int) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "topk": topk,
        "years": {
            str(y): [{"term": t, "count": c} for t, c in year_counters[y].most_common(topk)]
            for y in sorted(year_counters)
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_png(year_counters: dict[int, Counter], out_png: Path, topk: int) -> None:
    years = sorted(year_counters)
    if not years:
        return
    n = len(years)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]
    for ax, y in zip(axes, years):
        terms = year_counters[y].most_common(topk)
        labels = [t for t, _ in terms][::-1]
        values = [c for _, c in terms][::-1]
        ax.barh(labels, values)
        ax.set_title(f"{y} Top Terms")
        ax.set_xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def write_trend_heatmap(year_counters: dict[int, Counter], out_png: Path, topk: int) -> None:
    years = sorted(year_counters)
    if not years:
        return

    global_counter = Counter()
    for y in years:
        global_counter.update(year_counters[y])
    terms = [t for t, _ in global_counter.most_common(topk)]
    if not terms:
        return

    data = []
    for t in terms:
        row = [year_counters[y].get(t, 0) for y in years]
        data.append(row)

    plt.figure(figsize=(max(10, len(years) * 0.9), max(6, len(terms) * 0.45)))
    plt.imshow(data, aspect="auto")
    plt.colorbar(label="Frequency")
    plt.yticks(range(len(terms)), terms)
    plt.xticks(range(len(years)), years, rotation=45, ha="right")
    plt.title("Yearly Hotspot Trend (Top Terms Across All Years)")
    plt.xlabel("Year")
    plt.ylabel("Term")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def detect_themes(title: str) -> set[str]:
    t = title.lower()
    themes: set[str] = set()
    if any(k in t for k in ["graph neural", "gnn", "stgcn", "graph convolution"]):
        themes.add("GNN")
    if "transformer" in t or "attention" in t:
        themes.add("Transformer/Attention")
    if "federated" in t:
        themes.add("Federated")
    if any(k in t for k in ["transfer", "few-shot", "few shot", "meta-learning", "meta learning"]):
        themes.add("Transfer/Generalization")
    if any(k in t for k in ["probabilistic", "uncertainty", "bayesian"]):
        themes.add("Probabilistic")
    if any(k in t for k in ["physics-informed", "physics informed"]):
        themes.add("Physics-Informed")
    return themes


def write_theme_trends(items: list[tuple[int, str]], out_png: Path) -> None:
    years = sorted({y for y, _ in items})
    if not years:
        return
    theme_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for y, title in items:
        for th in detect_themes(title):
            theme_counts[th][y] += 1
    if not theme_counts:
        return

    # Keep most frequent themes for readability.
    ranked = sorted(
        theme_counts.keys(),
        key=lambda th: sum(theme_counts[th].values()),
        reverse=True,
    )[:8]

    plt.figure(figsize=(12, 6))
    for th in ranked:
        ys = [theme_counts[th].get(y, 0) for y in years]
        plt.plot(years, ys, marker="o", linewidth=2, label=th)
    plt.title("Yearly Theme Trends")
    plt.xlabel("Year")
    plt.ylabel("Paper Count")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract yearly hotspots from paper markdown list.")
    ap.add_argument("--input", required=True, help="Path to markdown list source (README.md or monthly candidates)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--topk", type=int, default=12, help="Top terms per year")
    ap.add_argument("--extra-noise", default="", help="Comma separated extra noise words")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_noise = {x.strip().lower() for x in args.extra_noise.split(",") if x.strip()}

    items = parse_candidates(in_path)
    year_counters = build_year_counters(items, extra_noise)

    out_md = out_dir / "yearly-hotspots.md"
    out_json = out_dir / "yearly-hotspots.json"
    out_png = out_dir / "yearly-hotspots.png"
    out_trend_png = out_dir / "yearly-hotspots-trend.png"
    out_theme_png = out_dir / "yearly-theme-trends.png"
    write_markdown(year_counters, out_md, args.topk)
    write_json(year_counters, out_json, args.topk)
    write_png(year_counters, out_png, args.topk)
    write_trend_heatmap(year_counters, out_trend_png, args.topk)
    write_theme_trends(items, out_theme_png)

    print(f"Saved: {out_md}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_trend_png}")
    print(f"Saved: {out_theme_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
