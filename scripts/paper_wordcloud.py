#!/usr/bin/env python3
"""Generate professional keyword visuals from markdown paper titles.

Input format:
- Markdown list lines containing paper titles and links.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
from wordcloud import WordCloud


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

# Generic noise words often present in paper titles.
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
}

DEFAULT_PHRASES = [
    "traffic flow",
    "traffic speed",
    "traffic forecasting",
    "traffic prediction",
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
    "traffic prediction": "traffic forecasting",
    "spatio temporal": "spatiotemporal",
    "graph neural network": "gnn",
    "graph convolutional network": "gcn",
    "few shot": "few-shot",
}

TOKEN_MAP = {
    "forecast": "forecasting",
    "predicting": "prediction",
    "predictions": "prediction",
    "graphs": "graph",
    "transformers": "transformer",
}


def read_titles(md_path: Path) -> List[str]:
    text = md_path.read_text(encoding="utf-8")
    titles: List[str] = []
    for line in text.splitlines():
        # Matches: 1. [Venue Year] Title [[paper]](...)
        m = re.match(r"^\s*\d+\.\s+\[[^\]]+\]\s+(.+?)\s+\[\[paper\]\]\(", line)
        if m:
            titles.append(m.group(1).strip())
    return titles


def tokenize(texts: Iterable[str]) -> List[str]:
    tokens: List[str] = []
    for t in texts:
        parts = re.findall(r"[a-zA-Z][a-zA-Z\-]+", t.lower())
        for p in parts:
            p = p.strip("-")
            if len(p) <= 2:
                continue
            tokens.append(p)
    return tokens


def normalize_token(token: str) -> str:
    return TOKEN_MAP.get(token, token)


def extract_phrases(texts: Iterable[str], phrases: List[str]) -> Counter:
    c = Counter()
    for t in texts:
        s = re.sub(r"[^a-zA-Z0-9\s\-]", " ", t.lower())
        s = re.sub(r"\s+", " ", s).strip()
        for p in phrases:
            p_norm = p.lower().strip()
            hits = len(re.findall(rf"\b{re.escape(p_norm)}\b", s))
            if hits:
                c[PHRASE_MAP.get(p_norm, p_norm)] += hits
    return c


def build_counter(tokens: Iterable[str], stopwords: set[str]) -> Counter:
    c = Counter()
    for tk in tokens:
        tk = normalize_token(tk)
        if tk in stopwords:
            continue
        c[tk] += 1
    return c


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate word cloud from markdown paper list.")
    ap.add_argument("--input", required=True, help="Input markdown file path.")
    ap.add_argument("--out-dir", required=True, help="Output directory path.")
    ap.add_argument("--prefix", default="monthly-paper", help="Output filename prefix.")
    ap.add_argument(
        "--extra-noise",
        default="",
        help="Comma-separated extra noise words to remove.",
    )
    ap.add_argument(
        "--extra-phrases",
        default="",
        help="Comma-separated extra domain phrases, e.g. 'meta learning,domain adaptation'.",
    )
    ap.add_argument("--bar-topk", type=int, default=15, help="Top-N terms for bar chart.")
    ap.add_argument("--topk", type=int, default=80, help="Max words to include.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    titles = read_titles(in_path)
    tokens = tokenize(titles)
    extra_phrases = [x.strip().lower() for x in args.extra_phrases.split(",") if x.strip()]
    extra_noise = {x.strip().lower() for x in args.extra_noise.split(",") if x.strip()}
    stopwords = set(DEFAULT_STOPWORDS) | set(DEFAULT_NOISE) | extra_noise
    token_freq = build_counter(tokens, stopwords)
    phrase_freq = extract_phrases(titles, DEFAULT_PHRASES + extra_phrases)
    freq = token_freq + phrase_freq

    if not freq:
        raise SystemExit("No valid tokens after filtering.")

    top_items = dict(freq.most_common(args.topk))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m")
    latest_png = out_dir / f"{args.prefix}-latest.png"
    dated_png = out_dir / f"{args.prefix}-{stamp}.png"
    latest_json = out_dir / f"{args.prefix}-latest.json"
    latest_bar = out_dir / f"{args.prefix}-topk-latest.png"

    wc = WordCloud(
        width=1800,
        height=1000,
        background_color="white",
        colormap="magma",
        prefer_horizontal=0.9,
        collocations=False,
        max_words=args.topk,
    )
    wc.generate_from_frequencies(top_items)

    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(latest_png, dpi=200)
    plt.savefig(dated_png, dpi=200)
    plt.close()

    # Professional companion view: top terms bar chart.
    bar_items = freq.most_common(args.bar_topk)
    labels = [x[0] for x in bar_items][::-1]
    values = [x[1] for x in bar_items][::-1]
    plt.figure(figsize=(12, 8))
    plt.barh(labels, values)
    plt.title("Top Terms in Monthly Paper Titles")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(latest_bar, dpi=180)
    plt.close()

    latest_json.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "input": str(in_path),
                "phrase_rules": DEFAULT_PHRASES + extra_phrases,
                "top_words": freq.most_common(100),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved: {latest_png}")
    print(f"Saved: {dated_png}")
    print(f"Saved: {latest_bar}")
    print(f"Saved: {latest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
