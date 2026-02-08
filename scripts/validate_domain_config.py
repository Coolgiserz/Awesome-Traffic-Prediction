#!/usr/bin/env python3
"""Validate domain config v2 for paper_search.py."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import paper_search


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate domain config v2.")
    ap.add_argument("config", help="Config JSON path")
    ap.add_argument("--summary", action="store_true", help="Print normalized summary")
    args = ap.parse_args()

    path = Path(args.config)
    if not path.exists():
        print(f"[FAIL] config not found: {path}", file=sys.stderr)
        return 2

    try:
        cfg = paper_search.load_config(str(path), datetime.now(timezone.utc).year)
    except Exception as exc:
        print(f"[FAIL] invalid config: {exc}", file=sys.stderr)
        return 1

    rule_ids = [r.rule_id for r in cfg.rules]
    dup_ids = sorted({x for x in rule_ids if rule_ids.count(x) > 1})
    if dup_ids:
        print(f"[FAIL] duplicate rule id(s): {', '.join(dup_ids)}", file=sys.stderr)
        return 1

    print(
        "[OK] %s version=2 years=%d-%d queries=%d rules=%d output_limit=%d"
        % (path, cfg.year_from, cfg.year_to, len(cfg.queries), len(cfg.rules), cfg.output_limit)
    )

    if args.summary:
        payload = {
            "queries": cfg.queries,
            "sources": sorted(cfg.sources),
            "year_from": cfg.year_from,
            "year_to": cfg.year_to,
            "min_score": cfg.min_score,
            "rank": {
                "recency_weight": cfg.rank.recency_weight,
                "source_weights": cfg.rank.source_weights,
                "tie_breakers": cfg.rank.tie_breakers,
            },
            "runtime": {
                "max_requests": cfg.runtime.max_requests,
                "min_interval_seconds": cfg.runtime.min_interval_seconds,
                "max_retries": cfg.runtime.max_retries,
                "retry_base_seconds": cfg.runtime.retry_base_seconds,
                "cache_dir": cfg.runtime.cache_dir,
                "cache_ttl_hours": cfg.runtime.cache_ttl_hours,
                "source_min_interval_seconds": cfg.runtime.source_min_interval_seconds,
            },
            "rules": [
                {
                    "id": r.rule_id,
                    "field": r.field,
                    "op": r.op,
                    "required": r.required,
                    "exclude": r.exclude,
                    "weight": r.weight,
                    "keywords": r.keywords,
                }
                for r in cfg.rules
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
