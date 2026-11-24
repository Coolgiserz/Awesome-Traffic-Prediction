#!/usr/bin/env python3
"""Catch venue labels that contradict the link they point at.

A venue tag is an inference. When the primary link is a DOI, the DOI prefix or
slug usually names the actual venue, so a mismatch means either the label or the
link is wrong — and both are worse than a missing field.

Example of what this catches: a record labelled `WWW` whose DOI is
`10.1109/iwcmc55113.2022.9825059` (an IWCMC proceedings paper). The rule that
produced it matched the substring "www" inside "iwcmc".

The check is deliberately conservative: it only reports a contradiction when a
known venue token appears in the DOI slug and a *different* known token is the
record's venue. Unknown venues are skipped rather than guessed at.

    python3 scripts/check_venue_consistency.py            # report
    python3 scripts/check_venue_consistency.py --strict   # non-zero on findings
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

# Tokens that unambiguously identify a venue inside a DOI slug or container name.
DOI_TOKENS = {
    "3219819": "KDD", "3292500": "KDD", "3583780": "CIKM",
    "3308558": "WWW", "3447548": "KDD", "3366423": "WWW",
    "iwcmc": "IWCMC", "icde": "ICDE", "icdm": "ICDM",
    "ijcai": "IJCAI", "aaai": "AAAI", "neurips": "NeurIPS",
    "trc": "TR-C", "tits": "TITS", "tkde": "TKDE",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/papers")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    findings: list[tuple[str, str, str, str]] = []
    for path in sorted(glob.glob(f"{args.data_dir}/*.json")):
        for rec in json.load(open(path, encoding="utf-8")):
            url = (rec.get("paper_url") or "").lower()
            venue = (rec.get("venue") or "").upper()
            if not url or not venue:
                continue
            for token, implied in DOI_TOKENS.items():
                if token not in url:
                    continue
                if venue != implied.upper():
                    findings.append((rec.get("id", "?"), venue, implied, url[-60:]))
                break

    if findings:
        print(f"[FAIL] {len(findings)} record(s) have a venue that contradicts their link:")
        for rid, venue, implied, tail in findings:
            print(f"- {rid}: labelled {venue}, link suggests {implied} ({tail})")
        return 1 if args.strict else 0

    print("[OK] venue labels agree with their links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
