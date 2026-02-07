#!/usr/bin/env python3
"""Check structural consistency between README.md and README.zh.md."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def find_section(lines: list[str], headings: tuple[str, ...]) -> int:
    for i, line in enumerate(lines):
        s = line.strip()
        for h in headings:
            if s == h or s.startswith(h):
                return i
    return -1


def get_section_block(lines: list[str], start_idx: int) -> list[str]:
    if start_idx < 0:
        return []
    out: list[str] = []
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if line.startswith("### "):
            break
        out.append(line)
    return out


def parse_year_counts(lines: list[str], deep_learning_headings: tuple[str, ...]) -> dict[int, int]:
    start_idx = find_section(lines, deep_learning_headings)
    if start_idx < 0:
        return {}

    counts: dict[int, int] = {}
    current_year: int | None = None

    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith("### ") and not line.startswith("#### "):
            break

        m_year = re.match(r"^####\s+(20\d{2})\s*$", line)
        if m_year:
            current_year = int(m_year.group(1))
            counts[current_year] = 0
            continue

        if current_year is not None and re.match(r"^\d+\.\s+\[", line):
            counts[current_year] += 1

    return counts


def count_table_rows(lines: list[str], headings: tuple[str, ...]) -> int:
    start_idx = find_section(lines, headings)
    block = get_section_block(lines, start_idx)
    table_lines = [line.strip() for line in block if line.strip().startswith("|")]
    if len(table_lines) < 3:
        return 0

    # Expected markdown table shape:
    # header row
    # separator row like | --- | --- |
    # data rows...
    sep = table_lines[1].replace("|", "").replace(" ", "")
    if not sep or any(ch not in "-:" for ch in sep):
        return 0
    return max(0, len(table_lines) - 2)


def report_year_diffs(en_counts: dict[int, int], zh_counts: dict[int, int]) -> list[str]:
    errs: list[str] = []
    all_years = sorted(set(en_counts) | set(zh_counts))
    for y in all_years:
        e = en_counts.get(y, -1)
        z = zh_counts.get(y, -1)
        if e != z:
            errs.append(f"Year {y}: README.md={e}, README.zh.md={z}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description="Check README bilingual structural sync.")
    ap.add_argument("--en", default="README.md", help="Path to English README")
    ap.add_argument("--zh", default="README.zh.md", help="Path to Chinese README")
    args = ap.parse_args()

    en_path = Path(args.en)
    zh_path = Path(args.zh)
    en_lines = read_lines(en_path)
    zh_lines = read_lines(zh_path)

    errors: list[str] = []

    en_years = parse_year_counts(en_lines, ("### Deep Learning Based Traffic Prediction Methods",))
    zh_years = parse_year_counts(zh_lines, ("### 基于深度学习的交通预测方法",))
    if not en_years or not zh_years:
        errors.append("Missing deep-learning yearly sections in one of the README files.")
    else:
        errors.extend(report_year_diffs(en_years, zh_years))

    en_digest_rows = count_table_rows(en_lines, ("### Curated Digest",))
    zh_digest_rows = count_table_rows(zh_lines, ("### 重点整理",))
    if en_digest_rows != zh_digest_rows:
        errors.append(
            f"Curated digest row mismatch: README.md={en_digest_rows}, README.zh.md={zh_digest_rows}"
        )

    en_bench_rows = count_table_rows(en_lines, ("### Benchmark Matrix",))
    zh_bench_rows = count_table_rows(zh_lines, ("### 基准矩阵",))
    if en_bench_rows != zh_bench_rows:
        errors.append(
            f"Benchmark matrix row mismatch: README.md={en_bench_rows}, README.zh.md={zh_bench_rows}"
        )

    if errors:
        print("[FAIL] README sync check failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("[OK] README sync check passed.")
    print(f"- Year sections checked: {len(en_years)}")
    print(f"- Curated digest rows: {en_digest_rows}")
    print(f"- Benchmark matrix rows: {en_bench_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
