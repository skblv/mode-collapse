#!/usr/bin/env python3
"""Summarize Zhang–Shasha caches into collision-ratio tables.

Writes ``codenet_collision_ratio.csv``, the per-problem file beside it, and
``codenet_same_problem_comparison.csv``. Existing tables are left in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mode_collapse.codenet.config import (
    CODENET_OUTPUT_DIR,
    PER_PROBLEM_PATH,
    PROCESSED_DIR,
    SAME_PROBLEM_PATH,
    SUMMARY_PATH,
    ZSS_DIR,
)
from mode_collapse.codenet.convergence import summarize_caches, write_collision_tables


def main() -> None:
    """Parse arguments and write the collision-ratio tables."""
    # setup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--cache-dir", type=Path, default=ZSS_DIR / "cache")
    parser.add_argument("--output-dir", type=Path, default=CODENET_OUTPUT_DIR)
    args = parser.parse_args()

    # data import
    per_problem_rows, summary_rows, comparisons = summarize_caches(
        args.processed_dir, args.cache_dir
    )

    # data export
    write_collision_tables(
        per_problem_rows,
        summary_rows,
        comparisons,
        per_problem_path=args.output_dir / PER_PROBLEM_PATH.name,
        summary_path=args.output_dir / SUMMARY_PATH.name,
        comparison_path=args.output_dir / SAME_PROBLEM_PATH.name,
    )


if __name__ == "__main__":
    main()
