#!/usr/bin/env python3
"""Build the accepted-Python solution split, or audit an existing split.

``--audit`` checks a split already written under ``--output-dir`` and does
not read the archive. Without it, the script streams accepted Python members
from the official archive.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mode_collapse.codenet.config import (
    CODENET_ARCHIVE,
    DATASET_ROOT,
    PROCESSED_DIR,
    SEED,
    VAL_FRACTION,
)
from mode_collapse.codenet.solution_split import (
    audit_processed_split,
    build_from_archive,
)


def main() -> None:
    """Parse arguments and build or audit the solution split."""
    # setup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--archive", type=Path, default=CODENET_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2)
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pending-batches", type=int, default=0)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--ast-sample-per-split", type=int, default=5000)
    args = parser.parse_args()

    # data import
    if args.audit:
        audit_processed_split(args.output_dir, args.ast_sample_per_split)
        return

    # data filtering
    build_from_archive(
        args.dataset_root,
        args.archive,
        args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        workers=args.workers,
        batch_size=args.batch_size,
        pending_batches=args.pending_batches,
    )


if __name__ == "__main__":
    main()
