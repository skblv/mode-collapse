#!/usr/bin/env python3
"""Score one CodeNet arm with the normalized Zhang–Shasha kernel.

A completed cache is reused, so an interrupted sweep can be rerun.
``--arm-index`` selects one arm of the grid in ``summary.json``, in the
order of ``scoring_arms``: every model, then base, then each training size
with ``early_stopped`` and ``fully_trained``. ``--human`` scores the
accepted-submission arm instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mode_collapse.codenet.config import (
    CHECKPOINTS,
    MEASURE_DIR,
    MODEL_SLUGS,
    PROCESSED_DIR,
    ZSS_DIR,
    scoring_arms,
)
from mode_collapse.codenet.zhang_shasha import score_human, score_model_arm


def main() -> None:
    """Parse arguments and score one human or model arm."""
    # setup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--measurement-dir", type=Path, default=MEASURE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=ZSS_DIR / "cache")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--model", choices=MODEL_SLUGS)
    parser.add_argument("--n", type=int)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS)
    parser.add_argument("--arm-index", type=int)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    # config
    if args.human:
        score_human(args.processed_dir, args.cache_dir, args.workers)
        return

    if args.arm_index is not None:
        summary = json.loads(
            (args.processed_dir / "summary.json").read_text(encoding="utf-8")
        )
        arms = scoring_arms(int(summary["n_train_rows"]))
        if not 0 <= args.arm_index < len(arms):
            raise SystemExit(
                f"--arm-index must be in 0..{len(arms) - 1}, received {args.arm_index}"
            )
        model, n, checkpoint = arms[args.arm_index]
    else:
        if args.model is None or args.n is None or args.checkpoint is None:
            raise SystemExit(
                "pass --human, --arm-index, or all of --model, --n, and --checkpoint"
            )
        model, n, checkpoint = args.model, args.n, args.checkpoint
        if checkpoint == "base" and n != 0:
            raise SystemExit("the base checkpoint requires n=0")
        if checkpoint != "base" and n <= 0:
            raise SystemExit(f"{checkpoint} requires n > 0")

    # model fitting and everything else
    score_model_arm(
        args.processed_dir,
        args.measurement_dir,
        args.cache_dir,
        model,
        n,
        checkpoint,
        args.workers,
    )


if __name__ == "__main__":
    main()
