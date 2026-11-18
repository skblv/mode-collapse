#!/usr/bin/env python3
"""LoRA-fine-tune one model on one point of the CodeNet halving grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from mode_collapse.codenet.config import (
    CHECKPOINTS_DIR,
    MODEL_REGISTRY,
    PROCESSED_DIR,
    SEED,
)
from mode_collapse.codenet.models import finetune


def main() -> None:
    """Parse arguments and run one fine-tuning job."""
    # setup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--run-suffix", default="")
    args = parser.parse_args()

    # model fitting and everything else
    finetune(args)


if __name__ == "__main__":
    main()
