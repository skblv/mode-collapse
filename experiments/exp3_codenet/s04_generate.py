#!/usr/bin/env python3
"""Sample 32 CodeNet programs per problem for one model checkpoint shard."""

from __future__ import annotations

import argparse
from pathlib import Path

from mode_collapse.codenet.config import (
    MAX_NEW_TOKENS,
    MEASURE_DIR,
    MODEL_REGISTRY,
    N_GENERATIONS,
    PROCESSED_DIR,
)
from mode_collapse.codenet.models import generate


def main() -> None:
    """Parse arguments and generate one measurement shard."""
    # setup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--checkpoint", default="base")
    parser.add_argument("--samples", type=int, default=N_GENERATIONS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--measure-dir", type=Path, default=MEASURE_DIR)
    args = parser.parse_args()

    # model fitting and everything else
    generate(args)


if __name__ == "__main__":
    main()
