"""Train or reduce the diffuse-prior decomposition grid for Experiment 1."""

from __future__ import annotations

import argparse

from mode_collapse.synthetic.diffuse_grid import N_LEVELS, run


def main() -> None:
    """Parse arguments and run one shard or the reduction."""
    parser = argparse.ArgumentParser(
        description="Diffuse-prior decomposition grid for Experiment 1."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny end-to-end smoke configuration.",
    )
    parser.add_argument(
        "--reduce",
        action="store_true",
        help="Concatenate part files and write the decomposition table.",
    )
    parser.add_argument(
        "--level-start",
        type=int,
        default=0,
        help="First sample-size index for this shard.",
    )
    parser.add_argument(
        "--level-count",
        type=int,
        default=N_LEVELS,
        help="Number of sample sizes for this shard.",
    )
    parser.add_argument(
        "--models-per-gpu",
        type=int,
        default=1,
        help="Models trained concurrently per GPU.",
    )
    args = parser.parse_args()
    run(
        smoke=args.smoke,
        reduce=args.reduce,
        level_start=args.level_start,
        level_count=args.level_count,
        models_per_gpu=args.models_per_gpu,
    )


if __name__ == "__main__":
    main()
