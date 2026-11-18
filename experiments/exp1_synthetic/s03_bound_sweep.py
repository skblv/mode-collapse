"""Train the diffuse-prior bound sweep for Experiment 1."""

from __future__ import annotations

import argparse

from mode_collapse.synthetic.bound_sweep import run


def main() -> None:
    """Parse arguments and run the bound sweep."""
    parser = argparse.ArgumentParser(description="Ell-2 bound sweep for Experiment 1.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny end-to-end smoke configuration.",
    )
    args = parser.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
