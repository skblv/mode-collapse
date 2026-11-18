"""Score the diffuse and mode-aligned initializations for Experiment 1."""

from __future__ import annotations

import argparse

from mode_collapse.synthetic.initial_alignment import run


def main() -> None:
    """Parse arguments and write the initialization decomposition."""
    parser = argparse.ArgumentParser(
        description="Initialization decomposition for Experiment 1."
    )
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
