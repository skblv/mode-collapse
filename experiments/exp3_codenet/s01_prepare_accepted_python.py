#!/usr/bin/env python3
"""List every accepted Python source member in the official CodeNet metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from mode_collapse.codenet.config import DATASET_ROOT
from mode_collapse.codenet.solution_split import prepare_accepted_python


def main() -> None:
    """Parse arguments and write the accepted-Python member list."""
    # setup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--member-list", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--verify-extracted", action="store_true")
    args = parser.parse_args()

    # data import
    # data filtering
    prepare_accepted_python(
        args.dataset_root,
        args.member_list,
        args.summary,
        verify_extracted=args.verify_extracted,
    )


if __name__ == "__main__":
    main()
