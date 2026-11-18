#!/usr/bin/env python3
"""Write the appendix source files and their complete canonical AST listings."""

from __future__ import annotations

import argparse
from pathlib import Path

from mode_collapse.codenet.canonical_ast import export_paper_ast_examples


def main() -> None:
    """Parse arguments and export the two selected pairs."""
    # setup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # data export
    export_paper_ast_examples(args.candidates, args.output_dir)


if __name__ == "__main__":
    main()
