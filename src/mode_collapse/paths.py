"""Locations of external inputs, run outputs, and committed results.

External inputs and run outputs default to ``data/`` and ``outputs/`` in the
repository and can be moved with the environment variables
``MODE_COLLAPSE_DATA_DIR`` and ``MODE_COLLAPSE_OUTPUT_DIR``.
"""

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MODE_COLLAPSE_DATA_DIR", REPO_DIR / "data"))
OUTPUT_DIR = Path(os.environ.get("MODE_COLLAPSE_OUTPUT_DIR", REPO_DIR / "outputs"))
RESULTS_DIR = REPO_DIR / "results"
