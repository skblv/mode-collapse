"""Checks of the committed result files and the figure script that reads them."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from mode_collapse.paths import REPO_DIR, RESULTS_DIR

FIGURE_SCRIPT = REPO_DIR / "experiments" / "figures" / "make_figures.py"
FIGURE_NAMES = ["decomposition.pdf", "bound_check.pdf", "surveys.pdf", "codenet.pdf"]
RESULT_TABLES = [
    ("synthetic/diffuse_terms.csv", 100),
    ("synthetic/aligned_terms.csv", 100),
    ("synthetic/bound_per_model.csv", 100),
    ("surveys/survey_collision_ratio.csv", 460),
    ("codenet/codenet_collision_ratio.csv", 148),
]


@pytest.mark.parametrize(("relative_path", "expected_rows"), RESULT_TABLES)
def test_result_table_has_the_expected_rows(relative_path, expected_rows):
    result_df = pd.read_csv(RESULTS_DIR / relative_path)
    assert len(result_df) == expected_rows
    assert not result_df.isna().any(axis=None)


def test_make_figures_writes_every_pdf(tmp_path):
    subprocess.run(
        [sys.executable, str(FIGURE_SCRIPT), "--out-dir", str(tmp_path)],
        check=True,
        cwd=REPO_DIR,
    )
    for figure_name in FIGURE_NAMES:
        header = (tmp_path / figure_name).read_bytes()[:4]
        assert header == b"%PDF", figure_name
