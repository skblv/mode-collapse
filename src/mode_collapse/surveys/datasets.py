"""Survey names, raw-file locations, and output paths for Experiment 2."""

from __future__ import annotations

import json

from mode_collapse.paths import DATA_DIR, OUTPUT_DIR
from mode_collapse.surveys import anes, gss, wvs

DATASETS = ("gss", "wvs", "anes")
MODEL_SLUGS = (
    "gemma-2-2b-it",
    "qwen3.5-2b",
    "qwen3.5-4b",
    "gemma-3-4b-it",
)
RAW_FILENAMES = {
    "gss": "gss7224_r3.dta",
    "wvs": "WVS_Cross-National_Wave_7_csv_v6_0.csv",
    "anes": "anes_timeseries_cdf_csv_20260205.csv",
}
DATASET_MODULES = {
    "gss": gss,
    "wvs": wvs,
    "anes": anes,
}
HALVING_SHIFTS = 20
MIN_HALVING_N = 8


def dataset_module(name: str):
    """Returns the survey module for ``name``.

    Args:
        name: ``gss``, ``wvs``, or ``anes``.

    Returns:
        The module that defines that survey's items and prompts.

    Raises:
        KeyError: If ``name`` is not one of the three surveys.
    """
    if name not in DATASET_MODULES:
        known = ", ".join(DATASETS)
        raise KeyError(f"unknown survey {name!r}; expected one of {known}")
    return DATASET_MODULES[name]


def raw_survey_path(name: str):
    """Returns the raw survey path, or raises if the file is absent.

    Args:
        name: ``gss``, ``wvs``, or ``anes``.

    Returns:
        ``DATA_DIR / "surveys" / <original filename>``.

    Raises:
        FileNotFoundError: If the raw file is not at that path. Survey
            microdata are not in the repository.
    """
    path = DATA_DIR / "surveys" / RAW_FILENAMES[name]
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw survey file not found at {path}. Place {RAW_FILENAMES[name]} "
            "there. Survey microdata are not redistributable and are not in "
            "the repository."
        )
    return path


def survey_output_dir():
    """Returns the Experiment 2 output directory."""
    return OUTPUT_DIR / "surveys"


def processed_dir(name: str):
    """Returns the processed-target directory for one survey."""
    return survey_output_dir() / name / "processed"


def summary_path(name: str):
    """Returns the ground-truth summary path for one survey.

    GSS keeps the unsuffixed filename. WVS and ANES keep the suffixed
    filenames used when those summaries were written.
    """
    if name == "gss":
        filename = "ground_truth_summary.json"
    else:
        filename = f"ground_truth_summary_{name}.json"
    return survey_output_dir() / name / filename


def item_config_path(name: str):
    """Returns the item-config JSON path for one survey."""
    return processed_dir(name) / "item_config.json"


def checkpoints_dir():
    """Returns the adapter checkpoint directory."""
    return survey_output_dir() / "checkpoints"


def measurements_dir():
    """Returns the per-pair measurement directory."""
    return survey_output_dir() / "measurements"


def aggregate_csv_path():
    """Returns the collision-ratio aggregate CSV path."""
    return survey_output_dir() / "survey_collision_ratio.csv"


def read_train_pool_size(name: str) -> int:
    """Reads the training-row count from a survey's ground-truth summary.

    Args:
        name: ``gss``, ``wvs``, or ``anes``.

    Returns:
        ``n_train_rows`` from that summary.

    Raises:
        FileNotFoundError: If the summary has not been built.
    """
    path = summary_path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Ground-truth summary not found at {path}. "
            "Build targets before fine-tuning."
        )
    with path.open() as handle:
        summary = json.load(handle)
    return int(summary["n_train_rows"])


def halving_grid(n_total: int) -> list[int]:
    """Returns fine-tuning sizes from ``n_total`` down to the smallest n >= 8.

    Args:
        n_total: Training-pool size. In the paper the pools are 4,706,940 (GSS),
            1,462,645 (WVS), and 1,662,788 (ANES).

    Returns:
        ``n_total >> shift`` for shift = 0, 1, ... while the size is at least
        8, and at most 20 sizes. Order is largest to smallest.
    """
    return [
        n_total >> shift
        for shift in range(HALVING_SHIFTS)
        if (n_total >> shift) >= MIN_HALVING_N
    ]
