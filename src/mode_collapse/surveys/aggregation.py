"""Quantiles of the survey collision ratio for Experiment 2.

For each question-group pair the human target is the distinct-pair U-statistic
``C_U = (m C_plug - 1) / (m - 1)``, with ``C_plug = 1 / N2_p`` the stored
plug-in collision. The model collision is ``C(q) = 1 / N2_q``. The reported
ratio is ``R_U = C(q) / C_U``, formed on every pair before any quantile is
taken. ``n_pairs`` is the number of pairs that enter those quantiles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from mode_collapse.surveys.datasets import DATASETS, MODEL_SLUGS

QUANTILE_LEVELS = (
    ("q05", 0.05),
    ("q25", 0.25),
    ("median", 0.5),
    ("q75", 0.75),
    ("q95", 0.95),
)
AGGREGATE_COLUMNS = [
    "dataset",
    "model",
    "n",
    "checkpoint",
    "n_pairs",
    "collision_ratio_q05",
    "collision_ratio_q25",
    "collision_ratio_median",
    "collision_ratio_q75",
    "collision_ratio_q95",
]
PAIR_COLUMNS = ["n_cell", "N2_p", "N2_q", "n", "checkpoint"]
SWEEP = [(dataset, model) for dataset in DATASETS for model in MODEL_SLUGS]


def pair_collision_ratio(pair_measure_df: pd.DataFrame) -> pd.Series:
    """Returns ``R_U = C(q) / C_U`` for each measurement row.

    Args:
        pair_measure_df: Rows with ``n_cell``, ``N2_p``, and ``N2_q``.

    Returns:
        Per-pair collision ratio. ``C_U`` is the distinct-pair U-statistic
        recovered from the plug-in inverse Simpson index.
    """
    n_cell = pair_measure_df["n_cell"].astype(float)
    n2_p = pair_measure_df["N2_p"].astype(float)
    n2_q = pair_measure_df["N2_q"].astype(float)
    assert (n_cell >= 2).all() and (n_cell > n2_p).all()
    assert (n2_p > 0).all() and (n2_q > 0).all()
    n2_p_unbiased = (n_cell - 1) * n2_p / (n_cell - n2_p)
    collision_ratio = n2_p_unbiased / n2_q
    assert np.isfinite(collision_ratio.to_numpy()).all()
    return collision_ratio


def summarize_collision_ratios(pair_measure_df: pd.DataFrame) -> pd.DataFrame:
    """Quantiles of per-pair ``R_U`` for one survey and model.

    Args:
        pair_measure_df: Measurement rows. ``collision_ratio`` is computed
            here; a pre-existing column of that name is not read.

    Returns:
        One row per ``(n, checkpoint)``. ``n_pairs`` is the number of pairs
        in that group, which is the set of pairs whose ratios are quantile'd.
    """
    pair_ratio_df = pair_measure_df.assign(
        collision_ratio=pair_collision_ratio(pair_measure_df)
    )
    rows = []
    arm_groups = pair_ratio_df.groupby(["n", "checkpoint"])
    for (n_train, checkpoint), pair_ratio_arm_df in arm_groups:
        levels = [level for _, level in QUANTILE_LEVELS]
        quantiles = pair_ratio_arm_df["collision_ratio"].quantile(levels)
        row = {
            "n": int(n_train),
            "checkpoint": checkpoint,
            "n_pairs": len(pair_ratio_arm_df),
        }
        for suffix, level in QUANTILE_LEVELS:
            row[f"collision_ratio_{suffix}"] = float(quantiles.loc[level])
        rows.append(row)
    summary_columns = [
        "n",
        "checkpoint",
        "n_pairs",
        "collision_ratio_q05",
        "collision_ratio_q25",
        "collision_ratio_median",
        "collision_ratio_q75",
        "collision_ratio_q95",
    ]
    return pd.DataFrame(rows)[summary_columns]


def aggregate_measurements(measure_dir: Path) -> pd.DataFrame:
    """Aggregates every survey measurement CSV in ``measure_dir``.

    Args:
        measure_dir: Directory of ``measure_{dataset}_{model}_n*_{checkpoint}.csv``
            files. All three surveys and four model slugs must be present.

    Returns:
        One row per survey, model, sample size, and checkpoint, with the
        collision-ratio quantiles.
    """
    arm_ratio_frames = []
    for dataset, model in SWEEP:
        pattern = f"measure_{dataset}_{model}_n*_*.csv"
        files = sorted(measure_dir.glob(pattern))
        assert files, (dataset, model)
        pair_measure_frames = []
        for path in tqdm(files, desc=f"{dataset}/{model}"):
            pair_measure_file_df = pd.read_csv(path, usecols=PAIR_COLUMNS)
            pair_measure_frames.append(pair_measure_file_df)
        pair_measure_df = pd.concat(pair_measure_frames, ignore_index=True)
        arm_ratio_combo_df = summarize_collision_ratios(pair_measure_df)
        arm_ratio_combo_df.insert(0, "model", model)
        arm_ratio_combo_df.insert(0, "dataset", dataset)
        print(
            f"{dataset}/{model}: {len(pair_measure_df):,} pairs, {len(files)} files",
            flush=True,
        )
        arm_ratio_frames.append(arm_ratio_combo_df)
    arm_ratio_df = pd.concat(arm_ratio_frames, ignore_index=True)
    key_columns = ["dataset", "model", "n", "checkpoint"]
    assert not arm_ratio_df.duplicated(key_columns).any()
    return arm_ratio_df[AGGREGATE_COLUMNS]


def final_adapter_table(arm_ratio_df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes final-adapter medians the way the appendix table does.

    Args:
        arm_ratio_df: Output of :func:`aggregate_measurements`. Every
            survey and model needs a ``base`` row at ``n = 0`` and at least
            one ``fully_trained`` row.

    Returns:
        One row per survey and model. ``max_over_grid_final`` is the largest
        final-adapter median collision ratio on the sample-size grid.
    """
    rows = []
    for dataset, model in SWEEP:
        is_combo = (arm_ratio_df["dataset"] == dataset) & (
            arm_ratio_df["model"] == model
        )
        arm_ratio_combo_df = arm_ratio_df[is_combo]
        is_base = (arm_ratio_combo_df["n"] == 0) & (
            arm_ratio_combo_df["checkpoint"] == "base"
        )
        base = arm_ratio_combo_df[is_base].iloc[0]
        is_final = arm_ratio_combo_df["checkpoint"] == "fully_trained"
        arm_ratio_final_df = arm_ratio_combo_df[is_final].sort_values("n")
        final_median = arm_ratio_final_df["collision_ratio_median"]
        first = arm_ratio_final_df.iloc[0]
        last = arm_ratio_final_df.iloc[-1]
        worst = arm_ratio_final_df.loc[final_median.idxmax()]
        best = arm_ratio_final_df.loc[final_median.idxmin()]
        near_sizes = arm_ratio_final_df.loc[final_median <= 1 / 0.95, "n"]
        within_sizes = arm_ratio_final_df.loc[final_median.between(0.95, 1.05), "n"]
        row = {
            "dataset": dataset,
            "model": model,
            "n_pairs": int(base["n_pairs"]),
            "base": float(base["collision_ratio_median"]),
            "max_over_grid_final": float(worst["collision_ratio_median"]),
            "n_at_max": int(worst["n"]),
            "min_over_grid_final": float(best["collision_ratio_median"]),
            "n_at_min": int(best["n"]),
            "largest_n_final": float(last["collision_ratio_median"]),
            "largest_n": int(last["n"]),
            "smallest_n_final": float(first["collision_ratio_median"]),
            "smallest_n": int(first["n"]),
            "first_n_at_or_below_inverse_095": (
                int(near_sizes.min()) if len(near_sizes) else None
            ),
            "first_n_within_005_of_one": (
                int(within_sizes.min()) if len(within_sizes) else None
            ),
        }
        row["max_grid_excess_above_base"] = row["max_over_grid_final"] - row["base"]
        rows.append(row)
    return pd.DataFrame(rows)
