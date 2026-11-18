"""Per-problem collision ratios and the summary across CodeNet problems.

A problem enters an arm's summary when its human and model kernel collisions
are both finite and the model collision is strictly positive. Its collision
ratio is ``C_ZS(q) / C_ZS(p)``, and the summary takes quantiles of that ratio
across problems. ``n_finite`` is the number of problems in the summary.
``human_collision_median`` and ``model_collision_median`` are medians of the
raw collisions over the same problems.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from mode_collapse.codenet.config import (
    CHECKPOINTS,
    COLLISION_RATIO_QUANTILES,
    MODEL_SLUGS,
    PER_PROBLEM_PATH,
    SAME_PROBLEM_PATH,
    SUMMARY_COLUMNS,
    SUMMARY_PATH,
    scoring_arms,
)
from mode_collapse.codenet.zhang_shasha import (
    human_cache_path,
    load_eval_problem_ids,
    model_cache_path,
    quantile,
    read_cache,
)


def finite_float(value: str) -> float | None:
    """Parse a finite floating-point value."""
    if value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def has_finite_ratio(human: float | None, model: float | None) -> bool:
    """Return whether a problem enters the collision-ratio quantiles.

    Both kernel collisions must be finite and the model collision strictly
    positive. The ratio stored for that problem is ``model / human``.
    """
    if human is None or model is None:
        return False
    if not math.isfinite(human) or not math.isfinite(model):
        return False
    if not model > 0:
        return False
    return math.isfinite(model / human)


def collision_ratio(human: float, model: float) -> float:
    """Return the model/human kernel-collision ratio."""
    return model / human


def summarize_loaded_arms(
    problem_ids: list[str],
    human_rows: list[dict[str, str]],
    model_rows_by_arm: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Join human and model caches into per-problem ratios and one summary row per arm."""
    human_by_problem = {row["problem_id"]: row for row in human_rows}
    per_problem_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for arm, model_rows in model_rows_by_arm.items():
        model_by_problem = {row["problem_id"]: row for row in model_rows}
        ratio_values: list[float] = []
        human_collisions: list[float] = []
        model_collisions: list[float] = []
        for problem_id in problem_ids:
            human_row = human_by_problem[problem_id]
            model_row = model_by_problem[problem_id]
            human_value = finite_float(str(human_row["zss_collision"]))
            model_value = finite_float(str(model_row["zss_collision"]))
            ratio = math.nan
            if has_finite_ratio(human_value, model_value):
                ratio = collision_ratio(float(human_value), float(model_value))
                ratio_values.append(ratio)
                human_collisions.append(float(human_value))
                model_collisions.append(float(model_value))
            per_problem_rows.append(
                {
                    "problem_id": problem_id,
                    "model": model_row["model"],
                    "n": int(model_row["n"]),
                    "checkpoint": model_row["checkpoint"],
                    "human_zss_collision": human_value,
                    "model_zss_collision": model_value,
                    "collision_ratio": ratio,
                    "human_n_candidates": int(human_row["n_candidates"]),
                    "human_n_pairs_scored": int(human_row["n_pairs_scored"]),
                    "model_n_parseable": int(model_row["n_parseable"]),
                    "model_n_pairs_scored": int(model_row["n_pairs_scored"]),
                }
            )
        if not ratio_values:
            raise AssertionError(f"no finite collision ratios for {arm}")
        summary_rows.append(
            _summary_row(
                model_rows,
                problem_ids,
                ratio_values,
                human_collisions,
                model_collisions,
            )
        )
    return per_problem_rows, ordered_summary(summary_rows)


def _summary_row(
    model_rows: list[dict[str, str]],
    problem_ids: list[str],
    ratio_values: list[float],
    human_collisions: list[float],
    model_collisions: list[float],
) -> dict[str, object]:
    """Quantiles of one arm's per-problem collision ratios."""
    ratio_quantiles = {
        suffix: quantile(ratio_values, probability)
        for suffix, probability in COLLISION_RATIO_QUANTILES
    }
    row: dict[str, object] = {
        "model": model_rows[0]["model"],
        "n": int(model_rows[0]["n"]),
        "checkpoint": model_rows[0]["checkpoint"],
        "n_problems": len(problem_ids),
        "n_finite": len(ratio_values),
        "human_collision_median": quantile(human_collisions, 0.50),
        "model_collision_median": quantile(model_collisions, 0.50),
    }
    for suffix, _probability in COLLISION_RATIO_QUANTILES:
        row[f"collision_ratio_{suffix}"] = ratio_quantiles[suffix]
    return row


def ordered_summary(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Sort summary rows by model, training size, and checkpoint."""
    return sorted(
        summary_rows,
        key=lambda row: (
            MODEL_SLUGS.index(str(row["model"])),
            int(row["n"]),
            CHECKPOINTS.index(str(row["checkpoint"])),
        ),
    )


def same_problem_comparison(
    per_problem_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Compare base and full fine-tuning on problems finite at both stages.

    Medians are medians of the per-problem collision ratio, restricted to
    problem ids that have a finite ratio at ``n=0`` / ``base`` and at
    the full training size / ``fully_trained``.
    """
    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in per_problem_rows:
        by_model[str(row["model"])].append(row)
    comparisons: list[dict[str, object]] = []
    for model in MODEL_SLUGS:
        rows = by_model[model]
        if not rows:
            continue
        n_full = max(int(row["n"]) for row in rows)
        base_by_problem = {
            str(row["problem_id"]): row
            for row in rows
            if int(row["n"]) == 0 and row["checkpoint"] == "base"
        }
        full_by_problem = {
            str(row["problem_id"]): row
            for row in rows
            if int(row["n"]) == n_full and row["checkpoint"] == "fully_trained"
        }
        matched_ids = [
            problem_id
            for problem_id, base_row in base_by_problem.items()
            if problem_id in full_by_problem
            and math.isfinite(float(base_row["collision_ratio"]))
            and math.isfinite(float(full_by_problem[problem_id]["collision_ratio"]))
        ]
        base_ratios = [
            float(base_by_problem[problem_id]["collision_ratio"])
            for problem_id in matched_ids
        ]
        full_ratios = [
            float(full_by_problem[problem_id]["collision_ratio"])
            for problem_id in matched_ids
        ]
        comparisons.append(
            {
                "model": model,
                "n_full": n_full,
                "n_matched": len(matched_ids),
                "base_collision_ratio_median": float(
                    np.median(np.asarray(base_ratios, dtype=float))
                ),
                "fully_trained_collision_ratio_median": float(
                    np.median(np.asarray(full_ratios, dtype=float))
                ),
            }
        )
    return comparisons


def summarize_caches(
    processed_dir: Path, cache_dir: Path
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Load completed caches and return per-problem rows, the summary, and the comparison."""
    summary = json.loads((processed_dir / "summary.json").read_text(encoding="utf-8"))
    n_train = int(summary["n_train_rows"])
    problem_ids = load_eval_problem_ids(processed_dir)
    if len(problem_ids) != int(summary["n_eval_problems"]):
        raise AssertionError("evaluation problem count does not match dataset summary")
    human_rows = read_cache(human_cache_path(cache_dir), problem_ids, "human")
    model_rows_by_arm: dict[str, list[dict[str, str]]] = {}
    for model, n, checkpoint in scoring_arms(n_train):
        arm = f"{model}_n{n}_{checkpoint}"
        cache_path = model_cache_path(cache_dir, model, n, checkpoint)
        if not cache_path.exists():
            raise FileNotFoundError(f"missing collision cache: {cache_path}")
        model_rows_by_arm[arm] = read_cache(cache_path, problem_ids, arm)
    per_problem_rows, summary_rows = summarize_loaded_arms(
        problem_ids, human_rows, model_rows_by_arm
    )
    comparisons = same_problem_comparison(per_problem_rows)
    return per_problem_rows, summary_rows, comparisons


def _write_new_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    """Write a CSV once. An existing file is left in place."""
    if path.exists():
        print(f"using completed table {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    temporary.replace(path)


def write_collision_tables(
    per_problem_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    per_problem_path: Path = PER_PROBLEM_PATH,
    summary_path: Path = SUMMARY_PATH,
    comparison_path: Path = SAME_PROBLEM_PATH,
) -> None:
    """Write the per-problem ratios, the summary, and the same-problem comparison."""
    per_problem_fields = [
        "problem_id",
        "model",
        "n",
        "checkpoint",
        "human_zss_collision",
        "model_zss_collision",
        "collision_ratio",
        "human_n_candidates",
        "human_n_pairs_scored",
        "model_n_parseable",
        "model_n_pairs_scored",
    ]
    comparison_fields = [
        "model",
        "n_full",
        "n_matched",
        "base_collision_ratio_median",
        "fully_trained_collision_ratio_median",
    ]
    _write_new_csv(per_problem_path, per_problem_rows, per_problem_fields)
    _write_new_csv(summary_path, summary_rows, list(SUMMARY_COLUMNS))
    _write_new_csv(comparison_path, comparisons, comparison_fields)
