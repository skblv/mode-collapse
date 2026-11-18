"""Aggregate Experiment 2 per-pair measurements into collision-ratio quantiles."""

from __future__ import annotations

import argparse

from mode_collapse.surveys.aggregation import (
    aggregate_measurements,
    final_adapter_table,
)
from mode_collapse.surveys.datasets import aggregate_csv_path, measurements_dir


def main() -> None:
    """Writes ``survey_collision_ratio.csv`` and prints the final-adapter table."""
    # --- config ---
    argparse.ArgumentParser(
        description=(
            "Aggregate Experiment 2 collision ratios. "
            "Reads OUTPUT_DIR/surveys/measurements and writes "
            "OUTPUT_DIR/surveys/survey_collision_ratio.csv."
        )
    ).parse_args()

    # --- data import ---
    arm_ratio_df = aggregate_measurements(measurements_dir())

    # --- data export ---
    out_path = aggregate_csv_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arm_ratio_df.to_csv(out_path, index=False)
    print(
        f"wrote {out_path} rows={len(arm_ratio_df)} "
        f"datasets={arm_ratio_df['dataset'].nunique()} "
        f"models={arm_ratio_df['model'].nunique()}",
        flush=True,
    )
    adapter_table_df = final_adapter_table(arm_ratio_df)
    print(adapter_table_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
