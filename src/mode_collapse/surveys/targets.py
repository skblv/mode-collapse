"""Processed survey targets for Experiment 2.

Each builder reads one raw survey file, assigns a respondent-level train or
validation split, and writes the question config, personas, respondent-question
rows, and per-group plug-in collision. The reported human target, the
distinct-pair U-statistic, is recovered later from these plug-in values.
Evaluation keeps question-group pairs with at least 60 respondents.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from mode_collapse.surveys import anes, gss, wvs
from mode_collapse.surveys.datasets import (
    item_config_path,
    processed_dir,
    raw_survey_path,
    summary_path,
)
from mode_collapse.surveys.gss import SEED

VAL_FRACTION = 0.10
MIN_EVAL_CELL_N = 60


def assign_split(survey_df: pd.DataFrame, rng: np.random.Generator) -> None:
    """Assigns a respondent-level train or validation split in place.

    Args:
        survey_df: Respondent frame. Gains a ``split`` column.
        rng: Generator whose next draw is this split.
    """
    is_val = rng.random(len(survey_df)) < VAL_FRACTION
    survey_df["split"] = np.where(is_val, "val", "train")


def build_cell_persona_df(
    survey_df: pd.DataFrame, cell_vars: list[str]
) -> pd.DataFrame:
    """Builds one persona row per demographic group.

    Args:
        survey_df: Respondent frame with ``cell_id`` and ``persona``.
        cell_vars: Group columns stored beside the persona text.

    Returns:
        One row per ``cell_id``.
    """
    keep = ["cell_id", "persona"] + cell_vars
    return survey_df[keep].drop_duplicates(subset="cell_id").reset_index(drop=True)


def cell_truth_record(
    item: str,
    cell_id: str,
    answer_counts: pd.Series,
    options: list[str],
) -> dict:
    """Builds one question-group plug-in collision record.

    Args:
        item: Question code.
        cell_id: Demographic group identifier.
        answer_counts: Counts of answer labels in the group.
        options: Ordered option labels.

    Returns:
        Plug-in collision, inverse Simpson index, probabilities, and whether
        the group has at least 60 respondents.
    """
    probs = np.array([answer_counts.get(option, 0) for option in options], dtype=float)
    n_cell = int(probs.sum())
    probs = probs / n_cell
    collision = float(np.sum(probs**2))
    probability_by_option = dict(zip(options, probs.round(6).tolist(), strict=True))
    return {
        "item": item,
        "cell_id": cell_id,
        "n_cell": n_cell,
        "C_p": collision,
        "N2_p": float(1.0 / collision),
        "probs_json": json.dumps(probability_by_option),
        "is_eval": n_cell >= MIN_EVAL_CELL_N,
    }


def per_item_summary(
    item: str,
    item_config: dict,
    cell_item_truth_df: pd.DataFrame,
    include_range: bool,
) -> dict:
    """Summarizes cells and eval cells for one question.

    Args:
        item: Question code.
        item_config: Per-question option config.
        cell_item_truth_df: All question-group truth rows.
        include_range: Whether to store the eval inverse-Simpson range.
            GSS and WVS do; ANES does not.

    Returns:
        Counts and eval inverse-Simpson summaries for the question.
    """
    item_truth_df = cell_item_truth_df[cell_item_truth_df["item"] == item]
    eval_truth_df = item_truth_df[item_truth_df["is_eval"]]
    if len(eval_truth_df):
        median_n2 = float(eval_truth_df["N2_p"].median())
        n2_min = float(eval_truth_df["N2_p"].min())
        n2_max = float(eval_truth_df["N2_p"].max())
    else:
        median_n2 = None
        n2_min = None
        n2_max = None
    summary = {
        "n_options": len(item_config[item]["options"]),
        "options": item_config[item]["options"],
        "n_cells": int(len(item_truth_df)),
        "n_eval_cells": int(len(eval_truth_df)),
        "median_N2_p_eval": median_n2,
    }
    if include_range:
        summary["N2_p_eval_range"] = [n2_min, n2_max]
    return summary


def load_gss(survey_path) -> pd.DataFrame:
    """Loads GSS group and question columns with labels and a respondent id.

    Age and year arrive as labelled strings, including the top-coded age
    ``89 OR OLDER``. The leading integer is parsed so decade and age-bucket
    arithmetic uses numbers.

    Args:
        survey_path: GSS cumulative Stata file.

    Returns:
        Respondents with complete group variables.
    """
    columns = sorted(set(gss.CELL_VARS + gss.ITEM_VARS))
    survey_raw_df = pd.read_stata(
        survey_path, columns=columns, convert_categoricals=True
    )
    for numeric_var in ["age", "year"]:
        digits = (
            survey_raw_df[numeric_var].astype(str).str.extract(r"(\d+)", expand=False)
        )
        survey_raw_df[numeric_var] = pd.to_numeric(digits, errors="coerce")
    survey_df = survey_raw_df.dropna(subset=gss.CELL_VARS).reset_index(drop=True)
    survey_df["respondent_uid"] = np.arange(len(survey_df))
    print(
        f"loaded {len(survey_raw_df)} rows; "
        f"{len(survey_df)} with complete cell variables",
        flush=True,
    )
    return survey_df


def build_gss_item_config(survey_df: pd.DataFrame) -> dict:
    """Derives ordered observed answer labels and letters for each GSS item.

    Args:
        survey_df: Wide respondent frame with categorical item columns.

    Returns:
        Mapping of item to question text, observed options in category order,
        and letters.
    """
    item_config = {}
    for item in gss.ITEM_VARS:
        column = survey_df[item]
        observed = set(column.dropna().unique().tolist())
        ordered_options = [
            str(category) for category in column.cat.categories if category in observed
        ]
        letters = gss.ANSWER_LETTERS[: len(ordered_options)]
        item_config[item] = {
            "question": gss.ITEM_QUESTIONS[item],
            "options": ordered_options,
            "letters": letters,
        }
    return item_config


def build_long_and_truth_gss(
    survey_df: pd.DataFrame, item_config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds GSS respondent-question rows and per-group plug-in collision.

    Args:
        survey_df: Wide frame with ``cell_id``, ``persona``, and ``split``.
        item_config: Per-item option and letter config.

    Returns:
        ``(respondent_item_long_df, cell_item_truth_df)``.
    """
    long_parts = []
    truth_rows = []
    for item in tqdm(gss.ITEM_VARS, desc="items"):
        options = item_config[item]["options"]
        letters = item_config[item]["letters"]
        option_to_letter = dict(zip(options, letters, strict=False))
        keep = ["respondent_uid", "cell_id", "split", item]
        item_df = survey_df[keep].copy()
        item_df = item_df.dropna(subset=[item])
        item_df["answer_label"] = item_df[item].astype(str)
        item_df["answer_letter"] = item_df["answer_label"].map(option_to_letter)
        assert item_df["answer_letter"].notna().all(), f"unmapped option in {item}"
        item_df["item"] = item
        long_columns = [
            "respondent_uid",
            "cell_id",
            "item",
            "answer_label",
            "answer_letter",
            "split",
        ]
        long_parts.append(item_df[long_columns])
        for cell_id, cell_df in item_df.groupby("cell_id"):
            answer_counts = cell_df["answer_label"].value_counts()
            truth_rows.append(cell_truth_record(item, cell_id, answer_counts, options))
    respondent_item_long_df = pd.concat(long_parts, ignore_index=True)
    cell_item_truth_df = pd.DataFrame(truth_rows)
    return respondent_item_long_df, cell_item_truth_df


def build_gss() -> None:
    """Reads the GSS cumulative file and writes processed targets."""
    # --- setup ---
    out_dir = processed_dir("gss")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path("gss").parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- data import ---
    survey_df = load_gss(raw_survey_path("gss"))

    # --- data augmentation ---
    survey_df["cell_id"] = survey_df.apply(gss.cell_id_of, axis=1)
    survey_df["persona"] = survey_df.apply(gss.build_persona, axis=1)

    # --- data filtering ---
    assign_split(survey_df, rng)

    # --- model fitting ---
    item_config = build_gss_item_config(survey_df)
    respondent_item_long_df, cell_item_truth_df = build_long_and_truth_gss(
        survey_df, item_config
    )

    # --- data export ---
    with item_config_path("gss").open("w") as handle:
        json.dump(item_config, handle, indent=2)
    cell_persona_df = build_cell_persona_df(survey_df, gss.CELL_VARS)
    cell_persona_df.to_csv(out_dir / "cell_persona_df.csv", index=False)
    respondent_item_long_df.to_csv(out_dir / "respondent_item_long_df.csv", index=False)
    cell_item_truth_df.to_csv(out_dir / "cell_item_truth_df.csv", index=False)
    summary = {
        "n_respondents": int(len(survey_df)),
        "n_cells": int(survey_df["cell_id"].nunique()),
        "val_fraction": VAL_FRACTION,
        "min_eval_cell_n": MIN_EVAL_CELL_N,
        "n_train_rows": int((respondent_item_long_df["split"] == "train").sum()),
        "n_val_rows": int((respondent_item_long_df["split"] == "val").sum()),
        "per_item": {},
    }
    for item in gss.ITEM_VARS:
        summary["per_item"][item] = per_item_summary(
            item, item_config, cell_item_truth_df, include_range=True
        )
    with summary_path("gss").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def load_wvs(survey_path) -> pd.DataFrame:
    """Loads WVS group and item columns and drops incomplete groups.

    Negative or zero ages are missing codes. Ages below 16 are removed before
    bucketing.

    Args:
        survey_path: WVS wave 7 CSV.

    Returns:
        Respondents with complete group labels.
    """
    item_cols = sorted({column for column, _labels in wvs.WVS_ITEM_CODES.values()})
    usecols = [
        wvs.COUNTRY_VAR,
        wvs.SEX_VAR,
        wvs.AGE_VAR,
        wvs.EDU_VAR,
        wvs.YEAR_VAR,
        wvs.URBRURAL_VAR,
    ] + item_cols
    survey_raw_df = pd.read_csv(survey_path, usecols=usecols)
    print(f"loaded {len(survey_raw_df)} raw rows", flush=True)

    survey_df = survey_raw_df.copy()
    survey_df["country"] = survey_df[wvs.COUNTRY_VAR].map(wvs.COUNTRY_NAMES)
    survey_df["year"] = pd.to_numeric(survey_df[wvs.YEAR_VAR], errors="coerce").astype(
        "Int64"
    )
    survey_df["sex"] = survey_df[wvs.SEX_VAR].map(wvs.SEX_LABELS)
    survey_df["education"] = survey_df[wvs.EDU_VAR].map(wvs.EDU_LABELS)
    urbrural_codes = pd.to_numeric(survey_df[wvs.URBRURAL_VAR], errors="coerce")
    survey_df["urbrural"] = urbrural_codes.map(wvs.URBRURAL_LABELS)
    age_numeric = pd.to_numeric(survey_df[wvs.AGE_VAR], errors="coerce").where(
        lambda age: age >= 16
    )
    survey_df["age"] = age_numeric.apply(
        lambda age: wvs.age_bucket_of(age) if pd.notna(age) else None
    )
    survey_df = survey_df.dropna(subset=wvs.CELL_VARS).reset_index(drop=True)
    survey_df["respondent_uid"] = np.arange(len(survey_df))
    print(
        f"{len(survey_df)} rows with complete group variables; "
        f"{survey_df['country'].nunique()} countries",
        flush=True,
    )
    return survey_df


def build_wvs_item_config(survey_df: pd.DataFrame, items: list[str]) -> dict:
    """Derives ordered observed WVS options for the chosen items.

    Args:
        survey_df: Wide frame carrying raw item-code columns.
        items: Item codes in the seeded one-third subset.

    Returns:
        Mapping of item to question, labels, letters, and numeric codes.
    """
    item_config = {}
    for item in items:
        column_name, code_to_label = wvs.WVS_ITEM_CODES[item]
        observed = set(
            pd.to_numeric(survey_df[column_name], errors="coerce").dropna().tolist()
        )
        ordered_codes = [code for code in code_to_label if code in observed]
        ordered_options = [code_to_label[code] for code in ordered_codes]
        letters = wvs.ANSWER_LETTERS[: len(ordered_options)]
        item_config[item] = {
            "question": wvs.ITEM_QUESTIONS[item],
            "options": ordered_options,
            "letters": letters,
            "codes": ordered_codes,
        }
    return item_config


def build_long_and_truth_coded(
    survey_df: pd.DataFrame,
    item_config: dict,
    code_column_of,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds respondent-question rows from numeric answer codes.

    Args:
        survey_df: Wide frame with ``cell_id``, ``split``, and raw codes.
        item_config: Per-item options, letters, and codes.
        code_column_of: Maps an item code to its raw column name.

    Returns:
        ``(respondent_item_long_df, cell_item_truth_df)``.
    """
    long_parts = []
    truth_rows = []
    for item in tqdm(list(item_config.keys()), desc="items"):
        column_name = code_column_of(item)
        options = item_config[item]["options"]
        letters = item_config[item]["letters"]
        codes = item_config[item]["codes"]
        code_to_label = dict(zip(codes, options, strict=True))
        option_to_letter = dict(zip(options, letters, strict=False))
        item_df = survey_df[["respondent_uid", "cell_id", "split", column_name]].copy()
        item_df[column_name] = pd.to_numeric(item_df[column_name], errors="coerce")
        item_df = item_df[item_df[column_name].isin(codes)]
        item_df["answer_label"] = item_df[column_name].map(code_to_label)
        item_df["answer_letter"] = item_df["answer_label"].map(option_to_letter)
        assert item_df["answer_letter"].notna().all(), f"unmapped option in {item}"
        item_df["item"] = item
        long_columns = [
            "respondent_uid",
            "cell_id",
            "item",
            "answer_label",
            "answer_letter",
            "split",
        ]
        long_parts.append(item_df[long_columns])
        for cell_id, cell_df in item_df.groupby("cell_id"):
            answer_counts = cell_df["answer_label"].value_counts()
            truth_rows.append(cell_truth_record(item, cell_id, answer_counts, options))
    respondent_item_long_df = pd.concat(long_parts, ignore_index=True)
    cell_item_truth_df = pd.DataFrame(truth_rows)
    return respondent_item_long_df, cell_item_truth_df


def build_wvs() -> None:
    """Reads the WVS wave 7 file and writes the seeded subset's targets."""
    # --- setup ---
    out_dir = processed_dir("wvs")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path("wvs").parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- data import ---
    survey_df = load_wvs(raw_survey_path("wvs"))

    # --- data filtering ---
    all_countries = survey_df["country"].dropna().unique().tolist()
    subset = wvs.select_wvs_subset(all_countries, wvs.ITEM_VARS, SEED)
    with (out_dir / "wvs_subset_seed42.json").open("w") as handle:
        json.dump(subset, handle, indent=2)
    chosen_countries = set(subset["countries"])
    chosen_items = subset["items"]
    print(
        f"WVS subset (seed {SEED}): "
        f"{subset['n_countries_kept']}/{subset['n_countries_full']} countries, "
        f"{subset['n_items_kept']}/{subset['n_items_full']} items",
        flush=True,
    )
    survey_df = survey_df[survey_df["country"].isin(chosen_countries)].reset_index(
        drop=True
    )
    survey_df["respondent_uid"] = np.arange(len(survey_df))
    print(
        f"{len(survey_df)} respondents after country subset; "
        f"{survey_df['country'].nunique()} countries",
        flush=True,
    )

    # --- data augmentation ---
    survey_df["cell_id"] = survey_df.apply(wvs.cell_id_of, axis=1)
    survey_df["persona"] = survey_df.apply(wvs.build_persona, axis=1)
    assign_split(survey_df, rng)

    # --- model fitting ---
    item_config = build_wvs_item_config(survey_df, chosen_items)
    respondent_item_long_df, cell_item_truth_df = build_long_and_truth_coded(
        survey_df,
        item_config,
        lambda item: wvs.WVS_ITEM_CODES[item][0],
    )

    # --- data export ---
    with item_config_path("wvs").open("w") as handle:
        json.dump(item_config, handle, indent=2)
    cell_persona_df = build_cell_persona_df(survey_df, wvs.CELL_VARS)
    cell_persona_df.to_csv(out_dir / "cell_persona_df.csv", index=False)
    respondent_item_long_df.to_csv(out_dir / "respondent_item_long_df.csv", index=False)
    cell_item_truth_df.to_csv(out_dir / "cell_item_truth_df.csv", index=False)
    summary = {
        "n_respondents": int(len(survey_df)),
        "n_countries": int(survey_df["country"].nunique()),
        "n_items": len(item_config),
        "n_cells": int(survey_df["cell_id"].nunique()),
        "val_fraction": VAL_FRACTION,
        "min_eval_cell_n": MIN_EVAL_CELL_N,
        "n_train_rows": int((respondent_item_long_df["split"] == "train").sum()),
        "n_val_rows": int((respondent_item_long_df["split"] == "val").sum()),
        "per_item": {},
    }
    for item in item_config:
        summary["per_item"][item] = per_item_summary(
            item, item_config, cell_item_truth_df, include_range=True
        )
    with summary_path("wvs").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def load_anes(survey_path) -> pd.DataFrame:
    """Loads ANES group and item columns and drops incomplete groups.

    Missing and non-positive ages are removed before bucketing. Calendar year
    is coarsened to decade.

    Args:
        survey_path: ANES time-series cumulative CSV.

    Returns:
        Respondents with complete group labels.
    """
    item_cols = sorted({column for column, _labels in anes.ANES_ITEM_CODES.values()})
    group_cols = [
        anes.YEAR_VAR,
        anes.AGE_VAR,
        anes.SEX_VAR,
        anes.RACE_VAR,
        anes.REGION_VAR,
        anes.EDU_VAR,
    ]
    usecols = group_cols + item_cols
    survey_raw_df = pd.read_csv(
        survey_path,
        usecols=lambda column: column in set(usecols),
        low_memory=False,
    )
    print(f"loaded {len(survey_raw_df)} raw rows", flush=True)

    survey_df = survey_raw_df.copy()
    year = pd.to_numeric(survey_df[anes.YEAR_VAR], errors="coerce")
    survey_df["decade"] = (year // 10 * 10).astype("Int64")
    survey_df["sex"] = pd.to_numeric(survey_df[anes.SEX_VAR], errors="coerce").map(
        anes.SEX_LABELS
    )
    survey_df["race"] = pd.to_numeric(survey_df[anes.RACE_VAR], errors="coerce").map(
        anes.RACE_LABELS
    )
    survey_df["region"] = pd.to_numeric(
        survey_df[anes.REGION_VAR], errors="coerce"
    ).map(anes.REGION_LABELS)
    survey_df["education"] = pd.to_numeric(
        survey_df[anes.EDU_VAR], errors="coerce"
    ).map(anes.EDU_LABELS)
    age_numeric = pd.to_numeric(survey_df[anes.AGE_VAR], errors="coerce").where(
        lambda age: age >= 16
    )
    survey_df["age"] = age_numeric.apply(
        lambda age: anes.age_bucket_of(age) if pd.notna(age) else None
    )
    survey_df = survey_df.dropna(subset=anes.CELL_VARS).reset_index(drop=True)
    survey_df["respondent_uid"] = np.arange(len(survey_df))
    decades = sorted(survey_df["decade"].dropna().unique().tolist())
    print(
        f"{len(survey_df)} rows with complete group variables; decades {decades}",
        flush=True,
    )
    return survey_df


def build_anes_item_config(survey_df: pd.DataFrame) -> dict:
    """Derives ordered observed ANES options from the codebook maps.

    Args:
        survey_df: Wide frame carrying raw item-code columns.

    Returns:
        Mapping of item to question, labels, letters, and numeric codes.
    """
    item_config = {}
    for item in anes.ITEM_VARS:
        column_name, code_to_label = anes.ANES_ITEM_CODES[item]
        observed = set(
            pd.to_numeric(survey_df[column_name], errors="coerce").dropna().tolist()
        )
        ordered_codes = [code for code in code_to_label if code in observed]
        ordered_options = [code_to_label[code] for code in ordered_codes]
        letters = anes.ANSWER_LETTERS[: len(ordered_options)]
        item_config[item] = {
            "question": anes.ITEM_QUESTIONS[item],
            "options": ordered_options,
            "letters": letters,
            "codes": ordered_codes,
        }
    return item_config


def build_anes() -> None:
    """Reads the ANES cumulative file and writes processed targets."""
    # --- setup ---
    out_dir = processed_dir("anes")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path("anes").parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- data import ---
    survey_df = load_anes(raw_survey_path("anes"))

    # --- data augmentation ---
    survey_df["cell_id"] = survey_df.apply(anes.cell_id_of, axis=1)
    survey_df["persona"] = survey_df.apply(anes.build_persona, axis=1)

    # --- data filtering ---
    assign_split(survey_df, rng)

    # --- model fitting ---
    item_config = build_anes_item_config(survey_df)
    respondent_item_long_df, cell_item_truth_df = build_long_and_truth_coded(
        survey_df,
        item_config,
        lambda item: anes.ANES_ITEM_CODES[item][0],
    )

    # --- data export ---
    with item_config_path("anes").open("w") as handle:
        json.dump(item_config, handle, indent=2)
    cell_persona_df = build_cell_persona_df(survey_df, anes.CELL_VARS)
    cell_persona_df.to_csv(out_dir / "cell_persona_df.csv", index=False)
    respondent_item_long_df.to_csv(out_dir / "respondent_item_long_df.csv", index=False)
    cell_item_truth_df.to_csv(out_dir / "cell_item_truth_df.csv", index=False)
    summary = {
        "n_respondents": int(len(survey_df)),
        "n_cells": int(survey_df["cell_id"].nunique()),
        "val_fraction": VAL_FRACTION,
        "min_eval_cell_n": MIN_EVAL_CELL_N,
        "n_train_rows": int((respondent_item_long_df["split"] == "train").sum()),
        "n_val_rows": int((respondent_item_long_df["split"] == "val").sum()),
        "per_item": {},
    }
    for item in anes.ITEM_VARS:
        summary["per_item"][item] = per_item_summary(
            item, item_config, cell_item_truth_df, include_range=False
        )
    with summary_path("anes").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "per_item"},
            indent=2,
        ),
        flush=True,
    )
    covered = sum(
        1 for value in summary["per_item"].values() if value["n_eval_cells"] > 0
    )
    n_eval_cells = sum(value["n_eval_cells"] for value in summary["per_item"].values())
    print(
        f"items={len(anes.ITEM_VARS)} with>=1 eval group={covered} "
        f"total_eval_cells={n_eval_cells}",
        flush=True,
    )
