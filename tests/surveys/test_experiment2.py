"""Permanent checks for Experiment 2 prompts, collision, and sample sizes."""

from __future__ import annotations

import math
import subprocess
import sys

import pandas as pd
import pytest
import torch

from mode_collapse.surveys.aggregation import (
    AGGREGATE_COLUMNS,
    aggregate_measurements,
    pair_collision_ratio,
    summarize_collision_ratios,
)
from mode_collapse.surveys.anes import build_chat_messages as anes_messages
from mode_collapse.surveys.anes import build_persona as anes_persona
from mode_collapse.surveys.datasets import DATASETS, MODEL_SLUGS, halving_grid
from mode_collapse.surveys.gss import build_chat_messages as gss_messages
from mode_collapse.surveys.gss import build_persona as gss_persona
from mode_collapse.surveys.gss import build_question_block
from mode_collapse.surveys.measurement import option_softmax_collision
from mode_collapse.surveys.wvs import build_chat_messages as wvs_messages
from mode_collapse.surveys.wvs import build_persona as wvs_persona

GSS_MESSAGE = (
    "It is the 1990s. You are a 30-44 year-old white female living in the "
    "new england region of the United States. Your highest education "
    "credential is: bachelor.\n"
    "\n"
    "Do you favor or oppose the death penalty for persons convicted of "
    "murder?\n"
    "\n"
    "Options:\n"
    "  A) favor\n"
    "  B) oppose\n"
    "\n"
    "Respond with only the single letter of the option that best matches "
    "your view."
)
WVS_MESSAGE = (
    "It is 2018. You live in an urban area of Canada. You are a 30-44 "
    "year-old woman. Your highest level of education is higher "
    "(university-level).\n"
    "\n"
    "Taking all things together, would you say you are:\n"
    "\n"
    "Options:\n"
    "  A) Very happy\n"
    "  B) Quite happy\n"
    "  C) Not very happy\n"
    "  D) Not at all happy\n"
    "\n"
    "Respond with only the single letter of the option that best matches "
    "your view."
)
ANES_MESSAGE = (
    "It is the 1990s. You are a 45-59 year-old man living in the South of "
    "the United States. You identify as Black and your highest level of "
    "education is high school.\n"
    "\n"
    "Do you approve or disapprove of the way that [the president] is "
    "handling his/her job as President?\n"
    "\n"
    "Options:\n"
    "  A) Approve\n"
    "  B) Disapprove\n"
    "\n"
    "Respond with only the single letter of the option that best matches "
    "your view."
)


def test_gss_prompt_matches_source_rendering():
    """The GSS death-penalty prompt is rendered as expected."""
    row = pd.Series(
        {
            "year": 1994,
            "age": 35,
            "sex": "female",
            "race": "white",
            "region": "new england",
            "degree": "bachelor",
        }
    )
    question = (
        "Do you favor or oppose the death penalty for persons convicted of murder?"
    )
    messages = gss_messages(gss_persona(row), question, ["favor", "oppose"])
    assert messages == [{"role": "user", "content": GSS_MESSAGE}]


def test_wvs_prompt_matches_source_rendering():
    """The WVS happiness prompt is rendered as expected."""
    row = pd.Series(
        {
            "year": 2018,
            "urbrural": "an urban area",
            "country": "Canada",
            "age": "30-44",
            "sex": "woman",
            "education": "higher (university-level)",
        }
    )
    question = "Taking all things together, would you say you are:"
    options = ["Very happy", "Quite happy", "Not very happy", "Not at all happy"]
    messages = wvs_messages(wvs_persona(row), question, options)
    assert messages == [{"role": "user", "content": WVS_MESSAGE}]


def test_anes_prompt_matches_source_rendering():
    """The ANES presidential-approval prompt is rendered as expected."""
    row = pd.Series(
        {
            "decade": 1990,
            "age": "45-59",
            "sex": "man",
            "region": "the South",
            "race": "Black",
            "education": "high school",
        }
    )
    question = (
        "Do you approve or disapprove of the way that [the president] is "
        "handling his/her job as President?"
    )
    messages = anes_messages(anes_persona(row), question, ["Approve", "Disapprove"])
    assert messages == [{"role": "user", "content": ANES_MESSAGE}]


def test_option_letters_are_uppercase_in_order():
    """Two options are lettered A and B, on their own indented lines."""
    block = build_question_block("Question?", ["first", "second"])
    assert "  A) first\n  B) second\n" in block


def test_option_softmax_collision_matches_hand_values():
    """Uniform two-way collision is 1/2, and (1/4, 3/4) collides at 0.625."""
    uniform = option_softmax_collision(torch.zeros(2))
    assert float(uniform) == pytest.approx(0.5)
    logits = torch.tensor([0.0, math.log(3.0)])
    skewed = option_softmax_collision(logits)
    assert float(skewed) == pytest.approx(0.25**2 + 0.75**2)


def test_distinct_pair_u_statistic_matches_hand_counts():
    """Three-of-four respondents agreeing has C_U = 1/2."""
    counts = (3, 1)
    total = 4
    c_u = sum(count * (count - 1) for count in counts) / (total * (total - 1))
    plug_in = (3 / 4) ** 2 + (1 / 4) ** 2
    c_u_from_plug_in = (total * plug_in - 1) / (total - 1)
    assert c_u == pytest.approx(0.5)
    assert c_u_from_plug_in == pytest.approx(c_u)
    pair_measure_df = pd.DataFrame(
        {"n_cell": [total], "N2_p": [1.0 / plug_in], "N2_q": [1.0 / c_u]}
    )
    ratio = float(pair_collision_ratio(pair_measure_df).iloc[0])
    assert ratio == pytest.approx(1.0)


def test_quantiles_are_taken_on_the_per_pair_ratio():
    """Median of ratios is not the reciprocal of the median reciprocal."""
    ratios = [1.0, 2.0, 3.0, 6.0]
    n_cell = 100.0
    n2_p = 2.0
    n2_unbiased = (n_cell - 1.0) * n2_p / (n_cell - n2_p)
    pair_measure_df = pd.DataFrame(
        {
            "n_cell": [n_cell] * 4,
            "N2_p": [n2_p] * 4,
            "N2_q": [n2_unbiased / ratio for ratio in ratios],
            "n": [8, 8, 8, 8],
            "checkpoint": ["early_stopped"] * 4,
        }
    )
    arm_ratio_df = summarize_collision_ratios(pair_measure_df)
    assert int(arm_ratio_df.loc[0, "n_pairs"]) == 4
    assert arm_ratio_df.loc[0, "collision_ratio_median"] == pytest.approx(2.5)
    reciprocal_median = float(pd.Series([1.0 / ratio for ratio in ratios]).median())
    assert arm_ratio_df.loc[0, "collision_ratio_median"] != pytest.approx(
        1.0 / reciprocal_median
    )
    assert list(arm_ratio_df.columns) == [
        "n",
        "checkpoint",
        "n_pairs",
        "collision_ratio_q05",
        "collision_ratio_q25",
        "collision_ratio_median",
        "collision_ratio_q75",
        "collision_ratio_q95",
    ]


def test_aggregate_columns_cover_the_reported_schema(tmp_path):
    """A full toy sweep writes only the collision-ratio columns."""
    row = "60,2.0,2.0,0,base\n"
    header = "n_cell,N2_p,N2_q,n,checkpoint\n"
    for dataset in DATASETS:
        for model in MODEL_SLUGS:
            path = tmp_path / f"measure_{dataset}_{model}_n0_base.csv"
            path.write_text(header + row)
    arm_ratio_df = aggregate_measurements(tmp_path)
    assert list(arm_ratio_df.columns) == AGGREGATE_COLUMNS
    assert len(arm_ratio_df) == 12
    assert set(arm_ratio_df["n_pairs"]) == {1}


def test_published_halving_grids():
    """The sample-size grids are the pools of the paper halved down to n >= 8."""
    gss_grid = [
        4706940,
        2353470,
        1176735,
        588367,
        294183,
        147091,
        73545,
        36772,
        18386,
        9193,
        4596,
        2298,
        1149,
        574,
        287,
        143,
        71,
        35,
        17,
        8,
    ]
    wvs_grid = [
        1462645,
        731322,
        365661,
        182830,
        91415,
        45707,
        22853,
        11426,
        5713,
        2856,
        1428,
        714,
        357,
        178,
        89,
        44,
        22,
        11,
    ]
    anes_grid = [
        1662788,
        831394,
        415697,
        207848,
        103924,
        51962,
        25981,
        12990,
        6495,
        3247,
        1623,
        811,
        405,
        202,
        101,
        50,
        25,
        12,
    ]
    assert halving_grid(4_706_940) == gss_grid
    assert halving_grid(1_462_645) == wvs_grid
    assert halving_grid(1_662_788) == anes_grid
    assert len(gss_grid) == 20
    assert len(wvs_grid) == 18
    assert len(anes_grid) == 18


def test_light_modules_do_not_import_transformers():
    """Dataset and aggregation imports stay free of transformers."""
    code = """
import sys
import mode_collapse.surveys.gss
import mode_collapse.surveys.wvs
import mode_collapse.surveys.anes
import mode_collapse.surveys.datasets
import mode_collapse.surveys.aggregation
loaded = [name for name in sys.modules if name.split(".")[0] == "transformers"]
assert loaded == [], loaded
"""
    subprocess.check_call([sys.executable, "-c", code])
