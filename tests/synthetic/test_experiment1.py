"""CPU checks for the Experiment 1 synthetic-language code."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from mode_collapse.synthetic.aligned_grid import SEEDS_PER_LEVEL as ALIGNED_SEEDS
from mode_collapse.synthetic.aligned_grid import (
    aggregate_aligned_level,
    base_config,
    grid_n_values,
    grid_spec,
)
from mode_collapse.synthetic.bound_sweep import build_tasks
from mode_collapse.synthetic.bound_sweep import n_values as bound_n_values
from mode_collapse.synthetic.diffuse_grid import (
    EXPECTED_PARAM_COUNT,
    FIT_SEED_OFFSET,
    FIXED_SIZE,
    aggregate_level,
    build_level_seed_tasks,
)
from mode_collapse.synthetic.diffuse_grid import n_values as diffuse_n_values
from mode_collapse.synthetic.initial_alignment import SEEDS_PER_LEVEL as INIT_SEEDS
from mode_collapse.synthetic.initial_alignment import fit_seeds
from mode_collapse.synthetic.kgram import GPT, Config, build_language, smoke_config
from mode_collapse.synthetic.kgram import true_conditionals as kgram_conditionals
from mode_collapse.synthetic.zipf_language import build_language as build_zipf
from mode_collapse.synthetic.zipf_language import (
    true_conditionals as zipf_conditionals,
)


def _members(
    n_seeds: int, n_ctx: int, vocab: int
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    members = []
    for seed_index in range(n_seeds):
        draw = rng.random((n_ctx, vocab))
        probs = draw / draw.sum(axis=1, keepdims=True)
        members.append(
            {
                "q_common": probs.astype(np.float32),
                "param_count": 11,
                "N": 32,
                "level_index": 2,
                "seed_index": seed_index,
                "prior_mass_on_support": 0.4 + 0.01 * seed_index,
                "missing_mass": 0.2,
            }
        )
    target = rng.random((n_ctx, vocab))
    p_common = target / target.sum(axis=1, keepdims=True)
    c_true = (p_common**2).sum(axis=1)
    return members, p_common, c_true


def test_decomposition_identity_on_a_tiny_ensemble():
    members, p_common, c_true = _members(5, 6, 7)
    row = aggregate_level(members, p_common, c_true, 5)
    assert abs(row["derived"] - row["measured"]) < 1e-9
    term_sum = row["var_term"] + row["bias2_term"] + row["cross_term"]
    assert row["derived"] == term_sum


def test_aligned_aggregate_keeps_the_identity():
    members, p_common, c_true = _members(4, 5, 6)
    diagnostics = {
        "prior_target_alignment": 0.01,
        "mean_support_rank_under_qstar": 2.5,
    }
    row = aggregate_aligned_level(members, p_common, c_true, 4, diagnostics)
    assert row["identity_residual"] < 1e-9
    assert abs(row["derived"] - row["measured"]) < 1e-9


def test_order16_conditionals_sum_to_one():
    cfg = smoke_config()
    lang = build_language(cfg, torch.device("cpu"))
    generator = torch.Generator().manual_seed(42)
    contexts = torch.randint(0, cfg.vocab_size, (5, cfg.order_k), generator=generator)
    probs = kgram_conditionals(lang, contexts)
    totals = probs.sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals))
    assert torch.all(probs >= 0)


def test_zipf_conditionals_sum_to_one():
    cfg = Config(
        seed=42,
        vocab_size=40,
        order_k=4,
        emb_dim=8,
        feature_dim=8,
        seq_len=5,
    )
    lang = build_zipf(cfg, torch.device("cpu"))
    generator = torch.Generator().manual_seed(42)
    contexts = torch.randint(0, cfg.vocab_size, (5, cfg.order_k), generator=generator)
    probs = zipf_conditionals(lang, contexts)
    totals = probs.sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals))
    assert torch.all(probs >= 0)


def test_diffuse_sample_size_grid():
    values = diffuse_n_values(False)
    assert len(values) == 100
    assert int(values[0]) == 32_000
    assert int(values[-1]) == 1_000_000
    assert len(np.unique(values)) == 100


def test_aligned_sample_size_grid():
    values = grid_n_values(False)
    assert len(values) == 100
    assert int(values[0]) == 500
    assert int(values[-1]) == 1_000_000
    assert len(np.unique(values)) == 100


def test_bound_sample_size_grid():
    values = bound_n_values(False)
    assert len(values) == 100
    assert int(values[0]) == 32_000
    assert int(values[-1]) == 1_000_000
    assert len(np.unique(values)) == 100


def test_diffuse_and_aligned_seed_formula():
    diffuse = build_level_seed_tasks(32_000, 3, 4, (384, 6, 3), EXPECTED_PARAM_COUNT)
    aligned = build_level_seed_tasks(500, 3, 4, (256, 4, 3), 1)
    for seed_index, (left, right) in enumerate(zip(diffuse, aligned, strict=True)):
        expected_data = 42 + 1000 * 3 + seed_index
        assert left["data_seed"] == expected_data
        assert left["fit_seed"] == expected_data + FIT_SEED_OFFSET
        assert right["data_seed"] == expected_data
        assert right["fit_seed"] == expected_data + FIT_SEED_OFFSET


def test_bound_seed_formula():
    tasks = build_tasks(smoke_config(), True)
    for index, task in enumerate(tasks):
        assert task["data_seed"] == 42 + 13 * (index + 1)
        assert task["fit_seed"] == 42 + 4242 * (index + 1)


def test_initialization_uses_level_zero_fit_seeds():
    seeds = fit_seeds()
    assert len(seeds) == 100
    assert seeds == [42 + seed_index + FIT_SEED_OFFSET for seed_index in range(100)]


def test_published_seed_counts():
    assert ALIGNED_SEEDS == 100
    assert INIT_SEEDS == 100


def test_diffuse_parameter_count():
    d_model, n_head, n_layer = FIXED_SIZE
    cfg = dataclasses.replace(Config(), d_model=d_model, n_head=n_head, n_layer=n_layer)
    count = int(sum(p.numel() for p in GPT(cfg).parameters()))
    assert count == EXPECTED_PARAM_COUNT


def test_aligned_parameter_count():
    base = base_config(False)
    _levels, seeds, size = grid_spec(False)
    assert seeds == 100
    cfg = dataclasses.replace(base, d_model=size[0], n_head=size[1], n_layer=size[2])
    count = int(sum(p.numel() for p in GPT(cfg).parameters()))
    assert count == 27_974_144
