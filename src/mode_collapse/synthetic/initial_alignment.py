"""Initialization decomposition for the diffuse and mode-aligned priors.

Before any fine-tuning on the target language, 100 seeds of each starting
distribution are scored on that distribution's common contexts. The three terms
of the collision decomposition, and bootstrap intervals over contexts, are
written to one JSON file.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from mode_collapse.paths import OUTPUT_DIR
from mode_collapse.synthetic.aligned_grid import (
    base_config,
    build_prior,
    grid_spec,
    inject_prior,
    pretrain_steps,
    support_size,
)
from mode_collapse.synthetic.aligned_grid import (
    build_eval_surfaces as build_aligned_surfaces,
)
from mode_collapse.synthetic.contexts import conditionals_on_contexts
from mode_collapse.synthetic.diffuse_grid import (
    BASE_SEED,
    EXPECTED_PARAM_COUNT,
    FIT_SEED_OFFSET,
    FIXED_SIZE,
    aggregate_level,
)
from mode_collapse.synthetic.diffuse_grid import (
    build_eval_surfaces as build_diffuse_surfaces,
)
from mode_collapse.synthetic.kgram import DTYPE, GPT, Config

# --- config ---

SEED = 42
SEEDS_PER_LEVEL = 100
IDENTITY_TOL = 1e-9
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5
DEVICE = torch.device("cuda:0")
OUTPUT_NAME = "initial_alignment.json"


def output_path() -> Path:
    """Returns the path of the initialization JSON."""
    return OUTPUT_DIR / "synthetic" / OUTPUT_NAME


@dataclasses.dataclass
class ExperimentSpec:
    """One starting distribution's initialization evaluation.

    Attributes:
        name: ``diffuse`` or ``mode_aligned``.
        cfg: Size-resolved configuration.
        common_ctx_np: Common context ids, shape (ctx, order_k).
        p_common: True conditionals, float64, shape (ctx, vocab).
        c_true_common: True collision per context, float64, shape (ctx,).
        injector: In-place map from a fresh model to the starting state, or
            ``None`` for random initialization.
    """

    name: str
    cfg: Config
    common_ctx_np: np.ndarray
    p_common: np.ndarray
    c_true_common: np.ndarray
    injector: object


def fit_seeds() -> list[int]:
    """Returns the level-0 fit seeds shared with the training grids.

    Returns:
        ``SEEDS_PER_LEVEL`` fit seeds. Seed ``i`` is
        ``42 + i + FIT_SEED_OFFSET``.
    """
    return [
        BASE_SEED + 1000 * 0 + seed_index + FIT_SEED_OFFSET
        for seed_index in range(SEEDS_PER_LEVEL)
    ]


def build_diffuse_spec() -> ExperimentSpec:
    """Builds the diffuse-prior specification, which is random initialization."""
    base = Config()
    d_model, n_head, n_layer = FIXED_SIZE
    cfg = dataclasses.replace(base, d_model=d_model, n_head=n_head, n_layer=n_layer)
    param_count = int(sum(p.numel() for p in GPT(cfg).parameters()))
    assert param_count == EXPECTED_PARAM_COUNT, (
        f"diffuse param_count {param_count} != expected {EXPECTED_PARAM_COUNT}"
    )
    _val, common_ctx_np, p_common, c_true_common = build_diffuse_surfaces(base)
    return ExperimentSpec("diffuse", cfg, common_ctx_np, p_common, c_true_common, None)


def build_mode_aligned_spec() -> ExperimentSpec:
    """Builds the mode-aligned specification, after prior injection."""
    base = base_config(smoke=False)
    _n_levels, _seeds, size = grid_spec(smoke=False)
    d_model, n_head, n_layer = size
    cfg = dataclasses.replace(base, d_model=d_model, n_head=n_head, n_layer=n_layer)
    _val, common_ctx_np, p_common, c_true_common = build_aligned_surfaces(
        base, smoke=False
    )
    support_m = support_size(smoke=False)
    steps = pretrain_steps(smoke=False)

    def injector(model: GPT, cfg: Config, fit_seed: int, device: torch.device) -> dict:
        """Injects the mode-aligned prior into a freshly initialized model.

        Args:
            model: GPT to modify in place.
            cfg: Configuration.
            fit_seed: Fit seed for this ensemble member.
            device: Target device.

        Returns:
            Pre-training loss and prior mass on the support.
        """
        q0, support = build_prior(cfg.vocab_size, support_m, device)
        pretrain_loss, prior_mass = inject_prior(
            model, cfg, fit_seed, device, steps, q0, support
        )
        return {"pretrain_loss": float(pretrain_loss), "prior_mass": float(prior_mass)}

    return ExperimentSpec(
        "mode_aligned", cfg, common_ctx_np, p_common, c_true_common, injector
    )


def build_pre_finetune_members(spec: ExperimentSpec) -> list[dict]:
    """Builds the starting ensemble and its common-context conditionals.

    Args:
        spec: Starting-distribution specification.

    Returns:
        Per-seed member dicts carrying ``q_common``, in the form
        ``aggregate_level`` expects.
    """
    cfg = spec.cfg
    param_count = int(sum(p.numel() for p in GPT(cfg).parameters()))
    common_ctx = torch.from_numpy(spec.common_ctx_np).to(DEVICE)

    members = []
    seeds = fit_seeds()
    for seed_index, fit_seed in enumerate(
        tqdm(seeds, desc=f"{spec.name} initialization")
    ):
        torch.manual_seed(fit_seed)
        model = GPT(cfg).to(device=DEVICE, dtype=DTYPE)
        assert next(model.parameters()).dtype == DTYPE, (
            "model dtype mismatch vs source DTYPE"
        )
        if spec.injector is not None:
            spec.injector(model, cfg, fit_seed, DEVICE)
        q_common = (
            conditionals_on_contexts(model, common_ctx, cfg)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        assert np.isfinite(q_common).all(), (
            f"{spec.name} seed {seed_index}: non-finite conditionals"
        )
        members.append(
            {
                "level_index": 0,
                "seed_index": seed_index,
                "N": 0,
                "param_count": param_count,
                "q_common": q_common,
            }
        )
    return members


def bootstrap_cis(spec: ExperimentSpec, members: list[dict], point: dict) -> dict:
    """Bootstraps 95% intervals by resampling the common contexts.

    The seed ensemble is held fixed. ``BOOTSTRAP_B`` resamples use
    ``BOOTSTRAP_SEED``. The per-context identity is asserted on every resample.

    Args:
        spec: Starting-distribution specification.
        members: Starting-ensemble member dicts.
        point: Point estimate from ``aggregate_level``.

    Returns:
        Dict mapping each quantity to its ``{lo, hi}`` interval, plus bootstrap
        metadata.
    """
    p_common = spec.p_common
    c_true_common = spec.c_true_common
    n_seeds = len(members)
    n_ctx = p_common.shape[0]
    inv_c = 1.0 / c_true_common

    q_bar = np.zeros_like(p_common)
    c_sh = np.empty((n_seeds, n_ctx), dtype=np.float64)
    for seed_index, member in enumerate(members):
        q = member["q_common"].astype(np.float64)
        q_bar += q
        c_sh[seed_index] = (q * q).sum(axis=1)
    q_bar /= n_seeds

    var_h = np.zeros(n_ctx, dtype=np.float64)
    for member in members:
        q = member["q_common"].astype(np.float64)
        var_h += ((q - q_bar) ** 2).sum(axis=1)
    var_h /= n_seeds

    b = q_bar - p_common
    stats = {
        "var_term": var_h * inv_c,
        "bias2_term": (b * b).sum(axis=1) * inv_c,
        "cross_term": 2.0 * (p_common * b).sum(axis=1) * inv_c,
        "measured": (c_sh * inv_c[None, :]).mean(axis=0) - 1.0,
        "qstar_dot_q0": (p_common * q_bar).sum(axis=1),
        "c_qstar": c_true_common,
    }
    for term in ("var_term", "bias2_term", "cross_term", "measured"):
        uniform_mean = float(stats[term].mean())
        assert abs(uniform_mean - point[term]) < IDENTITY_TOL, (
            f"{spec.name}: context-mean {term} {uniform_mean:.6e} != aggregate {point[term]:.6e}"
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n_ctx, size=(BOOTSTRAP_B, n_ctx))
    boot = {name: values[idx].mean(axis=1) for name, values in stats.items()}

    identity = boot["var_term"] + boot["bias2_term"] + boot["cross_term"]
    max_identity_resid = float(np.max(np.abs(identity - boot["measured"])))
    assert max_identity_resid < IDENTITY_TOL, (
        f"{spec.name}: bootstrap identity broken, max resid {max_identity_resid:.3e}"
    )

    cis = {
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_unit": "common_contexts",
        "bootstrap_max_identity_residual": max_identity_resid,
    }
    for term, samples in boot.items():
        cis[term] = {
            "lo": float(np.percentile(samples, CI_LOW_PCT)),
            "hi": float(np.percentile(samples, CI_HIGH_PCT)),
        }
    return cis


def decompose(spec: ExperimentSpec, members: list[dict]) -> dict:
    """Computes the initialization decomposition and its bootstrap intervals.

    Args:
        spec: Starting-distribution specification.
        members: Starting-ensemble member dicts.

    Returns:
        Dict with the three terms, the normalized gap, intervals, and the
        overlap of the mean starting distribution with the target.
    """
    p_common = spec.p_common
    c_true_common = spec.c_true_common
    row = aggregate_level(members, p_common, c_true_common, SEEDS_PER_LEVEL)

    residual = abs(row["derived"] - row["measured"])
    assert residual < IDENTITY_TOL, (
        f"{spec.name}: decomposition identity broken, |derived-measured|={residual:.3e}"
    )

    q_bar = np.stack([m["q_common"].astype(np.float64) for m in members]).mean(axis=0)
    q_star_dot_q0_h = (p_common * q_bar).sum(axis=1)
    overlap_norm = float((q_star_dot_q0_h / c_true_common).mean())
    cross_from_overlap = 2.0 * (overlap_norm - 1.0)
    assert abs(cross_from_overlap - row["cross_term"]) < IDENTITY_TOL, (
        f"{spec.name}: cross-term/overlap mismatch {cross_from_overlap:.3e} vs {row['cross_term']:.3e}"
    )

    q_star_dot_q0_mean = float(q_star_dot_q0_h.mean())
    c_qstar_mean = float(c_true_common.mean())

    cis = bootstrap_cis(spec, members, row)
    if row["cross_term"] > 0:
        cross_sign = "positive"
    elif row["cross_term"] < 0:
        cross_sign = "negative"
    else:
        cross_sign = "zero"

    return {
        "name": spec.name,
        "K": int(spec.cfg.vocab_size),
        "param_count": int(row["param_count"]),
        "param_millions": round(row["param_count"] / 1e6, 3),
        "n_seeds": int(row["n_seeds"]),
        "n_common": int(row["n_common"]),
        "var_term": row["var_term"],
        "bias2_term": row["bias2_term"],
        "cross_term": row["cross_term"],
        "cross_sign": cross_sign,
        "derived": row["derived"],
        "measured": row["measured"],
        "identity_residual": residual,
        "ci_95": cis,
        "qstar_dot_q0_mean": q_star_dot_q0_mean,
        "c_qstar_mean": c_qstar_mean,
        "overlap_norm_mean_qq0_over_cqstar": overlap_norm,
        "qstar_dot_q0_gt_c_qstar": bool(overlap_norm > 1.0),
    }


def format_table(results: list[dict]) -> str:
    """Formats the point estimate and 95% interval for every reported term.

    Args:
        results: Per-distribution decomposition dicts.

    Returns:
        The formatted report.
    """
    terms = [
        ("Tr Cov (variance)", "var_term", "var_term"),
        ("||b||^2 (sq bias)", "bias2_term", "bias2_term"),
        ("cross 2 p^T b", "cross_term", "cross_term"),
        ("measured (gap)", "measured", "measured"),
        ("p^T q0", "qstar_dot_q0_mean", "qstar_dot_q0"),
        ("C(p)", "c_qstar_mean", "c_qstar"),
    ]
    lines = []
    for res in results:
        lines.append(
            f"{res['name']}: K={res['K']:,}  params={res['param_count']:,} "
            f"({res['param_millions']}M)  seeds={res['n_seeds']}  common_ctx={res['n_common']}"
        )
        for label, point_key, ci_key in terms:
            ci = res["ci_95"][ci_key]
            lines.append(
                f"    {label:<20} = {res[point_key]:>+.6e}   "
                f"95% CI [{ci['lo']:>+.6e}, {ci['hi']:>+.6e}]"
            )
        lines.append(
            f"    cross sign = {res['cross_sign']}   "
            f"p^T q0 = {res['qstar_dot_q0_mean']:.6e}   C(p) = {res['c_qstar_mean']:.6e}   "
            f"p^T q0 > C(p) = {res['qstar_dot_q0_gt_c_qstar']}   "
            f"identity_residual = {res['identity_residual']:.2e}"
        )
        lines.append("")
    return "\n".join(lines)


def run() -> None:
    """Scores both starting distributions and writes the initialization JSON."""
    assert torch.cuda.is_available(), "CUDA is required."
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    output_path().parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[init] torch={torch.__version__} device={DEVICE} seeds/level={SEEDS_PER_LEVEL}",
        flush=True,
    )

    builders = [build_diffuse_spec, build_mode_aligned_spec]
    results = []
    for builder in builders:
        spec = builder()
        print(
            f"[init] {spec.name}: K={spec.cfg.vocab_size:,} "
            f"d_model={spec.cfg.d_model} n_head={spec.cfg.n_head} n_layer={spec.cfg.n_layer} "
            f"common_contexts={spec.common_ctx_np.shape[0]}",
            flush=True,
        )
        members = build_pre_finetune_members(spec)
        res = decompose(spec, members)
        results.append(res)
        print(
            f"[init] {res['name']} done: var={res['var_term']:+.6e} bias2={res['bias2_term']:+.6e} "
            f"cross={res['cross_term']:+.6e} ({res['cross_sign']}) measured={res['measured']:+.6e} "
            f"residual={res['identity_residual']:.2e}",
            flush=True,
        )

    table = format_table(results)
    print("\n=== INITIALIZATION DECOMPOSITION ===", flush=True)
    print(table, flush=True)

    payload = {
        "seed": SEED,
        "seeds_per_level": SEEDS_PER_LEVEL,
        "state": "initialization",
        "note": (
            "Initialization decomposition, before fine-tuning on the target language. "
            "diffuse = random initialization; mode_aligned = after the Zipf-head prior "
            "injection. Terms are C(p)-normalized and averaged over the common contexts. "
            "Models are float32 with float64 collision math."
        ),
        "experiments": results,
    }
    with output_path().open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\n[init] wrote {output_path()}", flush=True)
