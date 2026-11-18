"""Bound check for the diffuse-prior GPT.

One 6.135M-parameter model is trained at each of 100 log-spaced sample sizes
from 32,000 to 1,000,000. At the best-validation checkpoint, 1,500 held-in
contexts record the measured collision deviation and the ell-2 and chi-squared
bounds.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from mode_collapse.paths import OUTPUT_DIR
from mode_collapse.synthetic.kgram import (
    COMPUTE_DTYPE,
    GPT,
    Config,
    build_language,
    collision,
    extract_contexts,
    generate_sequences,
    smoke_config,
    train_model,
    true_conditionals,
    val_collisions,
)
from mode_collapse.synthetic.predictive import context_predictive

# --- config ---

N_MIN = 32_000
N_MAX = 1_000_000
N_COUNT = 100
FIXED_SIZE = (384, 6, 3)

SMOKE_N_MIN = 1_000
SMOKE_N_MAX = 8_000
SMOKE_N_COUNT = 8
SMOKE_FIXED_SIZE = (64, 2, 2)

CTX_SEQ = 1500
N_CTX = 1500
PROB_FLOOR = 1e-40
SUBSAMPLE_SEED = 42

POINTS_NAME = "bound_points.csv"
PER_MODEL_NAME = "bound_per_model.csv"
SUMMARY_NAME = "bound_summary.json"
PROGRESS_NAME = "bound_progress.txt"

WORKER_DEVICE: torch.device | None = None


def synthetic_dir() -> Path:
    """Returns the Experiment 1 output directory."""
    return OUTPUT_DIR / "synthetic"


def log_line(message: str) -> None:
    """Prints a timestamped line and appends it to the progress file.

    Args:
        message: Message to record.
    """
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with (synthetic_dir() / PROGRESS_NAME).open("a") as handle:
        handle.write(line + "\n")


def init_worker(gpu_queue) -> None:
    """Pins this process to one GPU from the shared queue.

    Args:
        gpu_queue: Shared queue of free GPU indices.
    """
    global WORKER_DEVICE
    gpu_id = gpu_queue.get()
    torch.cuda.set_device(gpu_id)
    WORKER_DEVICE = torch.device(f"cuda:{gpu_id}")


def sweep_spec(smoke: bool):
    """Returns the sample-size grid and the fixed model size.

    Args:
        smoke: Whether to use the tiny smoke sweep.

    Returns:
        Tuple (n_min, n_max, n_count, fixed_size).
    """
    if smoke:
        return SMOKE_N_MIN, SMOKE_N_MAX, SMOKE_N_COUNT, SMOKE_FIXED_SIZE
    return N_MIN, N_MAX, N_COUNT, FIXED_SIZE


def n_values(smoke: bool) -> np.ndarray:
    """Returns the distinct log-spaced training-set sizes.

    Level ``i`` is ``round(n_min * (n_max / n_min) ** (i / (n_count - 1)))``.

    Args:
        smoke: Whether to use the tiny smoke sweep.

    Returns:
        Integer array of sample sizes, length ``n_count``.
    """
    n_min, n_max, n_count, _size = sweep_spec(smoke)
    ratio = n_max / n_min
    vals = np.array(
        [round(n_min * ratio ** (i / (n_count - 1))) for i in range(n_count)],
        dtype=np.int64,
    )
    n_unique = len(np.unique(vals))
    assert n_unique == n_count, f"expected {n_count} distinct N, got {n_unique}"
    return vals


def build_tasks(base: Config, smoke: bool) -> list[dict]:
    """Builds one task per sample size, all at the fixed GPT size.

    The data seed of task ``i`` is ``base.seed + 13 * (i + 1)``. The fit seed
    is ``base.seed + 4242 * (i + 1)``.

    Args:
        base: Base configuration.
        smoke: Whether to use the tiny smoke sweep.

    Returns:
        List of task dicts.
    """
    _nmin, _nmax, _nc, size = sweep_spec(smoke)
    d_model, n_head, n_layer = size
    fixed_cfg = dataclasses.replace(
        base, d_model=d_model, n_head=n_head, n_layer=n_layer
    )
    param_count = int(sum(p.numel() for p in GPT(fixed_cfg).parameters()))
    tasks = []
    for i, num_seq in enumerate(n_values(smoke)):
        tasks.append(
            {
                "m": i,
                "N": int(num_seq),
                "d_model": d_model,
                "n_head": n_head,
                "n_layer": n_layer,
                "param_count": param_count,
                "data_seed": base.seed + 13 * (i + 1),
                "fit_seed": base.seed + 4242 * (i + 1),
            }
        )
    return tasks


def train_member(packed) -> dict:
    """Trains one model and records the per-context bound on held-in contexts.

    Args:
        packed: Tuple (base_cfg, task_dict, val_seqs_np, c_true_val_np).

    Returns:
        Dict with held-out collisions and the subsampled bound arrays from the
        best-validation checkpoint.
    """
    base, task, val_seqs_np, c_true_val_np = packed
    device = WORKER_DEVICE
    assert device is not None, "worker device was not initialized"

    cfg = dataclasses.replace(
        base, d_model=task["d_model"], n_head=task["n_head"], n_layer=task["n_layer"]
    )
    lang = build_language(cfg, device)
    train_seqs = generate_sequences(
        lang, task["N"], cfg.seq_len, task["data_seed"], device
    )
    val_seqs = torch.from_numpy(val_seqs_np).to(device)

    model, best_state, best_val, best_step, final_val, _traj = train_model(
        cfg,
        train_seqs,
        val_seqs,
        task["fit_seed"],
        device,
        c_true_val_np,
        record_trajectory=False,
    )
    assert best_state is not None, (
        f"m={task['m']} N={task['N']}: no best-validation checkpoint"
    )

    collisions_final = val_collisions(model, val_seqs, cfg)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    collisions_bestval = val_collisions(model, val_seqs, cfg)

    ctx_seqs = train_seqs[: min(CTX_SEQ, train_seqs.shape[0])]
    contexts = extract_contexts(ctx_seqs, cfg.order_k)
    p_h = true_conditionals(lang, contexts).to(COMPUTE_DTYPE)
    _hidden, q_h = context_predictive(model, ctx_seqs, cfg, device)
    assert p_h.shape == q_h.shape, "p_h and q_h misaligned"

    c_p = (p_h**2).sum(dim=-1)
    c_q = (q_h**2).sum(dim=-1)
    n2_true = 1.0 / c_p
    factor = c_q / c_p
    measured = (factor - 1.0).abs()

    delta = q_h - p_h
    l2sq = (delta * delta).sum(dim=-1)
    l2 = torch.sqrt(l2sq.clamp(min=0.0))
    p_safe = p_h.clamp(min=PROB_FLOOR)
    chi2 = ((delta * delta) / p_safe).sum(dim=-1)
    kl = (p_h * (torch.log(p_safe) - torch.log(q_h.clamp(min=PROB_FLOOR)))).sum(dim=-1)

    bound_l2 = 2.0 * torch.sqrt(n2_true) * l2 + n2_true * l2sq
    bound_chi2 = 2.0 * torch.sqrt(n2_true * chi2.clamp(min=0.0)) + n2_true * chi2
    ratio = measured / bound_l2.clamp(min=1e-12)

    n_contexts = measured.shape[0]
    rng = np.random.default_rng(SUBSAMPLE_SEED + task["m"])
    take = min(N_CTX, n_contexts)
    sel = rng.choice(n_contexts, size=take, replace=False)
    sel_t = torch.from_numpy(sel).to(device)

    def grab(tensor: torch.Tensor) -> np.ndarray:
        """Returns the subsampled tensor as a NumPy array.

        Args:
            tensor: Per-context values of shape (n_contexts,).

        Returns:
            The ``take`` selected values.
        """
        return tensor[sel_t].cpu().numpy()

    return {
        "m": task["m"],
        "num_seq": task["N"],
        "param_count": task["param_count"],
        "d_model": task["d_model"],
        "best_val": float(best_val),
        "final_val": float(final_val),
        "best_step": int(best_step),
        "collisions_bestval": collisions_bestval,
        "collisions_final": collisions_final,
        "n_contexts_full": int(n_contexts),
        "n_violations_full": int((measured > bound_l2 * (1.0 + 1e-9) + 1e-12).sum()),
        "kl_min_raw": float(kl.min()),
        "median_measured_full": float(measured.median()),
        "median_bound_l2_full": float(bound_l2.median()),
        "median_ratio_full": float(ratio.median()),
        "measured": grab(measured),
        "l2": grab(l2),
        "kl": grab(kl),
        "bound_l2": grab(bound_l2),
        "bound_chi2": grab(bound_chi2),
    }


def summarize_models(context_bound_df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes the measured-to-bound ratio of each model.

    Every model has its own sample size, so one row per sample size is one row
    per model. Models are indexed in increasing order of sample size.

    Args:
        context_bound_df: One row per held-in context with ``N``,
            ``param_count``, ``measured_dev`` and ``bound_l2``.

    Returns:
        One row per model with the medians of the measured deviation and of
        the bound, quantiles of their ratio, and the bound coverage.
    """
    rows = []
    for model_id, (num_seq, bound_df) in enumerate(
        context_bound_df.groupby("N", sort=True)
    ):
        measured = bound_df["measured_dev"].to_numpy()
        bound = bound_df["bound_l2"].to_numpy()
        ratio = measured / np.clip(bound, 1e-12, None)
        covered = measured <= bound * (1.0 + 1e-9) + 1e-12
        rows.append(
            {
                "m": model_id,
                "N": int(num_seq),
                "param_count": int(bound_df["param_count"].iloc[0]),
                "median_measured": float(np.median(measured)),
                "median_bound_l2": float(np.median(bound)),
                "min_ratio": float(np.min(ratio)),
                "q25_ratio": float(np.quantile(ratio, 0.25)),
                "median_ratio": float(np.median(ratio)),
                "q75_ratio": float(np.quantile(ratio, 0.75)),
                "max_ratio": float(np.max(ratio)),
                "frac_under": float(covered.mean()),
                "n_violations": int((~covered).sum()),
            }
        )
    return pd.DataFrame(rows)


def run(smoke: bool) -> None:
    """Trains the bound sweep and writes one row per held-in context.

    Args:
        smoke: Whether to use the tiny smoke sweep.
    """
    base = smoke_config() if smoke else Config()
    out_dir = synthetic_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / PROGRESS_NAME).write_text("")
    log_line(
        f"python started smoke={smoke} torch={torch.__version__} fixed_steps={base.fixed_steps}"
    )
    assert torch.cuda.is_available(), "CUDA is required."
    num_gpus = torch.cuda.device_count()

    torch.manual_seed(base.seed)
    np.random.seed(base.seed)

    _nmin, _nmax, n_count, fixed_size = sweep_spec(smoke)

    lang_cpu = build_language(base, torch.device("cpu"))
    val_seqs = generate_sequences(
        lang_cpu, base.val_seq, base.seq_len, base.seed + 99, torch.device("cpu")
    )
    val_seqs_np = val_seqs.numpy()
    val_contexts = extract_contexts(val_seqs, base.order_k)
    c_true_val = collision(true_conditionals(lang_cpu, val_contexts).numpy())
    n2 = 1.0 / c_true_val

    tasks = build_tasks(base, smoke)
    distinct_n = sorted({t["N"] for t in tasks})
    params_unique = sorted({t["param_count"] for t in tasks})
    assert len(tasks) == n_count, f"expected {n_count} models, built {len(tasks)}"
    assert len(distinct_n) == n_count, (
        f"expected {n_count} distinct N, got {len(distinct_n)}"
    )
    assert len(params_unique) == 1, (
        f"expected a single fixed size, got {len(params_unique)}: {params_unique}"
    )
    log_line(
        f"sweep: {len(tasks)} GPTs over {len(distinct_n)} distinct N "
        f"in [{distinct_n[0]:,}..{distinct_n[-1]:,}]; "
        f"fixed size d_model={fixed_size[0]} n_head={fixed_size[1]} n_layer={fixed_size[2]} "
        f"({params_unique[0] / 1e6:.3f}M params, {params_unique[0]:,}); 1 seed each; "
        f"N2 median={np.median(n2):.1f} range=[{n2.min():.1f}, {n2.max():.1f}]"
    )

    packed = [(base, task, val_seqs_np, c_true_val) for task in tasks]
    processes = min(num_gpus, len(packed))
    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Manager().Queue()
    for gpu_id in range(processes):
        gpu_queue.put(gpu_id)

    results = []
    with ctx.Pool(
        processes=processes, initializer=init_worker, initargs=(gpu_queue,)
    ) as pool:
        progress = pool.imap_unordered(train_member, packed)
        for res in tqdm(progress, total=len(packed), desc="bound sweep"):
            final_factor = float(np.mean(res["collisions_final"] / c_true_val))
            best_factor = float(np.mean(res["collisions_bestval"] / c_true_val))
            log_line(
                f"m={res['m']:>3} N={res['num_seq']:>9,} P={res['param_count'] / 1e6:.2f}M "
                f"best_val={res['best_val']:.3f}@{res['best_step']} final_val={res['final_val']:.3f} "
                f"final={final_factor:.3f} best={best_factor:.3f} "
                f"med_ratio={res['median_ratio_full']:.3f} viol={res['n_violations_full']}"
            )
            results.append(res)

    assert len(results) == len(tasks), f"trained {len(results)} of {len(tasks)} models"
    results.sort(key=lambda r: r["num_seq"])
    assert len({r["param_count"] for r in results}) == 1, (
        "results contain more than one parameter size"
    )
    assert len({r["num_seq"] for r in results}) == n_count, (
        "results do not cover all distinct N"
    )

    rows = []
    for res in results:
        point_cols = zip(
            res["measured"],
            res["l2"],
            res["kl"],
            res["bound_l2"],
            res["bound_chi2"],
            strict=True,
        )
        for measured, l2, kl, b_l2, b_chi2 in point_cols:
            rows.append(
                {
                    "N": res["num_seq"],
                    "param_count": res["param_count"],
                    "measured_dev": float(measured),
                    "delta_l2": float(l2),
                    "kl": float(kl),
                    "bound_l2": float(b_l2),
                    "bound_chi2": float(b_chi2),
                }
            )
    context_bound_df = pd.DataFrame(rows)
    point_cols = [
        "N",
        "param_count",
        "measured_dev",
        "delta_l2",
        "kl",
        "bound_l2",
        "bound_chi2",
    ]
    context_bound_df[point_cols].to_csv(out_dir / POINTS_NAME, index=False)

    measured_all = context_bound_df["measured_dev"].to_numpy()
    bound_all = context_bound_df["bound_l2"].to_numpy()
    under = measured_all <= bound_all * (1.0 + 1e-9) + 1e-12
    frac_under = float(under.mean())
    ratio_all = measured_all / np.clip(bound_all, 1e-12, None)
    med_ratio = float(np.median(ratio_all))
    max_ratio = float(np.max(ratio_all))
    n_viol_total = int((~under).sum())

    model_bound_df = summarize_models(context_bound_df)
    model_ids = dict(zip(model_bound_df["N"], model_bound_df["m"], strict=True))
    assert all(model_ids[res["num_seq"]] == res["m"] for res in results), (
        "model index does not follow the order of the sample sizes"
    )
    model_bound_df.to_csv(out_dir / PER_MODEL_NAME, index=False)

    summary = {
        "design": {
            "n_distinct_N": len(distinct_n),
            "n_gpts": len(results),
            "seeds_per_N": 1,
            "N_min": int(distinct_n[0]),
            "N_max": int(distinct_n[-1]),
            "fixed_size": {
                "d_model": fixed_size[0],
                "n_head": fixed_size[1],
                "n_layer": fixed_size[2],
                "param_count": int(params_unique[0]),
                "param_millions": round(params_unique[0] / 1e6, 3),
            },
            "fixed_steps_full_training": base.fixed_steps,
        },
        "n2_true": {
            "median": float(np.median(n2)),
            "min": float(n2.min()),
            "max": float(n2.max()),
        },
        "provable_bound": {
            "n_contexts_total": int(len(context_bound_df)),
            "frac_under_bound": frac_under,
            "n_violations": n_viol_total,
            "median_ratio": med_ratio,
            "worst_ratio": max_ratio,
            "kl_min_raw": float(min(r["kl_min_raw"] for r in results)),
        },
    }
    with (out_dir / SUMMARY_NAME).open("w") as handle:
        json.dump(summary, handle, indent=2)

    log_line("=== SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)
    if n_viol_total > 0:
        log_line(f"WARNING: {n_viol_total} ell-2 bound violations across the sweep.")
    log_line("DONE")
