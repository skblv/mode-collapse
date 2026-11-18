"""Diffuse-prior decomposition grid for Experiment 1.

One hundred log-spaced sample sizes from 32,000 to 1,000,000 sequences, with
100 independently seeded 6.135M-parameter GPTs at each size. On a shared set of
contexts the ensemble splits the normalized collision gap into variance,
squared bias, and the cross term.
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
from scipy import stats
from tqdm import tqdm

from mode_collapse.paths import OUTPUT_DIR
from mode_collapse.synthetic.contexts import (
    COMMON_CTX,
    SUBSAMPLE_SEED,
    conditionals_on_contexts,
)
from mode_collapse.synthetic.kgram import (
    COMPUTE_DTYPE,
    GPT,
    Config,
    build_language,
    extract_contexts,
    generate_sequences,
    smoke_config,
    train_model,
    true_conditionals,
)

# --- config ---

BASE_SEED = 42

N_MIN = 32_000
N_MAX = 1_000_000
N_LEVELS = 100
SEEDS_PER_LEVEL = 100

FIXED_SIZE = (384, 6, 3)
EXPECTED_PARAM_COUNT = 6_135_168

FIT_SEED_OFFSET = 500_000

SMOKE_N_LEVELS = 4
SMOKE_SEEDS_PER_LEVEL = 3
SMOKE_FIXED_SIZE = (64, 2, 2)

PARTS_DIRNAME = "diffuse_parts"
TERMS_NAME = "diffuse_terms.csv"
SUMMARY_NAME = "diffuse_summary.json"
PROGRESS_NAME = "diffuse_progress.txt"

WORKER_DEVICE: torch.device | None = None


def synthetic_dir() -> Path:
    """Returns the Experiment 1 output directory."""
    return OUTPUT_DIR / "synthetic"


def grid_spec(smoke: bool) -> tuple[int, int, int, int, tuple[int, int, int]]:
    """Returns the sample-size grid and the fixed model size.

    Args:
        smoke: Whether to use the tiny smoke grid.

    Returns:
        Tuple (n_min, n_max, n_levels, seeds_per_level, fixed_size). The smoke
        grid keeps the full sample-size endpoints and shortens the number of
        levels.
    """
    if smoke:
        return N_MIN, N_MAX, SMOKE_N_LEVELS, SMOKE_SEEDS_PER_LEVEL, SMOKE_FIXED_SIZE
    return N_MIN, N_MAX, N_LEVELS, SEEDS_PER_LEVEL, FIXED_SIZE


def n_values(smoke: bool) -> np.ndarray:
    """Returns the log-spaced training-set sizes.

    Level ``i`` is ``round(n_min * (n_max / n_min) ** (i / (n_levels - 1)))``.

    Args:
        smoke: Whether to use the tiny smoke grid.

    Returns:
        Integer array of sample sizes, length ``n_levels``, ascending.
    """
    n_min, n_max, n_levels, _seeds, _size = grid_spec(smoke)
    ratio = n_max / n_min
    vals = np.array(
        [round(n_min * ratio ** (i / (n_levels - 1))) for i in range(n_levels)],
        dtype=np.int64,
    )
    n_unique = len(np.unique(vals))
    assert n_unique == n_levels, f"expected {n_levels} distinct N, got {n_unique}"
    return vals


def parts_dir() -> Path:
    """Returns the per-level part-file directory."""
    return synthetic_dir() / PARTS_DIRNAME


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


def train_member(packed) -> dict:
    """Trains one GPT and returns its conditionals on the common contexts.

    Args:
        packed: Tuple (base_cfg, task_dict, val_seqs_np, common_ctx_np).

    Returns:
        Dict with the model's common-context conditionals and metadata.
    """
    base, task, val_seqs_np, common_ctx_np = packed
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

    c_true_dummy = np.zeros(1, dtype=np.float64)
    model, best_state, best_val, _bs, _fv, _tr = train_model(
        cfg,
        train_seqs,
        val_seqs,
        task["fit_seed"],
        device,
        c_true_dummy,
        record_trajectory=False,
    )
    assert best_state is not None, (
        f"level={task['level_index']} seed={task['seed_index']} N={task['N']}: "
        "no best-validation checkpoint"
    )
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    common_ctx = torch.from_numpy(common_ctx_np).to(device)
    q_common = (
        conditionals_on_contexts(model, common_ctx, cfg)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    assert np.isfinite(q_common).all(), (
        f"level={task['level_index']} seed={task['seed_index']}: non-finite conditionals"
    )

    return {
        "level_index": task["level_index"],
        "seed_index": task["seed_index"],
        "N": task["N"],
        "param_count": task["param_count"],
        "best_val": float(best_val),
        "q_common": q_common,
    }


def aggregate_level(
    members: list[dict],
    p_common: np.ndarray,
    c_true_common: np.ndarray,
    seeds_per_level: int,
) -> dict:
    """Averages the decomposition over one sample size's seed ensemble.

    Args:
        members: Per-seed result dicts for a single sample size.
        p_common: True conditionals on the common contexts, float64, shape
            (ctx, vocab).
        c_true_common: True collision per common context, float64, shape (ctx,).
        seeds_per_level: Expected number of seeds.

    Returns:
        Dict with the aggregated row: sample size, the three terms, their sum
        (``derived``), and the measured normalized gap.
    """
    assert len(members) == seeds_per_level, (
        f"level has {len(members)} seeds, expected {seeds_per_level}"
    )
    params = {m["param_count"] for m in members}
    assert len(params) == 1, f"level mixes parameter sizes: {sorted(params)}"
    nvals = {m["N"] for m in members}
    assert len(nvals) == 1, f"level mixes N values: {sorted(nvals)}"

    stacked = [m["q_common"].astype(np.float64) for m in members]
    q_stack = np.stack(stacked)
    q_bar = q_stack.mean(axis=0)
    b = q_bar - p_common

    centered = q_stack - q_bar[None, :, :]
    var_ens_h = (centered**2).sum(axis=-1).mean(axis=0)
    bsq_h = (b * b).sum(axis=-1)
    cross_h = 2.0 * (p_common * b).sum(axis=-1)

    var_term = float((var_ens_h / c_true_common).mean())
    bias2_term = float((bsq_h / c_true_common).mean())
    cross_term = float((cross_h / c_true_common).mean())
    derived = var_term + bias2_term + cross_term

    c_q = (q_stack**2).sum(axis=-1)
    meas_per_seed = (c_q / c_true_common[None, :]).mean(axis=-1) - 1.0
    measured = float(meas_per_seed.mean())

    return {
        "level_index": int(members[0]["level_index"]),
        "N": int(members[0]["N"]),
        "param_count": int(members[0]["param_count"]),
        "n_seeds": len(members),
        "n_common": int(c_true_common.shape[0]),
        "var_term": var_term,
        "bias2_term": bias2_term,
        "cross_term": cross_term,
        "derived": derived,
        "measured": measured,
    }


def build_eval_surfaces(
    base: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds validation sequences and the fixed common-context set.

    Both are deterministic from the base configuration and ``SUBSAMPLE_SEED``.

    Args:
        base: Base configuration.

    Returns:
        Tuple (val_seqs_np, common_ctx_np, p_common, c_true_common).
    """
    cpu = torch.device("cpu")
    lang_cpu = build_language(base, cpu)
    val_seqs = generate_sequences(
        lang_cpu, base.val_seq, base.seq_len, base.seed + 99, cpu
    )
    val_seqs_np = val_seqs.numpy()
    all_val_contexts = extract_contexts(val_seqs, base.order_k)
    common_rng = np.random.default_rng(SUBSAMPLE_SEED)
    n_common = min(COMMON_CTX, all_val_contexts.shape[0])
    common_idx = common_rng.choice(
        all_val_contexts.shape[0], size=n_common, replace=False
    )
    common_contexts = all_val_contexts[torch.from_numpy(common_idx)]
    common_ctx_np = common_contexts.numpy()
    p_common = true_conditionals(lang_cpu, common_contexts).to(COMPUTE_DTYPE).numpy()
    c_true_common = (p_common**2).sum(axis=-1)
    return val_seqs_np, common_ctx_np, p_common, c_true_common


def build_level_seed_tasks(
    nval: int,
    level_index: int,
    seeds_per_level: int,
    size: tuple[int, int, int],
    param_count: int,
) -> list[dict]:
    """Builds the per-seed task descriptors for one sample size.

    The data seed is ``42 + 1000 * level_index + seed_index``. The fit seed
    adds ``FIT_SEED_OFFSET``.

    Args:
        nval: Training-set size for this level.
        level_index: Index of the sample size.
        seeds_per_level: Number of independently seeded models.
        size: Fixed (d_model, n_head, n_layer).
        param_count: Parameter count of that size.

    Returns:
        List of per-seed task dicts.
    """
    d_model, n_head, n_layer = size
    tasks = []
    for seed_index in range(seeds_per_level):
        base_seed = BASE_SEED + 1000 * level_index + seed_index
        tasks.append(
            {
                "level_index": level_index,
                "seed_index": seed_index,
                "N": int(nval),
                "d_model": d_model,
                "n_head": n_head,
                "n_layer": n_layer,
                "param_count": param_count,
                "data_seed": base_seed,
                "fit_seed": base_seed + FIT_SEED_OFFSET,
            }
        )
    return tasks


def run_train_block(
    base: Config, smoke: bool, level_start: int, level_count: int, models_per_gpu: int
) -> None:
    """Trains the ensembles for a contiguous block of sample sizes.

    Args:
        base: Base configuration.
        smoke: Whether this is the smoke grid.
        level_start: First level index, inclusive.
        level_count: Number of levels for this shard.
        models_per_gpu: Models trained concurrently per GPU.
    """
    assert models_per_gpu >= 1, f"models_per_gpu must be >= 1, got {models_per_gpu}"
    _nmin, _nmax, n_levels, seeds_per_level, size = grid_spec(smoke)
    level_end = min(level_start + level_count, n_levels)
    assert 0 <= level_start < n_levels, (
        f"level_start {level_start} out of range [0, {n_levels})"
    )
    assert level_start < level_end, "empty level block"

    fixed_cfg = dataclasses.replace(
        base, d_model=size[0], n_head=size[1], n_layer=size[2]
    )
    param_count = int(sum(p.numel() for p in GPT(fixed_cfg).parameters()))
    if not smoke:
        assert param_count == EXPECTED_PARAM_COUNT, (
            f"fixed size param_count {param_count} != expected {EXPECTED_PARAM_COUNT}"
        )

    assert torch.cuda.is_available(), "CUDA is required."
    num_gpus = torch.cuda.device_count()
    parts_dir().mkdir(parents=True, exist_ok=True)

    worker_slots = num_gpus * models_per_gpu
    log_line(
        f"TRAIN block levels [{level_start},{level_end}) of {n_levels} | "
        f"size d_model={size[0]} n_head={size[1]} n_layer={size[2]} "
        f"({param_count / 1e6:.3f}M, {param_count:,}) | seeds/level={seeds_per_level} | "
        f"gpus={num_gpus} models_per_gpu={models_per_gpu} worker_slots={worker_slots}"
    )

    val_seqs_np, common_ctx_np, p_common, c_true_common = build_eval_surfaces(base)
    nvals = n_values(smoke)
    log_line(
        f"common context set: {common_ctx_np.shape[0]} contexts (fixed across levels)"
    )

    ctx = mp.get_context("spawn")
    gpu_queue = ctx.Manager().Queue()
    processes = min(worker_slots, seeds_per_level)
    for slot in range(processes):
        gpu_queue.put(slot % num_gpus)

    with ctx.Pool(
        processes=processes, initializer=init_worker, initargs=(gpu_queue,)
    ) as pool:
        for level_index in range(level_start, level_end):
            nval = int(nvals[level_index])
            tasks = build_level_seed_tasks(
                nval, level_index, seeds_per_level, size, param_count
            )
            packed = [(base, task, val_seqs_np, common_ctx_np) for task in tasks]
            members = []
            progress = pool.imap_unordered(train_member, packed)
            for res in tqdm(
                progress, total=len(packed), desc=f"level {level_index} N={nval}"
            ):
                members.append(res)
            row = aggregate_level(members, p_common, c_true_common, seeds_per_level)
            part_path = parts_dir() / f"level_{level_index:03d}.json"
            with part_path.open("w") as handle:
                json.dump(row, handle, indent=2)
            log_line(
                f"level {level_index:>3} N={nval:>9,} seeds={row['n_seeds']} "
                f"derived={row['derived']:+.5f} measured={row['measured']:+.5f} "
                f"var={row['var_term']:+.4f} bias2={row['bias2_term']:+.4f} "
                f"cross={row['cross_term']:+.4f} -> {part_path.name}"
            )

    log_line(f"TRAIN block [{level_start},{level_end}) DONE")


def run_reduce(base: Config, smoke: bool) -> None:
    """Concatenates the per-level part files and writes the decomposition table.

    Args:
        base: Base configuration. The part files already contain the rows.
        smoke: Whether this is the smoke grid.
    """
    _nmin, _nmax, n_levels, seeds_per_level, size = grid_spec(smoke)
    part_files = sorted(parts_dir().glob("level_*.json"))
    assert len(part_files) == n_levels, (
        f"expected {n_levels} level part files in {parts_dir()}, found {len(part_files)}"
    )

    rows = []
    for part in part_files:
        with part.open() as handle:
            rows.append(json.load(handle))

    level_indices = sorted(r["level_index"] for r in rows)
    assert level_indices == list(range(n_levels)), (
        "level part files are not a contiguous 0..L-1 set"
    )

    param_counts = {r["param_count"] for r in rows}
    assert len(param_counts) == 1, (
        f"reduce found multiple sizes: {sorted(param_counts)}"
    )
    param_count = param_counts.pop()
    if not smoke:
        assert param_count == EXPECTED_PARAM_COUNT, (
            f"reduce param_count {param_count} != expected {EXPECTED_PARAM_COUNT}"
        )

    assert all(r["n_seeds"] == seeds_per_level for r in rows), (
        "a level does not have the full seed count"
    )
    common_counts = {r["n_common"] for r in rows}
    assert len(common_counts) == 1, (
        f"levels evaluated on different common-context counts: {sorted(common_counts)}"
    )
    n_common = common_counts.pop()
    total_models = sum(r["n_seeds"] for r in rows)
    expected_models = n_levels * seeds_per_level
    assert total_models == expected_models, (
        f"trained {total_models} models, expected {expected_models} ({n_levels} x {seeds_per_level})"
    )

    out_cols = ["N", "derived", "measured", "var_term", "bias2_term", "cross_term"]
    extra_cols = ["level_index", "n_seeds", "param_count"]
    csv_cols = [
        "N",
        "derived",
        "measured",
        "var_term",
        "bias2_term",
        "cross_term",
        "param_count",
        "n_seeds",
    ]
    level_terms_df = (
        pd.DataFrame(rows)[out_cols + extra_cols]
        .sort_values("N")
        .reset_index(drop=True)
    )
    csv_path = synthetic_dir() / TERMS_NAME
    level_terms_df[csv_cols].to_csv(csv_path, index=False)
    assert len(level_terms_df) == n_levels, (
        f"aggregated CSV has {len(level_terms_df)} rows, expected {n_levels}"
    )

    pearson_r = float(
        stats.pearsonr(level_terms_df["derived"], level_terms_df["measured"])[0]
    )
    mean_abs_resid = float(
        np.mean(np.abs(level_terms_df["derived"] - level_terms_df["measured"]))
    )

    summary = {
        "design": {
            "n_levels": n_levels,
            "seeds_per_level": seeds_per_level,
            "total_models": total_models,
            "param_count": int(param_count),
            "param_millions": round(param_count / 1e6, 3),
            "fixed_size": {"d_model": size[0], "n_head": size[1], "n_layer": size[2]},
            "N_min": int(level_terms_df["N"].min()),
            "N_max": int(level_terms_df["N"].max()),
            "common_contexts": int(n_common),
        },
        "decomposition": {
            "pearson_r": pearson_r,
            "r2": pearson_r * pearson_r,
            "mean_abs_resid": mean_abs_resid,
            "n_points": int(len(level_terms_df)),
            "derived_range": [
                float(level_terms_df["derived"].min()),
                float(level_terms_df["derived"].max()),
            ],
            "measured_range": [
                float(level_terms_df["measured"].min()),
                float(level_terms_df["measured"].max()),
            ],
            "var_term_range": [
                float(level_terms_df["var_term"].min()),
                float(level_terms_df["var_term"].max()),
            ],
            "bias2_term_range": [
                float(level_terms_df["bias2_term"].min()),
                float(level_terms_df["bias2_term"].max()),
            ],
            "cross_term_range": [
                float(level_terms_df["cross_term"].min()),
                float(level_terms_df["cross_term"].max()),
            ],
            "cross_negative_levels": int((level_terms_df["cross_term"] < 0).sum()),
        },
    }
    summary_path = synthetic_dir() / SUMMARY_NAME
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    log_line("=== REDUCE SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)
    log_line(
        f"REDUCE DONE: {total_models} models, {n_levels} levels x {seeds_per_level} seeds, "
        f"single {param_count:,} size, Pearson r={pearson_r:.4f}"
    )


def run(
    smoke: bool, reduce: bool, level_start: int, level_count: int, models_per_gpu: int
) -> None:
    """Trains one shard of the diffuse-prior grid, or reduces finished parts.

    Args:
        smoke: Whether to use the tiny smoke grid.
        reduce: Whether to concatenate part files instead of training.
        level_start: First sample-size index for this shard.
        level_count: Number of sample sizes for this shard.
        models_per_gpu: Models trained concurrently per GPU.
    """
    base = smoke_config() if smoke else Config()
    synthetic_dir().mkdir(parents=True, exist_ok=True)
    log_line(
        f"python started smoke={smoke} reduce={reduce} torch={torch.__version__} "
        f"fixed_steps={base.fixed_steps} level_start={level_start} level_count={level_count}"
    )

    torch.manual_seed(base.seed)
    np.random.seed(base.seed)

    if reduce:
        run_reduce(base, smoke)
        return
    run_train_block(base, smoke, level_start, level_count, models_per_gpu)
