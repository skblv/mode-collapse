"""Mode-aligned prior grid for Experiment 1.

The target is a heavy-tailed order-16 Zipf-plus-context language on a
vocabulary of 50,000 tokens. Each GPT is pre-trained so that its next-token
distribution concentrates on the 32 highest-probability tokens of the global
Zipf component, then fine-tuned. The grid uses 100 seeds at each of 100
log-spaced sample sizes from 500 to 1,000,000.
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
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

from mode_collapse.synthetic.contexts import SUBSAMPLE_SEED, conditionals_on_contexts
from mode_collapse.synthetic.diffuse_grid import (
    aggregate_level,
    build_level_seed_tasks,
    synthetic_dir,
)
from mode_collapse.synthetic.kgram import (
    COMPUTE_DTYPE,
    DTYPE,
    GPT,
    Config,
    eval_loss,
    extract_contexts,
    loss_mask,
    lr_at_step,
)
from mode_collapse.synthetic.zipf_language import (
    LOG_ZIPF_WEIGHT,
    PERTURB_SCALE,
    ZIPF_S,
    build_language,
    generate_sequences,
    global_zipf_pmf,
    true_conditionals,
)

# --- config ---

SEED = 42

VOCAB_K = 50_000
ORDER_K = 16
EMB_DIM = 128
FEATURE_DIM = 256
SEQ_LEN = ORDER_K + 1
VAL_SEQ = 8192


PRIOR_SUPPORT_M = 32
SMOKE_PRIOR_SUPPORT_M = 8

PRIOR_PRETRAIN_STEPS = 400
SMOKE_PRIOR_PRETRAIN_STEPS = 120

D_MODEL = 256
N_HEAD = 4
N_LAYER = 3
FIXED_STEPS = 4000
BATCH = 128
EVAL_EVERY = 200

SMOKE_VOCAB_K = 2000
SMOKE_FIXED_SIZE = (64, 2, 2)

N_COMMON = 256
SMOKE_N_COMMON = 64

SEEDS_PER_LEVEL = 100
SMOKE_SEEDS_PER_LEVEL = 24

SMOKE_N_VALUES = np.array([200, 400, 800, 2000, 6000], dtype=np.int64)

N_MIN = 500
N_MAX = 1_000_000
N_LEVELS = 100

PARTS_DIRNAME = "aligned_parts"
TERMS_NAME = "aligned_terms.csv"
SUMMARY_NAME = "aligned_summary.json"
PROGRESS_NAME = "aligned_progress.txt"

WORKER_DEVICE: torch.device | None = None


def base_config(smoke: bool) -> Config:
    """Returns the mode-aligned configuration.

    Args:
        smoke: Whether to use the tiny smoke configuration.

    Returns:
        Configuration with the mode-aligned vocabulary, size, and step budget.
    """
    if smoke:
        return Config(
            seed=SEED,
            vocab_size=SMOKE_VOCAB_K,
            order_k=8,
            emb_dim=32,
            feature_dim=48,
            seq_len=9,
            val_seq=2048,
            d_model=SMOKE_FIXED_SIZE[0],
            n_head=SMOKE_FIXED_SIZE[1],
            n_layer=SMOKE_FIXED_SIZE[2],
            fixed_steps=400,
            batch=64,
            eval_every=50,
        )
    return Config(
        seed=SEED,
        vocab_size=VOCAB_K,
        order_k=ORDER_K,
        emb_dim=EMB_DIM,
        feature_dim=FEATURE_DIM,
        seq_len=SEQ_LEN,
        val_seq=VAL_SEQ,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        fixed_steps=FIXED_STEPS,
        batch=BATCH,
        eval_every=EVAL_EVERY,
    )


def grid_spec(smoke: bool) -> tuple[int, int, tuple[int, int, int]]:
    """Returns the grid size and the fixed model shape.

    Args:
        smoke: Whether to use the tiny smoke grid.

    Returns:
        Tuple (n_levels, seeds_per_level, fixed_size).
    """
    if smoke:
        return len(SMOKE_N_VALUES), SMOKE_SEEDS_PER_LEVEL, SMOKE_FIXED_SIZE
    return len(grid_n_values(False)), SEEDS_PER_LEVEL, (D_MODEL, N_HEAD, N_LAYER)


def grid_n_values(smoke: bool) -> np.ndarray:
    """Returns the training-set size at each level.

    The full grid is 100 log-spaced sizes from 500 to 1,000,000.

    Args:
        smoke: Whether to return the five smoke sizes.

    Returns:
        Integer array of sample sizes.
    """
    if smoke:
        return SMOKE_N_VALUES
    grid = np.unique(
        np.round(np.logspace(np.log10(N_MIN), np.log10(N_MAX), N_LEVELS)).astype(
            np.int64
        )
    )
    assert grid.size == N_LEVELS, (
        f"grid has {grid.size} unique levels, expected {N_LEVELS}"
    )
    return grid


def support_size(smoke: bool) -> int:
    """Returns how many tokens the injected prior concentrates on.

    Args:
        smoke: Whether to use the smoke support size.

    Returns:
        Support size.
    """
    return SMOKE_PRIOR_SUPPORT_M if smoke else PRIOR_SUPPORT_M


def pretrain_steps(smoke: bool) -> int:
    """Returns the number of prior-injection steps.

    Args:
        smoke: Whether to use the smoke step count.

    Returns:
        Step count.
    """
    return SMOKE_PRIOR_PRETRAIN_STEPS if smoke else PRIOR_PRETRAIN_STEPS


def common_count(smoke: bool) -> int:
    """Returns the number of fixed common evaluation contexts.

    Args:
        smoke: Whether to use the smoke context count.

    Returns:
        Context count.
    """
    return SMOKE_N_COMMON if smoke else N_COMMON


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


# --- prior ---


def build_prior(
    vocab: int, support_m: int, device: torch.device
) -> tuple[torch.Tensor, np.ndarray]:
    """Builds the mode-aligned prior and its support.

    The prior is the global Zipf distribution restricted to the first
    ``support_m`` token ids and renormalized. Those ids are the heaviest tokens
    of the global Zipf component.

    Args:
        vocab: Vocabulary size.
        support_m: Number of support tokens.
        device: Target device.

    Returns:
        Tuple (q0, support). ``q0`` is a float64 distribution of shape (vocab,).
        ``support`` is the int64 support ids.
    """
    assert 1 <= support_m < vocab, f"support_m={support_m} out of range [1, {vocab})"
    z = global_zipf_pmf(vocab, device)
    q0 = torch.zeros(vocab, dtype=COMPUTE_DTYPE, device=device)
    support = np.arange(support_m, dtype=np.int64)
    head = z[:support_m]
    q0[:support_m] = head / head.sum()
    assert torch.isfinite(q0).all(), "non-finite prior distribution"
    total = float(q0.sum())
    assert abs(total - 1.0) < 1e-10, f"prior distribution does not sum to 1: {total}"
    return q0, support


def inject_prior(
    model: GPT,
    cfg: Config,
    fit_seed: int,
    device: torch.device,
    steps: int,
    q0: torch.Tensor,
    support: np.ndarray,
) -> tuple[float, float]:
    """Pre-trains the model so every context concentrates on the prior.

    The objective is the soft cross-entropy of ``q0`` against the model softmax
    on random contexts. A fresh optimizer is used. Masking and clipping match
    the fine-tuning loop.

    Args:
        model: GPT to pre-train in place.
        cfg: Configuration.
        fit_seed: Seed controlling the pre-training minibatch stream.
        device: Target device.
        steps: Number of pre-training optimizer steps.
        q0: Target prior distribution of shape (vocab,).
        support: Support token ids.

    Returns:
        Tuple (final_pretrain_loss, prior_mass_on_support). The mass is the
        average softmax mass on the support for a fresh random batch.
    """
    assert steps >= 1, f"pretrain steps must be >= 1, got {steps}"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sampler = torch.Generator(device=device).manual_seed(fit_seed + 7)
    mask = loss_mask(cfg, device)
    flat_mask = mask[None, :].expand(cfg.batch, -1).reshape(-1)
    q0_model = q0.to(dtype=DTYPE)
    support_idx = torch.from_numpy(support).to(device)

    model.train()
    last_loss = float("nan")
    for _step in range(steps):
        rand_ctx = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch, cfg.seq_len),
            generator=sampler,
            device=device,
        )
        logits = model(rand_ctx[:, :-1])
        flat_logits = logits.reshape(-1, cfg.vocab_size)[flat_mask]
        log_probs = F.log_softmax(flat_logits, dim=-1)
        loss = -(q0_model[None, :] * log_probs).sum(dim=-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        last_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        probe = torch.randint(
            0,
            cfg.vocab_size,
            (cfg.batch, cfg.seq_len),
            generator=sampler,
            device=device,
        )
        probe_logits = model(probe[:, :-1]).reshape(-1, cfg.vocab_size).float()
        probe_probs = F.softmax(probe_logits, dim=-1)
        prior_mass_on_support = float(
            probe_probs[:, support_idx].sum(dim=-1).mean().item()
        )
    assert np.isfinite(last_loss), "non-finite pre-training loss"
    return last_loss, prior_mass_on_support


def prior_train_model(
    cfg: Config,
    train_seqs: torch.Tensor,
    val_seqs: torch.Tensor,
    fit_seed: int,
    device: torch.device,
    steps: int,
    q0: torch.Tensor,
    support: np.ndarray,
):
    """Injects the mode-aligned prior, then fine-tunes on the Zipf language.

    Args:
        cfg: Configuration.
        train_seqs: Training sequences.
        val_seqs: Held-out validation sequences.
        fit_seed: Seed controlling initialization, prior injection, and minibatches.
        device: Target device.
        steps: Prior-injection steps.
        q0: Target prior distribution.
        support: Support token ids.

    Returns:
        Tuple (model, best_state, best_val, best_step, pretrain_loss, prior_mass).
    """
    torch.manual_seed(fit_seed)
    model = GPT(cfg).to(device=device, dtype=DTYPE)

    pretrain_loss, prior_mass = inject_prior(
        model, cfg, fit_seed, device, steps, q0, support
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sampler = torch.Generator(device=device).manual_seed(fit_seed + 1)
    mask = loss_mask(cfg, device)
    flat_mask = mask[None, :].expand(cfg.batch, -1).reshape(-1)

    num_train = train_seqs.shape[0]
    best_val = float("inf")
    best_step = -1
    best_state = None

    model.train()
    for step in range(cfg.fixed_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at_step(cfg, step)
        idx = torch.randint(num_train, (cfg.batch,), generator=sampler, device=device)
        batch = train_seqs[idx]
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        flat_logits = logits.reshape(-1, cfg.vocab_size)
        flat_targets = targets.reshape(-1)
        loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.eval_every == 0 or step == cfg.fixed_steps - 1:
            val = eval_loss(model, val_seqs, mask, cfg)
            if val < best_val - 1e-4:
                best_val = val
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            model.train()

    return model, best_state, best_val, best_step, pretrain_loss, prior_mass


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
    """Trains one prior-injected GPT and returns its common-context conditionals.

    Args:
        packed: Tuple (base_cfg, task, val_seqs_np, common_ctx_np, steps, support_m, zlog_np).

    Returns:
        Dict with conditionals, missing mass, and prior mass.
    """
    base, task, val_seqs_np, common_ctx_np, steps, support_m, zlog_np = packed
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

    q0, support = build_prior(cfg.vocab_size, support_m, device)
    model, best_state, best_val, _bs, pretrain_loss, prior_mass = prior_train_model(
        cfg, train_seqs, val_seqs, task["fit_seed"], device, steps, q0, support
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

    z_global = torch.from_numpy(zlog_np).to(device).exp()
    observed = torch.unique(train_seqs)
    seen = torch.zeros(cfg.vocab_size, dtype=torch.bool, device=device)
    seen[observed] = True
    missing_mass = float(z_global[~seen].sum().item())

    return {
        "level_index": task["level_index"],
        "seed_index": task["seed_index"],
        "N": task["N"],
        "param_count": task["param_count"],
        "best_val": float(best_val),
        "pretrain_loss": float(pretrain_loss),
        "prior_mass_on_support": float(prior_mass),
        "missing_mass": missing_mass,
        "q_common": q_common,
    }


def build_eval_surfaces(
    base: Config, smoke: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds validation sequences and the fixed common-context set.

    Args:
        base: Base configuration.
        smoke: Whether this is the smoke grid.

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
    n_common = min(common_count(smoke), all_val_contexts.shape[0])
    common_idx = common_rng.choice(
        all_val_contexts.shape[0], size=n_common, replace=False
    )
    common_contexts = all_val_contexts[torch.from_numpy(common_idx)]
    common_ctx_np = common_contexts.numpy()
    p_common = true_conditionals(lang_cpu, common_contexts).to(COMPUTE_DTYPE).numpy()
    c_true_common = (p_common**2).sum(axis=-1)
    return val_seqs_np, common_ctx_np, p_common, c_true_common


def prior_diagnostics(
    vocab: int, support_m: int, p_common: np.ndarray, c_true_common: np.ndarray
) -> dict:
    """Computes prior-target alignment and the target rank of the prior support.

    Args:
        vocab: Vocabulary size.
        support_m: Number of support tokens.
        p_common: True conditionals on the common contexts, shape (ctx, vocab).
        c_true_common: True collision per common context, shape (ctx,).

    Returns:
        Dict with mean alignment ``p * (q0 - p)`` and the mean target-rank of
        the support tokens.
    """
    q0, support = build_prior(vocab, support_m, torch.device("cpu"))
    q0_np = q0.numpy()
    align_per_ctx = (p_common * q0_np[None, :]).sum(axis=1) - c_true_common
    descending = np.argsort(-p_common, axis=1)
    rank_pos = np.argsort(descending, axis=1)
    support_rank = float(rank_pos[:, support].mean())
    return {
        "prior_target_alignment": float(align_per_ctx.mean()),
        "mean_support_rank_under_qstar": support_rank,
    }


def aggregate_aligned_level(
    members: list[dict],
    p_common: np.ndarray,
    c_true_common: np.ndarray,
    seeds_per_level: int,
    diagnostics: dict,
) -> dict:
    """Adds mode-aligned diagnostics to the three-term decomposition.

    Args:
        members: Per-seed result dicts for one sample size.
        p_common: True conditionals on the common contexts.
        c_true_common: True collision per common context.
        seeds_per_level: Expected number of seeds.
        diagnostics: Prior diagnostics, constant across sample sizes.

    Returns:
        The aggregated row, including the identity residual and the diagnostics.
    """
    row = aggregate_level(members, p_common, c_true_common, seeds_per_level)
    residual = abs(row["derived"] - row["measured"])
    assert residual < 1e-9, (
        f"decomposition identity broken at level {row['level_index']}: "
        f"|derived-measured|={residual:.3e}"
    )
    row["identity_residual"] = residual
    row["mean_prior_mass_on_support"] = float(
        np.mean([m["prior_mass_on_support"] for m in members])
    )
    row["mean_missing_mass"] = float(np.mean([m["missing_mass"] for m in members]))
    row["prior_target_alignment"] = diagnostics["prior_target_alignment"]
    row["mean_support_rank_under_qstar"] = diagnostics["mean_support_rank_under_qstar"]
    return row


def run_train_block(
    base: Config,
    smoke: bool,
    level_start: int,
    level_count: int,
    models_per_gpu: int,
) -> None:
    """Trains the mode-aligned ensembles for a block of sample sizes.

    Args:
        base: Base configuration.
        smoke: Whether this is the smoke grid.
        level_start: First level index, inclusive.
        level_count: Number of levels for this shard.
        models_per_gpu: Models trained concurrently per GPU.
    """
    assert models_per_gpu >= 1, f"models_per_gpu must be >= 1, got {models_per_gpu}"
    n_levels, seeds_per_level, size = grid_spec(smoke)
    support_m = support_size(smoke)
    steps = pretrain_steps(smoke)
    level_end = min(level_start + level_count, n_levels)
    assert 0 <= level_start < n_levels, (
        f"level_start {level_start} out of range [0, {n_levels})"
    )
    assert level_start < level_end, "empty level block"

    fixed_cfg = dataclasses.replace(
        base, d_model=size[0], n_head=size[1], n_layer=size[2]
    )
    param_count = int(sum(p.numel() for p in GPT(fixed_cfg).parameters()))

    assert torch.cuda.is_available(), "CUDA is required."
    num_gpus = torch.cuda.device_count()
    parts_dir().mkdir(parents=True, exist_ok=True)

    worker_slots = num_gpus * models_per_gpu
    log_line(
        f"TRAIN block levels [{level_start},{level_end}) of {n_levels} | "
        f"size d_model={size[0]} n_head={size[1]} n_layer={size[2]} "
        f"({param_count / 1e6:.3f}M, {param_count:,}) | K={base.vocab_size} | "
        f"seeds/level={seeds_per_level} | "
        f"support_m={support_m} pretrain_steps={steps} fixed_steps={base.fixed_steps} | "
        f"gpus={num_gpus} models_per_gpu={models_per_gpu} worker_slots={worker_slots}"
    )

    val_seqs_np, common_ctx_np, p_common, c_true_common = build_eval_surfaces(
        base, smoke
    )
    diagnostics = prior_diagnostics(base.vocab_size, support_m, p_common, c_true_common)
    log_line(
        f"common contexts={common_ctx_np.shape[0]} | "
        f"prior_target_alignment={diagnostics['prior_target_alignment']:+.5e} "
        f"mean_support_rank_under_qstar={diagnostics['mean_support_rank_under_qstar']:.1f}"
    )

    zlog_np = build_language(base, torch.device("cpu")).log_z.numpy()
    nvals = grid_n_values(smoke)

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
            packed = [
                (base, task, val_seqs_np, common_ctx_np, steps, support_m, zlog_np)
                for task in tasks
            ]
            members = []
            level_t0 = time.time()
            progress = pool.imap_unordered(train_member, packed)
            for res in tqdm(
                progress, total=len(packed), desc=f"level {level_index} N={nval}"
            ):
                members.append(res)
                done = len(members)
                if done == 1 or done % 8 == 0 or done == seeds_per_level:
                    elapsed = time.time() - level_t0
                    log_line(
                        f"  level {level_index} N={nval:,} progress {done}/{seeds_per_level} "
                        f"models | elapsed {elapsed:.0f}s | {elapsed / done:.1f}s/model"
                    )
            row = aggregate_aligned_level(
                members, p_common, c_true_common, seeds_per_level, diagnostics
            )
            part_path = parts_dir() / f"level_{level_index:03d}.json"
            with part_path.open("w") as handle:
                json.dump(row, handle, indent=2)
            log_line(
                f"level {level_index:>3} N={nval:>9,} seeds={row['n_seeds']} "
                f"derived={row['derived']:+.5f} measured={row['measured']:+.5f} "
                f"var={row['var_term']:+.4f} bias2={row['bias2_term']:+.4f} "
                f"cross={row['cross_term']:+.4f} "
                f"align={row['prior_target_alignment']:+.3e} miss={row['mean_missing_mass']:.3f} "
                f"q0_supp={row['mean_prior_mass_on_support']:.3f} -> {part_path.name}"
            )

    log_line(f"TRAIN block [{level_start},{level_end}) DONE")


def run_reduce(base: Config, smoke: bool) -> None:
    """Concatenates the per-level part files and writes the decomposition table.

    Args:
        base: Base configuration.
        smoke: Whether this is the smoke grid.
    """
    n_levels, seeds_per_level, size = grid_spec(smoke)
    support_m = support_size(smoke)
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
    assert all(r["n_seeds"] == seeds_per_level for r in rows), (
        "a level lacks the full seed count"
    )
    common_counts = {r["n_common"] for r in rows}
    assert len(common_counts) == 1, (
        "levels evaluated on different common-context counts"
    )
    n_common = common_counts.pop()
    total_models = sum(r["n_seeds"] for r in rows)
    expected_models = n_levels * seeds_per_level
    assert total_models == expected_models, (
        f"trained {total_models} models, expected {expected_models}"
    )

    out_cols = ["N", "derived", "measured", "var_term", "bias2_term", "cross_term"]
    diag_cols = [
        "prior_target_alignment",
        "mean_support_rank_under_qstar",
        "mean_missing_mass",
        "mean_prior_mass_on_support",
        "identity_residual",
    ]
    extra_cols = ["level_index", "n_seeds", "param_count"]
    csv_cols = out_cols + diag_cols + ["param_count", "n_seeds"]
    level_terms_df = (
        pd.DataFrame(rows)[out_cols + diag_cols + extra_cols]
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
    max_identity_residual = float(level_terms_df["identity_residual"].max())

    cross = level_terms_df["cross_term"].to_numpy()
    summary = {
        "design": {
            "regime": "mode_aligned_prior",
            "vocab_size": int(base.vocab_size),
            "zipf_s": ZIPF_S,
            "log_zipf_weight": LOG_ZIPF_WEIGHT,
            "perturb_scale": PERTURB_SCALE,
            "prior_support_m": support_m,
            "pretrain_steps": pretrain_steps(smoke),
            "fixed_steps": base.fixed_steps,
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
        "prior_diagnostics": {
            "prior_target_alignment": float(
                level_terms_df["prior_target_alignment"].iloc[0]
            ),
            "mean_support_rank_under_qstar": float(
                level_terms_df["mean_support_rank_under_qstar"].iloc[0]
            ),
            "missing_mass_range": [
                float(level_terms_df["mean_missing_mass"].min()),
                float(level_terms_df["mean_missing_mass"].max()),
            ],
            "prior_mass_on_support_range": [
                float(level_terms_df["mean_prior_mass_on_support"].min()),
                float(level_terms_df["mean_prior_mass_on_support"].max()),
            ],
        },
        "decomposition": {
            "pearson_r": pearson_r,
            "r2": pearson_r * pearson_r,
            "mean_abs_resid": mean_abs_resid,
            "max_identity_residual": max_identity_residual,
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
            "cross_term_range": [float(cross.min()), float(cross.max())],
            "cross_positive_levels": int((cross > 0).sum()),
            "cross_negative_levels": int((cross < 0).sum()),
        },
    }
    summary_path = synthetic_dir() / SUMMARY_NAME
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    log_line("=== MODE-ALIGNED REDUCE SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)
    log_line(
        f"REDUCE DONE: {total_models} models, {n_levels} levels x {seeds_per_level} seeds, "
        f"cross positive in {int((cross > 0).sum())}/{n_levels} levels, "
        f"cross range [{cross.min():+.4f}, {cross.max():+.4f}], "
        f"max identity residual {max_identity_residual:.2e}"
    )


def run(
    smoke: bool, reduce: bool, level_start: int, level_count: int, models_per_gpu: int
) -> None:
    """Trains one shard of the mode-aligned grid, or reduces finished parts.

    Args:
        smoke: Whether to use the tiny smoke grid.
        reduce: Whether to concatenate part files instead of training.
        level_start: First sample-size index for this shard.
        level_count: Number of sample sizes for this shard.
        models_per_gpu: Models trained concurrently per GPU.
    """
    base = base_config(smoke)
    synthetic_dir().mkdir(parents=True, exist_ok=True)
    log_line(
        f"python started smoke={smoke} reduce={reduce} torch={torch.__version__} "
        f"K={base.vocab_size} fixed_steps={base.fixed_steps} "
        f"pretrain_steps={pretrain_steps(smoke)} support_m={support_size(smoke)} "
        f"level_start={level_start} level_count={level_count}"
    )

    torch.manual_seed(base.seed)
    np.random.seed(base.seed)

    if reduce:
        run_reduce(base, smoke)
        return
    run_train_block(base, smoke, level_start, level_count, models_per_gpu)
