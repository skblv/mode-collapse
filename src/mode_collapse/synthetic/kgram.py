"""Order-16 latent-factor language, the small GPT, and its fixed-budget training loop.

The diffuse-prior grid and the bound sweep train this model on the closed-form
language. The mode-aligned grid reuses the GPT, the learning-rate schedule, and
the masked cross-entropy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- config ---

SEED = 42

VOCAB_SIZE = 1024
ORDER_K = 16
EMB_DIM = 128
FEATURE_DIM = 256
TAU_GEN = 0.6

SEQ_LEN = 64
VAL_SEQ = 96

GEN_CHUNK = 50_000

SCALING_N_MIN = 32_000
SCALING_N_MAX = 1_000_000
SCALING_NUM = 120
REPRESENTATIVE_N = [64_000, 256_000, 1_000_000]

D_MODEL = 256
N_HEAD = 4
N_LAYER = 3
MLP_RATIO = 4
DROPOUT = 0.1

FIXED_STEPS = 8000
BATCH = 512
LR = 1e-3
WEIGHT_DECAY = 0.1
WARMUP = 100
EVAL_EVERY = 250
GRAD_CLIP = 1.0

CALIB_CONTEXTS = 4096

UNIFORM_CE = float(np.log(VOCAB_SIZE))

DTYPE = torch.float32
COMPUTE_DTYPE = torch.float64


@dataclass
class Config:
    """Hyper-parameters for one synthetic-language run.

    Attributes:
        seed: Master seed used to derive every other seed.
        vocab_size: Token vocabulary size.
        order_k: Markov order of the ground-truth language.
        emb_dim: Dimension of the fixed token embeddings used by the language.
        feature_dim: Dimension of the latent context feature.
        tau_gen: Generation temperature, scaled by the global logit standard deviation.
        seq_len: Generated sequence length, also the model context window.
        val_seq: Number of held-out sequences for evaluation.
        scaling_n_min: Smallest training-set size stored on the configuration.
        scaling_n_max: Largest training-set size stored on the configuration.
        scaling_num: Number of sample sizes stored on the configuration.
        representative_n: Sample sizes stored for trajectory recording.
        d_model: Transformer width.
        n_head: Number of attention heads.
        n_layer: Number of transformer blocks.
        mlp_ratio: MLP expansion ratio inside each block.
        dropout: Dropout probability.
        fixed_steps: Fixed optimizer-step budget applied to every model.
        batch: Minibatch size in sequences per step.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup: Linear-warmup steps.
        eval_every: Validation evaluation period in steps.
        grad_clip: Gradient-norm clip value.
        calib_contexts: Contexts used to calibrate the global logit scale.
    """

    seed: int = SEED
    vocab_size: int = VOCAB_SIZE
    order_k: int = ORDER_K
    emb_dim: int = EMB_DIM
    feature_dim: int = FEATURE_DIM
    tau_gen: float = TAU_GEN
    seq_len: int = SEQ_LEN
    val_seq: int = VAL_SEQ
    scaling_n_min: int = SCALING_N_MIN
    scaling_n_max: int = SCALING_N_MAX
    scaling_num: int = SCALING_NUM
    representative_n: list[int] = field(default_factory=lambda: list(REPRESENTATIVE_N))
    d_model: int = D_MODEL
    n_head: int = N_HEAD
    n_layer: int = N_LAYER
    mlp_ratio: int = MLP_RATIO
    dropout: float = DROPOUT
    fixed_steps: int = FIXED_STEPS
    batch: int = BATCH
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    warmup: int = WARMUP
    eval_every: int = EVAL_EVERY
    grad_clip: float = GRAD_CLIP
    calib_contexts: int = CALIB_CONTEXTS


def smoke_config() -> Config:
    """Returns a tiny configuration that exercises the diffuse-prior code paths."""
    return Config(
        vocab_size=64,
        order_k=8,
        emb_dim=32,
        feature_dim=48,
        seq_len=32,
        val_seq=24,
        scaling_n_min=1000,
        scaling_n_max=8000,
        scaling_num=6,
        representative_n=[1000, 8000],
        d_model=64,
        n_head=2,
        n_layer=2,
        fixed_steps=300,
        batch=128,
        eval_every=50,
        calib_contexts=512,
    )


# --- language ---


@dataclass
class Language:
    """Fixed closed-form latent-factor language on one device.

    Attributes:
        emb: Token embeddings of shape (vocab_size, emb_dim).
        pos_w: Fixed positional pooling weights of shape (order_k,).
        w1: First feature weight, shape (feature_dim, emb_dim).
        b1: First feature bias, shape (feature_dim,).
        w2: Second feature weight, shape (feature_dim, feature_dim).
        b2: Second feature bias, shape (feature_dim,).
        readout: Readout of shape (vocab_size, feature_dim).
        bias: Readout bias of shape (vocab_size,).
        scale: Global logit standard deviation.
        cfg: The originating configuration.
        device: Device the tensors live on.
    """

    emb: torch.Tensor
    pos_w: torch.Tensor
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    readout: torch.Tensor
    bias: torch.Tensor
    scale: float
    cfg: Config
    device: torch.device


def fixed_normal(generator: torch.Generator, shape, fan_in: int, device, dtype):
    """Draws a fixed normal tensor scaled by 1/sqrt(fan_in).

    Args:
        generator: Seeded CPU generator.
        shape: Output shape.
        fan_in: Fan-in for the 1/sqrt scaling.
        device: Target device.
        dtype: Target dtype.

    Returns:
        The drawn tensor on the requested device and dtype.
    """
    tensor = torch.randn(*shape, generator=generator, dtype=torch.float64)
    tensor = tensor / (fan_in**0.5)
    return tensor.to(device=device, dtype=dtype)


def build_language(cfg: Config, device: torch.device) -> Language:
    """Builds the seeded ground-truth language and calibrates the logit scale.

    Args:
        cfg: Configuration.
        device: Device to place the language tensors on.

    Returns:
        A fully specified language.
    """
    generator = torch.Generator().manual_seed(cfg.seed + 7)
    emb = fixed_normal(
        generator, (cfg.vocab_size, cfg.emb_dim), 1, device, COMPUTE_DTYPE
    )
    pos_w = fixed_normal(generator, (cfg.order_k,), 1, device, COMPUTE_DTYPE)
    w1 = fixed_normal(
        generator, (cfg.feature_dim, cfg.emb_dim), cfg.emb_dim, device, COMPUTE_DTYPE
    )
    b1 = fixed_normal(generator, (cfg.feature_dim,), 1, device, COMPUTE_DTYPE) * 0.0
    w2 = fixed_normal(
        generator,
        (cfg.feature_dim, cfg.feature_dim),
        cfg.feature_dim,
        device,
        COMPUTE_DTYPE,
    )
    b2 = fixed_normal(generator, (cfg.feature_dim,), 1, device, COMPUTE_DTYPE) * 0.0
    readout = fixed_normal(
        generator,
        (cfg.vocab_size, cfg.feature_dim),
        cfg.feature_dim,
        device,
        COMPUTE_DTYPE,
    )
    bias = fixed_normal(generator, (cfg.vocab_size,), 1, device, COMPUTE_DTYPE)

    lang = Language(
        emb, pos_w, w1, b1, w2, b2, readout, bias, scale=1.0, cfg=cfg, device=device
    )

    calib_seqs = generate_sequences(
        lang, cfg.calib_contexts, cfg.seq_len, cfg.seed + 555, device
    )
    contexts = extract_contexts(calib_seqs, cfg.order_k)
    logits = raw_logits(lang, contexts)
    lang.scale = float(logits.std().item())
    return lang


def raw_logits(lang: Language, contexts: torch.Tensor) -> torch.Tensor:
    """Computes the untempered true logits for a batch of contexts.

    The map is linear in the positional-weighted sum of the context embeddings.
    Its rank is at most ``feature_dim``.

    Args:
        lang: The language.
        contexts: Long tensor of shape (num, order_k) of token ids.

    Returns:
        Float64 logits of shape (num, vocab_size).
    """
    emb = lang.emb[contexts]
    pooled = torch.einsum("k,bke->be", lang.pos_w, emb)
    hidden = pooled @ lang.w1.T + lang.b1
    phi = hidden @ lang.w2.T + lang.b2
    return phi @ lang.readout.T + lang.bias


def true_conditionals(lang: Language, contexts: torch.Tensor) -> torch.Tensor:
    """Computes the exact closed-form next-token distribution for each context.

    Args:
        lang: The language.
        contexts: Long tensor of shape (num, order_k) of token ids.

    Returns:
        Float64 probability tensor of shape (num, vocab_size).
    """
    logits = raw_logits(lang, contexts)
    inv_temp = 1.0 / (lang.cfg.tau_gen * lang.scale)
    return torch.softmax(logits * inv_temp, dim=-1)


def generate_sequences(
    lang: Language, num_seq: int, seq_len: int, seed: int, device: torch.device
) -> torch.Tensor:
    """Generates sequences autoregressively from the closed-form language.

    Prefix tokens are drawn for every sequence in one call. Later tokens are
    sampled in blocks of ``GEN_CHUNK``. Each block uses a generator seeded by
    ``seed + 1 + start``.

    Args:
        lang: The language.
        num_seq: Number of sequences.
        seq_len: Sequence length.
        seed: RNG seed.
        device: Target device.

    Returns:
        Long tensor of shape (num_seq, seq_len) of token ids.
    """
    assert num_seq > 0, f"num_seq must be positive, got {num_seq}"
    order_k = lang.cfg.order_k
    prefix_generator = torch.Generator(device=device).manual_seed(seed)
    seqs = torch.empty(num_seq, seq_len, dtype=torch.long, device=device)
    seqs[:, :order_k] = torch.randint(
        0,
        lang.cfg.vocab_size,
        (num_seq, order_k),
        generator=prefix_generator,
        device=device,
    )
    for start in range(0, num_seq, GEN_CHUNK):
        stop = min(start + GEN_CHUNK, num_seq)
        chunk_generator = torch.Generator(device=device).manual_seed(seed + 1 + start)
        chunk = seqs[start:stop]
        for pos in range(order_k, seq_len):
            contexts = chunk[:, pos - order_k : pos]
            probs = true_conditionals(lang, contexts)
            nxt = torch.multinomial(probs.float(), 1, generator=chunk_generator)
            chunk[:, pos] = nxt[:, 0]
            del contexts, probs, nxt
    return seqs


def extract_contexts(seqs: torch.Tensor, order_k: int) -> torch.Tensor:
    """Extracts every order-k context at a prediction position.

    Args:
        seqs: Token sequences of shape (num_seq, seq_len).
        order_k: Markov order.

    Returns:
        Long tensor of shape (num_seq * (seq_len - order_k), order_k), flattened
        in sequence-then-position order.
    """
    seq_len = seqs.shape[1]
    windows = [seqs[:, pos - order_k : pos] for pos in range(order_k, seq_len)]
    stacked = torch.stack(windows, dim=1)
    return stacked.reshape(-1, order_k)


def collision(prob: np.ndarray) -> np.ndarray:
    """Returns the order-2 collision probability along the last axis.

    Args:
        prob: Probabilities.

    Returns:
        Sum of squared probabilities along the last axis.
    """
    return np.sum(prob**2, axis=-1)


# --- model ---


class Block(nn.Module):
    """A single pre-norm transformer block with causal self-attention."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the block.

        Args:
            x: Hidden states of shape (batch, seq, d_model).

        Returns:
            Hidden states of the same shape.
        """
        batch, seq, _ = x.shape
        head_dim = self.d_model // self.n_head
        qkv = self.qkv(self.ln1(x))
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(batch, seq, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_head, head_dim).transpose(1, 2)
        drop_p = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)
        attn = attn.transpose(1, 2).reshape(batch, seq, self.d_model)
        x = x + self.resid_dropout(self.proj(attn))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Small GPT mapping token ids to next-token logits."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.seq_len = cfg.seq_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Runs the model.

        Args:
            idx: Token ids of shape (batch, seq).

        Returns:
            Logits of shape (batch, seq, vocab_size).
        """
        seq = idx.shape[1]
        positions = torch.arange(seq, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(positions)[None, :, :])
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def lr_at_step(cfg: Config, step: int) -> float:
    """Returns the linear-warmup then cosine-decay learning rate.

    Args:
        cfg: Configuration.
        step: Current optimizer step, 0-indexed.

    Returns:
        Learning rate for the step.
    """
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.fixed_steps - cfg.warmup)
    return 0.5 * cfg.lr * (1.0 + np.cos(np.pi * progress))


def loss_mask(cfg: Config, device: torch.device) -> torch.Tensor:
    """Builds the mask over predictions that have a full order-k context.

    Args:
        cfg: Configuration.
        device: Target device.

    Returns:
        Bool tensor of shape (seq_len - 1,). True where the predicted token's
        original position is at least ``order_k``.
    """
    target_positions = torch.arange(1, cfg.seq_len, device=device)
    return target_positions >= cfg.order_k


@torch.no_grad()
def eval_loss(
    model: GPT, val_seqs: torch.Tensor, mask: torch.Tensor, cfg: Config
) -> float:
    """Computes masked next-token cross-entropy.

    Args:
        model: The model.
        val_seqs: Validation sequences of shape (num, seq_len).
        mask: Position mask over the (seq_len - 1) prediction slots.
        cfg: Configuration.

    Returns:
        Mean per-token cross-entropy over masked positions.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, val_seqs.shape[0], 512):
        batch = val_seqs[start : start + 512]
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        flat_logits = logits.reshape(-1, cfg.vocab_size)
        flat_targets = targets.reshape(-1)
        flat_mask = mask[None, :].expand(batch.shape[0], -1).reshape(-1)
        loss = F.cross_entropy(
            flat_logits[flat_mask], flat_targets[flat_mask], reduction="sum"
        )
        total_loss += float(loss.item())
        total_tokens += int(flat_mask.sum().item())
    return total_loss / total_tokens


@torch.no_grad()
def val_collisions(model: GPT, val_seqs: torch.Tensor, cfg: Config) -> np.ndarray:
    """Returns the model collision at every held-out prediction position.

    Args:
        model: The model.
        val_seqs: Validation sequences of shape (num, seq_len).
        cfg: Configuration.

    Returns:
        Float64 collisions of shape (num_contexts,), flattened like
        ``extract_contexts``.
    """
    model.eval()
    chunks = []
    start_pos = cfg.order_k - 1
    for start in range(0, val_seqs.shape[0], 512):
        batch = val_seqs[start : start + 512]
        logits = model(batch[:, :-1])
        sel = logits[:, start_pos:, :].to(COMPUTE_DTYPE)
        probs = torch.softmax(sel, dim=-1)
        coll = (probs**2).sum(dim=-1).reshape(-1)
        chunks.append(coll.cpu())
    return torch.cat(chunks, dim=0).numpy()


# --- fitting ---


def train_model(
    cfg: Config,
    train_seqs: torch.Tensor,
    val_seqs: torch.Tensor,
    fit_seed: int,
    device: torch.device,
    c_true: np.ndarray,
    record_trajectory: bool,
):
    """Trains one GPT for the fixed step budget.

    The returned model holds the final weights. ``best_state`` holds the
    best-validation weights from the same trajectory.

    Args:
        cfg: Configuration.
        train_seqs: Training sequences.
        val_seqs: Held-out validation sequences.
        fit_seed: Seed controlling model initialization and minibatch sampling.
        device: Target device.
        c_true: True per-context collision, used when a trajectory is recorded.
        record_trajectory: Whether to record the per-evaluation trajectory.

    Returns:
        Tuple (model, best_state, best_val, best_step, final_val, traj).
    """
    torch.manual_seed(fit_seed)
    model = GPT(cfg).to(device=device, dtype=DTYPE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sampler = torch.Generator(device=device).manual_seed(fit_seed + 1)
    mask = loss_mask(cfg, device)

    num_train = train_seqs.shape[0]
    traj = {"step": [], "train_loss": [], "val_ce": [], "factor": []}
    best_val = float("inf")
    best_step = -1
    best_state = None
    final_val = float("nan")

    model.train()
    last_train_loss = float("nan")
    for step in range(cfg.fixed_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at_step(cfg, step)
        idx = torch.randint(num_train, (cfg.batch,), generator=sampler, device=device)
        batch = train_seqs[idx]
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        flat_logits = logits.reshape(-1, cfg.vocab_size)
        flat_targets = targets.reshape(-1)
        flat_mask = mask[None, :].expand(cfg.batch, -1).reshape(-1)
        loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        last_train_loss = float(loss.item())

        if step % cfg.eval_every == 0 or step == cfg.fixed_steps - 1:
            val = eval_loss(model, val_seqs, mask, cfg)
            final_val = val
            if record_trajectory:
                c_model = val_collisions(model, val_seqs, cfg)
                traj["step"].append(step)
                traj["train_loss"].append(last_train_loss)
                traj["val_ce"].append(val)
                traj["factor"].append(float(np.mean(c_model / c_true)))
            if val < best_val - 1e-4:
                best_val = val
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            model.train()

    return model, best_state, best_val, best_step, final_val, traj
