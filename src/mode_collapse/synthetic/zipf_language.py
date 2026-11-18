"""Zipf-plus-context language of the mode-aligned case of Experiment 1.

Next-token logits are the log of a global Zipf law over token ids plus a
per-context perturbation, so every context shares the same few dominant tokens.
The conditionals are computable exactly for any context.
"""

from __future__ import annotations

import dataclasses

import torch

from mode_collapse.synthetic.kgram import COMPUTE_DTYPE, Config, fixed_normal

# --- config ---

ZIPF_S = 1.07
LOG_ZIPF_WEIGHT = 1.0
PERTURB_SCALE = 0.3
GEN_CHUNK = 2000


@dataclasses.dataclass
class ZipfLanguage:
    """Heavy-tailed language: a global Zipf law plus a per-context perturbation.

    Attributes:
        emb: Token embeddings, shape (vocab, emb_dim).
        pos_w: Positional pooling weights, shape (order_k,).
        w1: Feature weight, shape (feature_dim, emb_dim).
        w2: Readout weight, shape (vocab, feature_dim).
        log_z: Log of the global Zipf distribution, shape (vocab,).
        cfg: Originating configuration.
        device: Device the tensors live on.
    """

    emb: torch.Tensor
    pos_w: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    log_z: torch.Tensor
    cfg: Config
    device: torch.device


def global_zipf_pmf(vocab: int, device: torch.device) -> torch.Tensor:
    """Returns the fixed global Zipf distribution, decreasing in token id.

    Token id ``r`` has unnormalized weight ``1 / (r + 1) ** ZIPF_S``.

    Args:
        vocab: Vocabulary size.
        device: Target device.

    Returns:
        Float64 distribution of shape (vocab,) summing to 1.
    """
    ranks = torch.arange(vocab, device=device, dtype=COMPUTE_DTYPE)
    weights = (ranks + 1.0).pow(-ZIPF_S)
    return weights / weights.sum()


def build_language(cfg: Config, device: torch.device) -> ZipfLanguage:
    """Builds the seeded Zipf-plus-context language.

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
    w2 = fixed_normal(
        generator,
        (cfg.vocab_size, cfg.feature_dim),
        cfg.feature_dim,
        device,
        COMPUTE_DTYPE,
    )
    log_z = torch.log(global_zipf_pmf(cfg.vocab_size, device))
    return ZipfLanguage(emb, pos_w, w1, w2, log_z, cfg, device)


def true_conditionals(lang: ZipfLanguage, contexts: torch.Tensor) -> torch.Tensor:
    """Computes the exact next-token distribution for a batch of contexts.

    Logits are the global Zipf log-probabilities plus a per-context perturbation
    that is linear in a tanh feature of the pooled context embeddings.

    Args:
        lang: The language.
        contexts: Long tensor of shape (num, order_k).

    Returns:
        Float64 probabilities of shape (num, vocab).
    """
    emb = lang.emb[contexts]
    pooled = torch.einsum("k,nke->ne", lang.pos_w, emb)
    hidden = torch.tanh(pooled @ lang.w1.T)
    perturb = hidden @ lang.w2.T
    logits = LOG_ZIPF_WEIGHT * lang.log_z[None, :] + PERTURB_SCALE * perturb
    return torch.softmax(logits, dim=-1)


def generate_sequences(
    lang: ZipfLanguage, num_seq: int, seq_len: int, seed: int, device: torch.device
) -> torch.Tensor:
    """Generates sequences from the Zipf-plus-context language.

    Prefix tokens are drawn for every sequence first. Later tokens are sampled
    in blocks of ``GEN_CHUNK``, each block with generator seed ``seed + 1 + start``.

    Args:
        lang: The language.
        num_seq: Number of sequences.
        seq_len: Sequence length.
        seed: RNG seed.
        device: Target device.

    Returns:
        Long tensor of shape (num_seq, seq_len).
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
