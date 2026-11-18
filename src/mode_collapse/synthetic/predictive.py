"""Hidden states and next-token conditionals of a trained GPT.

These readers score held-in contexts for the bound sweep. They do not estimate
parameter curvature.
"""

from __future__ import annotations

import torch

from mode_collapse.synthetic.kgram import COMPUTE_DTYPE, Config


def forward_hidden(model, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs the GPT and returns the pre-head hidden state and the logits.

    Args:
        model: A GPT module.
        idx: Token ids of shape (batch, seq).

    Returns:
        Tuple (hidden, logits). Hidden has shape (batch, seq, d_model) and
        logits have shape (batch, seq, vocab).
    """
    seq = idx.shape[1]
    positions = torch.arange(seq, device=idx.device)
    x = model.wte(idx) + model.wpe(positions)[None, :, :]
    for block in model.blocks:
        x = block(x)
    hidden = model.ln_f(x)
    logits = model.head(hidden)
    return hidden, logits


@torch.no_grad()
def context_predictive(
    model, val_seqs: torch.Tensor, cfg: Config, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns per-context hidden states and model conditionals.

    Args:
        model: GPT, set to evaluation mode by this function.
        val_seqs: Sequences of shape (num, seq_len).
        cfg: Configuration.
        device: Device of ``val_seqs``.

    Returns:
        Tuple (hidden, probs). Hidden is float64 of shape (num_contexts, d_model).
        Probs is float64 of shape (num_contexts, vocab), flattened like
        context extraction.
    """
    model.eval()
    start_pos = cfg.order_k - 1
    hid_chunks = []
    prob_chunks = []
    for begin in range(0, val_seqs.shape[0], 256):
        batch = val_seqs[begin : begin + 256]
        hidden, logits = forward_hidden(model, batch[:, :-1])
        hid_chunks.append(
            hidden[:, start_pos:, :].reshape(-1, cfg.d_model).to(COMPUTE_DTYPE)
        )
        sel = logits[:, start_pos:, :].to(COMPUTE_DTYPE)
        prob_chunks.append(torch.softmax(sel, dim=-1).reshape(-1, cfg.vocab_size))
    return torch.cat(hid_chunks, dim=0), torch.cat(prob_chunks, dim=0)
