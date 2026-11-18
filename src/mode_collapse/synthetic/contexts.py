"""Conditionals on an explicit set of order-k contexts.

The diffuse-prior grid and the initialization comparison share one context set.
The mode-aligned grid uses the same conditional reader with its own context count.
"""

from __future__ import annotations

import torch

from mode_collapse.synthetic.kgram import COMPUTE_DTYPE, Config
from mode_collapse.synthetic.predictive import forward_hidden

# --- config ---

COMMON_CTX = 2000
SUBSAMPLE_SEED = 42


@torch.no_grad()
def conditionals_on_contexts(
    model, contexts: torch.Tensor, cfg: Config
) -> torch.Tensor:
    """Returns model next-token conditionals for explicit order-k contexts.

    Args:
        model: GPT. This function sets evaluation mode.
        contexts: Long tensor of shape (num, order_k).
        cfg: Configuration.

    Returns:
        Float64 conditionals of shape (num, vocab), the softmax of the
        last-position logits.
    """
    model.eval()
    chunks = []
    for begin in range(0, contexts.shape[0], 512):
        batch = contexts[begin : begin + 512]
        _hidden, logits = forward_hidden(model, batch)
        chunks.append(torch.softmax(logits[:, -1, :].to(COMPUTE_DTYPE), dim=-1))
    return torch.cat(chunks, dim=0)
