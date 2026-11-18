"""Measure option-softmax collision for one Experiment 2 checkpoint.

``WANDB_ENTITY`` and ``WANDB_PROJECT`` must be set. A CUDA device is required.
"""

from __future__ import annotations

import argparse

from mode_collapse.surveys.measurement import run_measure
from mode_collapse.surveys.models import DEFAULT_MODEL, MODEL_REGISTRY


def main() -> None:
    """Writes one per-pair measurement CSV."""
    # --- config ---
    parser = argparse.ArgumentParser(
        description="Measure Experiment 2 option-softmax collision."
    )
    parser.add_argument("--dataset", default="gss", choices=["gss", "wvs", "anes"])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODEL_REGISTRY))
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="LoRA adapter directory. Omit, or pass none, for the base model.",
    )
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--checkpoint", default="base")
    args = parser.parse_args()

    # --- model fitting ---
    if args.adapter_dir in (None, "none"):
        adapter_dir = None
    else:
        adapter_dir = args.adapter_dir
    run_measure(
        dataset=args.dataset,
        model_path=args.model,
        adapter_dir=adapter_dir,
        n=args.n,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
