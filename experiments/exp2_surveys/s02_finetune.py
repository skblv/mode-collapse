"""Fine-tune one Experiment 2 survey adapter.

``WANDB_ENTITY`` and ``WANDB_PROJECT`` must be set. A CUDA device is required.
"""

from __future__ import annotations

import argparse

from mode_collapse.surveys.finetune import DEFAULT_EPOCHS, run_finetune
from mode_collapse.surveys.gss import SEED
from mode_collapse.surveys.models import DEFAULT_MODEL, MODEL_REGISTRY


def main() -> None:
    """Trains one adapter for one survey, model, and sample size."""
    # --- config ---
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune one Experiment 2 survey adapter."
    )
    parser.add_argument("--dataset", default="gss", choices=["gss", "wvs", "anes"])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODEL_REGISTRY))
    parser.add_argument("--n", type=int, required=True, help="Number of training rows.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override the step budget; -1 runs the full epoch budget.",
    )
    args = parser.parse_args()

    # --- model fitting ---
    run_finetune(
        dataset=args.dataset,
        model_path=args.model,
        n=args.n,
        seed=args.seed,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
