"""Model registry and checkpoint loading for Experiment 2.

Text models load as causal language models. Qwen3.5 and gemma-3-4b-it load
with the image-text-to-text class; a text-only forward still returns next-token
logits. gemma-3-4b-it adapts only ``language_model`` projections. The vision
tower's wrapped linears are left fixed.
"""

from __future__ import annotations

import os

from mode_collapse.paths import OUTPUT_DIR

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DIR", str(OUTPUT_DIR / "surveys" / "wandb"))

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

DEFAULT_MODEL = "google/gemma-2-2b-it"
TEXT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
# Decoder projections only. The vision tower is not adapted.
GEMMA3_LORA_TARGETS = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|"
    r"gate_proj|up_proj|down_proj)$"
)
MODEL_REGISTRY = {
    "google/gemma-2-2b-it": {
        "slug": "gemma-2-2b-it",
        "auto": "causal",
        "attn": "eager",
        "multi_gpu": False,
        "per_device_batch": 8,
        "grad_accum": 2,
        "val_subset": 4000,
        "n_evals": 12,
        "lora_targets": TEXT_LORA_TARGETS,
    },
    "Qwen/Qwen3.5-2B": {
        "slug": "qwen3.5-2b",
        "auto": "img2text",
        "attn": "sdpa",
        "multi_gpu": False,
        "per_device_batch": 8,
        "grad_accum": 2,
        "val_subset": 4000,
        "n_evals": 12,
        "lora_targets": TEXT_LORA_TARGETS,
    },
    "Qwen/Qwen3.5-4B": {
        "slug": "qwen3.5-4b",
        "auto": "img2text",
        "attn": "sdpa",
        "multi_gpu": False,
        "per_device_batch": 4,
        "grad_accum": 4,
        "val_subset": 4000,
        "n_evals": 12,
        "lora_targets": TEXT_LORA_TARGETS,
    },
    "google/gemma-3-4b-it": {
        "slug": "gemma-3-4b-it",
        "auto": "img2text",
        "attn": "sdpa",
        "multi_gpu": False,
        "per_device_batch": 4,
        "grad_accum": 4,
        "val_subset": 4000,
        "n_evals": 12,
        "lora_targets": GEMMA3_LORA_TARGETS,
    },
}


def model_slug(model_path: str) -> str:
    """Returns the short model tag used in run names and file names.

    Args:
        model_path: Hugging Face model id present in MODEL_REGISTRY.

    Returns:
        Filesystem-safe model slug.
    """
    assert model_path in MODEL_REGISTRY, (
        f"unknown model {model_path}; add it to MODEL_REGISTRY"
    )
    return MODEL_REGISTRY[model_path]["slug"]


def run_name(dataset: str, model_path: str, n: int, seed: int) -> str:
    """Builds the ``{dataset}_{model}_n{n}_seed{seed}`` checkpoint name.

    Args:
        dataset: Survey name.
        model_path: Hugging Face model id.
        n: Number of fine-tuning rows.
        seed: Training seed.

    Returns:
        Run name shared by checkpoints and measurement tags.
    """
    return f"{dataset}_{model_slug(model_path)}_n{n}_seed{seed}"


def require_wandb_env() -> tuple[str, str]:
    """Returns the Weights & Biases entity and project from the environment.

    Returns:
        ``(WANDB_ENTITY, WANDB_PROJECT)``.

    Raises:
        RuntimeError: If either variable is missing or empty.
    """
    entity = os.environ.get("WANDB_ENTITY", "")
    project = os.environ.get("WANDB_PROJECT", "")
    if entity == "" or project == "":
        raise RuntimeError(
            "WANDB_ENTITY and WANDB_PROJECT must be set in the environment."
        )
    return entity, project


def load_tokenizer(model_path: str):
    """Loads the tokenizer with left padding and a pad token.

    Args:
        model_path: Hugging Face model id in MODEL_REGISTRY.

    Returns:
        Tokenizer with ``padding_side`` set to ``left``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(model_path: str):
    """Loads the base model in bfloat16 on CUDA.

    Args:
        model_path: Hugging Face model id in MODEL_REGISTRY.

    Returns:
        The base model, not yet wrapped and not yet switched to train or eval.
    """
    cfg = MODEL_REGISTRY[model_path]
    load_kwargs = {"dtype": torch.bfloat16, "attn_implementation": cfg["attn"]}
    if cfg["multi_gpu"]:
        load_kwargs["device_map"] = "auto"
    if cfg["auto"] == "causal":
        model_cls = AutoModelForCausalLM
    else:
        model_cls = AutoModelForImageTextToText
    model = model_cls.from_pretrained(model_path, **load_kwargs)
    if not cfg["multi_gpu"]:
        model = model.cuda()
    return model


def load_model_and_tokenizer(model_path: str, adapter_dir: str | None):
    """Loads a base model, optionally with a LoRA adapter, for measurement.

    Args:
        model_path: Hugging Face model id in MODEL_REGISTRY.
        adapter_dir: LoRA adapter directory, or None for the base model.

    Returns:
        ``(model, tokenizer)`` in eval mode with left padding.
    """
    tokenizer = load_tokenizer(model_path)
    model = load_base_model(model_path)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer
