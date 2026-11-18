"""Models, LoRA targets, the halving grid, and the code-generation prompt.

Directory locations come from ``mode_collapse.paths``. This module does not
load model weights.
"""

from __future__ import annotations

import os

from mode_collapse.paths import DATA_DIR, OUTPUT_DIR

SEED = 42
DATASET = "codenet"

TEXT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
GEMMA3_LORA_TARGETS = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)
MODEL_REGISTRY = {
    "google/gemma-2-2b-it": {
        "slug": "gemma-2-2b-it",
        "auto": "causal",
        "attn": "eager",
        "batch": 1,
        "accum": 16,
        "val_subset": 4000,
        "n_evals": 12,
        "lora_targets": TEXT_LORA_TARGETS,
    },
    "Qwen/Qwen3.5-2B": {
        "slug": "qwen3.5-2b",
        "auto": "img2text",
        "attn": "sdpa",
        "batch": 1,
        "accum": 16,
        "val_subset": 4000,
        "n_evals": 12,
        "lora_targets": TEXT_LORA_TARGETS,
    },
    "Qwen/Qwen3.5-4B": {
        "slug": "qwen3.5-4b",
        "auto": "img2text",
        "attn": "sdpa",
        "batch": 1,
        "accum": 16,
        "val_subset": 4000,
        "n_evals": 12,
        "lora_targets": TEXT_LORA_TARGETS,
    },
    "google/gemma-3-4b-it": {
        "slug": "gemma-3-4b-it",
        "auto": "img2text",
        "attn": "sdpa",
        "batch": 1,
        "accum": 16,
        "val_subset": 4000,
        "n_evals": 12,
        "lora_targets": GEMMA3_LORA_TARGETS,
    },
}
MODEL_SLUGS = tuple(str(cfg["slug"]) for cfg in MODEL_REGISTRY.values())
CHECKPOINTS = ("base", "early_stopped", "fully_trained")
TRAINED_CHECKPOINTS = ("early_stopped", "fully_trained")

MAX_LENGTH = 2048
MAX_PROMPT_TOKENS = 1024
MAX_NEW_TOKENS = 1024
N_GENERATIONS = 32
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_GRID_POINTS = 20
MIN_GRID_N = 8
N_MODELS = len(MODEL_REGISTRY)

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
SAVE_STEPS = 500
VAL_FRACTION = 0.10

HUMAN_SAMPLE = 32
PAIR_SAMPLE = 32
EXPECTED_SHARDS = 8
PROMPT_HEAD_TOKENS = 128

CODENET_DATA_DIR = DATA_DIR / "codenet"
CODENET_OUTPUT_DIR = OUTPUT_DIR / "codenet"
DATASET_ROOT = CODENET_DATA_DIR / "Project_CodeNet"
CODENET_ARCHIVE = CODENET_DATA_DIR / "Project_CodeNet.tar.gz"
PROCESSED_DIR = CODENET_OUTPUT_DIR / "processed"
CHECKPOINTS_DIR = CODENET_OUTPUT_DIR / "checkpoints"
MEASURE_DIR = CODENET_OUTPUT_DIR / "measurements"
ZSS_DIR = CODENET_OUTPUT_DIR / "zhang_shasha"
SUMMARY_PATH = CODENET_OUTPUT_DIR / "codenet_collision_ratio.csv"
PER_PROBLEM_PATH = CODENET_OUTPUT_DIR / "codenet_collision_ratio_per_problem.csv"
SAME_PROBLEM_PATH = CODENET_OUTPUT_DIR / "codenet_same_problem_comparison.csv"

SUMMARY_COLUMNS = (
    "model",
    "n",
    "checkpoint",
    "n_problems",
    "n_finite",
    "human_collision_median",
    "model_collision_median",
    "collision_ratio_q05",
    "collision_ratio_q25",
    "collision_ratio_median",
    "collision_ratio_q75",
    "collision_ratio_q95",
)
COLLISION_RATIO_QUANTILES = (
    ("q05", 0.05),
    ("q25", 0.25),
    ("median", 0.50),
    ("q75", 0.75),
    ("q95", 0.95),
)


def model_slug(model_path: str) -> str:
    """Return the short slug for a Hugging Face model id."""
    return str(MODEL_REGISTRY[model_path]["slug"])


def run_name(model_path: str, n: int, seed: int = SEED) -> str:
    """Return the checkpoint directory name for one model and training size."""
    return f"codenet_{model_slug(model_path)}_n{n}_seed{seed}"


def training_grid(n_total: int) -> list[int]:
    """Return the N-anchored halving grid, smallest point first.

    Each point is ``n_total`` right-shifted until the value drops below
    ``MIN_GRID_N``, keeping at most ``MAX_GRID_POINTS`` points. For a
    training pool of 1,587,907 rows this is 18 sizes from 12 through
    1,587,907.
    """
    if n_total < MIN_GRID_N:
        raise ValueError(
            f"training pool has {n_total} rows; at least {MIN_GRID_N} are required"
        )
    descending = [
        n_total >> index
        for index in range(MAX_GRID_POINTS)
        if (n_total >> index) >= MIN_GRID_N
    ]
    return list(reversed(descending))


def model_runs(n_total: int) -> list[tuple[int, str]]:
    """Return ``(n, checkpoint)`` pairs for one model, base first."""
    runs = [(0, "base")]
    for n in training_grid(n_total):
        for checkpoint in TRAINED_CHECKPOINTS:
            runs.append((n, checkpoint))
    return runs


def scoring_arms(n_total: int) -> list[tuple[str, int, str]]:
    """Return every ``(model, n, checkpoint)`` arm in the collision-ratio table."""
    runs = model_runs(n_total)
    return [(model, n, checkpoint) for model in MODEL_SLUGS for n, checkpoint in runs]


def build_messages(problem: str) -> list[dict[str, str]]:
    """Return the single user turn used for every CodeNet problem."""
    instruction = (
        "You are an expert competitive programmer. Return only a complete Python 3 program "
        "that solves the problem. Do not use Markdown fences or explanations.\n\n"
        f"{problem}"
    )
    return [{"role": "user", "content": instruction}]


def encode_generation_prompt(tokenizer, problem: str) -> list[int]:
    """Encode a prompt while retaining both its instruction head and assistant-turn suffix."""
    prompt = tokenizer.apply_chat_template(
        build_messages(problem),
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) <= MAX_PROMPT_TOKENS:
        return prompt_ids
    head = min(PROMPT_HEAD_TOKENS, MAX_PROMPT_TOKENS // 4)
    tail = MAX_PROMPT_TOKENS - head
    return prompt_ids[:head] + prompt_ids[-tail:]


def turn_terminator_ids(tokenizer) -> list[int]:
    """Derive the assistant-turn suffix from the tokenizer's own chat template."""
    messages = build_messages("Minimal placeholder problem.")
    sentinel = "ASSISTANT_CONTENT_SENTINEL_7F3A"
    completed_turn = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": sentinel}],
        tokenize=False,
        add_generation_prompt=False,
    )
    marker_index = completed_turn.rfind(sentinel)
    if marker_index < 0:
        raise ValueError("chat template did not preserve assistant sentinel content")
    suffix_text = completed_turn[marker_index + len(sentinel) :]
    suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
    if not suffix_text or not suffix:
        raise ValueError("chat template produced no assistant-turn terminator")
    return list(suffix)


def require_wandb_destination() -> tuple[str, str]:
    """Return ``WANDB_ENTITY`` and ``WANDB_PROJECT``, or raise if either is unset."""
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if not entity or not project:
        raise RuntimeError(
            "WANDB_API_KEY is set but WANDB_ENTITY and WANDB_PROJECT must both be set"
        )
    return entity, project
