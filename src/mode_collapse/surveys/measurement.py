"""Option-softmax collision measurement for Experiment 2.

For each evaluated question and demographic group, the model's answer
distribution is the softmax over the answer-option letters at the generation
position. No tokens are sampled. Each run writes one CSV per survey, model,
sample size, and checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm.auto import tqdm

from mode_collapse.surveys.datasets import (
    dataset_module,
    item_config_path,
    measurements_dir,
    processed_dir,
)
from mode_collapse.surveys.gss import SEED, build_chat_messages
from mode_collapse.surveys.models import (
    load_model_and_tokenizer,
    model_slug,
    require_wandb_env,
)

MAX_LENGTH = 256
EVAL_BATCH = 8


def option_softmax(option_logits: torch.Tensor) -> torch.Tensor:
    """Returns the softmax over answer-option logits.

    Args:
        option_logits: Logits with the option axis last. Values are cast to
            float32 before the softmax, matching the measurement kernel.

    Returns:
        Probabilities on the option axis.
    """
    return torch.softmax(option_logits.float(), dim=-1)


def option_softmax_collision(option_logits: torch.Tensor) -> torch.Tensor:
    """Returns the collision of an option-softmax distribution.

    Args:
        option_logits: Logits with the option axis last.

    Returns:
        ``sum_a q(a)^2`` for ``q = softmax(option_logits)``.
    """
    probabilities = option_softmax(option_logits)
    return torch.sum(probabilities**2, dim=-1)


def candidate_token_ids(
    tokenizer, options: list[str], answer_letters: list[str]
) -> list[int]:
    """Returns the first token id of each option letter.

    Args:
        tokenizer: Hugging Face tokenizer.
        options: Ordered answer-option labels. Only the count is used.
        answer_letters: Option alphabet, ``A`` onward.

    Returns:
        One token id per option letter. The ids are asserted distinct.
    """
    ids = []
    for letter in answer_letters[: len(options)]:
        letter_ids = tokenizer.encode(letter, add_special_tokens=False)
        assert letter_ids, f"empty tokenization for answer letter {letter!r}"
        ids.append(letter_ids[0])
    assert len(set(ids)) == len(ids), f"non-distinct letter tokens: {ids}"
    return ids


@torch.no_grad()
def measure_item(
    model,
    tokenizer,
    item: str,
    cfg: dict,
    pair_cell_df: pd.DataFrame,
    answer_letters: list[str],
) -> list[dict]:
    """Reads the option distribution for every eval group of one question.

    Args:
        model: Loaded model.
        tokenizer: Matching tokenizer with left padding.
        item: Question code.
        cfg: Item config with ``question`` and ``options``.
        pair_cell_df: Eval rows for this question. Columns include
            ``cell_id``, ``persona``, ``N2_p``, ``n_cell``.
        answer_letters: Option alphabet.

    Returns:
        One measurement dict per question-group row.
    """
    options = cfg["options"]
    question = cfg["question"]
    cand_ids = candidate_token_ids(tokenizer, options, answer_letters)
    cand_tensor = torch.tensor(cand_ids)
    input_device = model.get_input_embeddings().weight.device

    pair_rows = []
    pair_cell_indexed_df = pair_cell_df.reset_index(drop=True)
    for start in range(0, len(pair_cell_indexed_df), EVAL_BATCH):
        pair_cell_batch_df = pair_cell_indexed_df.iloc[start : start + EVAL_BATCH]
        prompts = []
        for _, row in pair_cell_batch_df.iterrows():
            messages = build_chat_messages(row["persona"], question, options)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=False,
        ).to(input_device)
        logits = model(**encoded).logits[:, -1, :]
        option_logits = logits.index_select(1, cand_tensor.to(logits.device))
        option_logits = option_logits.float()
        probabilities = option_softmax(option_logits).cpu().numpy()

        for i, (_, row) in enumerate(pair_cell_batch_df.iterrows()):
            probs = probabilities[i]
            collision = float(np.sum(probs**2))
            n2_q = float(1.0 / collision)
            probability_by_option = dict(
                zip(options, probs.round(6).tolist(), strict=False)
            )
            pair_rows.append(
                {
                    "item": item,
                    "cell_id": row["cell_id"],
                    "n_cell": int(row["n_cell"]),
                    "N2_p": float(row["N2_p"]),
                    "N2_q": n2_q,
                    "collision_ratio_plugin": float(row["N2_p"]) / n2_q,
                    "q_probs_json": json.dumps(probability_by_option),
                }
            )
    return pair_rows


def run_measure(
    dataset: str,
    model_path: str,
    adapter_dir: str | None,
    n: int,
    checkpoint: str,
) -> Path:
    """Measures one checkpoint on every eval pair and writes the CSV.

    Args:
        dataset: ``gss``, ``wvs``, or ``anes``.
        model_path: Hugging Face model id.
        adapter_dir: LoRA adapter directory, or None for the base model.
        n: Fine-tuning size this checkpoint corresponds to. Zero for the base.
        checkpoint: ``base``, ``early_stopped``, or ``fully_trained``.

    Returns:
        Path of the written per-pair CSV.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    assert torch.cuda.is_available(), "CUDA is required."

    common = dataset_module(dataset)
    measure_dir = measurements_dir()
    measure_dir.mkdir(parents=True, exist_ok=True)
    with item_config_path(dataset).open() as handle:
        item_config = json.load(handle)

    truth_df = pd.read_csv(processed_dir(dataset) / "cell_item_truth_df.csv")
    truth_df = truth_df[truth_df["is_eval"]]
    persona_columns = ["cell_id", "persona"]
    cell_persona_df = pd.read_csv(processed_dir(dataset) / "cell_persona_df.csv")
    cell_persona_df = cell_persona_df[persona_columns]
    pair_eval_df = truth_df.merge(cell_persona_df, on="cell_id", how="left")
    assert pair_eval_df["persona"].notna().all(), "missing persona for eval cells"

    slug = model_slug(model_path)
    model, tokenizer = load_model_and_tokenizer(model_path, adapter_dir)
    pair_rows = []
    items = list(item_config.keys())
    for item in tqdm(items, desc=f"items[{slug}/{checkpoint}]"):
        pair_cell_df = pair_eval_df[pair_eval_df["item"] == item]
        if pair_cell_df.empty:
            continue
        pair_rows.extend(
            measure_item(
                model,
                tokenizer,
                item,
                item_config[item],
                pair_cell_df,
                common.ANSWER_LETTERS,
            )
        )

    pair_measure_df = pd.DataFrame(pair_rows)
    pair_measure_df["n"] = n
    pair_measure_df["checkpoint"] = checkpoint
    pair_measure_df["model"] = slug
    pair_measure_df["dataset"] = common.DATASET
    out_path = measure_dir / f"measure_{common.DATASET}_{slug}_n{n}_{checkpoint}.csv"
    pair_measure_df.to_csv(out_path, index=False)
    median_ratio = float(pair_measure_df["collision_ratio_plugin"].median())
    print(
        f"wrote {out_path}  cells={len(pair_measure_df)}  "
        f"median plug-in collision ratio={median_ratio:.3f}",
        flush=True,
    )

    entity, project = require_wandb_env()
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    wandb.init(
        entity=entity,
        project=project,
        group=f"{common.DATASET}_{slug}",
        job_type="measure",
        name=f"{common.DATASET}_{slug}_measure_n{n}_{checkpoint}",
        config={
            "n": n,
            "checkpoint": checkpoint,
            "model": slug,
            "dataset": common.DATASET,
        },
    )
    per_item_median = pair_measure_df.groupby("item")["collision_ratio_plugin"].median()
    item_logs = {
        f"collision_ratio_plugin/{item}": float(value)
        for item, value in per_item_median.items()
    }
    wandb.log({"n": n, "collision_ratio_plugin_median": median_ratio, **item_logs})
    wandb.finish()
    return out_path
