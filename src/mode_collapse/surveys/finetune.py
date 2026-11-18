"""LoRA fine-tuning of one survey adapter for Experiment 2.

One call trains on ``n`` respondent-question rows and writes two adapters:
``early_stopped`` (lowest validation loss seen in this process) and
``fully_trained`` (the final step). The loss is the shifted cross-entropy on
the answer-letter tokens and EOS. Prompt and padding positions are masked.
Training and validation use that same loss. Checkpoints every 500 steps let an
interrupted run resume. Under ``torchrun`` the gradient accumulation is divided
by the number of processes, so the effective batch size and the number of
optimizer steps do not change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainerCallback, TrainingArguments

from mode_collapse.surveys.datasets import (
    checkpoints_dir,
    dataset_module,
    item_config_path,
    processed_dir,
)
from mode_collapse.surveys.gss import build_chat_messages
from mode_collapse.surveys.measurement import measure_item
from mode_collapse.surveys.models import (
    MODEL_REGISTRY,
    load_base_model,
    load_tokenizer,
    model_slug,
    require_wandb_env,
    run_name,
)

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_LENGTH = 256
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
DEFAULT_EPOCHS = 3
SAVE_STEPS = 500
COLLISION_CELLS_PER_ITEM = 40
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_MAIN_PROCESS = int(os.environ.get("RANK", "0")) == 0


class SurveyPairDataset(torch.utils.data.Dataset):
    """Tokenises persona and question to the respondent's answer letter."""

    def __init__(self, pair_answer_df: pd.DataFrame, item_config: dict, tokenizer):
        """Stores the rows, item config, and tokenizer.

        Args:
            pair_answer_df: Rows with ``persona``, ``item``, and ``answer_letter``.
            item_config: Per-question question text and options.
            tokenizer: Tokenizer with a chat template.
        """
        self.pair_answer_df = pair_answer_df.reset_index(drop=True)
        self.item_config = item_config
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Returns the number of rows."""
        return len(self.pair_answer_df)

    def __getitem__(self, idx: int) -> dict:
        """Tokenises one respondent-question row.

        Args:
            idx: Row index.

        Returns:
            ``input_ids``, ``attention_mask``, and ``labels``.
        """
        row = self.pair_answer_df.iloc[idx]
        cfg = self.item_config[row["item"]]
        messages = build_chat_messages(row["persona"], cfg["question"], cfg["options"])
        return encode_pair(messages, row["answer_letter"], self.tokenizer)


def encode_pair(user_messages: list[dict], answer_letter: str, tokenizer) -> dict:
    """Tokenises a chat pair and masks the prompt.

    The answer is appended to the generation prompt so the prompt is a strict
    prefix. Rendering the full conversation again is not safe: some chat
    templates emit a generation prompt that is longer than, and not a prefix
    of, the rendered assistant turn. The supervised token is then the letter
    at the position the measurement reads.

    Args:
        user_messages: Single user turn.
        answer_letter: Target answer letter.
        tokenizer: Tokenizer with a chat template.

    Returns:
        ``input_ids``, ``attention_mask``, and ``labels`` of length
        MAX_LENGTH.
    """
    prompt_text = tokenizer.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_letter, add_special_tokens=False)
    if tokenizer.eos_token_id:
        completion_ids = answer_ids + [tokenizer.eos_token_id]
    else:
        completion_ids = answer_ids

    full_ids = (prompt_ids + completion_ids)[:MAX_LENGTH]
    seq_len = len(full_ids)
    prompt_len = min(len(prompt_ids), seq_len)
    labels = [-100] * prompt_len + list(full_ids[prompt_len:])

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_len = MAX_LENGTH - seq_len
    input_ids = list(full_ids) + [pad_id] * pad_len
    attention_mask = [1] * seq_len + [0] * pad_len
    labels = labels + [-100] * pad_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class CausalLMTrainer(Trainer):
    """Next-token loss from logits, shared by training and validation.

    Multimodal checkpoints route labels through their own loss, which can
    disagree with the logit length by one. Shifting here keeps one loss for
    every model: mean cross-entropy on the answer-letter encoding and EOS.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Returns the shifted cross-entropy on unmasked labels.

        Args:
            model: Wrapped model.
            inputs: Batch with ``labels``.
            return_outputs: Whether to return ``(loss, outputs)``.
            num_items_in_batch: Unused. Accepted because Trainer passes it.

        Returns:
            The loss, or the loss and model outputs.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous().to(shift_logits.device)
        assert shift_logits.shape[1] == shift_labels.shape[1], (
            f"logits/labels length mismatch: {shift_logits.shape} vs {shift_labels.shape}"
        )
        flat_logits = shift_logits.reshape(-1, shift_logits.size(-1)).float()
        flat_labels = shift_labels.reshape(-1)
        loss = torch.nn.functional.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


class BestAdapterCallback(TrainerCallback):
    """Saves the adapter when validation loss improves.

    The best loss is process-local. A resumed run starts again at infinity,
    so a later, worse evaluation can overwrite the adapter saved before the
    interruption.
    """

    def __init__(self, early_dir: Path):
        """Stores the early-stopped adapter directory.

        Args:
            early_dir: Directory passed to ``save_pretrained``.
        """
        self.early_dir = early_dir
        self.best_loss = float("inf")
        self.best_step = -1

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        """Saves the adapter when ``eval_loss`` is the best so far.

        Args:
            args: Training arguments. Unused.
            state: Trainer state.
            control: Trainer control. Unused.
            metrics: Evaluation metrics.
            model: Model to save.
            **kwargs: Other callback arguments. Unused.
        """
        if not state.is_world_process_zero:
            return
        if metrics is None or "eval_loss" not in metrics:
            return
        if metrics["eval_loss"] < self.best_loss:
            self.best_loss = float(metrics["eval_loss"])
            self.best_step = int(state.global_step)
            model.save_pretrained(self.early_dir)


class CollisionEvalCallback(TrainerCallback):
    """Logs plug-in answer collision on a fixed eval subset at each evaluation."""

    def __init__(
        self,
        pair_eval_df: pd.DataFrame,
        item_config: dict,
        tokenizer,
        answer_letters: list[str],
    ):
        """Stores the capped eval rows and the tokenizer.

        Args:
            pair_eval_df: Eval question-group rows.
            item_config: Per-question config.
            tokenizer: Tokenizer used to build prompts.
            answer_letters: Option alphabet.
        """
        self.pair_eval_df = pair_eval_df
        self.item_config = item_config
        self.tokenizer = tokenizer
        self.answer_letters = answer_letters

    @torch.no_grad()
    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        """Logs the median plug-in ratio and inverse Simpson indexes.

        Args:
            args: Training arguments. Unused.
            state: Trainer state.
            control: Trainer control. Unused.
            metrics: Evaluation metrics. Unused.
            model: Model being evaluated.
            **kwargs: Other callback arguments. Unused.
        """
        if not state.is_world_process_zero:
            return
        prev_side = self.tokenizer.padding_side
        was_training = model.training
        self.tokenizer.padding_side = "left"
        model.eval()
        pair_rows = []
        for item, cfg in self.item_config.items():
            pair_cell_df = self.pair_eval_df[self.pair_eval_df["item"] == item]
            if not pair_cell_df.empty:
                pair_rows.extend(
                    measure_item(
                        model,
                        self.tokenizer,
                        item,
                        cfg,
                        pair_cell_df,
                        self.answer_letters,
                    )
                )
        model.train(was_training)
        self.tokenizer.padding_side = prev_side
        if not pair_rows:
            return
        pair_collision_df = pd.DataFrame(pair_rows)
        wandb.log(
            {
                "eval/global_step": state.global_step,
                "eval/collision_ratio_plugin_median": float(
                    pair_collision_df["collision_ratio_plugin"].median()
                ),
                "eval/N2_q_median": float(pair_collision_df["N2_q"].median()),
                "eval/N2_p_median": float(pair_collision_df["N2_p"].median()),
            }
        )


def evaluation_interval(planned_steps: int, n_evals: int) -> int:
    """Returns how often validation runs, in optimizer steps.

    The gap is about ``planned_steps / n_evals``, at least 20, and never
    longer than the run, so a short run still writes the early-stopped adapter.

    Args:
        planned_steps: Optimizer steps the run will take.
        n_evals: Requested number of evaluations.

    Returns:
        Evaluation interval in steps.
    """
    return min(max(20, planned_steps // n_evals), planned_steps)


def subsample_eval_cells(pair_eval_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Keeps at most a fixed number of eval groups per question.

    Rows are concatenated so the question column is kept.

    Args:
        pair_eval_df: Eval rows with an ``item`` column.
        seed: Sample seed.

    Returns:
        Capped eval rows.
    """
    parts = []
    for _item, item_cells_df in pair_eval_df.groupby("item"):
        n_keep = min(len(item_cells_df), COLLISION_CELLS_PER_ITEM)
        parts.append(item_cells_df.sample(n_keep, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def load_sft_frames(
    dataset: str,
    n: int,
    seed: int,
    item_config: dict,
    val_subset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and subsamples the train and validation rows.

    Args:
        dataset: Survey name.
        n: Number of training rows.
        seed: Subsample seed.
        item_config: Per-question config. Rows for other questions are dropped.
        val_subset: Number of validation rows.

    Returns:
        ``(pair_train_df, pair_val_df)`` with persona, item, and answer letter.
    """
    processed = processed_dir(dataset)
    respondent_item_long_df = pd.read_csv(processed / "respondent_item_long_df.csv")
    persona_columns = ["cell_id", "persona"]
    cell_persona_df = pd.read_csv(processed / "cell_persona_df.csv")[persona_columns]
    respondent_item_long_df = respondent_item_long_df[
        respondent_item_long_df["item"].isin(item_config.keys())
    ]
    respondent_persona_df = respondent_item_long_df.merge(
        cell_persona_df, on="cell_id", how="left"
    )
    assert respondent_persona_df["persona"].notna().all(), (
        "missing persona for some cells"
    )

    keep = ["persona", "item", "answer_letter"]
    pair_train_pool_df = respondent_persona_df[
        respondent_persona_df["split"] == "train"
    ][keep]
    pair_val_pool_df = respondent_persona_df[respondent_persona_df["split"] == "val"][
        keep
    ]
    n_train = min(n, len(pair_train_pool_df))
    pair_train_df = pair_train_pool_df.sample(n=n_train, random_state=seed).reset_index(
        drop=True
    )
    n_val = min(val_subset, len(pair_val_pool_df))
    pair_val_df = pair_val_pool_df.sample(n=n_val, random_state=seed).reset_index(
        drop=True
    )
    return pair_train_df, pair_val_df


def plot_loss_curve(
    log_history: list[dict], path: Path, n: int, slug: str, dataset: str
) -> None:
    """Plots training and validation loss against step.

    Args:
        log_history: Trainer log history.
        path: Output PNG path.
        n: Training-set size, for the title.
        slug: Model slug, for the title.
        dataset: Survey name, for the title.
    """
    train_steps = [
        (entry["step"], entry["loss"])
        for entry in log_history
        if "loss" in entry and "eval_loss" not in entry
    ]
    eval_steps = [
        (entry["step"], entry["eval_loss"])
        for entry in log_history
        if "eval_loss" in entry
    ]
    fig, axis = plt.subplots(figsize=(9, 5))
    if train_steps:
        train_x, train_y = zip(*train_steps, strict=True)
        axis.plot(train_x, train_y, color="C0", lw=1.6, label="train loss")
    if eval_steps:
        eval_x, eval_y = zip(*eval_steps, strict=True)
        axis.plot(
            eval_x,
            eval_y,
            color="C3",
            lw=1.8,
            marker="o",
            ms=4,
            label="validation loss",
        )
    axis.set_xlabel("training step")
    axis.set_ylabel("cross-entropy loss (answer token)")
    axis.set_title(f"{dataset.upper()} SFT loss curve  (n={n} train rows, {slug} LoRA)")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_finetune(
    dataset: str,
    model_path: str,
    n: int,
    seed: int,
    epochs: int,
    max_steps: int,
) -> None:
    """Trains one LoRA adapter and saves the early-stopped and final copies.

    Args:
        dataset: ``gss``, ``wvs``, or ``anes``.
        model_path: Hugging Face model id.
        n: Number of training rows.
        seed: Training and subsample seed.
        epochs: Epoch budget. Ignored as a step cap when ``max_steps`` is
            positive.
        max_steps: Positive values replace the epoch budget. ``-1`` runs the
            full epoch budget.
    """
    # --- setup ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    assert torch.cuda.is_available(), "CUDA is required."
    common = dataset_module(dataset)
    model_cfg = MODEL_REGISTRY[model_path]
    slug = model_slug(model_path)
    per_device_batch = model_cfg["per_device_batch"]
    grad_accum = model_cfg["grad_accum"]
    val_subset = model_cfg["val_subset"]
    n_evals = model_cfg["n_evals"]
    assert grad_accum % WORLD_SIZE == 0, (
        f"grad_accum={grad_accum} must be divisible by WORLD_SIZE={WORLD_SIZE} "
        "to keep the effective batch size unchanged"
    )
    grad_accum //= WORLD_SIZE

    # --- config ---
    with item_config_path(dataset).open() as handle:
        item_config = json.load(handle)
    name = run_name(dataset, model_path, n, seed)
    run_dir = checkpoints_dir() / name
    early_dir = run_dir / "early_stopped"
    full_dir = run_dir / "fully_trained"
    run_dir.mkdir(parents=True, exist_ok=True)
    entity, project = require_wandb_env()
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    if not IS_MAIN_PROCESS:
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(
        entity=entity,
        project=project,
        group=f"{common.DATASET}_{slug}",
        job_type="sft",
        name=name,
        config={
            "n": n,
            "seed": seed,
            "epochs": epochs,
            "model": model_path,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "dataset": common.DATASET,
        },
    )
    wandb.define_metric("eval/global_step")
    collision_metrics = [
        "eval/collision_ratio_plugin_median",
        "eval/N2_q_median",
        "eval/N2_p_median",
    ]
    for collision_metric in collision_metrics:
        wandb.define_metric(collision_metric, step_metric="eval/global_step")

    # --- data import ---
    tokenizer = load_tokenizer(model_path)
    model = load_base_model(model_path)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=model_cfg["lora_targets"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    pair_train_df, pair_val_df = load_sft_frames(
        dataset, n, seed, item_config, val_subset
    )
    train_dataset = SurveyPairDataset(pair_train_df, item_config, tokenizer)
    val_dataset = SurveyPairDataset(pair_val_df, item_config, tokenizer)

    # --- data filtering ---
    truth_df = pd.read_csv(processed_dir(dataset) / "cell_item_truth_df.csv")
    truth_df = truth_df[truth_df["is_eval"]]
    persona_columns = ["cell_id", "persona"]
    cell_persona_df = pd.read_csv(processed_dir(dataset) / "cell_persona_df.csv")
    cell_persona_df = cell_persona_df[persona_columns]
    pair_eval_df = truth_df.merge(cell_persona_df, on="cell_id", how="left")
    assert pair_eval_df["persona"].notna().all(), "missing persona for eval cells"
    pair_eval_df = subsample_eval_cells(pair_eval_df, seed)

    # --- model fitting ---
    effective_batch = per_device_batch * grad_accum * WORLD_SIZE
    steps_per_epoch = int(np.ceil(len(train_dataset) / effective_batch))
    full_steps = epochs * steps_per_epoch
    if max_steps > 0:
        planned_steps = max_steps
    else:
        planned_steps = full_steps
    eval_steps = evaluation_interval(planned_steps, n_evals)
    print(
        f"n_train={len(train_dataset)}  n_val={len(val_dataset)}  "
        f"steps_per_epoch={steps_per_epoch}  epochs={epochs}  "
        f"planned_steps={planned_steps}  max_steps_override={max_steps}  "
        f"world_size={WORLD_SIZE}  grad_accum={grad_accum}  "
        f"effective_batch={effective_batch}",
        flush=True,
    )
    training_args = TrainingArguments(
        output_dir=str(run_dir / "hf_trainer"),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        ddp_find_unused_parameters=False,
        report_to=["wandb"],
        run_name=name,
        seed=seed,
        dataloader_num_workers=4,
    )
    best_cb = BestAdapterCallback(early_dir)
    collision_cb = CollisionEvalCallback(
        pair_eval_df, item_config, tokenizer, common.ANSWER_LETTERS
    )
    trainer = CausalLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[best_cb, collision_cb],
    )
    existing_ckpts = list((run_dir / "hf_trainer").glob("checkpoint-*"))
    resume_from = True if existing_ckpts else None
    if resume_from:
        print(
            f"resuming from latest of {len(existing_ckpts)} checkpoints "
            f"in {run_dir / 'hf_trainer'}",
            flush=True,
        )
    trainer.train(resume_from_checkpoint=resume_from)
    if not trainer.is_world_process_zero():
        return

    # --- data export ---
    model.save_pretrained(full_dir)
    plot_loss_curve(
        trainer.state.log_history,
        run_dir / f"loss_curve_n{n}.png",
        n,
        slug,
        common.DATASET,
    )
    completed_steps = int(trainer.state.global_step)
    meta = {
        "n": n,
        "seed": seed,
        "epochs": epochs,
        "completed_steps": completed_steps,
        "steps_per_epoch": steps_per_epoch,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "best_val_loss": best_cb.best_loss,
        "best_val_step": best_cb.best_step,
        "model": model_path,
        "early_stopped_dir": str(early_dir),
        "fully_trained_dir": str(full_dir),
    }
    with (run_dir / "meta.json").open("w") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    wandb.summary.update(
        {
            "best_val_loss": best_cb.best_loss,
            "best_val_step": best_cb.best_step,
            "n_train": len(train_dataset),
            "completed_steps": completed_steps,
        }
    )
    wandb.finish()
