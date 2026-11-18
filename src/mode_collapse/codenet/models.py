"""Model loading, LoRA fine-tuning, and CodeNet program generation.

This is the only CodeNet module that imports ``transformers`` and ``peft``.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import wandb
from accelerate.utils import extract_model_from_parallel
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from mode_collapse.codenet.canonical_ast import analyze_source, feature_fingerprint
from mode_collapse.codenet.config import (
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MODEL_REGISTRY,
    SAVE_STEPS,
    SEED,
    TEMPERATURE,
    TOP_P,
    WARMUP_RATIO,
    encode_generation_prompt,
    model_slug,
    require_wandb_destination,
    run_name,
)
from mode_collapse.codenet.solution_split import CollisionAccumulator
from mode_collapse.codenet.training_data import CodeCollator, CodeDataset


def load_tokenizer(model_path: str):
    """Load a tokenizer and set the pad token to the eos token when missing."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_path: str):
    """Load one base model in bfloat16 and move it to CUDA."""
    cfg = MODEL_REGISTRY[model_path]
    cls = (
        AutoModelForCausalLM if cfg["auto"] == "causal" else AutoModelForImageTextToText
    )
    return cls.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=cfg["attn"],
    ).cuda()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """Read JSONL rows, optionally stopping after ``limit`` records."""
    with path.open(encoding="utf-8") as handle:
        rows = []
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
        return rows


class CausalTrainer(Trainer):
    """Shifted mean cross-entropy over code tokens and the assistant terminator."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute prompt-masked next-token cross-entropy in float32."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :].contiguous()
        targets = labels[:, 1:].contiguous().to(logits.device)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


class BestAdapter(TrainerCallback):
    """Save the adapter with the lowest validation loss seen so far."""

    def __init__(self, destination: Path):
        """Resume the best-loss record when a previous selection file exists."""
        self.destination = destination
        self.best_loss = math.inf
        self.best_step = -1
        self.state_path = destination.parent / "early_stopping_selection.json"
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text())
            self.best_loss = float(state["best_val_loss"])
            self.best_step = int(state["best_step"])

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        """Write the adapter when this evaluation improves the best validation loss."""
        if not args.should_save:
            return
        if metrics and metrics.get("eval_loss", math.inf) < self.best_loss:
            self.best_loss = float(metrics["eval_loss"])
            self.best_step = int(state.global_step)
            extract_model_from_parallel(model).save_pretrained(self.destination)
            self.state_path.write_text(
                json.dumps(
                    {"best_val_loss": self.best_loss, "best_step": self.best_step},
                    indent=2,
                )
                + "\n"
            )


def finetune(args) -> None:
    """LoRA-fine-tune one model on the first ``args.n`` ranked training rows.

    Validation loss is the same shifted cross-entropy as training loss. The
    ``fully_trained`` adapter is the one after the last step, and the adapter
    with the lowest validation loss is saved as ``early_stopped``. A completed
    run is left unchanged. An incomplete run resumes from the trainer
    checkpoint directory when one is present.
    """
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or local_rank < 0:
        raise SystemExit(
            "WORLD_SIZE must be positive and LOCAL_RANK must be nonnegative"
        )
    torch.cuda.set_device(local_rank)

    processed_dir = args.processed_dir
    summary = json.loads((processed_dir / "summary.json").read_text())
    valid_grid = [int(value) for value in summary["training_grid"]]
    if args.n not in valid_grid:
        raise SystemExit(f"n={args.n} is not on the N-anchored grid: {valid_grid}")

    torch.manual_seed(args.seed)
    rows = read_jsonl(processed_dir / "train.jsonl", limit=args.n)
    cfg = MODEL_REGISTRY[args.model]
    val_limit = int(cfg["val_subset"]) if args.val_limit is None else args.val_limit
    if val_limit < 1:
        raise SystemExit("--val-limit must be positive")
    val_rows = read_jsonl(processed_dir / "val.jsonl", limit=val_limit)
    descriptions = json.loads((processed_dir / "descriptions.json").read_text())

    name = run_name(args.model, len(rows), args.seed)
    run_dir = args.checkpoints_dir / f"{name}{args.run_suffix}"
    early_dir = run_dir / "early_stopped"
    final_dir = run_dir / "fully_trained"
    run_dir.mkdir(parents=True, exist_ok=True)
    completion_marker = run_dir / "COMPLETED"
    meta_path = run_dir / "meta.json"
    if completion_marker.is_file() and meta_path.is_file():
        print(meta_path.read_text(), flush=True)
        return

    tokenizer = load_tokenizer(args.model)
    tokenizer.padding_side = "right"
    model = load_base_model(args.model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=cfg["lora_targets"],
            task_type="CAUSAL_LM",
        ),
    )
    model.enable_input_require_grads()

    train_dataset = CodeDataset(rows, descriptions, tokenizer)
    val_dataset = CodeDataset(val_rows, descriptions, tokenizer)
    base_accumulation = int(cfg["accum"])
    if base_accumulation % world_size:
        raise SystemExit(
            f"world size {world_size} must divide the effective accumulation "
            f"factor {base_accumulation}"
        )
    accumulation = base_accumulation // world_size
    effective_batch = int(cfg["batch"]) * accumulation * world_size
    steps_per_epoch = math.ceil(len(rows) / effective_batch)
    planned = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
    eval_steps = min(max(20, planned // int(cfg["n_evals"])), planned)
    best = BestAdapter(early_dir)
    training_args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=int(cfg["batch"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        bf16=True,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        run_name=name,
        seed=args.seed,
        dataloader_num_workers=4,
    )
    if os.environ.get("WANDB_API_KEY"):
        entity, project = require_wandb_destination()
        wandb.init(
            entity=entity,
            project=project,
            group=f"codenet_{model_slug(args.model)}",
            name=name,
            config={"n": len(rows), "model": args.model, "epochs": args.epochs},
        )
    trainer = CausalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CodeCollator(tokenizer.pad_token_id),
        callbacks=[best],
    )
    resume = bool(list((run_dir / "trainer").glob("checkpoint-*"))) or None
    trainer.train(resume_from_checkpoint=resume)
    if trainer.is_world_process_zero():
        extract_model_from_parallel(model).save_pretrained(final_dir)
        if best.best_step < 0:
            extract_model_from_parallel(model).save_pretrained(early_dir)
        meta = {
            "model": args.model,
            "n": len(rows),
            "epochs": args.epochs,
            "steps": int(trainer.state.global_step),
            "best_val_loss": best.best_loss,
            "best_step": best.best_step,
            "world_size": world_size,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": effective_batch,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        completion_marker.write_text("ok\n")
        print(json.dumps(meta, indent=2), flush=True)
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def clean_code(text: str) -> str:
    """Strip a Markdown fence when the sample contains one."""
    text = text.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    return match.group(1).strip() if match else text


def generation_seed(problem_id: str, n: int, checkpoint: str) -> int:
    """Derive a 32-bit sample seed from the master seed, problem, size, and checkpoint."""
    raw = f"{SEED}\0{problem_id}\0{n}\0{checkpoint}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def load_eval_truth(processed_dir: Path) -> list[dict[str, str]]:
    """Load evaluation rows from ``problem_truth.csv``."""
    with (processed_dir / "problem_truth.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return [row for row in csv.DictReader(handle) if row["is_eval"] == "1"]


def generate(args) -> None:
    """Sample programs for one shard and write the generation archive.

    Sampling uses temperature 1.0, top-p 1.0, and at most 1,024 new tokens.
    Seeds derive from 42. A shard whose generation archive and companion
    table already exist is left unchanged.
    """
    if args.samples < 2 or not torch.cuda.is_available():
        raise SystemExit("at least two samples and CUDA are required")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid measurement shard")

    slug = model_slug(args.model)
    measure_dir = args.measure_dir
    measure_dir.mkdir(parents=True, exist_ok=True)
    shard_suffix = f"shard{args.shard_index:03d}-of-{args.num_shards:03d}"
    output_csv = measure_dir / (
        f"measure_codenet_{slug}_n{args.n}_{args.checkpoint}_{shard_suffix}.csv"
    )
    generation_path = measure_dir / (
        f"generations_codenet_{slug}_n{args.n}_{args.checkpoint}_{shard_suffix}.jsonl.gz"
    )
    if output_csv.exists() and generation_path.exists():
        print(f"measurement already complete: {output_csv}")
        return

    processed_dir = args.processed_dir
    descriptions = json.loads((processed_dir / "descriptions.json").read_text())
    truth_rows = [
        row
        for row in load_eval_truth(processed_dir)
        if generation_seed(row["problem_id"], 0, "measurement-shard") % args.num_shards
        == args.shard_index
    ]
    if not truth_rows:
        raise SystemExit(
            f"measurement shard {args.shard_index} has no evaluation problems"
        )
    tokenizer = load_tokenizer(args.model)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model = load_base_model(args.model)
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    results: list[dict[str, object]] = []
    generation_temporary = generation_path.with_suffix(generation_path.suffix + ".tmp")
    with gzip.open(generation_temporary, "wt", encoding="utf-8") as generation_file:
        for problem_index, truth in enumerate(truth_rows, start=1):
            problem_id = truth["problem_id"]
            prompt_ids = encode_generation_prompt(tokenizer, descriptions[problem_id])
            encoded = {
                "input_ids": torch.tensor(
                    [prompt_ids], dtype=torch.long, device=input_device
                ),
                "attention_mask": torch.ones(
                    (1, len(prompt_ids)), dtype=torch.long, device=input_device
                ),
            }
            torch.manual_seed(generation_seed(problem_id, args.n, args.checkpoint))
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    num_return_sequences=args.samples,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_tokens = encoded["input_ids"].shape[1]
            texts = tokenizer.batch_decode(
                generated[:, prompt_tokens:],
                skip_special_tokens=True,
            )

            accumulator = CollisionAccumulator()
            errors: dict[str, int] = {}
            for sample_index, raw_text in enumerate(texts):
                code = clean_code(raw_text)
                parse_ok = True
                error = ""
                try:
                    features, signature = analyze_source(
                        code, f"{problem_id}/sample{sample_index}"
                    )
                    accumulator.add(features, signature, feature_fingerprint(features))
                except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
                    parse_ok = False
                    error = type(exc).__name__
                    errors[error] = errors.get(error, 0) + 1
                generation_file.write(
                    json.dumps(
                        {
                            "problem_id": problem_id,
                            "sample": sample_index,
                            "parse_ok": parse_ok,
                            "error": error,
                            "code": code,
                        }
                    )
                    + "\n"
                )

            metrics = accumulator.metrics() if accumulator.n else {}
            model_ast = float(metrics.get("ast_code_collision_distinct", math.nan))
            model_strategy = float(metrics.get("strategy_collision_distinct", math.nan))
            human_ast = float(truth["ast_collision"])
            human_strategy = float(truth["strategy_collision"])
            ast_collision_ratio = (
                model_ast / human_ast if math.isfinite(model_ast) else math.nan
            )
            strategy_ratio = (
                model_strategy / human_strategy if human_strategy > 0 else math.nan
            )
            results.append(
                {
                    "problem_id": problem_id,
                    "model": slug,
                    "n": args.n,
                    "checkpoint": args.checkpoint,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "n_requested": args.samples,
                    "n_parseable": accumulator.n,
                    "parse_rate": accumulator.n / args.samples,
                    "human_ast_collision": human_ast,
                    "model_ast_collision": model_ast,
                    "ast_collision_ratio": ast_collision_ratio,
                    "human_strategy_collision": human_strategy,
                    "model_strategy_collision": model_strategy,
                    "strategy_collision_ratio": strategy_ratio,
                    "errors_json": json.dumps(errors, sort_keys=True),
                }
            )
            print(
                f"[{problem_index}/{len(truth_rows)}] {problem_id} parse="
                f"{accumulator.n}/{args.samples} ast_ratio={ast_collision_ratio:.3f}",
                flush=True,
            )

    output_temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with output_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    generation_temporary.replace(generation_path)
    output_temporary.replace(output_csv)
    finite = sorted(
        float(row["ast_collision_ratio"])
        for row in results
        if math.isfinite(float(row["ast_collision_ratio"]))
    )
    median = statistics.median(finite) if finite else math.nan
    print(
        f"wrote {output_csv}; finite problems={len(finite)}/{len(results)}; "
        f"median AST collision ratio={median:.3f}",
        flush=True,
    )
