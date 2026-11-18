"""Tokenization and dynamic padding for accepted CodeNet SFT programs."""

from __future__ import annotations

from pathlib import Path

import torch

from mode_collapse.codenet.config import (
    MAX_LENGTH,
    encode_generation_prompt,
    turn_terminator_ids,
)


class CodeDataset(torch.utils.data.Dataset):
    """Prompt-masked token sequences for one accepted Python submission each."""

    def __init__(self, rows: list[dict], descriptions: dict[str, str], tokenizer):
        """Store rows, problem descriptions, and the tokenizer's terminator ids."""
        self.rows = rows
        self.descriptions = descriptions
        self.tokenizer = tokenizer
        self.terminator_ids = turn_terminator_ids(tokenizer)

    def __len__(self) -> int:
        """Return the number of training rows."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """Return prompt-masked token ids for one accepted submission."""
        row = self.rows[index]
        if "source" in row:
            source = str(row["source"])
        else:
            source = Path(row["source_path"]).read_text(
                encoding="utf-8", errors="replace"
            )
        prompt_ids = encode_generation_prompt(
            self.tokenizer, self.descriptions[row["problem_id"]]
        )
        code_ids = self.tokenizer.encode(source, add_special_tokens=False)
        available_code_tokens = MAX_LENGTH - len(prompt_ids) - len(self.terminator_ids)
        if available_code_tokens < 1:
            raise ValueError(f"prompt leaves no code tokens for {row['problem_id']}")
        code_ids = code_ids[:available_code_tokens]
        ids = prompt_ids + code_ids + self.terminator_ids
        labels = [-100] * len(prompt_ids) + code_ids + self.terminator_ids
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": labels,
        }


class CodeCollator:
    """Dynamically pad code batches instead of padding every example to 2,048 tokens."""

    def __init__(self, pad_token_id: int):
        """Store the pad token id used to fill a batch."""
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """Pad a batch to the next multiple of eight, capped at ``MAX_LENGTH``."""
        maximum = min(
            MAX_LENGTH, max(len(feature["input_ids"]) for feature in features)
        )
        padded_length = min(MAX_LENGTH, ((maximum + 7) // 8) * 8)
        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for feature in features:
            pad = padded_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            batch["labels"].append(feature["labels"] + [-100] * pad)
        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()
        }
