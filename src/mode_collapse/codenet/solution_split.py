"""Accepted-Python solution split, subtree-cosine truth, and the split audit.

The builder streams sources from the official archive. Training order is a
hash rank, so every smaller grid point is a prefix of every larger one.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import re
import tarfile
import warnings
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from html.parser import HTMLParser
from pathlib import Path

from tqdm import tqdm

from mode_collapse.codenet.canonical_ast import (
    StrategySignature,
    analyze_source,
    feature_fingerprint,
    normalized_features,
)
from mode_collapse.codenet.config import SEED, VAL_FRACTION, training_grid


class TextExtractor(HTMLParser):
    """Extract visible text from a CodeNet problem-description HTML file."""

    def __init__(self) -> None:
        """Start with an empty text buffer."""
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        """Skip script and style blocks and break lines at block tags."""
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"p", "br", "li", "h1", "h2", "h3", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Close a hidden block and break lines after paragraphs and list items."""
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
        if tag in {"p", "li", "pre"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Append visible text."""
        if not self.hidden:
            self.parts.append(data)

    def text(self) -> str:
        """Return normalized visible text."""
        value = html.unescape(" ".join(self.parts))
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n\s*\n+", "\n\n", value)
        return value.strip()


def stable_int(*parts: str) -> int:
    """Return a stable 64-bit integer from the given key parts."""
    return int.from_bytes(hashlib.sha256("\0".join(parts).encode()).digest()[:8], "big")


def solution_is_validation(submission_id: str, seed: int, val_fraction: float) -> bool:
    """Assign one solution to the deterministic Bernoulli validation split."""
    threshold = math.floor(val_fraction * (1 << 64))
    return stable_int("split", str(seed), submission_id) < threshold


def solution_rank(submission_id: str, seed: int) -> int:
    """Give each solution a stable rank so every smaller n is nested in every larger n."""
    return stable_int("sample-order", str(seed), submission_id)


def read_description(dataset_root: Path, problem_id: str) -> str | None:
    """Return the problem statement text, or None when the HTML file is missing or empty."""
    description = dataset_root / "problem_descriptions" / f"{problem_id}.html"
    if not description.exists():
        return None
    parser = TextExtractor()
    parser.feed(description.read_text(encoding="utf-8", errors="replace"))
    text = parser.text()
    return text or None


def write_ranked_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows in the stable nested-sampling order without the internal rank field."""
    rows.sort(key=lambda row: (int(row["rank"]), str(row["submission_id"])))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            public_row = {key: value for key, value in row.items() if key != "rank"}
            handle.write(json.dumps(public_row, separators=(",", ":")) + "\n")
    temporary.replace(path)


class CollisionAccumulator:
    """Streaming sufficient statistics for exact AST and signature collisions."""

    def __init__(self) -> None:
        """Start an empty accumulator."""
        self.n = 0
        self.summed_unit: Counter[str] = Counter()
        self.signature_counts: Counter[StrategySignature] = Counter()
        self.fingerprints: set[str] = set()

    def add(
        self, features: Counter[str], signature: StrategySignature, fingerprint: str
    ) -> None:
        """Add one parsed program."""
        self.n += 1
        self.summed_unit.update(normalized_features(features))
        self.signature_counts[signature] += 1
        self.fingerprints.add(fingerprint)

    def metrics(self) -> dict[str, object]:
        """Return with-replacement and distinct-pair collision statistics."""
        n = self.n
        if n == 0:
            raise ValueError("cannot compute collision for zero programs")

        ast_numerator = math.fsum(value * value for value in self.summed_unit.values())
        ast_collision = min(1.0, max(0.0, ast_numerator / (n * n)))

        strategy_numerator = sum(
            count * count for count in self.signature_counts.values()
        )
        strategy_collision = strategy_numerator / (n * n)

        if n > 1:
            ast_distinct = min(1.0, max(0.0, (ast_numerator - n) / (n * (n - 1))))
            strategy_distinct = (strategy_numerator - n) / (n * (n - 1))
        else:
            ast_distinct = math.nan
            strategy_distinct = math.nan

        return {
            "ast_code_collision": ast_collision,
            "ast_code_collision_distinct": ast_distinct,
            "ast_effective_diversity": 1.0 / ast_collision,
            "strategy_collision": strategy_collision,
            "strategy_collision_distinct": strategy_distinct,
            "strategy_effective_diversity": 1.0 / strategy_collision,
            "n_strategy_classes": len(self.signature_counts),
            "largest_strategy_share": max(self.signature_counts.values()) / n,
        }


def exact_collisions(
    feature_rows: Sequence[Counter[str]], signatures: Sequence[StrategySignature]
) -> dict[str, object]:
    """Compute with-replacement and distinct-pair collisions without materializing pairs."""
    if not feature_rows or len(signatures) != len(feature_rows):
        raise ValueError("feature/signature rows must have the same positive length")
    accumulator = CollisionAccumulator()
    for index, (features, signature) in enumerate(
        zip(feature_rows, signatures, strict=True)
    ):
        accumulator.add(features, signature, str(index))
    return accumulator.metrics()


def member_name(row: dict[str, str]) -> str:
    """Return the archive member path of one accepted Python submission."""
    extension = row["filename_ext"].lstrip(".")
    if not extension:
        raise ValueError(f"missing filename extension for {row['submission_id']}")
    return (
        f"Project_CodeNet/data/{row['problem_id']}/Python/"
        f"{row['submission_id']}.{extension}"
    )


def parse_batch(records: list[tuple[str, str, str, str, int, float]]) -> list[dict]:
    """Parse one bounded batch in a worker process."""
    results: list[dict] = []
    for problem_id, submission_id, user_id, source, seed, val_fraction in records:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                features, signature = analyze_source(
                    source, f"{problem_id}/{submission_id}"
                )
        except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
            results.append({"error": type(exc).__name__})
            continue
        split = (
            "val"
            if solution_is_validation(submission_id, seed, val_fraction)
            else "train"
        )
        results.append(
            {
                "problem_id": problem_id,
                "split": split,
                "row": {
                    "problem_id": problem_id,
                    "submission_id": submission_id,
                    "user_id": user_id,
                    "source": source,
                    "rank": solution_rank(submission_id, seed),
                },
                "features": features,
                "signature": signature,
                "fingerprint": feature_fingerprint(features),
            }
        )
    return results


def scan_metadata(
    dataset_root: Path,
) -> tuple[Counter[str], dict[str, tuple[str, str, str]], dict[str, str]]:
    """Scan metadata for accepted Python submissions that have a description."""
    scan_counts: Counter[str] = Counter()
    wanted: dict[str, tuple[str, str, str]] = {}
    descriptions: dict[str, str] = {}
    metadata_paths = sorted((dataset_root / "metadata").glob("p?????.csv"))
    if not metadata_paths:
        raise RuntimeError(f"no CodeNet metadata found under {dataset_root}")
    for metadata_path in tqdm(metadata_paths, desc="metadata"):
        problem_id = metadata_path.stem
        description = read_description(dataset_root, problem_id)
        accepted_rows: list[dict[str, str]] = []
        with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                scan_counts["all_submission_rows"] += 1
                if row["language"] == "Python":
                    scan_counts["python_submission_rows"] += 1
                if row["language"] != "Python" or row["status"] != "Accepted":
                    continue
                scan_counts["accepted_python_rows"] += 1
                accepted_rows.append(row)
        if not accepted_rows:
            continue
        scan_counts["problems_with_accepted_python"] += 1
        if description is None:
            scan_counts["accepted_python_rows_without_description"] += len(
                accepted_rows
            )
            scan_counts["accepted_python_problems_without_description"] += 1
            continue
        descriptions[problem_id] = description
        for row in accepted_rows:
            name = member_name(row)
            if name in wanted:
                raise RuntimeError(f"duplicate accepted archive member: {name}")
            wanted[name] = (problem_id, row["submission_id"], row["user_id"])
    return scan_counts, wanted, descriptions


def _write_split_outputs(
    output_dir: Path,
    train_rows: list[dict[str, object]],
    val_rows: list[dict[str, object]],
    truth_rows: list[dict[str, object]],
    descriptions: dict[str, str],
    summary: dict[str, object],
) -> None:
    """Write the ranked splits, truth table, descriptions, and summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_ranked_jsonl(output_dir / "train.jsonl", train_rows)
    write_ranked_jsonl(output_dir / "val.jsonl", val_rows)
    truth_rows.sort(key=lambda row: str(row["problem_id"]))
    truth_path = output_dir / "problem_truth.csv"
    truth_temporary = truth_path.with_suffix(".csv.tmp")
    with truth_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(truth_rows[0]))
        writer.writeheader()
        writer.writerows(truth_rows)
    truth_temporary.replace(truth_path)
    descriptions_path = output_dir / "descriptions.json"
    descriptions_temporary = descriptions_path.with_suffix(".json.tmp")
    descriptions_temporary.write_text(
        json.dumps(dict(sorted(descriptions.items())), separators=(",", ":")),
        encoding="utf-8",
    )
    descriptions_temporary.replace(descriptions_path)
    summary_path = output_dir / "summary.json"
    summary_temporary = summary_path.with_suffix(".json.tmp")
    summary_temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_temporary.replace(summary_path)


def build_from_archive(
    dataset_root: Path,
    archive: Path,
    output_dir: Path,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
    workers: int | None = None,
    batch_size: int = 64,
    pending_batches: int = 0,
) -> dict[str, object]:
    """Stream accepted Python sources from the archive and write the solution split."""
    # setup
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) // 2)
    if not 0 < val_fraction < 1:
        raise SystemExit("--val-fraction must be strictly between zero and one")
    if workers < 1 or batch_size < 1:
        raise SystemExit("worker and batch counts must be positive")
    if not archive.is_file():
        raise SystemExit(f"CodeNet archive is missing: {archive}")

    # data import
    scan_counts, wanted, all_descriptions = scan_metadata(dataset_root)
    expected_archive_sources = len(wanted)
    max_pending = pending_batches or max(2, workers * 2)
    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    accumulators: dict[str, CollisionAccumulator] = {}
    per_problem_train: Counter[str] = Counter()
    per_problem_val: Counter[str] = Counter()
    processing_counts: Counter[str] = Counter()
    parse_errors: Counter[str] = Counter()

    def consume_results(results: list[dict], progress) -> None:
        """Fold one parsed batch into the split and the per-problem accumulators."""
        for result in results:
            progress.update(1)
            if "error" in result:
                processing_counts["ast_parse_failures"] += 1
                parse_errors[str(result["error"])] += 1
                continue
            problem_id = str(result["problem_id"])
            split = str(result["split"])
            if split == "train":
                train_rows.append(result["row"])
                per_problem_train[problem_id] += 1
            else:
                val_rows.append(result["row"])
                per_problem_val[problem_id] += 1
            accumulator = accumulators.setdefault(problem_id, CollisionAccumulator())
            accumulator.add(
                result["features"], result["signature"], str(result["fingerprint"])
            )

    found_sources = 0

    def source_records(progress):
        """Yield accepted sources in one sequential pass over the archive."""
        nonlocal found_sources
        with tarfile.open(archive, mode="r|gz") as archive_handle:
            for member in archive_handle:
                if not wanted:
                    break
                metadata = wanted.pop(member.name, None)
                if metadata is None:
                    continue
                found_sources += 1
                extracted = archive_handle.extractfile(member)
                if extracted is None:
                    processing_counts["source_read_failures"] += 1
                    progress.update(1)
                    continue
                problem_id, submission_id, user_id = metadata
                source = extracted.read().decode("utf-8", errors="replace")
                yield (
                    problem_id,
                    submission_id,
                    user_id,
                    source,
                    seed,
                    val_fraction,
                )

    # data filtering
    with tqdm(total=expected_archive_sources, desc="accepted source ASTs") as progress:
        batch: list[tuple[str, str, str, str, int, float]] = []
        if workers == 1:
            for record in source_records(progress):
                batch.append(record)
                if len(batch) == batch_size:
                    consume_results(parse_batch(batch), progress)
                    batch = []
            if batch:
                consume_results(parse_batch(batch), progress)
        else:
            pending: set[Future[list[dict]]] = set()
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for record in source_records(progress):
                    batch.append(record)
                    if len(batch) < batch_size:
                        continue
                    pending.add(executor.submit(parse_batch, batch))
                    batch = []
                    if len(pending) >= max_pending:
                        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in finished:
                            consume_results(future.result(), progress)
                if batch:
                    pending.add(executor.submit(parse_batch, batch))
                for future in pending:
                    consume_results(future.result(), progress)

    if wanted:
        processing_counts["source_read_failures"] += len(wanted)
        sample = sorted(wanted)[:5]
        raise RuntimeError(
            f"archive is missing {len(wanted):,} accepted Python sources; sample={sample}"
        )
    if not train_rows or not val_rows or not accumulators:
        raise RuntimeError("the accepted-Python solution split is empty")
    processing_counts["problems_without_parseable_solutions"] = len(
        all_descriptions
    ) - len(accumulators)

    # model fitting and everything else
    truth_rows: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}
    for problem_id, accumulator in accumulators.items():
        metrics = accumulator.metrics()
        n_train = per_problem_train[problem_id]
        n_val = per_problem_val[problem_id]
        truth_rows.append(
            {
                "problem_id": problem_id,
                "is_eval": int(accumulator.n >= 2),
                "n_human": accumulator.n,
                "n_train": n_train,
                "n_val": n_val,
                "ast_collision": metrics["ast_code_collision_distinct"],
                "strategy_collision": metrics["strategy_collision_distinct"],
                "ast_collision_with_self": metrics["ast_code_collision"],
                "strategy_collision_with_self": metrics["strategy_collision"],
                "n_strategy_classes": metrics["n_strategy_classes"],
            }
        )
        descriptions[problem_id] = all_descriptions[problem_id]

    n_usable = len(train_rows) + len(val_rows)
    n_eval_problems = sum(int(row["is_eval"]) for row in truth_rows)
    summary = {
        "experiment": "accepted Python CodeNet solution-level split",
        "seed": seed,
        "split_unit": "submission",
        "train_fraction_target": 1 - val_fraction,
        "val_fraction_target": val_fraction,
        "problem_count_cutoff": None,
        "status_filter": "Accepted",
        "language_filter": "Python",
        "duplicate_user_problem_submissions_retained": True,
        "source_access": "single sequential pass over official tar.gz archive",
        "source_storage": "embedded in processed train/val JSONL",
        "source_archive": str(archive),
        **scan_counts,
        **processing_counts,
        "accepted_archive_sources_expected": expected_archive_sources,
        "accepted_archive_sources_found": found_sources,
        "usable_accepted_python_rows": n_usable,
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "observed_val_fraction": len(val_rows) / n_usable,
        "n_problems": len(truth_rows),
        "n_eval_problems": n_eval_problems,
        "problems_in_both_splits": sum(
            int(int(row["n_train"]) > 0 and int(row["n_val"]) > 0) for row in truth_rows
        ),
        "human_truth_pool": "all usable accepted Python solutions across both splits",
        "truth_collision_estimator": "distinct-pair U-statistic",
        "training_grid": training_grid(len(train_rows)),
        "parse_errors": dict(parse_errors.most_common()),
    }

    # data export
    _write_split_outputs(
        output_dir, train_rows, val_rows, truth_rows, descriptions, summary
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def prepare_accepted_python(
    dataset_root: Path,
    member_list: Path,
    summary_path: Path,
    verify_extracted: bool = False,
) -> dict[str, object]:
    """List every accepted Python source member in the official CodeNet metadata."""
    metadata_paths = sorted((dataset_root / "metadata").glob("p?????.csv"))
    if not metadata_paths:
        raise SystemExit(f"no CodeNet metadata found under {dataset_root}")

    member_list.parent.mkdir(parents=True, exist_ok=True)
    accepted_count = 0
    accepted_problems: set[str] = set()
    all_rows = 0
    python_rows = 0
    with member_list.open("w", encoding="utf-8") as members:
        for metadata_path in tqdm(metadata_paths, desc="accepted Python members"):
            with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    all_rows += 1
                    if row["language"] == "Python":
                        python_rows += 1
                    if row["language"] != "Python" or row["status"] != "Accepted":
                        continue
                    extension = row["filename_ext"].lstrip(".")
                    if not extension:
                        raise ValueError(
                            f"missing filename extension for {row['submission_id']}"
                        )
                    accepted_count += 1
                    accepted_problems.add(row["problem_id"])
                    members.write(
                        f"Project_CodeNet/data/{row['problem_id']}/Python/"
                        f"{row['submission_id']}.{extension}\n"
                    )

    summary: dict[str, object] = {
        "scope": "all official CodeNet metadata",
        "metadata_files": len(metadata_paths),
        "all_submission_rows": all_rows,
        "python_submission_rows": python_rows,
        "accepted_python_rows": accepted_count,
        "accepted_python_problems": len(accepted_problems),
        "problem_count_cutoff": None,
        "per_problem_solution_cutoff": None,
        "one_per_user_filter": False,
    }
    if verify_extracted:
        extracted_count = sum(
            1
            for path in (dataset_root / "data").glob("p?????/Python/*")
            if path.is_file()
        )
        summary["extracted_source_files"] = extracted_count
        summary["archive_extraction_scope"] = "all Python sources"
        summary["downstream_training_filter"] = "language=Python and status=Accepted"
        if extracted_count != python_rows:
            raise RuntimeError(
                f"found {extracted_count:,} Python source files; expected {python_rows:,}"
            )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _audit_split(
    path: Path,
    split: str,
    seed: int,
    val_fraction: float,
    ast_sample: int,
) -> tuple[int, set[str], Counter[str], int]:
    """Check one JSONL split, including order, assignment, paths, and sampled AST parsing."""
    submission_ids: set[str] = set()
    per_problem: Counter[str] = Counter()
    previous_key: tuple[int, str] | None = None
    parsed = 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            submission_id = row["submission_id"]
            if submission_id in submission_ids:
                raise AssertionError(
                    f"duplicate submission in {split}: {submission_id}"
                )
            submission_ids.add(submission_id)
            per_problem[row["problem_id"]] += 1

            expected_val = solution_is_validation(submission_id, seed, val_fraction)
            if (split == "val") != expected_val:
                raise AssertionError(f"incorrect split assignment: {submission_id}")
            key = (solution_rank(submission_id, seed), submission_id)
            if previous_key is not None and key < previous_key:
                raise AssertionError(f"{split} is not in stable nested-sampling order")
            previous_key = key

            if "source" in row:
                source = row["source"]
                if not isinstance(source, str) or not source.strip():
                    raise AssertionError(f"invalid embedded source: {submission_id}")
                source_name = f"{row['problem_id']}/{submission_id}"
            else:
                source_path = Path(row["source_path"])
                if not source_path.is_file() or source_path.parent.name != "Python":
                    raise AssertionError(f"invalid Python source path: {source_path}")
                source = source_path.read_text(encoding="utf-8", errors="replace")
                source_name = str(source_path)
            if parsed < ast_sample:
                analyze_source(source, source_name)
                parsed += 1
    return count, submission_ids, per_problem, parsed


def audit_processed_split(
    processed_dir: Path, ast_sample_per_split: int = 5000
) -> dict:
    """Check the processed split against its summary and a sample of reparsed sources."""
    summary = json.loads((processed_dir / "summary.json").read_text())
    seed = int(summary["seed"])
    val_fraction = float(summary["val_fraction_target"])
    train = _audit_split(
        processed_dir / "train.jsonl",
        "train",
        seed,
        val_fraction,
        ast_sample_per_split,
    )
    val = _audit_split(
        processed_dir / "val.jsonl",
        "val",
        seed,
        val_fraction,
        ast_sample_per_split,
    )
    if train[1] & val[1]:
        raise AssertionError("a submission appears in both train and validation")
    if train[0] != int(summary["n_train_rows"]):
        raise AssertionError("training row count does not match summary")
    if val[0] != int(summary["n_val_rows"]):
        raise AssertionError("validation row count does not match summary")
    if summary["problem_count_cutoff"] is not None:
        raise AssertionError("a problem-count cutoff is still active")
    if [int(value) for value in summary["training_grid"]] != training_grid(train[0]):
        raise AssertionError("training grid does not match the N-anchored halving rule")

    descriptions = json.loads((processed_dir / "descriptions.json").read_text())
    with (processed_dir / "problem_truth.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        truth_rows = list(csv.DictReader(handle))
    if set(descriptions) != {row["problem_id"] for row in truth_rows}:
        raise AssertionError("description and truth problem sets differ")
    if len(truth_rows) != int(summary["n_problems"]):
        raise AssertionError("problem count does not match summary")

    eval_problems = 0
    for row in truth_rows:
        problem_id = row["problem_id"]
        n_train = train[2][problem_id]
        n_val = val[2][problem_id]
        if n_train != int(row["n_train"]) or n_val != int(row["n_val"]):
            raise AssertionError(f"per-problem split count mismatch for {problem_id}")
        n_human = int(row["n_human"])
        if n_human != n_train + n_val:
            raise AssertionError(
                f"human truth does not use the full pool for {problem_id}"
            )
        expected_eval = n_human >= 2
        if bool(int(row["is_eval"])) != expected_eval:
            raise AssertionError(f"incorrect evaluation flag for {problem_id}")
        if expected_eval:
            eval_problems += 1
            for field in ("ast_collision", "strategy_collision"):
                if not math.isfinite(float(row[field])):
                    raise AssertionError(f"non-finite {field} for {problem_id}")
    if eval_problems != int(summary["n_eval_problems"]):
        raise AssertionError("evaluation problem count does not match summary")

    observed_val = val[0] / (train[0] + val[0])
    if not math.isclose(
        observed_val, float(summary["observed_val_fraction"]), abs_tol=1e-15
    ):
        raise AssertionError("observed validation fraction does not match summary")

    report = {
        "status": "ok",
        "split_unit": "submission",
        "problem_count_cutoff": None,
        "train_rows": train[0],
        "val_rows": val[0],
        "observed_val_fraction": observed_val,
        "submission_overlap": 0,
        "problems": len(truth_rows),
        "evaluation_problems": eval_problems,
        "sampled_sources_reparsed": train[3] + val[3],
        "training_grid": summary["training_grid"],
    }
    print(json.dumps(report, indent=2))
    return report
