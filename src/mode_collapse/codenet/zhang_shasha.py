"""Unit-cost Zhang–Shasha distance and the normalized CodeNet kernel.

Trees are built from the canonical AST in ``canonical_ast``. Node labels are
``edge:label`` with the same ordered, field-typed children as that AST.
The distance is the Zhang–Shasha recurrence with unit insert, delete, and
relabel costs. The kernel is ``1 - distance / (left nodes + right nodes)``.
The postorder annotation (leftmost descendants and keyroots) is laid out as in
the ``zss`` package, whose license is ``LICENSE-zss.txt`` beside this module.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mode_collapse.codenet.canonical_ast import (
    _ast_children,
    _node_label,
    parse_and_canonicalize,
)
from mode_collapse.codenet.config import (
    EXPECTED_SHARDS,
    HUMAN_SAMPLE,
    N_GENERATIONS,
    PAIR_SAMPLE,
    SEED,
)

EXPECTED_GENERATIONS = N_GENERATIONS


@dataclass(frozen=True, slots=True)
class TreeNode:
    """One labeled node in an ordered canonical AST."""

    label: str
    children: tuple[TreeNode, ...]


@dataclass(frozen=True, slots=True)
class AnnotatedTree:
    """Postorder metadata used by the Zhang–Shasha dynamic program."""

    nodes: tuple[TreeNode, ...]
    leftmost_descendants: tuple[int, ...]
    keyroots: tuple[int, ...]

    @classmethod
    def from_root(cls, root: TreeNode) -> AnnotatedTree:
        """Construct postorder indices, leftmost descendants, and keyroots."""
        postorder: list[TreeNode] = []
        stack: list[tuple[TreeNode, bool]] = [(root, False)]
        while stack:
            node, visited = stack.pop()
            if visited:
                postorder.append(node)
                continue
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(node.children))

        node_indices = {id(node): index for index, node in enumerate(postorder)}
        leftmost_descendants: list[int] = []
        for index, node in enumerate(postorder):
            if node.children:
                first_child_index = node_indices[id(node.children[0])]
                leftmost_descendants.append(leftmost_descendants[first_child_index])
            else:
                leftmost_descendants.append(index)

        keyroot_by_leftmost = {
            leftmost: index for index, leftmost in enumerate(leftmost_descendants)
        }
        return cls(
            nodes=tuple(postorder),
            leftmost_descendants=tuple(leftmost_descendants),
            keyroots=tuple(sorted(keyroot_by_leftmost.values())),
        )


@dataclass(frozen=True, slots=True)
class PreparedTree:
    """A canonical tree and its stable structural fingerprint."""

    annotated: AnnotatedTree
    fingerprint: str


CACHE_FIELDS = (
    "problem_id",
    "arm",
    "model",
    "n",
    "checkpoint",
    "n_candidates",
    "n_parseable",
    "n_pairs_available",
    "n_pairs_scored",
    "zss_collision",
    "parse_errors_json",
)


def stable_rank(*parts: object) -> int:
    """Return a deterministic rank keyed by the global seed."""
    payload = "\0".join([str(SEED), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def tree_fingerprint(root: TreeNode) -> str:
    """Hash a complete ordered labeled tree without recursion."""
    digest = hashlib.sha256()
    stack = [root]
    while stack:
        node = stack.pop()
        label = node.label.encode("utf-8")
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(node.children).to_bytes(4, "big"))
        stack.extend(reversed(node.children))
    return digest.hexdigest()


def build_tree(node, edge: str, ast_children, node_label) -> TreeNode:
    """Convert a canonical Python AST into an ordered edge-typed tree."""
    children = tuple(
        build_tree(child, child_edge, ast_children, node_label)
        for child_edge, child in ast_children(node)
    )
    return TreeNode(label=f"{edge}:{node_label(node)}", children=children)


def prepare_source(source: str, filename: str) -> PreparedTree:
    """Parse and canonicalize source code for tree-edit comparison."""
    tree = parse_and_canonicalize(source, filename)
    root = build_tree(tree, "root", _ast_children, _node_label)
    return PreparedTree(
        annotated=AnnotatedTree.from_root(root),
        fingerprint=tree_fingerprint(root),
    )


def zhang_shasha_distance(left: AnnotatedTree, right: AnnotatedTree) -> int:
    """Return unit-cost ordered tree edit distance using Zhang–Shasha."""
    left_count = len(left.nodes)
    right_count = len(right.nodes)
    tree_distances = np.zeros((left_count, right_count), dtype=np.int32)

    for left_keyroot in left.keyroots:
        for right_keyroot in right.keyroots:
            left_start = left.leftmost_descendants[left_keyroot]
            right_start = right.leftmost_descendants[right_keyroot]
            row_count = left_keyroot - left_start + 2
            column_count = right_keyroot - right_start + 2
            forest_distances = np.zeros((row_count, column_count), dtype=np.int32)
            forest_distances[:, 0] = np.arange(row_count, dtype=np.int32)
            forest_distances[0, :] = np.arange(column_count, dtype=np.int32)
            left_offset = left_start - 1
            right_offset = right_start - 1

            for row in range(1, row_count):
                left_index = row + left_offset
                left_node = left.nodes[left_index]
                for column in range(1, column_count):
                    right_index = column + right_offset
                    right_node = right.nodes[right_index]
                    same_subtree = (
                        left.leftmost_descendants[left_keyroot]
                        == left.leftmost_descendants[left_index]
                        and right.leftmost_descendants[right_keyroot]
                        == right.leftmost_descendants[right_index]
                    )
                    if same_subtree:
                        update_cost = int(left_node.label != right_node.label)
                        value = min(
                            forest_distances[row - 1, column] + 1,
                            forest_distances[row, column - 1] + 1,
                            forest_distances[row - 1, column - 1] + update_cost,
                        )
                        forest_distances[row, column] = value
                        tree_distances[left_index, right_index] = value
                    else:
                        left_prefix = (
                            left.leftmost_descendants[left_index] - 1 - left_offset
                        )
                        right_prefix = (
                            right.leftmost_descendants[right_index] - 1 - right_offset
                        )
                        forest_distances[row, column] = min(
                            forest_distances[row - 1, column] + 1,
                            forest_distances[row, column - 1] + 1,
                            forest_distances[left_prefix, right_prefix]
                            + tree_distances[left_index, right_index],
                        )

    return int(tree_distances[-1, -1])


def tree_similarity(
    left: PreparedTree,
    right: PreparedTree,
    distance_cache: dict[tuple[str, str], int],
) -> float:
    """Return sum-size-normalized Zhang–Shasha tree similarity."""
    if left.fingerprint == right.fingerprint:
        return 1.0
    fingerprints = tuple(sorted((left.fingerprint, right.fingerprint)))
    if fingerprints not in distance_cache:
        distance_cache[fingerprints] = zhang_shasha_distance(
            left.annotated, right.annotated
        )
    distance = distance_cache[fingerprints]
    total_nodes = len(left.annotated.nodes) + len(right.annotated.nodes)
    if not 0 <= distance <= total_nodes:
        raise AssertionError(
            f"invalid tree edit distance {distance} for {total_nodes} total nodes"
        )
    return 1.0 - distance / total_nodes


def selected_pairs(
    prepared: list[tuple[str, PreparedTree]],
    arm: str,
    problem_id: str,
) -> list[tuple[int, int]]:
    """Select a deterministic subset of distinct pairs without replacement."""
    pairs = list(combinations(range(len(prepared)), 2))
    pairs.sort(
        key=lambda pair: stable_rank(
            "pair",
            arm,
            problem_id,
            prepared[pair[0]][0],
            prepared[pair[1]][0],
        )
    )
    return pairs[:PAIR_SAMPLE]


def analyze_problem(
    task: tuple[str, str, str, int, str, list[tuple[str, str]]],
) -> dict[str, object]:
    """Compute one problem-arm kernel collision from up to 32 unordered pairs."""
    problem_id, arm, model, n, checkpoint, candidates = task
    prepared: list[tuple[str, PreparedTree]] = []
    parse_errors: dict[str, int] = defaultdict(int)
    for candidate_id, source in candidates:
        try:
            prepared.append(
                (candidate_id, prepare_source(source, f"{problem_id}/{candidate_id}"))
            )
        except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
            parse_errors[type(exc).__name__] += 1

    pairs = selected_pairs(prepared, arm, problem_id) if len(prepared) >= 2 else []
    distance_cache: dict[tuple[str, str], int] = {}
    similarities = [
        tree_similarity(prepared[left][1], prepared[right][1], distance_cache)
        for left, right in pairs
    ]
    collision = (
        math.fsum(similarities) / len(similarities) if similarities else math.nan
    )
    return {
        "problem_id": problem_id,
        "arm": arm,
        "model": model,
        "n": n,
        "checkpoint": checkpoint,
        "n_candidates": len(candidates),
        "n_parseable": len(prepared),
        "n_pairs_available": len(prepared) * (len(prepared) - 1) // 2,
        "n_pairs_scored": len(similarities),
        "zss_collision": collision,
        "parse_errors_json": json.dumps(parse_errors, sort_keys=True),
    }


def quantile(values: Iterable[float], probability: float) -> float:
    """Return a linearly interpolated empirical quantile."""
    return float(np.quantile(np.asarray(list(values), dtype=float), probability))


def load_eval_problem_ids(processed_dir: Path) -> list[str]:
    """Load the fixed evaluation problem identifiers."""
    truth_path = processed_dir / "problem_truth.csv"
    with truth_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    problem_ids = sorted(row["problem_id"] for row in rows if row["is_eval"] == "1")
    if len(problem_ids) != len(set(problem_ids)):
        raise AssertionError("duplicate evaluation problem identifiers")
    return problem_ids


def load_human_candidates(
    processed_dir: Path,
    problem_ids: list[str],
) -> dict[str, list[tuple[str, str]]]:
    """Select the lowest stable-rank accepted programs for every problem."""
    eval_problem_ids = set(problem_ids)
    heaps: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for split in ("train", "val"):
        split_path = processed_dir / f"{split}.jsonl"
        with split_path.open(encoding="utf-8") as handle:
            for line in tqdm(handle, desc=f"human {split}", unit="program"):
                row = json.loads(line)
                problem_id = row["problem_id"]
                if problem_id not in eval_problem_ids:
                    continue
                submission_id = row["submission_id"]
                rank = stable_rank("human-program", submission_id)
                entry = (-rank, submission_id, row["source"])
                heap = heaps[problem_id]
                if len(heap) < HUMAN_SAMPLE:
                    heapq.heappush(heap, entry)
                elif rank < -heap[0][0]:
                    heapq.heapreplace(heap, entry)

    missing = [problem_id for problem_id in problem_ids if len(heaps[problem_id]) < 2]
    if missing:
        raise AssertionError(
            f"evaluation problems with fewer than two human programs: {missing}"
        )
    return {
        problem_id: sorted(
            ((submission_id, source) for _, submission_id, source in heaps[problem_id]),
            key=lambda item: stable_rank("human-program", item[0]),
        )
        for problem_id in problem_ids
    }


def generation_paths(
    measurement_dir: Path,
    model: str,
    n: int,
    checkpoint: str,
) -> list[Path]:
    """Resolve and validate all saved generation shards for one arm."""
    pattern = f"generations_codenet_{model}_n{n}_{checkpoint}_shard*-of-*.jsonl.gz"
    paths = sorted(measurement_dir.glob(pattern))
    if len(paths) != EXPECTED_SHARDS:
        raise AssertionError(
            f"expected {EXPECTED_SHARDS} generation shards for "
            f"{model}/{n}/{checkpoint}, found {len(paths)}"
        )
    return paths


def load_model_candidates(
    measurement_dir: Path,
    problem_ids: list[str],
    model: str,
    n: int,
    checkpoint: str,
) -> dict[str, list[tuple[str, str]]]:
    """Load every saved generation for one model checkpoint."""
    eval_problem_ids = set(problem_ids)
    candidates: dict[str, dict[int, str]] = defaultdict(dict)
    paths = generation_paths(measurement_dir, model, n, checkpoint)
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in tqdm(handle, desc=path.name, unit="generation"):
                row = json.loads(line)
                problem_id = row["problem_id"]
                if problem_id not in eval_problem_ids:
                    raise AssertionError(f"unexpected generation problem {problem_id}")
                sample = int(row["sample"])
                if sample in candidates[problem_id]:
                    raise AssertionError(
                        f"duplicate generation {problem_id}/{sample} in {path}"
                    )
                candidates[problem_id][sample] = row["code"]

    invalid_counts = {
        problem_id: len(candidates[problem_id])
        for problem_id in problem_ids
        if len(candidates[problem_id]) != EXPECTED_GENERATIONS
    }
    if invalid_counts:
        raise AssertionError(f"invalid generation counts: {invalid_counts}")
    return {
        problem_id: [
            (f"sample{sample:02d}", source)
            for sample, source in sorted(candidates[problem_id].items())
        ]
        for problem_id in problem_ids
    }


def write_csv_once(path: Path, rows: list[dict[str, object]]) -> None:
    """Atomically write a new CSV and refuse to overwrite existing results."""
    if not rows:
        raise AssertionError(f"cannot write an empty table: {path}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {path}")
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if path.exists():
        raise FileExistsError(f"result appeared while writing: {path}")
    temporary.rename(path)


def read_cache(path: Path, problem_ids: list[str], arm: str) -> list[dict[str, str]]:
    """Read and validate one completed arm cache."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != list(CACHE_FIELDS):
        raise AssertionError(f"unexpected cache schema: {path}")
    if {row["problem_id"] for row in rows} != set(problem_ids):
        raise AssertionError(f"cache problem coverage mismatch: {path}")
    if {row["arm"] for row in rows} != {arm}:
        raise AssertionError(f"cache arm mismatch: {path}")
    return rows


def compute_arm(
    cache_path: Path,
    problem_ids: list[str],
    candidates: dict[str, list[tuple[str, str]]],
    arm: str,
    model: str,
    n: int,
    checkpoint: str,
    workers: int,
) -> list[dict[str, str]]:
    """Compute or load one restartable problem-level cache."""
    if cache_path.exists():
        print(f"using completed cache {cache_path}", flush=True)
        return read_cache(cache_path, problem_ids, arm)
    tasks = [
        (problem_id, arm, model, n, checkpoint, candidates[problem_id])
        for problem_id in problem_ids
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(
                executor.map(analyze_problem, tasks, chunksize=4),
                total=len(tasks),
                desc=arm,
                unit="problem",
            )
        )
    results.sort(key=lambda row: str(row["problem_id"]))
    write_csv_once(cache_path, results)
    return read_cache(cache_path, problem_ids, arm)


def human_cache_path(cache_dir: Path) -> Path:
    """Return the human-arm cache path."""
    return cache_dir / f"human_h{HUMAN_SAMPLE}_pairs{PAIR_SAMPLE}_seed{SEED}.csv"


def model_cache_path(cache_dir: Path, model: str, n: int, checkpoint: str) -> Path:
    """Return one model-arm cache path."""
    arm = f"{model}_n{n}_{checkpoint}"
    return cache_dir / f"{arm}_pairs{PAIR_SAMPLE}_seed{SEED}.csv"


def score_human(processed_dir: Path, cache_dir: Path, workers: int) -> Path:
    """Score accepted human programs, reusing a completed cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    problem_ids = load_eval_problem_ids(processed_dir)
    cache_path = human_cache_path(cache_dir)
    if cache_path.exists():
        read_cache(cache_path, problem_ids, "human")
        print(f"using completed cache {cache_path}", flush=True)
        return cache_path
    candidates = load_human_candidates(processed_dir, problem_ids)
    compute_arm(
        cache_path,
        problem_ids,
        candidates,
        "human",
        "human",
        -1,
        "human",
        workers,
    )
    return cache_path


def score_model_arm(
    processed_dir: Path,
    measurement_dir: Path,
    cache_dir: Path,
    model: str,
    n: int,
    checkpoint: str,
    workers: int,
) -> Path:
    """Score one model arm, reusing a completed cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    problem_ids = load_eval_problem_ids(processed_dir)
    arm = f"{model}_n{n}_{checkpoint}"
    cache_path = model_cache_path(cache_dir, model, n, checkpoint)
    if cache_path.exists():
        read_cache(cache_path, problem_ids, arm)
        print(f"using completed cache {cache_path}", flush=True)
        return cache_path
    candidates = load_model_candidates(
        measurement_dir, problem_ids, model, n, checkpoint
    )
    compute_arm(
        cache_path,
        problem_ids,
        candidates,
        arm,
        model,
        n,
        checkpoint,
        workers,
    )
    return cache_path
