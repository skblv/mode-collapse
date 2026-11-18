"""CPU checks for the CodeNet canonical AST, kernel, grid, and summary."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mode_collapse.codenet.canonical_ast import (
    analyze_source,
    canonical_ast_listing,
    cosine_similarity,
    parse_and_canonicalize,
    strategy_signature,
)
from mode_collapse.codenet.config import (
    build_messages,
    scoring_arms,
    training_grid,
    turn_terminator_ids,
)
from mode_collapse.codenet.convergence import (
    has_finite_ratio,
    same_problem_comparison,
    summarize_loaded_arms,
)
from mode_collapse.codenet.solution_split import (
    audit_processed_split,
    build_from_archive,
    exact_collisions,
    prepare_accepted_python,
)
from mode_collapse.codenet.training_data import CodeCollator, CodeDataset
from mode_collapse.codenet.zhang_shasha import (
    prepare_source,
    tree_similarity,
    zhang_shasha_distance,
)

FIXTURE = Path(__file__).parent / "fixture"
EXAMPLES = FIXTURE / "examples"
REPORTED_GRID = [
    12,
    24,
    48,
    96,
    193,
    387,
    775,
    1550,
    3101,
    6202,
    12405,
    24811,
    49622,
    99244,
    198488,
    396976,
    793953,
    1587907,
]


def _listing_walk(source: str) -> list[tuple[int, str, str]]:
    rows = []
    for line in canonical_ast_listing(source).splitlines():
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        edge, label = body.split(": ", 1)
        rows.append((indent // 2, edge, label))
    return rows


def _tree_walk(node, depth: int = 0) -> list[tuple[int, str, str]]:
    edge, label = node.label.split(":", 1)
    rows = [(depth, edge, label)]
    for child in node.children:
        rows.extend(_tree_walk(child, depth + 1))
    return rows


def test_reported_halving_grid() -> None:
    assert training_grid(1_587_907) == REPORTED_GRID
    assert scoring_arms(1_587_907)[0] == ("gemma-2-2b-it", 0, "base")
    assert scoring_arms(1_587_907)[-1] == ("gemma-3-4b-it", 1_587_907, "fully_trained")
    assert len(scoring_arms(1_587_907)) == 148


def test_halving_rule_on_the_survey_pool_size() -> None:
    assert len(training_grid(4_706_940)) == 20
    assert training_grid(4_706_940)[0] == 8
    assert training_grid(4_706_940)[-1] == 4_706_940
    assert training_grid(1_462_645)[0] == 11


def test_prompt_is_one_user_turn() -> None:
    messages = build_messages("Add two integers.")
    assert [message["role"] for message in messages] == ["user"]
    assert "complete Python 3 program" in messages[0]["content"]


def test_turn_terminator_comes_from_chat_template() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            return f"prompt{messages[-1]['content']}<end>"

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            assert text == "<end>"
            return [107, 13]

    assert turn_terminator_ids(FakeTokenizer()) == [107, 13]


def test_code_dataset_preserves_prompt_head_and_dynamic_padding() -> None:
    class FakeTokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            content = "".join(message["content"] for message in messages)
            suffix = "<assistant>" if add_generation_prompt else "<end>"
            return "HEAD:" + content + suffix

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return [1 + (ord(character) % 251) for character in text]

    fixture_source = FIXTURE / "data" / "p00001" / "Python" / "s000000001.py"
    rows = [{"problem_id": "p00001", "source_path": str(fixture_source)}]
    dataset = CodeDataset(rows, {"p00001": "x" * 5000}, FakeTokenizer())
    feature = dataset[0]
    assert len(feature["input_ids"]) <= 2048
    assert len(feature["input_ids"]) == len(feature["labels"])
    assert feature["labels"][:1024] == [-100] * 1024
    assert feature["labels"][-1] != -100
    batch = CodeCollator(0)([feature, feature])
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] % 8 == 0


def test_renaming_formatting_comments_and_literals_are_invariant() -> None:
    first = """
import heapq as h

def solve(values, limit):
    # This wording and all whitespace must disappear.
    total = 17
    for item in values:
        total = total + item
    h.heappush(values, total)
    return total > limit
"""
    second = """import heapq as priority
def renamed(xs, boundary):
    accumulator=999
    for element in xs:
        accumulator=accumulator+element
    priority.heappush(xs,accumulator)
    return accumulator>boundary
"""
    first_features, first_signature = analyze_source(first)
    second_features, second_signature = analyze_source(second)
    assert first_features == second_features
    assert first_signature == second_signature
    assert cosine_similarity(first_features, second_features) == 1.0


def test_literal_types_and_operators_remain_distinguishable() -> None:
    integer, _signature = analyze_source("result = 123 + 456")
    floating, _signature = analyze_source("result = 1.5 + 9.25")
    subtraction, _signature = analyze_source("result = 123 - 456")
    assert cosine_similarity(integer, floating) < 1.0
    assert cosine_similarity(integer, subtraction) < 1.0


def test_complete_ast_listing_uses_canonical_metric_tree() -> None:
    first = "value = 1\nprint(value + 1)\n"
    second = "renamed = 9\nprint(renamed + 4)  # ignored\n"
    listing = canonical_ast_listing(first)
    assert listing == canonical_ast_listing(second)
    assert "root: Module" in listing
    assert "Name:VAR_0_0" in listing
    assert "Constant:INT" in listing
    assert "Load" not in listing
    assert "lineno" not in listing


def test_identifier_renaming_is_scope_aware() -> None:
    reused_name = """
def first(value):
    return value + 1
def second(value):
    return value * 2
"""
    independently_renamed = """
def alpha(left):
    return left + 9
def beta(right):
    return right * 7
"""
    reused_features, _signature = analyze_source(reused_name)
    renamed_features, _signature = analyze_source(independently_renamed)
    assert reused_features == renamed_features


def test_strategy_signature_components() -> None:
    source = """
import heapq
from collections import deque

def dfs(graph, node):
    for neighbor in graph[node]:
        dfs(graph, neighbor)

table = [[0] * 4 for _ in range(3)]
table[0][0] = 1
mapping = {1: 2}
items = sorted(mapping.items())
heapq.heappush(items, (2, 3))
queue = deque(items)
while queue:
    queue.popleft()
"""
    signature = strategy_signature(parse_and_canonicalize(source))
    assert signature.recursion
    assert signature.loop_nesting_depth == 1
    assert signature.sorting
    assert signature.hashing
    assert signature.heap
    assert signature.graph_traversal
    assert signature.dynamic_programming_table


def test_mutual_recursion_is_detected() -> None:
    tree = parse_and_canonicalize(
        """
def even(number):
    return True if number == 0 else odd(number - 1)
def odd(number):
    return False if number == 0 else even(number - 1)
"""
    )
    assert strategy_signature(tree).recursion


def test_exact_linear_aggregation_matches_explicit_pairs() -> None:
    sources = [
        "x = [i + 1 for i in range(4)]",
        "renamed = [element + 7 for element in range(8)]",
        "x = {i: i * 2 for i in range(4)}",
    ]
    analyzed = [analyze_source(source) for source in sources]
    features = [row[0] for row in analyzed]
    signatures = [row[1] for row in analyzed]
    metrics = exact_collisions(features, signatures)
    n = len(features)
    explicit = math.fsum(
        cosine_similarity(left, right) for left in features for right in features
    ) / (n * n)
    explicit_distinct = math.fsum(
        cosine_similarity(features[i], features[j])
        for i in range(n)
        for j in range(n)
        if i != j
    ) / (n * (n - 1))
    assert float(metrics["ast_code_collision"]) == pytest.approx(explicit)
    assert float(metrics["ast_code_collision_distinct"]) == pytest.approx(
        explicit_distinct
    )


def test_listing_and_zhang_shasha_trees_match_on_real_programs() -> None:
    sources = [path.read_text(encoding="utf-8") for path in EXAMPLES.glob("*.py")]
    sources.extend(
        path.read_text(encoding="utf-8")
        for path in (FIXTURE / "data").glob("p?????/Python/*.py")
    )
    assert sources
    compared = 0
    for source in sources:
        try:
            prepared = prepare_source(source, "<fixture>")
        except SyntaxError:
            continue
        assert _listing_walk(source) == _tree_walk(prepared.annotated.nodes[-1])
        compared += 1
    assert compared >= len(list(EXAMPLES.glob("*.py")))


def test_paper_example_kernel() -> None:
    high_a = prepare_source(
        (EXAMPLES / "high_a.py").read_text(encoding="utf-8"), "high_a.py"
    )
    high_b = prepare_source(
        (EXAMPLES / "high_b.py").read_text(encoding="utf-8"), "high_b.py"
    )
    low_a = prepare_source(
        (EXAMPLES / "low_a.py").read_text(encoding="utf-8"), "low_a.py"
    )
    low_b = prepare_source(
        (EXAMPLES / "low_b.py").read_text(encoding="utf-8"), "low_b.py"
    )
    assert len(high_a.annotated.nodes) == 34
    assert len(high_b.annotated.nodes) == 34
    assert len(low_a.annotated.nodes) == 44
    assert len(low_b.annotated.nodes) == 64
    high_distance = zhang_shasha_distance(high_a.annotated, high_b.annotated)
    low_distance = zhang_shasha_distance(low_a.annotated, low_b.annotated)
    assert high_distance == 2
    assert low_distance == 64
    assert tree_similarity(high_a, high_b, {}) == 1.0 - 2 / 68
    assert tree_similarity(low_a, low_b, {}) == 1.0 - 64 / 108


def test_kernel_is_one_for_identical_trees_symmetric_and_in_range() -> None:
    left = prepare_source("print(1)\n", "left.py")
    same = prepare_source("print(1)\n", "same.py")
    other = prepare_source("value = 1\nprint(value)\n", "other.py")
    assert tree_similarity(left, same, {}) == 1.0
    forward = tree_similarity(left, other, {})
    backward = tree_similarity(other, left, {})
    assert forward == backward
    assert 0.0 <= forward <= 1.0
    assert zhang_shasha_distance(left.annotated, left.annotated) == 0


def test_summary_uses_finite_ratio_problems_on_a_synthetic_fixture() -> None:
    problem_ids = ["p1", "p2", "p3"]
    human_rows = [
        _cache_row("p1", "human", -1, "human", 0.5, 4, 2),
        _cache_row("p2", "human", -1, "human", 0.25, 4, 2),
        _cache_row("p3", "human", -1, "human", 0.8, 4, 1),
    ]
    model_rows = [
        _cache_row("p1", "gemma-2-2b-it", 0, "base", 1.0, 4, 2),
        _cache_row("p2", "gemma-2-2b-it", 0, "base", 0.5, 4, 2),
        _cache_row("p3", "gemma-2-2b-it", 0, "base", math.nan, 4, 0),
    ]
    per_problem, summary = summarize_loaded_arms(
        problem_ids,
        human_rows,
        {"gemma-2-2b-it_n0_base": model_rows},
    )
    assert [row["problem_id"] for row in per_problem] == problem_ids
    assert has_finite_ratio(0.5, 1.0)
    assert not has_finite_ratio(0.8, math.nan)
    ratios = [2.0, 2.0]
    assert summary[0]["n_problems"] == 3
    assert summary[0]["n_finite"] == 2
    assert summary[0]["collision_ratio_median"] == float(np.quantile(ratios, 0.5))
    assert summary[0]["human_collision_median"] == float(np.quantile([0.5, 0.25], 0.5))
    assert summary[0]["model_collision_median"] == float(np.quantile([1.0, 0.5], 0.5))
    assert math.isnan(float(per_problem[2]["collision_ratio"]))
    assert list(summary[0]) == [
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
    ]


def test_same_problem_comparison_keeps_the_intersection() -> None:
    rows = []
    for problem_id, base_ratio, full_ratio in (
        ("p1", 2.0, 1.0),
        ("p2", 0.5, 1.5),
        ("p3", math.nan, 1.0),
    ):
        rows.append(_ratio_row(problem_id, 0, "base", base_ratio))
        rows.append(_ratio_row(problem_id, 8, "fully_trained", full_ratio))
    comparison = same_problem_comparison(rows)
    assert comparison[0]["model"] == "gemma-2-2b-it"
    assert comparison[0]["n_matched"] == 2
    assert comparison[0]["base_collision_ratio_median"] == float(np.median([2.0, 0.5]))
    assert comparison[0]["fully_trained_collision_ratio_median"] == float(
        np.median([1.0, 1.5])
    )


def test_archive_split_and_audit_on_the_fixture() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        archive_path = temporary_path / "Project_CodeNet.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for source_path in sorted((FIXTURE / "data").glob("p?????/Python/*")):
                archive.add(
                    source_path,
                    arcname=str(
                        Path("Project_CodeNet") / source_path.relative_to(FIXTURE)
                    ),
                )
        output = temporary_path / "processed"
        summary = build_from_archive(
            FIXTURE,
            archive_path,
            output,
            workers=1,
            batch_size=2,
            pending_batches=2,
        )
        assert summary["accepted_python_rows"] == 13
        assert summary["accepted_archive_sources_expected"] == 12
        assert summary["accepted_archive_sources_found"] == 12
        assert summary["usable_accepted_python_rows"] == 12
        assert summary["n_train_rows"] == 11
        assert summary["n_val_rows"] == 1
        assert summary["training_grid"] == [11]
        first_train = json.loads(
            (output / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert "source" in first_train
        assert "source_path" not in first_train
        report = audit_processed_split(output, ast_sample_per_split=2)
        assert report["status"] == "ok"


def test_prepare_lists_accepted_python_without_a_cutoff() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        member_list = temporary_path / "members.txt"
        summary_path = temporary_path / "summary.json"
        summary = prepare_accepted_python(FIXTURE, member_list, summary_path)
        members = member_list.read_text(encoding="utf-8").splitlines()
        assert summary["accepted_python_rows"] == 13
        assert len(members) == 13
        assert summary["problem_count_cutoff"] is None
        assert summary["per_problem_solution_cutoff"] is None
        assert summary["one_per_user_filter"] is False


def test_prepare_verifies_extracted_python_sources() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "Project_CodeNet"
        metadata = root / "metadata"
        data = root / "data" / "p00001" / "Python"
        metadata.mkdir(parents=True)
        data.mkdir(parents=True)
        fields = [
            "submission_id",
            "problem_id",
            "user_id",
            "language",
            "status",
            "filename_ext",
        ]
        with (metadata / "p00001.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "submission_id": "s1",
                    "problem_id": "p00001",
                    "user_id": "u1",
                    "language": "Python",
                    "status": "Accepted",
                    "filename_ext": "py",
                }
            )
            writer.writerow(
                {
                    "submission_id": "s2",
                    "problem_id": "p00001",
                    "user_id": "u2",
                    "language": "Python",
                    "status": "Wrong Answer",
                    "filename_ext": "py",
                }
            )
        (data / "s1.py").write_text("print(1)\n", encoding="utf-8")
        (data / "s2.py").write_text("print(2)\n", encoding="utf-8")
        summary = prepare_accepted_python(
            root,
            Path(temporary) / "members.txt",
            Path(temporary) / "summary.json",
            verify_extracted=True,
        )
        assert summary["accepted_python_rows"] == 1
        assert summary["extracted_source_files"] == 2
        assert (
            summary["downstream_training_filter"]
            == "language=Python and status=Accepted"
        )


def test_light_modules_do_not_import_transformers() -> None:
    script = (
        "import sys\n"
        "import mode_collapse.codenet.config\n"
        "import mode_collapse.codenet.canonical_ast\n"
        "import mode_collapse.codenet.zhang_shasha\n"
        "import mode_collapse.codenet.convergence\n"
        "assert 'transformers' not in sys.modules\n"
        "assert 'peft' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def _cache_row(
    problem_id: str,
    model: str,
    n: int,
    checkpoint: str,
    collision: float,
    n_candidates: int,
    n_pairs: int,
) -> dict[str, str]:
    text = "" if math.isnan(collision) else repr(collision)
    return {
        "problem_id": problem_id,
        "arm": f"{model}_n{n}_{checkpoint}",
        "model": model,
        "n": str(n),
        "checkpoint": checkpoint,
        "n_candidates": str(n_candidates),
        "n_parseable": str(n_pairs and n_candidates),
        "n_pairs_available": str(n_pairs),
        "n_pairs_scored": str(n_pairs),
        "zss_collision": text,
        "parse_errors_json": "{}",
    }


def _ratio_row(
    problem_id: str, n: int, checkpoint: str, ratio: float
) -> dict[str, object]:
    return {
        "problem_id": problem_id,
        "model": "gemma-2-2b-it",
        "n": n,
        "checkpoint": checkpoint,
        "collision_ratio": ratio,
    }
