"""Canonical Python AST features and strategy signatures for Project CodeNet.

The representation keeps syntax and standard-library or data-structure calls while
discarding comments, formatting, identifier spelling, and literal values. Method
names in ``PRESERVED_METHODS`` stay literal because they name the operation.
Every feature is an ordered, edge-typed rooted AST subtree of height 1, 2, or 3.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

SUBTREE_DEPTHS = (1, 2, 3)
LOOP_NODES = (ast.For, ast.AsyncFor, ast.While)
BUILTIN_NAMES = frozenset(dir(builtins)) | {"__name__", "__file__", "__package__"}

PRESERVED_METHODS = frozenset(
    {
        "add",
        "append",
        "appendleft",
        "clear",
        "copy",
        "count",
        "discard",
        "extend",
        "extendleft",
        "get",
        "heappop",
        "heappush",
        "index",
        "insert",
        "intersection",
        "items",
        "join",
        "keys",
        "lower",
        "popleft",
        "pop",
        "popitem",
        "push",
        "put",
        "qsize",
        "remove",
        "replace",
        "reverse",
        "setdefault",
        "sort",
        "split",
        "strip",
        "union",
        "update",
        "upper",
        "values",
    }
)


def _literal_category(value: object) -> str:
    if value is None:
        return "NONE"
    if value is Ellipsis:
        return "ELLIPSIS"
    if isinstance(value, bool):
        return "BOOL"
    if isinstance(value, int):
        return "INT"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, complex):
        return "COMPLEX"
    if isinstance(value, str):
        return "STRING"
    if isinstance(value, bytes):
        return "BYTES"
    return type(value).__name__.upper()


def _dotted_name(node: ast.AST) -> str | None:
    """Returns a simple dotted expression name, or None for a computed expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


@dataclass
class Scope:
    """Canonical names bound in one lexical scope.

    Attributes:
        scope_id: Position of the scope in opening order.
        imports: Local import name to the imported dotted name.
        functions: Function name to its ``FUNC_i`` placeholder.
        classes: Class name to its ``CLASS_i`` placeholder.
        variables: Variable name to its ``VAR_scope_i`` placeholder.
        attributes: Attribute name to its ``ATTR_scope_i`` placeholder.
    """

    scope_id: int
    imports: dict[str, str] = field(default_factory=dict)
    functions: dict[str, str] = field(default_factory=dict)
    classes: dict[str, str] = field(default_factory=dict)
    variables: dict[str, str] = field(default_factory=dict)
    attributes: dict[str, str] = field(default_factory=dict)


class Canonicalizer(ast.NodeTransformer):
    """Renames user-controlled identifiers and categorizes literal values deterministically."""

    def __init__(self, tree: ast.AST):
        self.scopes: list[Scope] = []
        self.next_scope = 0
        self.next_function = 0
        self.next_class = 0
        self._push_scope(getattr(tree, "body", []))

    def _push_scope(self, body: Iterable[ast.AST] = ()) -> None:
        """Open a scope and predeclare its direct definitions.

        Predeclaring functions, classes, and imports makes forward calls and
        mutual recursion invariant to source order.
        """
        self.scopes.append(Scope(scope_id=self.next_scope))
        self.next_scope += 1
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._bind_function(node.name)
            elif isinstance(node, ast.ClassDef):
                self._bind_class(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    self.scopes[-1].imports[local] = alias.name
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    local = alias.asname or alias.name
                    full = f"{node.module}.{alias.name}" if node.module else alias.name
                    self.scopes[-1].imports[local] = full

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _bind_function(self, name: str) -> str:
        functions = self.scopes[-1].functions
        if name not in functions:
            functions[name] = f"FUNC_{self.next_function}"
            self.next_function += 1
        return functions[name]

    def _bind_class(self, name: str) -> str:
        classes = self.scopes[-1].classes
        if name not in classes:
            classes[name] = f"CLASS_{self.next_class}"
            self.next_class += 1
        return classes[name]

    def _bind_variable(self, name: str) -> str:
        scope = self.scopes[-1]
        variables = scope.variables
        return variables.setdefault(name, f"VAR_{scope.scope_id}_{len(variables)}")

    def _lookup(self, name: str) -> str | None:
        for scope in reversed(self.scopes):
            if name in scope.imports:
                return f"LIB_{scope.imports[name]}"
            if name in scope.functions:
                return scope.functions[name]
            if name in scope.classes:
                return scope.classes[name]
            if name in scope.variables:
                return scope.variables[name]
        return None

    def _resolve_import(self, name: str) -> str | None:
        for scope in reversed(self.scopes):
            if name in scope.imports:
                return scope.imports[name]
        return None

    def _variable(self, name: str, bind: bool = False) -> str:
        if bind:
            return self._bind_variable(name)
        known = self._lookup(name)
        if known is not None:
            return known
        if name in BUILTIN_NAMES:
            return name
        return self._bind_variable(name)

    def _attribute(self, name: str) -> str:
        if name in PRESERVED_METHODS:
            return name
        scope = self.scopes[-1]
        attributes = scope.attributes
        return attributes.setdefault(name, f"ATTR_{scope.scope_id}_{len(attributes)}")

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Rename a variable load or bind it on store and delete."""
        bind = isinstance(node.ctx, (ast.Store, ast.Del))
        return ast.copy_location(
            ast.Name(id=self._variable(node.id, bind=bind), ctx=node.ctx), node
        )

    def visit_arg(self, node: ast.arg) -> ast.AST:
        """Bind a parameter and drop its type comment."""
        node.arg = self._bind_variable(node.arg)
        node.type_comment = None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Rename a function and canonicalize its body in a new scope."""
        node.name = self._bind_function(node.name)
        node.type_comment = None
        self._push_scope(node.body)
        try:
            return self.generic_visit(node)
        finally:
            self._pop_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Rename an async function and canonicalize its body in a new scope."""
        node.name = self._bind_function(node.name)
        node.type_comment = None
        self._push_scope(node.body)
        try:
            return self.generic_visit(node)
        finally:
            self._pop_scope()

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Rename a class and canonicalize its body in a new scope."""
        node.name = self._bind_class(node.name)
        self._push_scope(node.body)
        try:
            return self.generic_visit(node)
        finally:
            self._pop_scope()

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        """Canonicalize a lambda body in its own scope."""
        self._push_scope()
        try:
            return self.generic_visit(node)
        finally:
            self._pop_scope()

    def _visit_comprehension_scope(self, node: ast.AST) -> ast.AST:
        self._push_scope()
        try:
            return self.generic_visit(node)
        finally:
            self._pop_scope()

    visit_ListComp = _visit_comprehension_scope
    visit_SetComp = _visit_comprehension_scope
    visit_DictComp = _visit_comprehension_scope
    visit_GeneratorExp = _visit_comprehension_scope

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Keep library attributes and rename other attribute spellings."""
        original = _dotted_name(node)
        root = original.split(".", 1)[0] if original else None
        library_attribute = root is not None and self._resolve_import(root) is not None
        node.value = self.visit(node.value)
        if not library_attribute:
            node.attr = self._attribute(node.attr)
        return node

    def visit_alias(self, node: ast.alias) -> ast.AST:
        """Keep the imported object and drop the local alias spelling."""
        return ast.copy_location(ast.alias(name=node.name, asname=None), node)

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        """Replace a keyword name with a fixed label and visit its value."""
        if node.arg is not None:
            node.arg = "KW"
        node.value = self.visit(node.value)
        return node

    def visit_Global(self, node: ast.Global) -> ast.AST:
        """Rename names declared global."""
        node.names = [self._variable(name) for name in node.names]
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
        """Rename names declared nonlocal."""
        node.names = [self._variable(name) for name in node.names]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        """Bind an exception target and visit the handler body."""
        if node.name:
            node.name = self._bind_variable(node.name)
        return self.generic_visit(node)


def parse_and_canonicalize(source: str, filename: str = "<unknown>") -> ast.AST:
    """Parses Python source and returns its canonicalized AST."""
    tree = ast.parse(source, filename=filename, type_comments=False)
    tree = Canonicalizer(tree).visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def _node_label(node: ast.AST) -> str:
    label = type(node).__name__
    if isinstance(node, ast.Name):
        return f"{label}:{node.id}"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return f"{label}:{node.name}"
    if isinstance(node, ast.arg):
        return f"{label}:{node.arg}"
    if isinstance(node, ast.Attribute):
        return f"{label}:{node.attr}"
    if isinstance(node, ast.alias):
        return f"{label}:{node.name}"
    if isinstance(node, ast.keyword):
        return f"{label}:{node.arg or 'EXPAND'}"
    if isinstance(node, ast.Constant):
        return f"{label}:{_literal_category(node.value)}"
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return f"{label}:{','.join(node.names)}"
    if isinstance(node, ast.ExceptHandler) and node.name:
        return f"{label}:{node.name}"
    return label


def _ast_children(node: ast.AST) -> Iterator[tuple[str, ast.AST]]:
    """Yields ordered AST children with field-name edge types, excluding Load/Store context."""
    for field_name, value in ast.iter_fields(node):
        if field_name in {"ctx", "type_comment", "type_ignores"}:
            continue
        if isinstance(value, ast.AST):
            yield field_name, value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    yield f"{field_name}[]", item


def _subtree(node: ast.AST, depth: int) -> str:
    label = _node_label(node)
    if depth == 1:
        return label
    children = list(_ast_children(node))
    if not children:
        return label
    body = ",".join(f"{edge}:{_subtree(child, depth - 1)}" for edge, child in children)
    return f"{label}({body})"


def typed_subtree_counts(
    tree: ast.AST, depths: Iterable[int] = SUBTREE_DEPTHS
) -> Counter[str]:
    """Counts ordered, edge-typed rooted AST subtrees at each requested height."""
    depths = tuple(depths)
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("subtree depths must be positive")
    counts: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Load, ast.Store, ast.Del)):
            continue
        for depth in depths:
            counts[f"d{depth}:{_subtree(node, depth)}"] += 1
    return counts


def canonical_ast_listing(source: str, filename: str = "<unknown>") -> str:
    """Return the complete canonical AST that supplies the metric's subtree features.

    The listing uses the same node labels and ordered, field-typed child edges as
    ``typed_subtree_counts``. Context-only nodes and source-location attributes are absent because
    they are not part of the metric.
    """
    tree = parse_and_canonicalize(source, filename)
    lines: list[str] = []

    def visit(node: ast.AST, edge: str, depth: int) -> None:
        lines.append(f"{'  ' * depth}{edge}: {_node_label(node)}")
        for child_edge, child in _ast_children(node):
            visit(child, child_edge, depth + 1)

    visit(tree, "root", 0)
    return "\n".join(lines) + "\n"


def feature_norm(features: Counter[str]) -> float:
    """Return the Euclidean norm of a subtree-count vector."""
    return math.sqrt(math.fsum(value * value for value in features.values()))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    """Returns exact sparse cosine similarity, bounded to [0, 1] up to roundoff."""
    left_norm = feature_norm(left)
    right_norm = feature_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine similarity is undefined for an empty feature vector")
    if len(left) > len(right):
        left, right = right, left
    dot = math.fsum(value * right.get(key, 0) for key, value in left.items())
    return min(1.0, max(0.0, dot / (left_norm * right_norm)))


def normalized_features(features: Counter[str]) -> dict[str, float]:
    """Return the feature vector divided by its Euclidean norm."""
    norm = feature_norm(features)
    if norm == 0.0:
        raise ValueError("cannot normalize an empty feature vector")
    return {key: value / norm for key, value in features.items()}


def feature_fingerprint(features: Counter[str]) -> str:
    """Returns a stable digest of the canonical feature multiset for audit/deduplication."""
    digest = hashlib.sha256()
    for key, value in sorted(features.items()):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _call_name(node: ast.Call) -> str | None:
    return _dotted_name(node.func)


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_collection_allocation(node: ast.AST) -> bool:
    if isinstance(node, (ast.List, ast.ListComp, ast.Dict, ast.DictComp)):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _is_collection_allocation(node.left) or _is_collection_allocation(
            node.right
        )
    if isinstance(node, ast.Call):
        name = _dotted_name(node.func) or ""
        return name.rsplit(".", 1)[-1] in {"list", "dict", "defaultdict"}
    return False


def _max_loop_depth(tree: ast.AST) -> int:
    maximum = 0

    def walk(node: ast.AST, depth: int, root: bool = False) -> None:
        nonlocal maximum
        if not root and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ):
            depth = 0
        next_depth = depth + 1 if isinstance(node, LOOP_NODES) else depth
        maximum = max(maximum, next_depth)
        for child in ast.iter_child_nodes(node):
            walk(child, next_depth)

    walk(tree, 0, root=True)
    return maximum


def _scoped_calls(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[str, bool]]:
    """Returns calls in one function body, excluding nested lexical scopes."""
    found: list[tuple[str, bool]] = []

    def walk(node: ast.AST, loop_depth: int, root: bool = False) -> None:
        if not root and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            return
        next_depth = loop_depth + 1 if isinstance(node, LOOP_NODES) else loop_depth
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name:
                found.append((name, next_depth > 0))
        for child in ast.iter_child_nodes(node):
            walk(child, next_depth)

    walk(function, 0, root=True)
    return found


def _cyclic_functions(graph: dict[str, set[str]]) -> set[str]:
    """Returns all function names that participate in a directed call-graph cycle."""
    cyclic: set[str] = set()

    def reaches(origin: str, current: str, visited: set[str]) -> bool:
        for target in graph.get(current, set()):
            if target == origin:
                return True
            if target not in visited and reaches(origin, target, visited | {target}):
                return True
        return False

    for name in graph:
        if reaches(name, name, {name}):
            cyclic.add(name)
    return cyclic


@dataclass(frozen=True, order=True)
class StrategySignature:
    """Coarse deterministic algorithm/data-structure signature."""

    recursion: bool
    loop_nesting_depth: int
    sorting: bool
    hashing: bool
    heap: bool
    graph_traversal: bool
    dynamic_programming_table: bool

    def as_tuple(self) -> tuple[bool, int, bool, bool, bool, bool, bool]:
        """Return the signature fields in a fixed order."""
        return (
            self.recursion,
            self.loop_nesting_depth,
            self.sorting,
            self.hashing,
            self.heap,
            self.graph_traversal,
            self.dynamic_programming_table,
        )

    def compact(self) -> str:
        """Return the signature as a pipe-separated bit string."""
        return "|".join(str(int(value)) for value in self.as_tuple())


def strategy_signature(tree: ast.AST) -> StrategySignature:
    """Detects the documented coarse strategy components from a canonical AST.

    Graph traversal is a deliberately conservative structural proxy: either a deque/queue
    worklist consumed in a loop, or a recursive call made from within a loop.  A DP table is a
    named list/dict-like allocation that is subsequently written through a subscript.
    """
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    call_names = {_call_name(node) or "" for node in calls}

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls_by_function = {
        name: _scoped_calls(function) for name, function in functions.items()
    }
    call_graph = {
        name: {target for target, _ in calls if target in functions}
        for name, calls in calls_by_function.items()
    }
    recursive_functions = _cyclic_functions(call_graph)
    recursive_inside_loop = any(
        caller in recursive_functions and target in recursive_functions and inside_loop
        for caller, calls in calls_by_function.items()
        for target, inside_loop in calls
    )

    last_call_names = {name.rsplit(".", 1)[-1] for name in call_names}
    sorting = bool(last_call_names & {"sort", "sorted"})
    hashing = any(
        isinstance(node, (ast.Dict, ast.Set, ast.DictComp, ast.SetComp))
        for node in ast.walk(tree)
    )
    hashing |= bool(last_call_names & {"dict", "set", "defaultdict", "Counter"})
    heap = any(
        "heapq" in name or name.rsplit(".", 1)[-1].startswith("heap")
        for name in call_names
    )

    queue_created = bool(
        last_call_names & {"deque", "Queue", "LifoQueue", "PriorityQueue"}
    )
    queue_consumed = bool(last_call_names & {"popleft", "get"})
    has_loop = any(isinstance(node, LOOP_NODES) for node in ast.walk(tree))
    graph_traversal = (
        queue_created and queue_consumed and has_loop
    ) or recursive_inside_loop

    allocated: set[str] = set()
    subscript_writes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not None and _is_collection_allocation(value):
                allocated.update(
                    name for name in (_root_name(target) for target in targets) if name
                )
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            name = _root_name(node)
            if name:
                subscript_writes.add(name)

    return StrategySignature(
        recursion=bool(recursive_functions),
        loop_nesting_depth=_max_loop_depth(tree),
        sorting=sorting,
        hashing=hashing,
        heap=heap,
        graph_traversal=graph_traversal,
        dynamic_programming_table=bool(allocated & subscript_writes),
    )


def analyze_source(
    source: str, filename: str = "<unknown>"
) -> tuple[Counter[str], StrategySignature]:
    """Returns the primary AST feature vector and secondary strategy signature."""
    tree = parse_and_canonicalize(source, filename)
    return typed_subtree_counts(tree), strategy_signature(tree)


PAPER_EXAMPLE_SELECTIONS = {
    "high": {
        "problem_id": "p03125",
        "left_submission_id": "s613176273",
        "right_submission_id": "s053364634",
        "candidate_group": "high",
    },
    "low": {
        "problem_id": "p03135",
        "left_submission_id": "s227737564",
        "right_submission_id": "s089174134",
        "candidate_group": "low",
    },
}


def selected_pair(
    candidates: dict[str, object], specification: dict[str, str]
) -> dict[str, object]:
    """Return the candidate pair named by a paper-example specification."""
    problem_groups = candidates["per_problem_candidates"]
    problem = problem_groups[specification["problem_id"]]
    pairs = problem[specification["candidate_group"]]
    for pair in pairs:
        if (
            pair["left"]["submission_id"] == specification["left_submission_id"]
            and pair["right"]["submission_id"] == specification["right_submission_id"]
        ):
            return pair
    raise ValueError(f"selected pair not present: {specification}")


def export_paper_ast_examples(candidates_path: Path, output_dir: Path) -> None:
    """Write the two appendix pairs and their complete canonical AST listings."""
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "generator": "export_ast_examples.py",
        "ast_export": "complete canonical metric tree; no pruning",
        "examples": {},
    }
    examples = manifest["examples"]
    for name, specification in PAPER_EXAMPLE_SELECTIONS.items():
        pair = selected_pair(candidates, specification)
        examples[name] = {
            "problem_id": pair["problem_id"],
            "similarity": pair["similarity"],
            "same_strategy_signature": pair["same_strategy_signature"],
            "left_submission_id": pair["left"]["submission_id"],
            "right_submission_id": pair["right"]["submission_id"],
        }
        for side, suffix in (("left", "a"), ("right", "b")):
            source = pair[side]["source"] + "\n"
            source_path = output_dir / f"{name}_{suffix}.py"
            ast_path = output_dir / f"{name}_{suffix}.ast"
            source_path.write_text(source, encoding="utf-8")
            ast_path.write_text(
                canonical_ast_listing(source, str(source_path)), encoding="utf-8"
            )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
