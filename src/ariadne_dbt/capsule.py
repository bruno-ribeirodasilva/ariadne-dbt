"""Capsule builder: search → DAG traversal → skeletonization → token budgeting."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .config import CapsuleConfig, IntentDepth
from .graph import GraphOps
from .models import (
    ContextCapsule,
    FullModelContext,
    MinimalModelContext,
    SkeletonColumn,
    SkeletonModelContext,
)
from .patterns import PatternExtractor
from .search import HybridSearch


# ─── Intent detection ─────────────────────────────────────────────────────────

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "debug": ["debug", "fix", "error", "fail", "broken", "wrong", "incorrect", "issue", "bug", "problem", "test failing"],
    "add_feature": ["add", "create", "new", "build", "implement", "feature", "metric", "measure", "calculate"],
    "refactor": ["refactor", "restructure", "reorganize", "rename", "move", "split", "merge", "optimize", "performance"],
    "test": ["test", "coverage", "validate", "assert", "check", "verify"],
    "document": ["document", "describe", "description", "docs", "comment", "explain"],
    "explore": ["explore", "understand", "find", "search", "show", "list", "what", "how", "which"],
}


def detect_intent(task: str) -> str:
    task_lower = task.lower()
    scores: dict[str, int] = {}
    for intent, keywords in _INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in task_lower)
        if score > 0:
            scores[intent] = score
    if not scores:
        return "explore"
    return max(scores, key=lambda k: scores[k])


# ─── Token estimation ─────────────────────────────────────────────────────────

_CHARS_PER_TOKEN = 4  # rough approximation


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _estimate_dict_tokens(d: Any) -> int:
    return _estimate_tokens(json.dumps(d, default=str))


# ─── Skeletonization ──────────────────────────────────────────────────────────

def _build_full_model(model_row: dict[str, Any], columns: list[dict[str, Any]], tests: list[dict[str, Any]]) -> FullModelContext:
    col_objects = []
    test_by_col: dict[str, list[str]] = {}
    for t in tests:
        col = t.get("column_name", "")
        if col:
            test_by_col.setdefault(col, []).append(t["test_type"])

    for c in columns:
        col_objects.append(SkeletonColumn(
            name=c["name"],
            data_type=c.get("data_type", ""),
            description=c.get("description", ""),
            tests=test_by_col.get(c["name"], []),
        ))

    depends_on: list[str] = []
    raw = model_row.get("depends_on_nodes", "[]")
    try:
        deps = json.loads(raw)
        depends_on = [d.split(".")[-1] for d in deps if d.startswith("model.")]
    except (json.JSONDecodeError, TypeError):
        pass

    return FullModelContext(
        unique_id=model_row["unique_id"],
        name=model_row["name"],
        layer=model_row.get("layer", "other"),
        materialization=model_row.get("materialization", "view"),
        file_path=model_row.get("file_path", ""),
        compiled_sql=model_row.get("compiled_code") or model_row.get("raw_code", ""),
        description=model_row.get("description", ""),
        columns=col_objects,
        tags=json.loads(model_row.get("tags", "[]") or "[]"),
        depends_on=depends_on,
    )


def _build_skeleton_model(model_row: dict[str, Any], columns: list[dict[str, Any]]) -> SkeletonModelContext:
    return SkeletonModelContext(
        unique_id=model_row["unique_id"],
        name=model_row["name"],
        layer=model_row.get("layer", "other"),
        materialization=model_row.get("materialization", "view"),
        columns=[{"name": c["name"], "type": c.get("data_type", "")} for c in columns],
    )


def _build_minimal_model(model_row: dict[str, Any], columns: list[dict[str, Any]]) -> MinimalModelContext:
    key_cols = [c["name"] for c in columns if c.get("is_primary_key") or c.get("is_foreign_key")]
    return MinimalModelContext(
        unique_id=model_row["unique_id"],
        name=model_row["name"],
        layer=model_row.get("layer", "other"),
        column_count=len(columns),
        key_columns=key_cols[:5],
    )


# ─── Capsule builder ──────────────────────────────────────────────────────────

class CapsuleBuilder:
    """Builds a ContextCapsule for a given task query."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        config: CapsuleConfig | None = None,
    ) -> None:
        self._conn = conn
        self._config = config or CapsuleConfig()
        self._search = HybridSearch(conn)
        self._graph = GraphOps(conn)
        self._patterns = PatternExtractor(conn)

    def build(
        self,
        task: str,
        focus_model: str | None = None,
        token_budget: int | None = None,
    ) -> ContextCapsule:
        budget = token_budget or self._config.default_token_budget
        intent = detect_intent(task)
        depths: IntentDepth = self._config.intent_depths.get(intent, IntentDepth())

        # ── Step 1: Select pivot models ───────────────────────────────────────
        pivot_ids = self._select_pivots(task, intent, focus_model)

        # ── Step 2: DAG traversal ─────────────────────────────────────────────
        upstream_ids: dict[str, int] = {}
        downstream_ids: dict[str, int] = {}
        for pid in pivot_ids:
            for uid, dist in self._graph.upstream(pid, depth=depths.upstream):
                if uid.startswith("model.") and uid not in pivot_ids:
                    upstream_ids[uid] = min(upstream_ids.get(uid, 999), dist)
            for uid, dist in self._graph.downstream(pid, depth=depths.downstream):
                if uid.startswith("model.") and uid not in pivot_ids:
                    downstream_ids[uid] = min(downstream_ids.get(uid, 999), dist)

        # ── Step 3: Collect related context ───────────────────────────────────
        tests_map: dict[str, list[dict[str, Any]]] = {}
        macros_map: dict[str, list[dict[str, Any]]] = {}
        sources_list: list[dict[str, Any]] = []

        for pid in pivot_ids:
            tests_map[pid] = self._search.get_tests_for_model(pid)
            macros_map[pid] = self._search.get_macros_for_model(pid)
            sources_list.extend(self._search.get_sources_for_model(pid))

        # Similar models (awareness only, not in pivot/upstream/downstream)
        all_known = set(pivot_ids) | set(upstream_ids) | set(downstream_ids)
        similar = self._search.search(task, intent=intent, limit=5, exclude_ids=all_known)
        similar_names = [s.name for s in similar]

        # Patterns
        try:
            patterns = self._patterns.get_patterns()
            patterns_dict = {
                "naming": patterns.naming.model_dump(),
                "common_materializations": patterns.common_materializations,
            }
        except Exception:
            patterns_dict = {}

        # ── Step 4: Assemble with token budgeting ─────────────────────────────
        capsule = self._assemble(
            task=task,
            intent=intent,
            pivot_ids=pivot_ids,
            upstream_ids=upstream_ids,
            downstream_ids=downstream_ids,
            tests_map=tests_map,
            macros_map=macros_map,
            sources=sources_list,
            similar_models=similar_names,
            patterns=patterns_dict,
            budget=budget,
        )
        return capsule

    # ── Pivot selection ───────────────────────────────────────────────────────

    def _select_pivots(self, task: str, intent: str, focus_model: str | None) -> list[str]:
        pivot_ids: list[str] = []

        if focus_model:
            # Try by name first, then by unique_id
            row = self._search.get_model_by_name(focus_model) or self._search.get_model_by_id(focus_model)
            if row:
                pivot_ids.append(row["unique_id"])

        if len(pivot_ids) < self._config.max_pivots:
            exclude = set(pivot_ids)
            results = self._search.search(
                task, intent=intent,
                limit=self._config.max_pivots - len(pivot_ids) + 2,
                exclude_ids=exclude,
            )
            for r in results:
                if len(pivot_ids) >= self._config.max_pivots:
                    break
                pivot_ids.append(r.unique_id)

        return pivot_ids

    # ── Assembly + token budgeting ────────────────────────────────────────────

    def _assemble(
        self,
        task: str,
        intent: str,
        pivot_ids: list[str],
        upstream_ids: dict[str, int],
        downstream_ids: dict[str, int],
        tests_map: dict[str, list],
        macros_map: dict[str, list],
        sources: list[dict[str, Any]],
        similar_models: list[str],
        patterns: dict[str, Any],
        budget: int,
    ) -> ContextCapsule:
        # Budget allocation
        alloc = {
            "pivot": int(budget * 0.45),
            "upstream": int(budget * 0.20),
            "downstream": int(budget * 0.10),
            "tests_macros": int(budget * 0.10),
            "patterns": int(budget * 0.10),
            "session": int(budget * 0.05),
        }

        # Build pivot models
        pivot_models = []
        pivot_tokens = 0
        for pid in pivot_ids:
            row = self._get_model_row(pid)
            if not row:
                continue
            cols = self._search.get_columns(pid)
            tests = tests_map.get(pid, [])
            full = _build_full_model(row, cols, tests)
            cost = _estimate_dict_tokens(full.model_dump())
            if pivot_tokens + cost <= alloc["pivot"]:
                pivot_models.append(full)
                pivot_tokens += cost

        # Build upstream (skeleton level)
        upstream_models = []
        upstream_tokens = 0
        for uid, dist in sorted(upstream_ids.items(), key=lambda x: x[1]):
            row = self._get_model_row(uid)
            if not row:
                continue
            cols = self._search.get_columns(uid)
            skel = _build_skeleton_model(row, cols)
            cost = _estimate_dict_tokens(skel.model_dump())
            if upstream_tokens + cost <= alloc["upstream"]:
                upstream_models.append(skel)
                upstream_tokens += cost
            else:
                break

        # Build downstream (minimal level)
        downstream_models = []
        downstream_tokens = 0
        for uid, dist in sorted(downstream_ids.items(), key=lambda x: x[1]):
            row = self._get_model_row(uid)
            if not row:
                continue
            cols = self._search.get_columns(uid)
            mini = _build_minimal_model(row, cols)
            cost = _estimate_dict_tokens(mini.model_dump())
            if downstream_tokens + cost <= alloc["downstream"]:
                downstream_models.append(mini)
                downstream_tokens += cost
            else:
                break

        # Tests and macros
        relevant_tests: list[dict[str, Any]] = []
        relevant_macros: list[dict[str, Any]] = []
        tm_tokens = 0
        for pid, tests in tests_map.items():
            for t in tests:
                cost = _estimate_dict_tokens(t)
                if tm_tokens + cost <= alloc["tests_macros"] // 2:
                    relevant_tests.append(t)
                    tm_tokens += cost
        for pid, macros in macros_map.items():
            for m in macros:
                cost = _estimate_dict_tokens(m)
                if tm_tokens + cost <= alloc["tests_macros"]:
                    relevant_macros.append(m)
                    tm_tokens += cost

        # Deduplicate sources
        seen_sources: set[str] = set()
        unique_sources = []
        for s in sources:
            sid = s.get("unique_id", "")
            if sid not in seen_sources:
                seen_sources.add(sid)
                unique_sources.append(s)

        total_tokens = (
            pivot_tokens + upstream_tokens + downstream_tokens + tm_tokens
            + _estimate_dict_tokens(patterns)
        )

        return ContextCapsule(
            task=task,
            intent=intent,
            pivot_models=pivot_models,
            upstream_models=upstream_models,
            downstream_models=downstream_models,
            relevant_tests=relevant_tests,
            relevant_macros=relevant_macros,
            relevant_sources=unique_sources,
            project_patterns=patterns,
            similar_models=similar_models,
            session_context={},
            token_estimate=total_tokens,
            token_budget=budget,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_model_row(self, unique_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM models WHERE unique_id = ?", (unique_id,)
        ).fetchone()
        return dict(row) if row else None
