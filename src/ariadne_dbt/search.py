"""Hybrid search: SQLite FTS5 BM25 + graph centrality re-ranking."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from .models import SearchResult


# Intent → layer affinity weights
INTENT_LAYER_WEIGHTS: dict[str, dict[str, float]] = {
    "debug":       {"staging": 0.10, "intermediate": 0.05, "marts": 0.0,  "other": 0.0},
    "add_feature": {"staging": 0.0,  "intermediate": 0.05, "marts": 0.10, "other": 0.0},
    "refactor":    {"staging": 0.0,  "intermediate": 0.10, "marts": 0.05, "other": 0.0},
    "test":        {"staging": 0.05, "intermediate": 0.05, "marts": 0.05, "other": 0.0},
    "document":    {"staging": 0.0,  "intermediate": 0.0,  "marts": 0.0,  "other": 0.0},
    "explore":     {"staging": 0.0,  "intermediate": 0.0,  "marts": 0.0,  "other": 0.0},
}


def _tokenize_query(query: str) -> str:
    """Convert a natural-language query to an FTS5 MATCH expression.

    Handles multi-word queries with OR and basic stemming via porter tokenizer.
    """
    # Strip non-alphanumeric (keep spaces/underscores)
    tokens = re.sub(r"[^\w\s]", " ", query).split()
    # Filter stopwords
    stopwords = {"a", "an", "the", "to", "for", "in", "of", "on", "at", "with", "and", "or", "is", "it"}
    tokens = [t for t in tokens if t.lower() not in stopwords and len(t) > 1]
    if not tokens:
        return query
    return " OR ".join(tokens)


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    mn, mx = min(values), max(values)
    r = mx - mn
    if r == 0:
        return [1.0] * len(values)
    return [(v - mn) / r for v in values]


class HybridSearch:
    """Two-phase search: broad FTS5 recall → precise re-ranking with graph signals."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def search(
        self,
        query: str,
        intent: str = "explore",
        limit: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        """Return top ``limit`` models ranked by BM25 + centrality + layer affinity."""
        fts_query = _tokenize_query(query)
        candidates = self._fts_phase(fts_query, limit=limit * 4)

        if not candidates:
            # Fallback: name LIKE search
            candidates = self._fallback_search(query, limit=limit * 4)

        if exclude_ids:
            candidates = [c for c in candidates if c["unique_id"] not in exclude_ids]

        # Re-rank
        layer_weights = INTENT_LAYER_WEIGHTS.get(intent, {})
        bm25_scores = [c["bm25_score"] for c in candidates]
        norm_bm25 = _normalize(bm25_scores)
        query_lower = query.lower()

        results = []
        for i, c in enumerate(candidates):
            centrality = c.get("centrality", 0.0) or 0.0
            layer_boost = layer_weights.get(c.get("layer", "other"), 0.0)
            name_bonus = 0.15 if query_lower in c["name"].lower() else 0.0

            score = (
                norm_bm25[i] * 0.55
                + centrality * 0.20
                + layer_boost * 0.10
                + name_bonus * 0.15
            )

            results.append(SearchResult(
                unique_id=c["unique_id"],
                name=c["name"],
                layer=c.get("layer", "other"),
                description=c.get("description", ""),
                bm25_score=bm25_scores[i],
                centrality=centrality,
                layer_boost=layer_boost,
                name_bonus=name_bonus,
                score=score,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _fts_phase(self, fts_query: str, limit: int) -> list[dict[str, Any]]:
        try:
            rows = self._conn.execute(
                """
                SELECT
                    s.unique_id,
                    m.name,
                    m.layer,
                    m.description,
                    m.centrality,
                    -- BM25: lower is better (negative), so negate to get higher-is-better
                    -bm25(search_index, 5, 3, 2, 1, 1) AS bm25_score
                FROM search_index s
                JOIN models m ON m.unique_id = s.unique_id
                WHERE search_index MATCH ?
                ORDER BY bm25_score DESC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def _fallback_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Simple LIKE search when FTS returns nothing."""
        pattern = f"%{query}%"
        rows = self._conn.execute(
            """
            SELECT unique_id, name, layer, description, centrality,
                   0.5 AS bm25_score
            FROM models
            WHERE lower(name) LIKE lower(?) OR lower(description) LIKE lower(?)
            ORDER BY centrality DESC
            LIMIT ?
            """,
            (pattern, pattern, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Model lookup ──────────────────────────────────────────────────────────

    def get_model_by_id(self, unique_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT m.*, GROUP_CONCAT(c.name, ',') as col_names
            FROM models m
            LEFT JOIN columns c ON c.model_id = m.unique_id
            WHERE m.unique_id = ?
            GROUP BY m.unique_id
            """,
            (unique_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_model_by_name(self, name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM models WHERE lower(name) = lower(?)", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_columns(self, model_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT c.name, c.data_type, c.description, c.is_primary_key, c.is_foreign_key,
                   GROUP_CONCAT(t.test_type, ',') as test_types
            FROM columns c
            LEFT JOIN tests t ON t.model_id = c.model_id AND t.column_name = c.name
            WHERE c.model_id = ?
            GROUP BY c.name
            """,
            (model_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_tests_for_model(self, model_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT unique_id, name, test_type, column_name, severity, last_status
            FROM tests WHERE model_id = ?
            """,
            (model_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_macros_for_model(self, model_id: str) -> list[dict[str, Any]]:
        """Return macros used by a model (via SQL reference matching)."""
        row = self._conn.execute(
            "SELECT raw_code, compiled_code FROM models WHERE unique_id = ?", (model_id,)
        ).fetchone()
        if not row:
            return []
        sql = row["compiled_code"] or row["raw_code"] or ""
        macro_rows = self._conn.execute(
            "SELECT unique_id, name, package_name, description FROM macros"
        ).fetchall()
        used = []
        for m in macro_rows:
            if m["name"] in sql:
                used.append(dict(m))
        return used

    def get_sources_for_model(self, model_id: str) -> list[dict[str, Any]]:
        """Return sources that feed into a model (direct upstream sources)."""
        rows = self._conn.execute(
            """
            SELECT s.unique_id, s.name, s.source_name, s.schema_name, s.description
            FROM edges e
            JOIN sources s ON s.unique_id = e.parent_id
            WHERE e.child_id = ?
            """,
            (model_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_test_coverage(self, model_id: str) -> dict[str, Any]:
        """Return test coverage summary for a model."""
        columns = self.get_columns(model_id)
        tests = self.get_tests_for_model(model_id)

        tested_cols = {t["column_name"] for t in tests if t["column_name"]}
        total_cols = len(columns)
        tested_count = sum(1 for c in columns if c["name"] in tested_cols)
        untested = [c["name"] for c in columns if c["name"] not in tested_cols]

        coverage_pct = int(tested_count / total_cols * 100) if total_cols > 0 else 0

        # Suggest missing tests
        suggestions = []
        col_names_lower = {c["name"].lower() for c in columns}
        tested_types = {t["test_type"] for t in tests}
        if "not_null" not in tested_types:
            suggestions.append("Add not_null tests to key columns")
        if "unique" not in tested_types:
            pk_candidates = [c["name"] for c in columns if "id" in c["name"].lower() or c["is_primary_key"]]
            if pk_candidates:
                suggestions.append(f"Add unique test to: {', '.join(pk_candidates[:3])}")
        if untested and total_cols > 5:
            suggestions.append(f"Untested columns ({len(untested)}): {', '.join(untested[:5])}")

        return {
            "model_id": model_id,
            "total_columns": total_cols,
            "tested_columns": tested_count,
            "coverage_pct": coverage_pct,
            "untested_columns": untested,
            "suggestions": suggestions,
            "tests": tests,
        }
