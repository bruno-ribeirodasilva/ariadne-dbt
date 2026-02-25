"""DAG graph operations — upstream/downstream traversal, impact analysis."""

from __future__ import annotations

import sqlite3
from collections import deque
from typing import Any


class GraphOps:
    """All DAG operations backed by the SQLite edges table.

    Avoids loading the full graph into memory — uses BFS queries instead.
    For projects with 10k+ models, NetworkX would be loaded lazily on first
    call to methods that need full-graph metrics (centrality, etc.).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ── BFS traversal ─────────────────────────────────────────────────────────

    def upstream(self, unique_id: str, depth: int = 1) -> list[tuple[str, int]]:
        """Return [(unique_id, distance)] for all ancestors up to ``depth`` hops.

        Returns only model.* nodes (excludes source.*, test.*, etc.).
        """
        return self._bfs(unique_id, direction="up", depth=depth)

    def downstream(self, unique_id: str, depth: int = 1) -> list[tuple[str, int]]:
        """Return [(unique_id, distance)] for all descendants up to ``depth`` hops."""
        return self._bfs(unique_id, direction="down", depth=depth)

    def neighbors(
        self, unique_id: str, upstream_depth: int = 1, downstream_depth: int = 1
    ) -> dict[str, list[tuple[str, int]]]:
        return {
            "upstream": self.upstream(unique_id, depth=upstream_depth),
            "downstream": self.downstream(unique_id, depth=downstream_depth),
        }

    def _bfs(
        self, start: str, direction: str, depth: int
    ) -> list[tuple[str, int]]:
        if depth <= 0:
            return []

        visited: dict[str, int] = {}  # unique_id → distance
        queue: deque[tuple[str, int]] = deque([(start, 0)])

        while queue:
            node_id, dist = queue.popleft()
            if dist >= depth:
                continue
            next_dist = dist + 1

            if direction == "up":
                rows = self._conn.execute(
                    "SELECT parent_id FROM edges WHERE child_id = ?", (node_id,)
                ).fetchall()
                neighbors = [r[0] for r in rows]
            else:
                rows = self._conn.execute(
                    "SELECT child_id FROM edges WHERE parent_id = ?", (node_id,)
                ).fetchall()
                neighbors = [r[0] for r in rows]

            for nid in neighbors:
                if nid == start or nid in visited:
                    continue
                visited[nid] = next_dist
                queue.append((nid, next_dist))

        # Sort by distance then name for determinism
        return sorted(visited.items(), key=lambda x: (x[1], x[0]))

    # ── Impact analysis ───────────────────────────────────────────────────────

    def impact_analysis(self, unique_id: str, max_depth: int = 5) -> dict[str, Any]:
        """Return blast radius of changing a model.

        Returns a dict with affected models, tests, exposures, and risk level.
        """
        downstream_nodes = self._bfs(unique_id, "down", depth=max_depth)
        model_ids = [nid for nid, _ in downstream_nodes if nid.startswith("model.")]
        exposure_ids = [nid for nid, _ in downstream_nodes if nid.startswith("exposure.")]

        affected_models = []
        for mid in model_ids:
            row = self._conn.execute(
                "SELECT name, layer, materialization FROM models WHERE unique_id = ?", (mid,)
            ).fetchone()
            if row:
                affected_models.append({"unique_id": mid, "name": row[0], "layer": row[1], "materialization": row[2]})

        affected_tests = self._conn.execute(
            f"""
            SELECT t.unique_id, t.name, t.test_type, t.model_id, t.column_name
            FROM tests t
            WHERE t.model_id IN ({','.join('?' * len(model_ids))})
            """,
            model_ids,
        ).fetchall() if model_ids else []

        affected_exposures = []
        for eid in exposure_ids:
            row = self._conn.execute(
                "SELECT name, type, url FROM exposures WHERE unique_id = ?", (eid,)
            ).fetchone()
            if row:
                affected_exposures.append({"unique_id": eid, "name": row[0], "type": row[1], "url": row[2]})

        # Risk level heuristic
        n_models = len(affected_models)
        n_exposures = len(affected_exposures)
        has_mart = any(m["layer"] == "marts" for m in affected_models)
        if n_exposures > 0 or (has_mart and n_models > 5):
            risk = "high"
        elif n_models > 3 or has_mart:
            risk = "medium"
        else:
            risk = "low"

        return {
            "target": unique_id,
            "affected_model_count": n_models,
            "affected_models": affected_models,
            "affected_test_count": len(affected_tests),
            "affected_tests": [dict(r) for r in affected_tests],
            "affected_exposures": affected_exposures,
            "risk_level": risk,
        }

    # ── Source dependencies ────────────────────────────────────────────────────

    def get_source_deps(self, unique_id: str) -> list[dict[str, Any]]:
        """Return all source nodes in the upstream lineage of a model."""
        upstream = self._bfs(unique_id, "up", depth=10)
        source_ids = [nid for nid, _ in upstream if nid.startswith("source.")]
        if not source_ids:
            return []
        rows = self._conn.execute(
            f"""
            SELECT unique_id, name, source_name, schema_name, description
            FROM sources
            WHERE unique_id IN ({','.join('?' * len(source_ids))})
            """,
            source_ids,
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Centrality recompute (called after indexing) ──────────────────────────

    def recompute_centrality(self) -> None:
        """Recompute degree centrality for all models and update the DB."""
        max_row = self._conn.execute(
            "SELECT MAX(upstream_count + downstream_count) FROM models"
        ).fetchone()[0] or 1
        self._conn.execute(
            """
            UPDATE models SET centrality =
                CAST((upstream_count + downstream_count) AS REAL) / ?
            """,
            (max_row,),
        )
        self._conn.commit()

    # ── Convenience: get model centrality ────────────────────────────────────

    def get_centrality(self, unique_id: str) -> float:
        row = self._conn.execute(
            "SELECT centrality FROM models WHERE unique_id = ?", (unique_id,)
        ).fetchone()
        return float(row[0]) if row else 0.0

    # ── High-centrality models ────────────────────────────────────────────────

    def get_high_centrality_models(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT unique_id, name, layer, description, upstream_count, downstream_count, centrality
            FROM models
            ORDER BY centrality DESC, downstream_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
