"""FastMCP server with tools for Ariadne."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from .capsule import CapsuleBuilder, detect_intent
from .config import EngineConfig, load_config
from .graph import GraphOps
from .indexer import Indexer
from .patterns import PatternExtractor
from .search import HybridSearch
from .usage import UsageLogger

# Tracks the most-recent capsule log_id per db_path for rate_capsule
_last_capsule_log_id: dict[Path, int] = {}


# ── Server factory ────────────────────────────────────────────────────────────

def create_server(config: EngineConfig | None = None) -> FastMCP:
    cfg = config or load_config()
    db_path = cfg.absolute_index_path

    mcp = FastMCP(
        name="ariadne",
        instructions=(
            "Intelligent context server for dbt projects. "
            "Start with get_context_capsule — pass entry_models or entry_paths if you "
            "already know which models are involved (e.g., from a PR diff or file path). "
            "Use find_models_by_path or find_models_by_column to discover models when "
            "the task is vague. Use get_impact_analysis before refactoring. "
            "Check the 'confidence' field — if low, refine with a more specific query. "
            "Call refresh_index after running dbt compile to keep the index current. "
            "Call rate_capsule after completing a task to record whether the context was useful."
        ),
    )

    # ── Tool: get_context_capsule ─────────────────────────────────────────────

    @mcp.tool()
    def get_context_capsule(
        task: str,
        focus_model: str | None = None,
        entry_models: list[str] | None = None,
        entry_paths: list[str] | None = None,
        token_budget: int = 10000,
    ) -> dict[str, Any]:
        """THE primary tool. Call this first for any dbt task.

        Returns a pre-filtered, token-budgeted context package containing:
        - Pivot models (full SQL + all columns + tests)
        - Upstream models (column schemas)
        - Downstream models (names + column counts)
        - Relevant tests, macros, sources
        - Project patterns and naming conventions
        - Similar models for awareness

        Args:
            task: Natural language description of what you want to do.
                  Examples: "add monthly revenue metric", "debug failing test on fct_orders",
                  "refactor stg_stripe__charges", "document the fct_revenue model"
            focus_model: Optional model name to anchor the search (e.g., "fct_orders")
            entry_models: Optional list of model names to include as pivots
                          (e.g., discovered from a PR diff or Jira ticket).
                          These become guaranteed pivots alongside focus_model.
            entry_paths: Optional list of file paths to resolve to models
                         (e.g., ["models/marts/finance/fct_revenue.sql"]).
                         Resolved to model unique_ids by matching file_path or basename.
            token_budget: Maximum tokens for the response (default: 10000)

        Returns:
            Structured context capsule with pivot_models, upstream_models,
            downstream_models, relevant_tests, relevant_macros, relevant_sources,
            project_patterns, similar_models, confidence, suggested_refinements,
            token_estimate
        """
        conn = _get_conn(db_path)
        builder = CapsuleBuilder(conn, cfg.capsule)
        t0 = time.perf_counter()
        capsule = builder.build(
            task=task,
            focus_model=focus_model,
            entry_models=entry_models,
            entry_paths=entry_paths,
            token_budget=token_budget,
        )
        duration_ms = round((time.perf_counter() - t0) * 1000)
        log_id = UsageLogger(conn).log(
            "get_context_capsule",
            task_text=task,
            intent=capsule.intent,
            focus_model=focus_model,
            pivot_count=len(capsule.pivot_models),
            token_estimate=capsule.token_estimate,
            duration_ms=duration_ms,
        )
        _last_capsule_log_id[db_path] = log_id
        return capsule.model_dump()

    # ── Tool: get_model_details ───────────────────────────────────────────────

    @mcp.tool()
    def get_model_details(model_name: str) -> dict[str, Any]:
        """Get full details for a specific model.

        Returns the complete model definition including:
        - Compiled SQL and raw SQL
        - All columns with types, descriptions, tests
        - Upstream and downstream model names
        - Test coverage summary
        - File path for editing

        Args:
            model_name: Model name (e.g., "fct_orders") or unique_id
        """
        conn = _get_conn(db_path)
        search = HybridSearch(conn)
        graph = GraphOps(conn)

        row = search.get_model_by_name(model_name) or search.get_model_by_id(model_name)
        if not row:
            return {"error": f"Model '{model_name}' not found. Use search_models to find similar names."}

        uid = row["unique_id"]
        columns = search.get_columns(uid)
        tests = search.get_tests_for_model(uid)
        sources = search.get_sources_for_model(uid)
        macros = search.get_macros_for_model(uid)
        coverage = search.get_test_coverage(uid)
        upstream = graph.upstream(uid, depth=1)
        downstream = graph.downstream(uid, depth=1)

        upstream_names = []
        for up_id, _ in upstream:
            up_row = conn.execute("SELECT name FROM models WHERE unique_id = ?", (up_id,)).fetchone()
            if up_row:
                upstream_names.append(up_row[0])

        downstream_names = []
        for dn_id, _ in downstream:
            dn_row = conn.execute("SELECT name FROM models WHERE unique_id = ?", (dn_id,)).fetchone()
            if dn_row:
                downstream_names.append(dn_row[0])

        return {
            "unique_id": uid,
            "name": row["name"],
            "layer": row.get("layer", ""),
            "materialization": row.get("materialization", ""),
            "file_path": row.get("file_path", ""),
            "description": row.get("description", ""),
            "compiled_sql": row.get("compiled_code") or row.get("raw_code", ""),
            "columns": columns,
            "tests": tests,
            "test_coverage": coverage,
            "upstream_models": upstream_names,
            "downstream_models": downstream_names,
            "sources": sources,
            "macros_used": macros,
        }

    # ── Tool: get_lineage ─────────────────────────────────────────────────────

    @mcp.tool()
    def get_lineage(
        model_name: str,
        direction: str = "both",
        depth: int = 3,
    ) -> dict[str, Any]:
        """Get DAG lineage for a model.

        Args:
            model_name: Model name or unique_id
            direction: "upstream", "downstream", or "both" (default: "both")
            depth: Number of hops to traverse (default: 3, max: 10)

        Returns:
            Dict with upstream and/or downstream lists, each containing
            [{unique_id, name, layer, distance}]
        """
        conn = _get_conn(db_path)
        search = HybridSearch(conn)
        graph = GraphOps(conn)

        row = search.get_model_by_name(model_name) or search.get_model_by_id(model_name)
        if not row:
            return {"error": f"Model '{model_name}' not found."}

        uid = row["unique_id"]
        depth = min(max(1, depth), 10)

        result: dict[str, Any] = {
            "model": {"unique_id": uid, "name": row["name"], "layer": row.get("layer", "")},
        }

        def enrich(nodes: list[tuple[str, int]]) -> list[dict[str, Any]]:
            enriched = []
            for node_id, dist in nodes:
                node_row = conn.execute(
                    "SELECT name, layer, materialization FROM models WHERE unique_id = ?",
                    (node_id,),
                ).fetchone()
                if node_row:
                    enriched.append({
                        "unique_id": node_id,
                        "name": node_row[0],
                        "layer": node_row[1],
                        "materialization": node_row[2],
                        "distance": dist,
                    })
                elif node_id.startswith("source."):
                    src_row = conn.execute(
                        "SELECT name, source_name FROM sources WHERE unique_id = ?",
                        (node_id,),
                    ).fetchone()
                    if src_row:
                        enriched.append({
                            "unique_id": node_id,
                            "name": f"{src_row[1]}.{src_row[0]}",
                            "layer": "source",
                            "materialization": "external",
                            "distance": dist,
                        })
            return enriched

        if direction in ("upstream", "both"):
            result["upstream"] = enrich(graph.upstream(uid, depth=depth))
        if direction in ("downstream", "both"):
            result["downstream"] = enrich(graph.downstream(uid, depth=depth))

        return result

    # ── Tool: get_impact_analysis ────────────────────────────────────────────

    @mcp.tool()
    def get_impact_analysis(
        model_name: str,
        depth: int = 5,
    ) -> dict[str, Any]:
        """Blast radius analysis for changing a model.

        Returns affected downstream models, tests, exposures, and a risk level
        (low/medium/high). Use this before refactoring or modifying a model.

        Args:
            model_name: Model name (e.g., "fct_orders") or unique_id
            depth: Maximum downstream hops to analyze (default: 5, max: 10)

        Returns:
            Dict with target, affected_models, affected_tests,
            affected_exposures, and risk_level
        """
        conn = _get_conn(db_path)
        search = HybridSearch(conn)
        graph = GraphOps(conn)

        row = search.get_model_by_name(model_name) or search.get_model_by_id(model_name)
        if not row:
            return {"error": f"Model '{model_name}' not found. Use search_models to find similar names."}

        uid = row["unique_id"]
        depth = min(max(1, depth), 10)
        return graph.impact_analysis(uid, max_depth=depth)

    # ── Tool: search_models ───────────────────────────────────────────────────

    @mcp.tool()
    def search_models(
        query: str,
        limit: int = 10,
        layer: str | None = None,
    ) -> dict[str, Any]:
        """Hybrid search for models by name, description, column names, or SQL content.

        Uses BM25 full-text search re-ranked with graph centrality scores.

        Args:
            query: Search query (e.g., "revenue", "customer orders", "payment_method")
            limit: Maximum number of results (default: 10, max: 50)
            layer: Optional filter by layer ("staging", "intermediate", "marts", "other")

        Returns:
            List of matching models with name, layer, description, and relevance score
        """
        conn = _get_conn(db_path)
        search = HybridSearch(conn)
        limit = min(max(1, limit), 50)

        t0 = time.perf_counter()
        results = search.search(query, intent="explore", limit=limit * 2)

        if layer:
            results = [r for r in results if r.layer == layer]

        results = results[:limit]
        duration_ms = round((time.perf_counter() - t0) * 1000)

        UsageLogger(conn).log(
            "search_models",
            task_text=query,
            token_estimate=sum(len(r.description or "") // 4 for r in results),
            duration_ms=duration_ms,
        )

        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "unique_id": r.unique_id,
                    "name": r.name,
                    "layer": r.layer,
                    "description": r.description[:200] if r.description else "",
                    "score": round(r.score, 4),
                }
                for r in results
            ],
        }

    # ── Tool: find_models_by_column ──────────────────────────────────────────

    @mcp.tool()
    def find_models_by_column(column_name: str, limit: int = 20) -> dict[str, Any]:
        """Find all models containing a specific column name.

        Useful when you know a business concept (e.g., "revenue", "customer_id")
        but not which model contains it. Supports partial matching.

        Args:
            column_name: Column name or partial name to search for
            limit: Maximum results (default: 20, max: 50)

        Returns:
            List of models with matching column details
        """
        conn = _get_conn(db_path)
        search = HybridSearch(conn)
        limit = min(max(1, limit), 50)
        results = search.find_by_column(column_name, limit=limit)
        return {
            "column_name": column_name,
            "count": len(results),
            "results": results,
        }

    # ── Tool: find_models_by_path ─────────────────────────────────────────

    @mcp.tool()
    def find_models_by_path(path_pattern: str, limit: int = 20) -> dict[str, Any]:
        """Find models by file path or directory pattern.

        Examples: "models/marts/finance/%", "stg_stripe%", "%revenue%"

        Args:
            path_pattern: SQL LIKE pattern for file_path matching (use % as wildcard)
            limit: Maximum results (default: 20, max: 50)

        Returns:
            List of models matching the path pattern
        """
        conn = _get_conn(db_path)
        search = HybridSearch(conn)
        limit = min(max(1, limit), 50)
        results = search.find_by_path(path_pattern, limit=limit)
        return {
            "path_pattern": path_pattern,
            "count": len(results),
            "results": results,
        }

    # ── Tool: refresh_index ───────────────────────────────────────────────────

    @mcp.tool()
    def refresh_index(full: bool = False) -> dict[str, Any]:
        """Re-index from the latest dbt artifacts.

        Run this after `dbt compile` or `dbt build` to keep the context engine
        synchronized with your project.

        Args:
            full: If True, rebuild the entire index from scratch (default: False = incremental)

        Returns:
            Summary of what was indexed
        """
        manifest_path = cfg.manifest_path
        if not manifest_path.exists():
            return {
                "success": False,
                "error": (
                    f"manifest.json not found at {manifest_path}. "
                    "Run `dbt compile` or `dbt build` first."
                ),
            }

        with Indexer(db_path) as idx:
            idx.index_manifest(manifest_path)
            if cfg.catalog_path.exists():
                idx.index_catalog(cfg.catalog_path)
            if cfg.run_results_path.exists():
                idx.index_run_results(cfg.run_results_path)

            conn = idx.conn
            model_count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
            source_count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            test_count = conn.execute("SELECT COUNT(*) FROM tests").fetchone()[0]

        return {
            "success": True,
            "manifest": str(manifest_path),
            "models_indexed": model_count,
            "sources_indexed": source_count,
            "tests_indexed": test_count,
            "catalog_included": cfg.catalog_path.exists(),
            "run_results_included": cfg.run_results_path.exists(),
        }

    # ── Tool: rate_capsule ────────────────────────────────────────────────────

    @mcp.tool()
    def rate_capsule(
        rating: int,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Rate the most recent get_context_capsule call (1–5 stars).

        Call this after completing a task to record whether the context capsule
        was useful. Ratings are stored locally and used to improve Ariadne over time.

        Args:
            rating: 1 (useless) to 5 (perfect — had exactly what I needed)
            notes: Optional free-text note, e.g. "missing upstream source tables"

        Returns:
            Confirmation with the log_id that was rated
        """
        log_id = _last_capsule_log_id.get(db_path)
        if log_id is None:
            return {"success": False, "error": "No capsule call found in this session yet."}
        conn = _get_conn(db_path)
        UsageLogger(conn).rate(log_id, rating, notes)
        return {"success": True, "log_id": log_id, "rating": rating}

    return mcp


# ── Connection helper ─────────────────────────────────────────────────────────

_connections: dict[Path, sqlite3.Connection] = {}


def _get_conn(db_path: Path) -> sqlite3.Connection:
    """Return a cached SQLite connection for the given database path."""
    if db_path not in _connections or not _is_alive(_connections[db_path]):
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        _connections[db_path] = conn
    return _connections[db_path]


def _is_alive(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT 1")
        return True
    except Exception:
        return False
