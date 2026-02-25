"""manifest.json / catalog.json / run_results.json parser → SQLite indexer."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import (
    ColumnInfo,
    ExposureNode,
    MacroNode,
    ModelNode,
    SourceNode,
    TestNode,
)


# ─── Layer detection ─────────────────────────────────────────────────────────

_LAYER_KEYWORDS: dict[str, list[str]] = {
    "staging": ["staging", "stg"],
    "intermediate": ["intermediate", "int"],
    "marts": ["marts", "mart", "fct", "dim", "agg", "rpt", "report"],
}


def _detect_layer(fqn: list[str], name: str, config: dict[str, Any]) -> str:
    tags = config.get("tags", []) or []
    path_parts = fqn[1:] if len(fqn) > 1 else []  # skip package name
    candidates = [p.lower() for p in path_parts] + [name.lower()] + [t.lower() for t in tags]
    for layer, keywords in _LAYER_KEYWORDS.items():
        if any(c.startswith(kw) or ("/" + kw) in c or c == kw for c in candidates for kw in keywords):
            return layer
    return "other"


# ─── Indexer class ────────────────────────────────────────────────────────────

class Indexer:
    """Parse dbt artifacts and populate the SQLite index."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._apply_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Indexer":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _apply_schema(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        with schema_path.open() as f:
            self._conn.executescript(f.read())
        self._conn.commit()

    # ── Public API ────────────────────────────────────────────────────────────

    def index_manifest(self, manifest_path: Path) -> None:
        """Parse manifest.json and populate all tables."""
        with manifest_path.open() as f:
            manifest = json.load(f)

        self._store_metadata(manifest)
        nodes: dict[str, Any] = manifest.get("nodes", {})
        sources: dict[str, Any] = manifest.get("sources", {})
        macros: dict[str, Any] = manifest.get("macros", {})
        exposures: dict[str, Any] = manifest.get("exposures", {})
        parent_map: dict[str, list[str]] = manifest.get("parent_map", {})
        child_map: dict[str, list[str]] = manifest.get("child_map", {})

        models, tests = self._parse_nodes(nodes)
        source_nodes = self._parse_sources(sources)
        macro_nodes = self._parse_macros(macros)
        exposure_nodes = self._parse_exposures(exposures)

        with self._conn:
            self._insert_models(models)
            self._insert_sources(source_nodes)
            self._insert_tests(tests)
            self._insert_macros(macro_nodes)
            self._insert_exposures(exposure_nodes)
            self._insert_edges(parent_map)
            self._update_degree_counts()
            self._populate_fts(models)

    def index_catalog(self, catalog_path: Path) -> None:
        """Parse catalog.json and enrich model rows with warehouse stats."""
        if not catalog_path.exists():
            return
        with catalog_path.open() as f:
            catalog = json.load(f)

        nodes: dict[str, Any] = catalog.get("nodes", {})
        with self._conn:
            for unique_id, node in nodes.items():
                meta = node.get("metadata", {})
                stats_raw = node.get("stats", {})
                row_count = self._extract_stat(stats_raw, "num_rows", "row_count")
                bytes_val = self._extract_stat(stats_raw, "num_bytes", "bytes")
                last_modified = meta.get("last_modified")

                # Enrich columns with catalog types
                catalog_cols: dict[str, Any] = node.get("columns", {})
                for col_name, col_data in catalog_cols.items():
                    self._conn.execute(
                        """
                        UPDATE columns SET data_type = ?
                        WHERE model_id = ? AND lower(name) = lower(?)
                        """,
                        (col_data.get("type", ""), unique_id, col_name),
                    )

                self._conn.execute(
                    """
                    UPDATE models
                    SET row_count = ?, bytes = ?, last_modified = ?
                    WHERE unique_id = ?
                    """,
                    (row_count, bytes_val, last_modified, unique_id),
                )

    def index_run_results(self, run_results_path: Path) -> None:
        """Parse run_results.json and update test statuses."""
        if not run_results_path.exists():
            return
        with run_results_path.open() as f:
            run_results = json.load(f)

        results: list[dict[str, Any]] = run_results.get("results", [])
        with self._conn:
            for result in results:
                unique_id: str = result.get("unique_id", "")
                if not unique_id.startswith("test."):
                    continue
                status = result.get("status", "")
                timing = result.get("timing", [])
                exec_time = sum(t.get("completed_at", 0) - t.get("started_at", 0) for t in timing if isinstance(t, dict))
                failures = result.get("failures", 0) or 0
                self._conn.execute(
                    """
                    UPDATE tests
                    SET last_status = ?, last_execution_time = ?, last_failures = ?
                    WHERE unique_id = ?
                    """,
                    (status, exec_time if exec_time > 0 else None, failures, unique_id),
                )

    # ── Parsing helpers ───────────────────────────────────────────────────────

    def _store_metadata(self, manifest: dict[str, Any]) -> None:
        meta: dict[str, Any] = manifest.get("metadata", {})
        rows = [
            ("dbt_schema_version", str(manifest.get("metadata", {}).get("dbt_schema_version", ""))),
            ("dbt_version", str(meta.get("dbt_version", ""))),
            ("adapter_type", str(meta.get("adapter_type", ""))),
            ("project_name", str(meta.get("project_name", ""))),
            ("generated_at", str(meta.get("generated_at", ""))),
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?, ?)", rows
        )
        self._conn.commit()

    def _parse_nodes(
        self, nodes: dict[str, Any]
    ) -> tuple[list[ModelNode], list[TestNode]]:
        models: list[ModelNode] = []
        tests: list[TestNode] = []
        for uid, node in nodes.items():
            resource_type = node.get("resource_type", "")
            if resource_type == "model":
                models.append(self._node_to_model(uid, node))
            elif resource_type == "test":
                tests.append(self._node_to_test(uid, node))
        return models, tests

    def _node_to_model(self, uid: str, node: dict[str, Any]) -> ModelNode:
        config: dict[str, Any] = node.get("config", {})
        fqn: list[str] = node.get("fqn", [])
        name: str = node.get("name", "")
        depends_on_nodes: list[str] = node.get("depends_on", {}).get("nodes", [])
        refs: list[str] = [
            r["name"] if isinstance(r, dict) else (r[0] if isinstance(r, list) else r)
            for r in node.get("refs", [])
        ]
        sources_list: list[str] = [
            ".".join(s) if isinstance(s, list) else s for s in node.get("sources", [])
        ]
        columns_raw: dict[str, Any] = node.get("columns", {})
        columns = [
            ColumnInfo(
                name=c.get("name") or cname,
                data_type=c.get("data_type") or "",
                description=c.get("description") or "",
                meta=c.get("meta") or {},
                tags=c.get("tags") or [],
            )
            for cname, c in columns_raw.items()
        ]
        return ModelNode(
            unique_id=uid,
            name=name,
            fqn=fqn,
            package_name=node.get("package_name", ""),
            database=node.get("database", ""),
            db_schema=node.get("schema", ""),
            alias=node.get("alias", name),
            file_path=node.get("original_file_path", ""),
            raw_code=node.get("raw_code", node.get("raw_sql", "")),
            compiled_code=node.get("compiled_code", node.get("compiled_sql", "")),
            language=node.get("language", "sql"),
            description=node.get("description", ""),
            layer=_detect_layer(fqn, name, config),
            materialization=config.get("materialized", "view"),
            tags=node.get("tags", []) + config.get("tags", []),
            meta=node.get("meta", {}),
            config=config,
            depends_on_nodes=depends_on_nodes,
            refs=refs,
            sources=sources_list,
            columns=columns,
        )

    def _node_to_test(self, uid: str, node: dict[str, Any]) -> TestNode:
        test_meta = node.get("test_metadata", {})
        test_type = test_meta.get("name", node.get("name", "")).lower()
        # Classify
        known = {"not_null", "unique", "accepted_values", "relationships"}
        if test_type not in known:
            test_type = "generic" if test_meta else "singular"

        model_id: str | None = None
        for dep in node.get("depends_on", {}).get("nodes", []):
            if dep.startswith("model."):
                model_id = dep
                break

        return TestNode(
            unique_id=uid,
            name=node.get("name", ""),
            test_type=test_type,
            model_id=model_id,
            column_name=test_meta.get("kwargs", {}).get("column_name", ""),
            depends_on=node.get("depends_on", {}).get("nodes", []),
            severity=node.get("config", {}).get("severity", "error"),
        )

    def _parse_sources(self, sources: dict[str, Any]) -> list[SourceNode]:
        result = []
        for uid, src in sources.items():
            cols_raw: dict[str, Any] = src.get("columns", {})
            columns = [
                ColumnInfo(
                    name=c.get("name") or cname,
                    data_type=c.get("data_type") or "",
                    description=c.get("description") or "",
                )
                for cname, c in cols_raw.items()
            ]
            freshness = src.get("freshness", {}) or {}
            result.append(SourceNode(
                unique_id=uid,
                name=src.get("name", ""),
                source_name=src.get("source_name", ""),
                schema_name=src.get("schema", ""),
                database=src.get("database", ""),
                description=src.get("description", ""),
                loader=src.get("loader", ""),
                freshness_warn=freshness.get("warn_after"),
                freshness_error=freshness.get("error_after"),
                tags=src.get("tags", []),
                meta=src.get("meta", {}),
                columns=columns,
            ))
        return result

    def _parse_macros(self, macros: dict[str, Any]) -> list[MacroNode]:
        result = []
        for uid, macro in macros.items():
            result.append(MacroNode(
                unique_id=uid,
                name=macro.get("name", ""),
                package_name=macro.get("package_name", ""),
                file_path=macro.get("original_file_path", ""),
                description=macro.get("description", ""),
                arguments=macro.get("arguments", []),
                macro_sql=macro.get("macro_sql", ""),
            ))
        return result

    def _parse_exposures(self, exposures: dict[str, Any]) -> list[ExposureNode]:
        result = []
        for uid, exp in exposures.items():
            owner: dict[str, Any] = exp.get("owner", {})
            result.append(ExposureNode(
                unique_id=uid,
                name=exp.get("name", ""),
                label=exp.get("label", ""),
                type=exp.get("type", ""),
                url=exp.get("url", ""),
                description=exp.get("description", ""),
                owner_name=owner.get("name", ""),
                owner_email=owner.get("email", ""),
                depends_on=exp.get("depends_on", {}).get("nodes", []),
                tags=exp.get("tags", []),
            ))
        return result

    # ── Insertion helpers ─────────────────────────────────────────────────────

    def _insert_models(self, models: list[ModelNode]) -> None:
        self._conn.execute("DELETE FROM models")
        for m in models:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO models (
                    unique_id, name, fqn, package_name, database, db_schema,
                    alias, file_path, raw_code, compiled_code, language, description,
                    layer, materialization, tags, meta, config,
                    depends_on_nodes, refs, sources
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    m.unique_id, m.name, json.dumps(m.fqn), m.package_name,
                    m.database, m.db_schema, m.alias, m.file_path,
                    m.raw_code, m.compiled_code, m.language, m.description,
                    m.layer, m.materialization,
                    json.dumps(m.tags), json.dumps(m.meta), json.dumps(m.config),
                    json.dumps(m.depends_on_nodes), json.dumps(m.refs), json.dumps(m.sources),
                ),
            )
            # Columns
            for col in m.columns:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO columns
                        (model_id, name, data_type, description, meta, tags)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (m.unique_id, col.name, col.data_type, col.description,
                     json.dumps(col.meta), json.dumps(col.tags)),
                )

    def _insert_sources(self, sources: list[SourceNode]) -> None:
        self._conn.execute("DELETE FROM sources")
        for s in sources:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO sources (
                    unique_id, name, source_name, schema_name, database,
                    description, loader, freshness_warn, freshness_error, tags, meta
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    s.unique_id, s.name, s.source_name, s.schema_name, s.database,
                    s.description, s.loader,
                    json.dumps(s.freshness_warn), json.dumps(s.freshness_error),
                    json.dumps(s.tags), json.dumps(s.meta),
                ),
            )
            for col in s.columns:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO source_columns
                        (source_id, name, data_type, description)
                    VALUES (?,?,?,?)
                    """,
                    (s.unique_id, col.name, col.data_type, col.description),
                )

    def _insert_tests(self, tests: list[TestNode]) -> None:
        self._conn.execute("DELETE FROM tests")
        for t in tests:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO tests (
                    unique_id, name, test_type, model_id, column_name,
                    depends_on, severity
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    t.unique_id, t.name, t.test_type, t.model_id, t.column_name,
                    json.dumps(t.depends_on), t.severity,
                ),
            )
        # Mark is_primary_key / is_foreign_key on columns
        self._conn.executescript("""
            UPDATE columns SET is_primary_key = 1
            WHERE rowid IN (
                SELECT c.rowid FROM columns c
                JOIN tests t ON t.model_id = c.model_id AND t.column_name = c.name
                WHERE t.test_type IN ('unique', 'not_null')
                GROUP BY c.model_id, c.name
                HAVING COUNT(DISTINCT t.test_type) >= 2
            );

            UPDATE columns SET is_foreign_key = 1
            WHERE rowid IN (
                SELECT c.rowid FROM columns c
                JOIN tests t ON t.model_id = c.model_id AND t.column_name = c.name
                WHERE t.test_type = 'relationships'
            );
        """)

    def _insert_macros(self, macros: list[MacroNode]) -> None:
        self._conn.execute("DELETE FROM macros")
        for m in macros:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO macros (
                    unique_id, name, package_name, file_path, description,
                    arguments, macro_sql
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (m.unique_id, m.name, m.package_name, m.file_path, m.description,
                 json.dumps(m.arguments), m.macro_sql),
            )

    def _insert_exposures(self, exposures: list[ExposureNode]) -> None:
        self._conn.execute("DELETE FROM exposures")
        for e in exposures:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO exposures (
                    unique_id, name, label, type, url, description,
                    owner_name, owner_email, depends_on, tags
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (e.unique_id, e.name, e.label, e.type, e.url, e.description,
                 e.owner_name, e.owner_email, json.dumps(e.depends_on), json.dumps(e.tags)),
            )

    def _insert_edges(self, parent_map: dict[str, list[str]]) -> None:
        """Build edges table from parent_map (child_id → [parent_ids])."""
        self._conn.execute("DELETE FROM edges")
        rows = []
        for child_id, parents in parent_map.items():
            for parent_id in parents:
                # Only include model-to-model, model-to-source, source-to-model edges
                if parent_id.startswith(("model.", "source.")) and child_id.startswith(("model.", "source.", "exposure.", "test.")):
                    rows.append((parent_id, child_id))
        self._conn.executemany("INSERT OR IGNORE INTO edges (parent_id, child_id) VALUES (?,?)", rows)

    def _update_degree_counts(self) -> None:
        self._conn.executescript("""
            UPDATE models SET upstream_count = (
                SELECT COUNT(*) FROM edges WHERE child_id = models.unique_id
            );
            UPDATE models SET downstream_count = (
                SELECT COUNT(*) FROM edges WHERE parent_id = models.unique_id
            );
            UPDATE models SET centrality = CAST(
                (upstream_count + downstream_count) AS REAL
            ) / NULLIF((SELECT MAX(upstream_count + downstream_count) FROM models), 0);
        """)

    def _populate_fts(self, models: list[ModelNode]) -> None:
        self._conn.execute("DELETE FROM search_index")
        for m in models:
            col_names = " ".join(c.name for c in m.columns)
            # Truncate SQL to avoid bloating FTS index
            sql_snippet = (m.compiled_code or m.raw_code)[:2000]
            self._conn.execute(
                """
                INSERT INTO search_index
                    (unique_id, name, description, column_names, sql_text, tags)
                VALUES (?,?,?,?,?,?)
                """,
                (m.unique_id, m.name, m.description, col_names,
                 sql_snippet, " ".join(m.tags)),
            )

    # ── Stats helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_stat(
        stats: dict[str, Any], *keys: str
    ) -> int | None:
        for key in keys:
            val = stats.get(key, {})
            if isinstance(val, dict):
                v = val.get("value")
            else:
                v = val
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
        return None

    # ── Connection access for other modules ───────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn
