"""Pattern extraction: naming conventions, materializations, test patterns from the project."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from typing import Any

from .models import NamingPatterns, ProjectPatterns, ProjectStats


class PatternExtractor:
    """Analyze the indexed project and extract patterns and statistics."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ── Public API ────────────────────────────────────────────────────────────

    def get_stats(self) -> ProjectStats:
        meta = self._get_meta()
        counts = self._get_layer_counts()
        source_info = self._get_source_info()
        test_info = self._get_test_info()
        macro_info = self._get_macro_info()
        exposure_count = self._conn.execute("SELECT COUNT(*) FROM exposures").fetchone()[0]

        total_models = sum(counts.values())
        total_cols = self._conn.execute("SELECT COUNT(*) FROM columns").fetchone()[0]
        tested_cols = self._conn.execute(
            "SELECT COUNT(DISTINCT model_id || ':' || column_name) FROM tests WHERE column_name != ''"
        ).fetchone()[0]
        coverage_pct = int(tested_cols / total_cols * 100) if total_cols > 0 else 0

        return ProjectStats(
            project_name=meta.get("project_name", ""),
            adapter_type=meta.get("adapter_type", ""),
            dbt_schema_version=meta.get("dbt_schema_version", ""),
            model_count=total_models,
            staging_count=counts.get("staging", 0),
            intermediate_count=counts.get("intermediate", 0),
            marts_count=counts.get("marts", 0),
            other_count=counts.get("other", 0),
            source_count=source_info["source_count"],
            source_schema_count=source_info["source_schema_count"],
            test_count=test_info["total"],
            test_coverage_pct=coverage_pct,
            macro_count=macro_info["total"],
            project_macro_count=macro_info["project"],
            exposure_count=exposure_count,
        )

    def get_patterns(self) -> ProjectPatterns:
        naming = self._extract_naming_patterns()
        layer_counts = self._get_layer_counts()
        materializations = self._extract_materializations()
        coverage = self._extract_coverage_by_layer()
        common_tags = self._extract_common_tags()

        return ProjectPatterns(
            naming=naming,
            has_staging=layer_counts.get("staging", 0) > 0,
            has_intermediate=layer_counts.get("intermediate", 0) > 0,
            has_marts=layer_counts.get("marts", 0) > 0,
            common_tags=common_tags,
            common_materializations=materializations,
            test_coverage_by_layer=coverage,
        )

    def get_example_model(self, layer: str) -> dict[str, Any] | None:
        """Return a representative well-documented model for a given layer."""
        row = self._conn.execute(
            """
            SELECT m.unique_id, m.name, m.raw_code, m.compiled_code, m.file_path
            FROM models m
            WHERE m.layer = ?
            ORDER BY
                (SELECT COUNT(*) FROM columns c WHERE c.model_id = m.unique_id) DESC,
                length(m.description) DESC
            LIMIT 1
            """,
            (layer,),
        ).fetchone()
        return dict(row) if row else None

    def get_example_test_yaml(self) -> str:
        """Return an example YAML snippet showing test patterns."""
        # Find a model with good test coverage
        row = self._conn.execute(
            """
            SELECT m.name, m.layer,
                GROUP_CONCAT(DISTINCT t.test_type) as test_types,
                GROUP_CONCAT(DISTINCT t.column_name) as tested_columns
            FROM models m
            JOIN tests t ON t.model_id = m.unique_id
            WHERE t.column_name != ''
            GROUP BY m.unique_id
            ORDER BY COUNT(DISTINCT t.test_type) DESC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return ""

        lines = [f"models:", f"  - name: {row['name']}", "    columns:"]
        cols = self._conn.execute(
            """
            SELECT t.column_name, GROUP_CONCAT(t.test_type, ', ') as test_types
            FROM tests t
            JOIN models m ON m.unique_id = t.model_id
            WHERE m.name = ? AND t.column_name != ''
            GROUP BY t.column_name
            LIMIT 3
            """,
            (row["name"],),
        ).fetchall()
        for col in cols:
            lines.append(f"      - name: {col['column_name']}")
            lines.append(f"        tests:")
            for tt in (col["test_types"] or "").split(", "):
                if tt:
                    lines.append(f"          - {tt}")
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_meta(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM index_metadata").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def _get_layer_counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT layer, COUNT(*) as cnt FROM models GROUP BY layer"
        ).fetchall()
        return {r["layer"]: r["cnt"] for r in rows}

    def _get_source_info(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        schemas = self._conn.execute(
            "SELECT COUNT(DISTINCT source_name) FROM sources"
        ).fetchone()[0]
        return {"source_count": total, "source_schema_count": schemas}

    def _get_test_info(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM tests").fetchone()[0]
        return {"total": total}

    def _get_macro_info(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM macros").fetchone()[0]
        project_name = self._conn.execute(
            "SELECT value FROM index_metadata WHERE key = 'project_name'"
        ).fetchone()
        project_pkg = project_name[0] if project_name else ""
        project = self._conn.execute(
            "SELECT COUNT(*) FROM macros WHERE package_name = ?", (project_pkg,)
        ).fetchone()[0]
        return {"total": total, "project": project}

    def _extract_naming_patterns(self) -> NamingPatterns:
        """Infer naming patterns from model names in each layer."""
        patterns = NamingPatterns()
        layers: dict[str, list[str]] = {}
        rows = self._conn.execute("SELECT name, layer FROM models").fetchall()
        for r in rows:
            layers.setdefault(r["layer"], []).append(r["name"])

        staging_names = layers.get("staging", [])
        if staging_names:
            example = staging_names[0]
            patterns.staging_example = example
            # Detect separator (__ vs _)
            if "__" in example:
                patterns.staging_pattern = "stg_{source}__{entity}"
            else:
                patterns.staging_pattern = "stg_{source}_{entity}"

        intermediate_names = layers.get("intermediate", [])
        if intermediate_names:
            # Check for int_ prefix
            if any(n.startswith("int_") for n in intermediate_names):
                patterns.intermediate_pattern = "int_{entity}_{verb}"

        marts_names = layers.get("marts", [])
        has_fct = any(n.startswith("fct_") for n in marts_names)
        has_dim = any(n.startswith("dim_") for n in marts_names)
        if has_fct and has_dim:
            patterns.marts_pattern = "fct_{entity} | dim_{entity}"
        elif has_fct:
            patterns.marts_pattern = "fct_{entity}"
        elif has_dim:
            patterns.marts_pattern = "dim_{entity}"

        # Materializations per layer
        for layer in ("staging", "intermediate", "marts"):
            mat_rows = self._conn.execute(
                "SELECT materialization, COUNT(*) as cnt FROM models WHERE layer = ? GROUP BY materialization ORDER BY cnt DESC LIMIT 1",
                (layer,),
            ).fetchone()
            if mat_rows:
                setattr(patterns, f"{layer}_materialization", mat_rows["materialization"])

        # YAML pattern (look at file paths)
        yaml_rows = self._conn.execute(
            "SELECT file_path FROM models WHERE file_path != '' LIMIT 20"
        ).fetchall()
        yaml_pattern = self._infer_yaml_pattern([r["file_path"] for r in yaml_rows])
        if yaml_pattern:
            patterns.yaml_pattern = yaml_pattern

        patterns.naming_summary = (
            f"staging: {patterns.staging_pattern}, "
            f"intermediate: {patterns.intermediate_pattern}, "
            f"marts: {patterns.marts_pattern}"
        )
        patterns.directory_summary = (
            "models/staging/{source}/, models/intermediate/, models/marts/{domain}/"
        )
        patterns.yaml_requirements = (
            "Each model needs description + column descriptions + not_null/unique on PK"
        )

        return patterns

    @staticmethod
    def _infer_yaml_pattern(file_paths: list[str]) -> str:
        """Guess the YAML schema file naming from model file paths."""
        # Not directly extractable from manifest — return sensible default
        return "__{folder_name}_models.yml"

    def _extract_materializations(self) -> dict[str, str]:
        """Return most common materialization per layer."""
        result = {}
        for layer in ("staging", "intermediate", "marts", "other"):
            row = self._conn.execute(
                """
                SELECT materialization, COUNT(*) as cnt
                FROM models WHERE layer = ?
                GROUP BY materialization ORDER BY cnt DESC LIMIT 1
                """,
                (layer,),
            ).fetchone()
            if row:
                result[layer] = row["materialization"]
        return result

    def _extract_coverage_by_layer(self) -> dict[str, float]:
        """Return test coverage percentage per layer."""
        result = {}
        for layer in ("staging", "intermediate", "marts", "other"):
            total_cols = self._conn.execute(
                "SELECT COUNT(*) FROM columns c JOIN models m ON m.unique_id = c.model_id WHERE m.layer = ?",
                (layer,),
            ).fetchone()[0]
            tested_cols = self._conn.execute(
                """
                SELECT COUNT(DISTINCT t.model_id || ':' || t.column_name)
                FROM tests t
                JOIN models m ON m.unique_id = t.model_id
                WHERE m.layer = ? AND t.column_name != ''
                """,
                (layer,),
            ).fetchone()[0]
            result[layer] = round(tested_cols / total_cols * 100, 1) if total_cols > 0 else 0.0
        return result

    def _extract_common_tags(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT tags FROM models WHERE tags != '[]'"
        ).fetchall()
        counter: Counter = Counter()
        for row in rows:
            try:
                tags = json.loads(row["tags"])
                counter.update(tags)
            except (json.JSONDecodeError, TypeError):
                pass
        return [tag for tag, _ in counter.most_common(10)]
