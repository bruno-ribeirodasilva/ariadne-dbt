"""Tests for manifest.json parsing and SQLite indexing."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ariadne_dbt.indexer import Indexer, _detect_layer


class TestLayerDetection:
    def test_staging_prefix(self):
        assert _detect_layer(["project", "staging", "stg_orders"], "stg_orders", {}) == "staging"

    def test_intermediate_prefix(self):
        assert _detect_layer(["project", "intermediate", "int_orders"], "int_orders", {}) == "intermediate"

    def test_marts_fct_prefix(self):
        assert _detect_layer(["project", "marts", "fct_orders"], "fct_orders", {}) == "marts"

    def test_marts_dim_prefix(self):
        assert _detect_layer(["project", "marts", "dim_customers"], "dim_customers", {}) == "marts"

    def test_unknown_layer(self):
        assert _detect_layer(["project", "utils", "util_helper"], "util_helper", {}) == "other"

    def test_tag_override(self):
        # Tags can trigger layer detection too
        result = _detect_layer(["project", "custom", "my_model"], "my_model", {"tags": ["staging"]})
        assert result == "staging"


class TestIndexer:
    def test_models_indexed(self, indexed_db: sqlite3.Connection):
        count = indexed_db.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        assert count == 5  # stg_orders, stg_customers, stg_payments, fct_orders, dim_customers

    def test_sources_indexed(self, indexed_db: sqlite3.Connection):
        count = indexed_db.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        assert count == 3  # orders, customers, payments

    def test_tests_indexed(self, indexed_db: sqlite3.Connection):
        count = indexed_db.execute("SELECT COUNT(*) FROM tests").fetchone()[0]
        assert count == 5

    def test_macros_indexed(self, indexed_db: sqlite3.Connection):
        count = indexed_db.execute("SELECT COUNT(*) FROM macros").fetchone()[0]
        assert count == 1

    def test_exposures_indexed(self, indexed_db: sqlite3.Connection):
        count = indexed_db.execute("SELECT COUNT(*) FROM exposures").fetchone()[0]
        assert count == 1

    def test_model_layers(self, indexed_db: sqlite3.Connection):
        rows = indexed_db.execute("SELECT layer, COUNT(*) FROM models GROUP BY layer").fetchall()
        layer_counts = {r[0]: r[1] for r in rows}
        assert layer_counts.get("staging", 0) == 3
        assert layer_counts.get("marts", 0) == 2

    def test_edges_built(self, indexed_db: sqlite3.Connection):
        # fct_orders has 2 upstream model parents (stg_orders, stg_payments)
        count = indexed_db.execute(
            "SELECT COUNT(*) FROM edges WHERE child_id = 'model.jaffle_shop.fct_orders'"
        ).fetchone()[0]
        # stg_orders + stg_payments (source edges might also be present)
        assert count >= 2

    def test_fts_index_populated(self, indexed_db: sqlite3.Connection):
        count = indexed_db.execute("SELECT COUNT(*) FROM search_index").fetchone()[0]
        assert count == 5  # one row per model

    def test_metadata_stored(self, indexed_db: sqlite3.Connection):
        adapter = indexed_db.execute(
            "SELECT value FROM index_metadata WHERE key = 'adapter_type'"
        ).fetchone()[0]
        assert adapter == "duckdb"

    def test_columns_indexed(self, indexed_db: sqlite3.Connection):
        # fct_orders has 5 columns
        count = indexed_db.execute(
            "SELECT COUNT(*) FROM columns WHERE model_id = 'model.jaffle_shop.fct_orders'"
        ).fetchone()[0]
        assert count == 5

    def test_degree_counts_updated(self, indexed_db: sqlite3.Connection):
        row = indexed_db.execute(
            "SELECT upstream_count, downstream_count FROM models WHERE name = 'fct_orders'"
        ).fetchone()
        assert row["upstream_count"] >= 2  # stg_orders, stg_payments
        assert row["downstream_count"] >= 1  # dim_customers

    def test_is_primary_key_flagged(self, indexed_db: sqlite3.Connection):
        # fct_orders.order_id has both not_null and unique tests → primary key
        row = indexed_db.execute(
            "SELECT is_primary_key FROM columns WHERE model_id = 'model.jaffle_shop.fct_orders' AND name = 'order_id'"
        ).fetchone()
        assert row["is_primary_key"] == 1

    def test_idempotent_reindex(self, tmp_db: Path, manifest_path: Path):
        """Re-indexing the same manifest should not duplicate rows."""
        with Indexer(tmp_db) as idx:
            idx.index_manifest(manifest_path)
            count1 = idx.conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]

        with Indexer(tmp_db) as idx:
            idx.index_manifest(manifest_path)
            count2 = idx.conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]

        assert count1 == count2
