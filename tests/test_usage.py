"""Tests for UsageLogger."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ariadne_dbt.indexer import Indexer
from ariadne_dbt.usage import UsageLogger

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"


@pytest.fixture()
def usage_db(tmp_path: Path) -> sqlite3.Connection:
    """Indexed DB with usage_log table available."""
    db_path = tmp_path / "usage.db"
    with Indexer(db_path) as idx:
        idx.index_manifest(MANIFEST_PATH)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


class TestUsageLogger:
    def test_log_returns_id(self, usage_db: sqlite3.Connection) -> None:
        logger = UsageLogger(usage_db)
        log_id = logger.log("get_context_capsule", task_text="debug fct_orders", intent="debug")
        assert isinstance(log_id, int)
        assert log_id > 0

    def test_log_stores_all_fields(self, usage_db: sqlite3.Connection) -> None:
        logger = UsageLogger(usage_db)
        logger.log(
            "get_context_capsule",
            task_text="add revenue metric",
            intent="add_feature",
            focus_model="fct_orders",
            pivot_count=2,
            token_estimate=3500,
            duration_ms=120,
        )
        row = usage_db.execute("SELECT * FROM usage_log ORDER BY id DESC LIMIT 1").fetchone()
        assert row["tool_name"] == "get_context_capsule"
        assert row["task_text"] == "add revenue metric"
        assert row["intent"] == "add_feature"
        assert row["focus_model"] == "fct_orders"
        assert row["pivot_count"] == 2
        assert row["token_estimate"] == 3500
        assert row["duration_ms"] == 120
        assert row["rating"] is None

    def test_rate_updates_row(self, usage_db: sqlite3.Connection) -> None:
        logger = UsageLogger(usage_db)
        log_id = logger.log("get_context_capsule", task_text="test")
        logger.rate(log_id, 5, "perfect context")
        row = usage_db.execute("SELECT rating, notes FROM usage_log WHERE id = ?", (log_id,)).fetchone()
        assert row["rating"] == 5
        assert row["notes"] == "perfect context"

    def test_rate_clamps_to_range(self, usage_db: sqlite3.Connection) -> None:
        logger = UsageLogger(usage_db)
        log_id = logger.log("search_models", task_text="revenue")
        logger.rate(log_id, 99)
        row = usage_db.execute("SELECT rating FROM usage_log WHERE id = ?", (log_id,)).fetchone()
        assert row["rating"] == 5  # clamped

    def test_get_stats_empty(self, usage_db: sqlite3.Connection) -> None:
        logger = UsageLogger(usage_db)
        stats = logger.get_stats(days=30)
        assert stats["total_calls"] == 0
        assert stats["by_tool"] == {}
        assert stats["by_intent"] == {}

    def test_get_stats_counts(self, usage_db: sqlite3.Connection) -> None:
        logger = UsageLogger(usage_db)
        logger.log("get_context_capsule", task_text="t1", intent="debug", token_estimate=1000, duration_ms=100)
        logger.log("get_context_capsule", task_text="t2", intent="explore", token_estimate=2000, duration_ms=200)
        logger.log("search_models", task_text="revenue", token_estimate=500, duration_ms=20)

        stats = logger.get_stats(days=30)
        assert stats["total_calls"] == 3
        assert stats["by_tool"]["get_context_capsule"] == 2
        assert stats["by_tool"]["search_models"] == 1
        assert stats["by_intent"]["debug"] == 1
        assert stats["by_intent"]["explore"] == 1
        assert stats["avg_token_estimate"] == round((1000 + 2000 + 500) / 3)

    def test_get_stats_top_models(self, usage_db: sqlite3.Connection) -> None:
        logger = UsageLogger(usage_db)
        logger.log("get_context_capsule", focus_model="fct_orders")
        logger.log("get_context_capsule", focus_model="fct_orders")
        logger.log("get_context_capsule", focus_model="dim_customers")

        stats = logger.get_stats(days=30)
        models = {m["model"]: m["calls"] for m in stats["top_models"]}
        assert models["fct_orders"] == 2
        assert models["dim_customers"] == 1

    def test_recent_queries(self, usage_db: sqlite3.Connection) -> None:
        logger = UsageLogger(usage_db)
        for i in range(5):
            logger.log("search_models", task_text=f"query {i}")
        rows = logger.recent_queries(limit=3)
        assert len(rows) == 3
        # Most recent first
        assert rows[0]["task_text"] == "query 4"
