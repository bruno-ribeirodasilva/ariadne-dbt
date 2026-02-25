"""Shared pytest fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ariadne_dbt.indexer import Indexer

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"
REAL_MANIFEST_PATH = FIXTURES_DIR / "manifest_real.json"
REAL_CATALOG_PATH = FIXTURES_DIR / "catalog_real.json"
REAL_RUN_RESULTS_PATH = FIXTURES_DIR / "run_results_real.json"


@pytest.fixture(scope="session")
def manifest_path() -> Path:
    return MANIFEST_PATH


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Return path to a fresh temporary SQLite database."""
    return tmp_path / "test.db"


@pytest.fixture()
def indexed_db(tmp_db: Path) -> sqlite3.Connection:
    """Return a connection to a fully-indexed test database (jaffle_shop)."""
    with Indexer(tmp_db) as idx:
        idx.index_manifest(MANIFEST_PATH)
    conn = sqlite3.connect(str(tmp_db))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


# ── Real-manifest fixtures (skipped when file absent) ─────────────────────────

@pytest.fixture(scope="session")
def real_manifest_path() -> Path:
    """Path to the real manifest fixture."""
    return REAL_MANIFEST_PATH


@pytest.fixture(scope="session")
def real_catalog_path() -> Path:
    """Path to the real catalog fixture (optional enrichment)."""
    return REAL_CATALOG_PATH


@pytest.fixture(scope="session")
def real_run_results_path() -> Path:
    """Path to the real run_results fixture (optional enrichment)."""
    return REAL_RUN_RESULTS_PATH
