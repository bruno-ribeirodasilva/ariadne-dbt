"""pytest-benchmark tests for Ariadne performance targets.

Requires: tests/fixtures/manifest_real.json
Install benchmark extras: pip install -e ".[dev]"

Run:
    .venv/bin/pytest tests/test_benchmarks.py -v --benchmark-only

Performance targets:
    - Full index rebuild: <5s for 500 models
    - Capsule build (P95): <500ms
    - Search query (P95): <100ms
    - Lineage traversal depth=3 (P95): <50ms
    - Pattern extraction (P95): <200ms
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ariadne_dbt.capsule import CapsuleBuilder
from ariadne_dbt.config import CapsuleConfig
from ariadne_dbt.graph import GraphOps
from ariadne_dbt.indexer import Indexer
from ariadne_dbt.patterns import PatternExtractor
from ariadne_dbt.search import HybridSearch

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_MANIFEST = FIXTURES_DIR / "manifest_real.json"

_SKIP = pytest.mark.skipif(
    not REAL_MANIFEST.exists(),
    reason="No real manifest at tests/fixtures/manifest_real.json",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Pre-built database for benchmark reuse."""
    db_path = tmp_path_factory.mktemp("bench_db") / "bench.db"
    with Indexer(db_path) as idx:
        idx.index_manifest(REAL_MANIFEST)
    return db_path


@pytest.fixture(scope="module")
def real_conn(real_db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(real_db_path))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def top_model_uid(real_conn: sqlite3.Connection) -> str:
    graph = GraphOps(real_conn)
    top = graph.high_centrality_models(limit=1)
    assert top, "No models found in benchmark DB"
    return top[0][0]


# ── Benchmarks ────────────────────────────────────────────────────────────────

@_SKIP
def test_bench_index_full_rebuild(benchmark, tmp_path: Path) -> None:
    """Target: full index rebuild <5s for 500 models."""
    db_path = tmp_path / "bench_rebuild.db"

    def rebuild() -> None:
        with Indexer(db_path) as idx:
            idx.index_manifest(REAL_MANIFEST)
        db_path.unlink(missing_ok=True)

    benchmark(rebuild)


@_SKIP
def test_bench_capsule_build(benchmark, real_conn: sqlite3.Connection) -> None:
    """Target: capsule build P95 <500ms."""
    cfg = CapsuleConfig()
    builder = CapsuleBuilder(real_conn, cfg)

    def build() -> None:
        builder.build(task="debug failing test on revenue model", token_budget=8000)

    benchmark(build)


@_SKIP
def test_bench_search_query(benchmark, real_conn: sqlite3.Connection) -> None:
    """Target: search query P95 <100ms."""
    search = HybridSearch(real_conn)

    def search_fn() -> None:
        search.search("revenue customer order", intent="explore", limit=10)

    benchmark(search_fn)


@_SKIP
def test_bench_lineage_traversal(
    benchmark, real_conn: sqlite3.Connection, top_model_uid: str
) -> None:
    """Target: lineage traversal depth=3 P95 <50ms."""
    graph = GraphOps(real_conn)

    def traverse() -> None:
        graph.upstream(top_model_uid, depth=3)
        graph.downstream(top_model_uid, depth=3)

    benchmark(traverse)


@_SKIP
def test_bench_pattern_extraction(benchmark, real_conn: sqlite3.Connection) -> None:
    """Target: pattern extraction P95 <200ms."""
    extractor = PatternExtractor(real_conn)

    def extract() -> None:
        extractor.get_stats()
        extractor.get_patterns()

    benchmark(extract)
