"""Integration tests against a real dbt manifest.

Copy your manifest to tests/fixtures/manifest_real.json to enable these tests.
Optionally copy catalog.json → tests/fixtures/catalog_real.json.

Run:
    .venv/bin/pytest tests/test_integration.py -v
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ariadne_dbt.capsule import CapsuleBuilder
from ariadne_dbt.config import CapsuleConfig
from ariadne_dbt.generator import ContextGenerator
from ariadne_dbt.graph import GraphOps
from ariadne_dbt.indexer import Indexer
from ariadne_dbt.patterns import PatternExtractor
from ariadne_dbt.search import HybridSearch

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_MANIFEST = FIXTURES_DIR / "manifest_real.json"
REAL_CATALOG = FIXTURES_DIR / "catalog_real.json"
REAL_RUN_RESULTS = FIXTURES_DIR / "run_results_real.json"

_SKIP = pytest.mark.skipif(
    not REAL_MANIFEST.exists(),
    reason="No real manifest at tests/fixtures/manifest_real.json",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_indexed_db(tmp_path_factory: pytest.TempPathFactory) -> sqlite3.Connection:
    """Index the real manifest into a temporary SQLite database."""
    db_path = tmp_path_factory.mktemp("real_db") / "real.db"
    with Indexer(db_path) as idx:
        idx.index_manifest(REAL_MANIFEST)
        if REAL_CATALOG.exists():
            idx.index_catalog(REAL_CATALOG)
        if REAL_RUN_RESULTS.exists():
            idx.index_run_results(REAL_RUN_RESULTS)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


# ── Tests ─────────────────────────────────────────────────────────────────────

@_SKIP
def test_real_index_build(real_indexed_db: sqlite3.Connection) -> None:
    """Real manifest indexes with more than 50 models."""
    count = real_indexed_db.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    assert count > 50, f"Expected >50 models, got {count}"


@_SKIP
def test_real_search_relevance(real_indexed_db: sqlite3.Connection) -> None:
    """Common domain terms return results in a real project."""
    search = HybridSearch(real_indexed_db)
    for term in ("revenue", "customer", "order"):
        results = search.search(term, intent="explore", limit=5)
        assert len(results) > 0, f"No results for '{term}'"
        assert all(r.score > 0 for r in results), f"Non-positive scores for '{term}'"


@_SKIP
def test_real_capsule_builds(real_indexed_db: sqlite3.Connection) -> None:
    """Capsules build successfully for five distinct intents."""
    cfg = CapsuleConfig()
    builder = CapsuleBuilder(real_indexed_db, cfg)
    intents = [
        ("debug a failing test", None),
        ("add a new revenue metric", None),
        ("refactor staging model", None),
        ("document all columns", None),
        ("explore the project structure", None),
    ]
    for task, focus in intents:
        capsule = builder.build(task=task, focus_model=focus, token_budget=8000)
        assert capsule.intent in ("debug", "add_feature", "refactor", "document", "explore"), (
            f"Unexpected intent '{capsule.intent}' for task '{task}'"
        )
        assert capsule.token_estimate > 0
        assert isinstance(capsule.pivot_models, list)


@_SKIP
def test_real_lineage_depth(real_indexed_db: sqlite3.Connection) -> None:
    """Highest-centrality model has upstream and downstream nodes."""
    graph = GraphOps(real_indexed_db)
    top_models = graph.high_centrality_models(limit=1)
    assert top_models, "No high-centrality models found"
    uid = top_models[0][0]

    upstream = graph.upstream(uid, depth=3)
    downstream = graph.downstream(uid, depth=3)
    assert len(upstream) > 0, "Top model has no upstream nodes"
    assert len(downstream) > 0, "Top model has no downstream nodes"


@_SKIP
def test_real_pattern_extraction(real_indexed_db: sqlite3.Connection) -> None:
    """Pattern extractor returns non-empty stats and patterns for real project."""
    extractor = PatternExtractor(real_indexed_db)
    stats = extractor.get_stats()
    assert stats.model_count > 50
    assert stats.adapter_type != ""

    patterns = extractor.get_patterns()
    assert patterns.naming_conventions, "No naming conventions extracted"


@_SKIP
def test_real_generator(real_indexed_db: sqlite3.Connection, tmp_path: Path) -> None:
    """Generator produces non-trivial .md files from a real manifest."""
    generator = ContextGenerator(real_indexed_db)
    written = generator.generate_all(tmp_path, targets=["claude_code"])
    assert written, "No files were generated"
    for path in written:
        assert path.exists(), f"Generated file missing: {path}"
        content = path.read_text()
        assert len(content) > 100, f"File too short (<100 chars): {path}"


@_SKIP
def test_real_token_reduction(real_indexed_db: sqlite3.Connection) -> None:
    """Capsule token estimate is less than naive all-models estimate."""
    cfg = CapsuleConfig()
    builder = CapsuleBuilder(real_indexed_db, cfg)
    capsule = builder.build(task="explore the project", token_budget=10000)

    # Naive estimate: sum of all model descriptions + SQL snippets
    rows = real_indexed_db.execute(
        "SELECT COALESCE(description,'') || ' ' || COALESCE(raw_code,'') FROM models"
    ).fetchall()
    naive_tokens = sum(len(r[0].split()) for r in rows)

    assert capsule.token_estimate < naive_tokens, (
        f"Capsule ({capsule.token_estimate}) should be smaller than naive ({naive_tokens})"
    )
