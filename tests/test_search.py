"""Tests for hybrid search."""

from __future__ import annotations

import pytest

from ariadne_dbt.search import HybridSearch, _tokenize_query


class TestTokenizeQuery:
    def test_simple(self):
        result = _tokenize_query("revenue")
        assert "revenue" in result

    def test_multi_word(self):
        result = _tokenize_query("monthly revenue metric")
        # Should produce OR-joined tokens
        assert "OR" in result
        assert "revenue" in result
        assert "monthly" in result

    def test_stopwords_removed(self):
        result = _tokenize_query("add a new metric to the project")
        assert " a " not in result
        assert " the " not in result

    def test_punctuation_stripped(self):
        result = _tokenize_query("fct_orders!")
        assert "!" not in result


class TestHybridSearch:
    def test_search_returns_results(self, indexed_db):
        search = HybridSearch(indexed_db)
        results = search.search("orders", limit=5)
        assert len(results) > 0

    def test_search_revenue_finds_orders(self, indexed_db):
        search = HybridSearch(indexed_db)
        results = search.search("orders amount", limit=5)
        names = [r.name for r in results]
        assert any("order" in n for n in names)

    def test_search_customer_finds_customers(self, indexed_db):
        search = HybridSearch(indexed_db)
        results = search.search("customer", limit=5)
        names = [r.name for r in results]
        assert any("customer" in n for n in names)

    def test_search_scores_are_positive(self, indexed_db):
        search = HybridSearch(indexed_db)
        results = search.search("orders", limit=5)
        for r in results:
            assert r.score >= 0

    def test_search_respects_limit(self, indexed_db):
        search = HybridSearch(indexed_db)
        results = search.search("orders", limit=2)
        assert len(results) <= 2

    def test_search_exclude_ids(self, indexed_db):
        search = HybridSearch(indexed_db)
        exclude = {"model.jaffle_shop.fct_orders"}
        results = search.search("orders", limit=5, exclude_ids=exclude)
        result_ids = {r.unique_id for r in results}
        assert "model.jaffle_shop.fct_orders" not in result_ids

    def test_get_model_by_name(self, indexed_db):
        search = HybridSearch(indexed_db)
        row = search.get_model_by_name("fct_orders")
        assert row is not None
        assert row["name"] == "fct_orders"

    def test_get_model_by_name_case_insensitive(self, indexed_db):
        search = HybridSearch(indexed_db)
        row = search.get_model_by_name("FCT_ORDERS")
        assert row is not None

    def test_get_model_by_id(self, indexed_db):
        search = HybridSearch(indexed_db)
        row = search.get_model_by_id("model.jaffle_shop.fct_orders")
        assert row is not None

    def test_get_columns(self, indexed_db):
        search = HybridSearch(indexed_db)
        cols = search.get_columns("model.jaffle_shop.fct_orders")
        assert len(cols) == 5
        col_names = [c["name"] for c in cols]
        assert "order_id" in col_names
        assert "amount" in col_names

    def test_get_tests_for_model(self, indexed_db):
        search = HybridSearch(indexed_db)
        tests = search.get_tests_for_model("model.jaffle_shop.fct_orders")
        assert len(tests) == 3  # not_null, unique, accepted_values
        test_types = {t["test_type"] for t in tests}
        assert "not_null" in test_types
        assert "unique" in test_types
        assert "accepted_values" in test_types

    def test_get_test_coverage(self, indexed_db):
        search = HybridSearch(indexed_db)
        coverage = search.get_test_coverage("model.jaffle_shop.fct_orders")
        assert coverage["total_columns"] == 5
        assert coverage["coverage_pct"] >= 0
        assert coverage["coverage_pct"] <= 100

    def test_get_sources_for_model(self, indexed_db):
        search = HybridSearch(indexed_db)
        sources = search.get_sources_for_model("model.jaffle_shop.stg_orders")
        assert len(sources) >= 1
        source_names = [s["name"] for s in sources]
        assert "orders" in source_names

    # ── resolve_file_paths ───────────────────────────────────────────────

    def test_resolve_file_paths_by_basename(self, indexed_db):
        search = HybridSearch(indexed_db)
        resolved = search.resolve_file_paths(["models/marts/fct_orders.sql"])
        assert len(resolved) == 1
        assert resolved[0] == "model.jaffle_shop.fct_orders"

    def test_resolve_file_paths_multiple(self, indexed_db):
        search = HybridSearch(indexed_db)
        resolved = search.resolve_file_paths([
            "models/marts/fct_orders.sql",
            "models/staging/stg_orders.sql",
        ])
        assert len(resolved) == 2
        names = set(resolved)
        assert "model.jaffle_shop.fct_orders" in names
        assert "model.jaffle_shop.stg_orders" in names

    def test_resolve_file_paths_yaml_skipped(self, indexed_db):
        search = HybridSearch(indexed_db)
        resolved = search.resolve_file_paths(["models/marts/_marts.yml"])
        assert len(resolved) == 0

    def test_resolve_file_paths_nonexistent(self, indexed_db):
        search = HybridSearch(indexed_db)
        resolved = search.resolve_file_paths(["models/nonexistent_model.sql"])
        assert len(resolved) == 0

    def test_resolve_file_paths_deduplicates(self, indexed_db):
        search = HybridSearch(indexed_db)
        resolved = search.resolve_file_paths([
            "models/marts/fct_orders.sql",
            "some/other/path/fct_orders.sql",
        ])
        assert len(resolved) == 1

    # ── find_by_column ───────────────────────────────────────────────────

    def test_find_by_column_order_id(self, indexed_db):
        search = HybridSearch(indexed_db)
        results = search.find_by_column("order_id")
        assert len(results) > 0
        model_names = [r["name"] for r in results]
        assert any("order" in name for name in model_names)

    def test_find_by_column_partial_match(self, indexed_db):
        search = HybridSearch(indexed_db)
        results = search.find_by_column("amount")
        assert len(results) > 0

    def test_find_by_column_no_results(self, indexed_db):
        search = HybridSearch(indexed_db)
        results = search.find_by_column("zzz_nonexistent_column_zzz")
        assert len(results) == 0

    def test_find_by_column_respects_limit(self, indexed_db):
        search = HybridSearch(indexed_db)
        results = search.find_by_column("id", limit=1)
        assert len(results) <= 1

    # ── find_by_path ─────────────────────────────────────────────────────

    def test_find_by_path_wildcard(self, indexed_db):
        search = HybridSearch(indexed_db)
        results = search.find_by_path("%staging%")
        assert len(results) > 0
        for r in results:
            assert "staging" in r["file_path"].lower()

    def test_find_by_path_no_results(self, indexed_db):
        search = HybridSearch(indexed_db)
        results = search.find_by_path("nonexistent_dir/%")
        assert len(results) == 0

    def test_find_by_path_respects_limit(self, indexed_db):
        search = HybridSearch(indexed_db)
        results = search.find_by_path("%", limit=2)
        assert len(results) <= 2
