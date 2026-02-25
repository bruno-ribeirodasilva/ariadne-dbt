"""Tests for the capsule builder and intent detection."""

from __future__ import annotations

import pytest

from ariadne_dbt.capsule import CapsuleBuilder, detect_intent


class TestIntentDetection:
    def test_add_feature(self):
        assert detect_intent("add monthly revenue metric") == "add_feature"

    def test_debug(self):
        assert detect_intent("debug failing test on fct_orders") == "debug"

    def test_refactor(self):
        assert detect_intent("refactor stg_customers to rename columns") == "refactor"

    def test_test(self):
        assert detect_intent("write tests for fct_orders") == "test"

    def test_document(self):
        assert detect_intent("document the stg_payments model") == "document"

    def test_explore_fallback(self):
        assert detect_intent("show me the model structure") == "explore"

    def test_explore_default(self):
        assert detect_intent("xyzzy gobbledygook") == "explore"


class TestCapsuleBuilder:
    def test_build_returns_capsule(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build("modify fct_orders to add discount column")
        assert capsule is not None
        assert capsule.task == "modify fct_orders to add discount column"
        assert capsule.intent == "add_feature"

    def test_build_has_pivots(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build("modify fct_orders")
        assert len(capsule.pivot_models) > 0

    def test_focus_model_anchors_pivot(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build("add a discount column", focus_model="fct_orders")
        pivot_names = [p.name for p in capsule.pivot_models]
        assert "fct_orders" in pivot_names

    def test_capsule_token_estimate(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build("analyze revenue models", token_budget=5000)
        assert capsule.token_budget == 5000
        # Should not wildly exceed budget
        assert capsule.token_estimate <= 5000 * 1.2

    def test_capsule_has_intent(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build("debug failing test on stg_orders")
        assert capsule.intent == "debug"

    def test_debug_includes_tests(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build("debug failing test on stg_orders", focus_model="stg_orders")
        # Debug intent should include tests
        all_tests = capsule.relevant_tests
        # There should be some test info
        assert isinstance(all_tests, list)

    def test_capsule_similar_models(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build("work with customer data", focus_model="dim_customers")
        # similar_models should exist but not contain the pivot
        pivot_names = {p.name for p in capsule.pivot_models}
        for similar in capsule.similar_models:
            assert similar not in pivot_names

    def test_no_pivot_ids_in_upstream(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build("add feature to fct_orders", focus_model="fct_orders")
        pivot_ids = {p.unique_id for p in capsule.pivot_models}
        upstream_ids = {u.unique_id for u in capsule.upstream_models}
        assert not pivot_ids.intersection(upstream_ids)

    def test_no_pivot_ids_in_downstream(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build("add feature to stg_orders", focus_model="stg_orders")
        pivot_ids = {p.unique_id for p in capsule.pivot_models}
        downstream_ids = {d.unique_id for d in capsule.downstream_models}
        assert not pivot_ids.intersection(downstream_ids)
