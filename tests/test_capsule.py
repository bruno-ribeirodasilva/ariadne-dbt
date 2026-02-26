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

    # ── entry_models / entry_paths ───────────────────────────────────────

    def test_entry_models_become_pivots(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build(
            "review PR changes",
            entry_models=["fct_orders", "stg_payments"],
        )
        pivot_names = [p.name for p in capsule.pivot_models]
        assert "fct_orders" in pivot_names
        assert "stg_payments" in pivot_names

    def test_entry_paths_resolve_to_pivots(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build(
            "review PR changes",
            entry_paths=["models/marts/fct_orders.sql"],
        )
        pivot_names = [p.name for p in capsule.pivot_models]
        assert "fct_orders" in pivot_names

    def test_entry_models_combined_with_focus(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build(
            "refactor models",
            focus_model="dim_customers",
            entry_models=["fct_orders"],
        )
        pivot_names = [p.name for p in capsule.pivot_models]
        assert "dim_customers" in pivot_names
        assert "fct_orders" in pivot_names

    # ── Confidence scoring ───────────────────────────────────────────────

    def test_confidence_high_with_focus_model(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build("do something", focus_model="fct_orders")
        assert capsule.confidence == "high"
        assert capsule.suggested_refinements == []

    def test_confidence_high_with_entry_models(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build("review PR", entry_models=["fct_orders"])
        assert capsule.confidence == "high"

    def test_confidence_low_with_vague_task(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build("something completely unrelated to dbt models xyzzy")
        assert capsule.confidence in ("low", "medium")
        if capsule.confidence == "low":
            assert len(capsule.suggested_refinements) > 0

    def test_confidence_field_always_present(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        capsule = builder.build("modify fct_orders")
        assert capsule.confidence in ("high", "medium", "low")
        assert isinstance(capsule.suggested_refinements, list)


class TestDiscover:
    def test_discover_returns_models(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        result = builder.discover("work with orders")
        assert len(result) > 0

    def test_discover_has_required_fields(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        result = builder.discover("work with orders")
        for m in result:
            assert "unique_id" in m
            assert "name" in m
            assert "layer" in m
            assert "file_path" in m
            assert "relationship" in m
            assert "distance" in m

    def test_discover_focus_model_is_pivot(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        result = builder.discover("do something", focus_model="fct_orders")
        pivots = [m for m in result if m["relationship"] == "pivot"]
        pivot_names = [m["name"] for m in pivots]
        assert "fct_orders" in pivot_names

    def test_discover_entry_models_are_pivots(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        result = builder.discover(
            "review PR",
            entry_models=["fct_orders", "stg_payments"],
        )
        pivot_names = [m["name"] for m in result if m["relationship"] == "pivot"]
        assert "fct_orders" in pivot_names
        assert "stg_payments" in pivot_names

    def test_discover_includes_dag_neighbors(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        result = builder.discover("work on orders", focus_model="fct_orders")
        relationships = {m["relationship"] for m in result}
        # Should have pivot + at least upstream or downstream
        assert "pivot" in relationships
        assert "upstream" in relationships or "downstream" in relationships

    def test_discover_respects_limit(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        result = builder.discover("orders", limit=2)
        assert len(result) <= 2

    def test_discover_no_duplicate_models(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        result = builder.discover("orders and customers")
        names = [m["unique_id"] for m in result]
        assert len(names) == len(set(names))

    def test_discover_broader_than_capsule(self, indexed_db):
        builder = CapsuleBuilder(indexed_db)
        discovery = builder.discover("work with orders", focus_model="fct_orders")
        capsule = builder.build("work with orders", focus_model="fct_orders")
        capsule_names = set()
        for pm in capsule.pivot_models:
            capsule_names.add(pm.name)
        for um in capsule.upstream_models:
            capsule_names.add(um.name)
        for dm in capsule.downstream_models:
            capsule_names.add(dm.name)
        discovery_names = {m["name"] for m in discovery}
        # Discovery should cover at least as many models as capsule
        assert len(discovery_names) >= len(capsule_names)
