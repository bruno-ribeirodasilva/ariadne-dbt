"""Tests for DAG graph operations."""

from __future__ import annotations

import pytest

from ariadne_dbt.graph import GraphOps


class TestGraphOps:
    def test_upstream_stg_orders(self, indexed_db):
        graph = GraphOps(indexed_db)
        upstream = graph.upstream("model.jaffle_shop.stg_orders", depth=1)
        upstream_ids = [uid for uid, _ in upstream]
        assert "source.jaffle_shop.jaffle_shop.orders" in upstream_ids

    def test_upstream_fct_orders(self, indexed_db):
        graph = GraphOps(indexed_db)
        upstream = graph.upstream("model.jaffle_shop.fct_orders", depth=1)
        upstream_ids = [uid for uid, _ in upstream]
        assert "model.jaffle_shop.stg_orders" in upstream_ids
        assert "model.jaffle_shop.stg_payments" in upstream_ids

    def test_upstream_depth_2(self, indexed_db):
        """dim_customers at depth 2 should reach staging models."""
        graph = GraphOps(indexed_db)
        upstream = graph.upstream("model.jaffle_shop.dim_customers", depth=2)
        upstream_ids = [uid for uid, _ in upstream]
        # depth 1: stg_customers, fct_orders
        # depth 2: stg_orders, stg_payments
        assert "model.jaffle_shop.stg_customers" in upstream_ids
        assert "model.jaffle_shop.fct_orders" in upstream_ids
        assert "model.jaffle_shop.stg_orders" in upstream_ids

    def test_downstream_stg_orders(self, indexed_db):
        graph = GraphOps(indexed_db)
        downstream = graph.downstream("model.jaffle_shop.stg_orders", depth=1)
        downstream_ids = [uid for uid, _ in downstream]
        assert "model.jaffle_shop.fct_orders" in downstream_ids

    def test_downstream_depth_2(self, indexed_db):
        """stg_orders downstream at depth 2 should reach dim_customers."""
        graph = GraphOps(indexed_db)
        downstream = graph.downstream("model.jaffle_shop.stg_orders", depth=2)
        downstream_ids = [uid for uid, _ in downstream]
        assert "model.jaffle_shop.dim_customers" in downstream_ids

    def test_depth_zero(self, indexed_db):
        graph = GraphOps(indexed_db)
        assert graph.upstream("model.jaffle_shop.fct_orders", depth=0) == []
        assert graph.downstream("model.jaffle_shop.fct_orders", depth=0) == []

    def test_impact_analysis_fct_orders(self, indexed_db):
        graph = GraphOps(indexed_db)
        impact = graph.impact_analysis("model.jaffle_shop.fct_orders", max_depth=3)
        assert impact["target"] == "model.jaffle_shop.fct_orders"
        assert impact["affected_model_count"] >= 1
        model_names = [m["name"] for m in impact["affected_models"]]
        assert "dim_customers" in model_names

    def test_impact_analysis_risk_level(self, indexed_db):
        graph = GraphOps(indexed_db)
        impact = graph.impact_analysis("model.jaffle_shop.stg_orders", max_depth=5)
        # stg_orders → fct_orders → dim_customers + exposure → high risk
        assert impact["risk_level"] in ("low", "medium", "high")

    def test_high_centrality_models(self, indexed_db):
        graph = GraphOps(indexed_db)
        models = graph.get_high_centrality_models(limit=3)
        assert len(models) > 0
        names = [m["name"] for m in models]
        # fct_orders is the most connected node
        assert "fct_orders" in names

    def test_no_self_in_traversal(self, indexed_db):
        """The start node should never appear in traversal results."""
        graph = GraphOps(indexed_db)
        uid = "model.jaffle_shop.fct_orders"
        for result_id, _ in graph.upstream(uid, depth=3):
            assert result_id != uid
        for result_id, _ in graph.downstream(uid, depth=3):
            assert result_id != uid
