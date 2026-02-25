"""Tests for pattern extraction."""

from __future__ import annotations

import pytest

from ariadne_dbt.patterns import PatternExtractor


class TestPatternExtractor:
    def test_get_stats(self, indexed_db):
        extractor = PatternExtractor(indexed_db)
        stats = extractor.get_stats()
        assert stats.project_name == "jaffle_shop"
        assert stats.adapter_type == "duckdb"
        assert stats.model_count == 5
        assert stats.staging_count == 3
        assert stats.marts_count == 2
        assert stats.source_count == 3
        assert stats.test_count == 5

    def test_get_patterns_naming(self, indexed_db):
        extractor = PatternExtractor(indexed_db)
        patterns = extractor.get_patterns()
        # stg_ naming should be detected
        assert "stg_" in patterns.naming.staging_pattern or "staging" in patterns.naming.staging_pattern.lower()

    def test_get_patterns_has_staging(self, indexed_db):
        extractor = PatternExtractor(indexed_db)
        patterns = extractor.get_patterns()
        assert patterns.has_staging is True

    def test_get_patterns_has_marts(self, indexed_db):
        extractor = PatternExtractor(indexed_db)
        patterns = extractor.get_patterns()
        assert patterns.has_marts is True

    def test_materializations_detected(self, indexed_db):
        extractor = PatternExtractor(indexed_db)
        patterns = extractor.get_patterns()
        assert "staging" in patterns.common_materializations
        assert "marts" in patterns.common_materializations
        assert patterns.common_materializations["staging"] == "view"
        assert patterns.common_materializations["marts"] == "table"

    def test_get_example_model_staging(self, indexed_db):
        extractor = PatternExtractor(indexed_db)
        example = extractor.get_example_model("staging")
        assert example is not None
        assert example["name"].startswith("stg_")

    def test_get_example_model_marts(self, indexed_db):
        extractor = PatternExtractor(indexed_db)
        example = extractor.get_example_model("marts")
        assert example is not None

    def test_get_example_test_yaml(self, indexed_db):
        extractor = PatternExtractor(indexed_db)
        yaml_str = extractor.get_example_test_yaml()
        # May be empty if no tests, or contain yaml structure
        assert isinstance(yaml_str, str)

    def test_test_coverage_by_layer(self, indexed_db):
        extractor = PatternExtractor(indexed_db)
        patterns = extractor.get_patterns()
        assert "staging" in patterns.test_coverage_by_layer
        assert "marts" in patterns.test_coverage_by_layer
        # Coverage should be between 0 and 100
        for layer, pct in patterns.test_coverage_by_layer.items():
            assert 0.0 <= pct <= 100.0
