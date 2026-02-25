"""Tests for context file generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ariadne_dbt.generator import ContextGenerator


class TestContextGenerator:
    def test_generate_claude_md(self, indexed_db, tmp_path):
        generator = ContextGenerator(indexed_db)
        written = generator.generate_all(tmp_path, targets=["claude_code"])
        assert len(written) > 0

        claude_md = tmp_path / ".claude" / "CLAUDE.md"
        assert claude_md.exists()
        content = claude_md.read_text()
        assert "jaffle_shop" in content
        assert "get_context_capsule" in content

    def test_generate_skills_new_model(self, indexed_db, tmp_path):
        generator = ContextGenerator(indexed_db)
        generator.generate_all(tmp_path, targets=["claude_code"])

        new_model_md = tmp_path / ".claude" / "skills" / "new_model.md"
        assert new_model_md.exists()
        content = new_model_md.read_text()
        assert "Naming" in content or "naming" in content.lower()

    def test_generate_skills_debug_test(self, indexed_db, tmp_path):
        generator = ContextGenerator(indexed_db)
        generator.generate_all(tmp_path, targets=["claude_code"])

        debug_md = tmp_path / ".claude" / "skills" / "debug_test.md"
        assert debug_md.exists()
        content = debug_md.read_text()
        assert "debug" in content.lower() or "Debug" in content

    def test_generate_dag_summary(self, indexed_db, tmp_path):
        generator = ContextGenerator(indexed_db)
        generator.generate_all(tmp_path, targets=["claude_code"])

        dag_md = tmp_path / ".claude" / "context" / "dag_summary.md"
        assert dag_md.exists()
        content = dag_md.read_text()
        assert "jaffle_shop" in content

    def test_generate_memory_md_created(self, indexed_db, tmp_path):
        generator = ContextGenerator(indexed_db)
        generator.generate_all(tmp_path, targets=["claude_code"])

        memory_md = tmp_path / ".claude" / "memory.md"
        assert memory_md.exists()

    def test_memory_md_not_overwritten(self, indexed_db, tmp_path):
        """memory.md should not be overwritten on subsequent runs."""
        generator = ContextGenerator(indexed_db)
        generator.generate_all(tmp_path, targets=["claude_code"])

        # Write custom content
        memory_md = tmp_path / ".claude" / "memory.md"
        memory_md.write_text("# Custom content\nDo not overwrite me.")

        # Run again
        generator.generate_all(tmp_path, targets=["claude_code"])
        assert "Custom content" in memory_md.read_text()

    def test_generate_cursor_rules(self, indexed_db, tmp_path):
        generator = ContextGenerator(indexed_db)
        generator.generate_all(tmp_path, targets=["cursor"])

        rules_file = tmp_path / ".cursor" / "rules" / "ariadne.mdc"
        assert rules_file.exists()
        content = rules_file.read_text()
        assert "dbt" in content.lower()

    def test_generate_multiple_targets(self, indexed_db, tmp_path):
        generator = ContextGenerator(indexed_db)
        written = generator.generate_all(tmp_path, targets=["claude_code", "cursor"])
        paths = [str(p) for p in written]
        assert any(".claude" in p for p in paths)
        assert any(".cursor" in p for p in paths)

    def test_key_models_in_claude_md(self, indexed_db, tmp_path):
        generator = ContextGenerator(indexed_db)
        generator.generate_all(tmp_path, targets=["claude_code"])
        content = (tmp_path / ".claude" / "CLAUDE.md").read_text()
        # Most connected model should appear
        assert "fct_orders" in content or "dim_customers" in content
