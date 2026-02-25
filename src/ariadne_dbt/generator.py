"""Auto-generate .claude/, .cursor/, and .windsurf/ context files from the project index."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .models import ProjectPatterns, ProjectStats
from .patterns import PatternExtractor


TEMPLATES_DIR = Path(__file__).parent / "templates"


def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


class ContextGenerator:
    """Generate .claude/ and agent-rules files from the indexed project."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._extractor = PatternExtractor(conn)
        self._env = _jinja_env()

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_all(
        self,
        project_root: Path,
        targets: list[str] | None = None,
    ) -> list[Path]:
        """Generate all context files. Returns list of written file paths."""
        targets = targets or ["claude_code"]
        stats = self._extractor.get_stats()
        patterns = self._extractor.get_patterns()
        key_models = self._get_key_models()
        ctx = self._build_context(stats, patterns, key_models)

        written: list[Path] = []

        if "claude_code" in targets:
            written += self._write_claude_files(project_root, ctx, stats, patterns)

        if "cursor" in targets:
            written += self._write_cursor_files(project_root, ctx)

        if "windsurf" in targets:
            written += self._write_windsurf_files(project_root, ctx)

        return written

    # ── File writers ──────────────────────────────────────────────────────────

    def _write_claude_files(
        self,
        project_root: Path,
        ctx: dict[str, Any],
        stats: ProjectStats,
        patterns: ProjectPatterns,
    ) -> list[Path]:
        claude_dir = project_root / ".claude"
        skills_dir = claude_dir / "skills"
        context_dir = claude_dir / "context"
        for d in (claude_dir, skills_dir, context_dir):
            d.mkdir(parents=True, exist_ok=True)

        written = []

        # CLAUDE.md
        claude_md = claude_dir / "CLAUDE.md"
        rendered = self._render("claude_md.j2", ctx)
        _write_file(claude_md, rendered)
        written.append(claude_md)

        # memory.md — only create if absent
        memory_md = claude_dir / "memory.md"
        if not memory_md.exists():
            _write_file(memory_md, _initial_memory_md(stats.project_name))
            written.append(memory_md)

        # skills/new_model.md
        example_model = self._extractor.get_example_model("staging") or self._extractor.get_example_model("marts")
        example_yaml = self._extractor.get_example_test_yaml()
        skill_ctx = {**ctx, "example_model": example_model, "example_yaml": example_yaml}
        rendered = self._render("skill_new_model.j2", skill_ctx)
        _write_file(skills_dir / "new_model.md", rendered)
        written.append(skills_dir / "new_model.md")

        # skills/debug_test.md
        rendered = self._render("skill_debug_test.j2", ctx)
        _write_file(skills_dir / "debug_test.md", rendered)
        written.append(skills_dir / "debug_test.md")

        # context/dag_summary.md
        rendered = self._render("dag_summary.j2", ctx)
        _write_file(context_dir / "dag_summary.md", rendered)
        written.append(context_dir / "dag_summary.md")

        return written

    def _write_cursor_files(self, project_root: Path, ctx: dict[str, Any]) -> list[Path]:
        cursor_dir = project_root / ".cursor" / "rules"
        cursor_dir.mkdir(parents=True, exist_ok=True)
        rules_file = cursor_dir / "ariadne.mdc"
        rendered = self._render("cursor_rules.j2", ctx)
        _write_file(rules_file, rendered)
        return [rules_file]

    def _write_windsurf_files(self, project_root: Path, ctx: dict[str, Any]) -> list[Path]:
        windsurf_dir = project_root / ".windsurf" / "rules"
        windsurf_dir.mkdir(parents=True, exist_ok=True)
        rules_file = windsurf_dir / "ariadne.md"
        # Windsurf uses same format as Cursor rules (Markdown)
        rendered = self._render("cursor_rules.j2", ctx)
        _write_file(rules_file, rendered)
        return [rules_file]

    # ── Template rendering ────────────────────────────────────────────────────

    def _render(self, template_name: str, context: dict[str, Any]) -> str:
        template = self._env.get_template(template_name)
        return template.render(**context)

    def _build_context(
        self,
        stats: ProjectStats,
        patterns: ProjectPatterns,
        key_models: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "stats": stats,
            "patterns": patterns,
            "key_models": key_models,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _get_key_models(self, limit: int = 8) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT unique_id, name, layer, description, upstream_count, downstream_count, centrality
            FROM models
            ORDER BY centrality DESC, downstream_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _initial_memory_md(project_name: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""# Project Memory (auto-updated by ariadne)
> Initialized: {today}

## Recent Changes
<!-- ariadne will append entries here after significant sessions -->

## Known Issues
<!-- Add known data quality issues, source problems, or model gotchas here -->

## Domain Knowledge
<!-- Add project-specific business rules and definitions here -->
<!-- Example: "Revenue" means net revenue (after refunds) -->

## Agent Notes
<!-- Persistent notes for AI agents about this project's conventions -->
"""
