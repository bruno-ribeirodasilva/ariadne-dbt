#!/usr/bin/env python3
"""Analyze real Claude Code session transcripts to benchmark Ariadne's context capsule.

Parses JSONL session files from ~/.claude/projects/-Users-taxfix-projects-data-dbt-models/
and compares the agent's context-gathering behavior against what Ariadne would provide
in a single capsule call.

Usage:
    python benchmarks/session_analysis.py
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ariadne_dbt.capsule import CapsuleBuilder
from ariadne_dbt.config import CapsuleConfig
from ariadne_dbt.indexer import Indexer

console = Console()

# ── Configuration ──────────────────────────────────────────────────────────────

SESSIONS_DIR = Path.home() / ".claude" / "projects" / "-Users-taxfix-projects-data-dbt-models"
MANIFEST_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "manifest.json"
MIN_CONTEXT_CALLS = 5

# Tools that gather context (read/search)
CONTEXT_TOOLS = {"Read", "Grep", "Glob", "Bash", "ToolSearch", "WebSearch", "WebFetch"}
# Tools that modify files (implementation)
IMPL_TOOLS = {"Edit", "Write"}
# Bash commands that are implementation (not context-gathering)
IMPL_BASH_PATTERNS = [
    re.compile(r"\bgit\s+(commit|push|add|checkout|merge|rebase|cherry-pick)\b"),
    re.compile(r"\bdbt\s+(run|build|test|seed|snapshot)\b"),
    re.compile(r"\bmkdir\b"),
    re.compile(r"\btouch\b"),
    re.compile(r"\brm\s"),
    re.compile(r"\bmv\s"),
    re.compile(r"\bcp\s"),
]

# Pattern to extract dbt model names from file paths
MODEL_PATH_RE = re.compile(r"models/.*?/([a-z_][a-z0-9_]*)\.(?:sql|yml|yaml)", re.IGNORECASE)
# Also catch model names from ref() calls in bash/grep
REF_RE = re.compile(r"""ref\(\s*['"]([a-z_][a-z0-9_]*)['"]""", re.IGNORECASE)


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    name: str
    input: dict
    is_context: bool
    models_referenced: list[str] = field(default_factory=list)


@dataclass
class SessionAnalysis:
    session_id: str
    task: str
    context_calls: list[ToolCall]
    impl_calls: list[ToolCall]
    models_explored: set[str]
    total_tool_calls: int


@dataclass
class AriadneComparison:
    session: SessionAnalysis
    ariadne_models: set[str]
    overlap: set[str]
    overlap_pct: float
    potential_savings: int  # context calls that could have been skipped


# ── Session parsing ────────────────────────────────────────────────────────────


def _extract_models_from_input(tool_name: str, tool_input: dict) -> list[str]:
    """Extract dbt model names from a tool call's input."""
    models = set()

    if tool_name == "Read":
        fp = tool_input.get("file_path", "")
        for m in MODEL_PATH_RE.findall(fp):
            models.add(m)

    elif tool_name == "Grep":
        path = tool_input.get("path", "")
        pattern = tool_input.get("pattern", "")
        glob_val = tool_input.get("glob", "")
        for text in [path, pattern, glob_val]:
            for m in MODEL_PATH_RE.findall(text):
                models.add(m)
            for m in REF_RE.findall(text):
                models.add(m)

    elif tool_name == "Glob":
        path = tool_input.get("path", "")
        pattern = tool_input.get("pattern", "")
        for text in [path, pattern]:
            for m in MODEL_PATH_RE.findall(text):
                models.add(m)

    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        for m in MODEL_PATH_RE.findall(cmd):
            models.add(m)
        for m in REF_RE.findall(cmd):
            models.add(m)

    elif tool_name == "Edit":
        fp = tool_input.get("file_path", "")
        for m in MODEL_PATH_RE.findall(fp):
            models.add(m)

    elif tool_name == "Write":
        fp = tool_input.get("file_path", "")
        for m in MODEL_PATH_RE.findall(fp):
            models.add(m)

    return list(models)


def _is_context_bash(command: str) -> bool:
    """Determine if a Bash command is context-gathering vs implementation."""
    for pattern in IMPL_BASH_PATTERNS:
        if pattern.search(command):
            return False
    return True


def _extract_initial_task(messages: list[dict]) -> str:
    """Find the first substantial user message as the task description."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        # Content can be a string or list of blocks
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            # Find text blocks, skip tool_result blocks
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        continue
                elif isinstance(block, str):
                    texts.append(block)
            text = " ".join(texts).strip()
        else:
            continue

        # Skip very short messages (likely commands like "/clear" etc.)
        if len(text) > 15:
            return text

    return ""


def parse_session(filepath: Path) -> SessionAnalysis | None:
    """Parse a JSONL session file into a SessionAnalysis."""
    messages: list[dict] = []  # (role, content) from type=user/assistant
    tool_calls: list[ToolCall] = []
    session_id = filepath.stem

    with filepath.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            obj_type = obj.get("type")

            # Collect messages for task extraction
            if obj_type in ("user", "assistant"):
                msg = obj.get("message", {})
                if msg.get("role") and msg.get("content"):
                    messages.append(msg)

            # Extract tool calls from assistant messages
            if obj_type == "assistant":
                msg = obj.get("message", {})
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            name = block.get("name", "")
                            inp = block.get("input", {})
                            models_ref = _extract_models_from_input(name, inp)

                            # Classify: context vs implementation
                            if name in IMPL_TOOLS:
                                is_context = False
                            elif name == "Bash":
                                cmd = inp.get("command", "")
                                is_context = _is_context_bash(cmd)
                            elif name in CONTEXT_TOOLS:
                                is_context = True
                            else:
                                # Other tools (TaskCreate, TaskUpdate, Skill, etc.) - skip
                                is_context = None

                            if is_context is not None:
                                tool_calls.append(ToolCall(
                                    name=name,
                                    input=inp,
                                    is_context=is_context,
                                    models_referenced=models_ref,
                                ))

    task = _extract_initial_task(messages)
    if not task:
        return None

    # Split into context-gathering and implementation phases
    # Context calls = all context tool calls before the first implementation call
    context_calls: list[ToolCall] = []
    impl_calls: list[ToolCall] = []
    first_impl_seen = False

    for tc in tool_calls:
        if not tc.is_context:
            first_impl_seen = True
            impl_calls.append(tc)
        elif first_impl_seen:
            # After first impl, context calls are part of the implementation phase
            # (e.g., reading a file to verify an edit) but we count them differently
            impl_calls.append(tc)
        else:
            context_calls.append(tc)

    # Collect all models explored across the entire session
    models_explored: set[str] = set()
    for tc in tool_calls:
        models_explored.update(tc.models_referenced)

    return SessionAnalysis(
        session_id=session_id,
        task=task,
        context_calls=context_calls,
        impl_calls=impl_calls,
        models_explored=models_explored,
        total_tool_calls=len(tool_calls),
    )


# ── Ariadne comparison ────────────────────────────────────────────────────────


def build_ariadne_index(manifest_path: Path) -> tuple[Path, any]:
    """Build the Ariadne index and return (db_path, tmpdir)."""
    tmpdir = tempfile.mkdtemp(prefix="ariadne_bench_")
    db_path = Path(tmpdir) / "ariadne.db"
    with Indexer(db_path) as idx:
        idx.index_manifest(manifest_path)
    return db_path, tmpdir


def compare_with_ariadne(
    session: SessionAnalysis,
    builder: CapsuleBuilder,
    index_model_names: set[str],
) -> AriadneComparison | None:
    """Run Ariadne's capsule builder and compare with agent behavior."""
    try:
        capsule = builder.build(task=session.task, token_budget=10000)
    except Exception as e:
        console.print(f"  [dim red]Capsule build failed for {session.session_id}: {e}[/dim red]")
        return None

    # Collect all model names from Ariadne's capsule
    ariadne_models: set[str] = set()
    for pm in capsule.pivot_models:
        ariadne_models.add(pm.name)
    for um in capsule.upstream_models:
        ariadne_models.add(um.name)
    for dm in capsule.downstream_models:
        ariadne_models.add(dm.name)
    for name in capsule.similar_models:
        ariadne_models.add(name)

    # Only compare models that exist in the index (the agent may reference
    # models from a different branch or that aren't in the test manifest)
    agent_models_in_index = session.models_explored & index_model_names
    ariadne_models_in_index = ariadne_models & index_model_names

    overlap = agent_models_in_index & ariadne_models_in_index

    # Overlap percentage: what fraction of the agent's explored models
    # were found by Ariadne
    if agent_models_in_index:
        overlap_pct = len(overlap) / len(agent_models_in_index) * 100
    else:
        overlap_pct = 0.0

    # Potential savings: context calls where the agent was reading/searching
    # models that Ariadne already covers
    savings = 0
    for tc in session.context_calls:
        if tc.models_referenced:
            # If all models in this call are covered by Ariadne, it's a "saved" call
            tc_models = set(tc.models_referenced) & index_model_names
            if tc_models and tc_models.issubset(ariadne_models_in_index):
                savings += 1

    return AriadneComparison(
        session=session,
        ariadne_models=ariadne_models,
        overlap=overlap,
        overlap_pct=overlap_pct,
        potential_savings=savings,
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    console.print(Panel(
        "[bold]Ariadne Session Analysis Benchmark[/bold]\n"
        f"Sessions dir: [cyan]{SESSIONS_DIR}[/cyan]\n"
        f"Manifest: [cyan]{MANIFEST_PATH}[/cyan]\n"
        f"Min context calls: [cyan]{MIN_CONTEXT_CALLS}[/cyan]",
        border_style="blue",
    ))

    if not SESSIONS_DIR.exists():
        console.print(f"[red]Error:[/red] Sessions directory not found: {SESSIONS_DIR}")
        sys.exit(1)

    if not MANIFEST_PATH.exists():
        console.print(f"[red]Error:[/red] Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    # Parse all sessions
    console.print("\n[bold]Phase 1:[/bold] Parsing session transcripts...")
    session_files = sorted(SESSIONS_DIR.glob("*.jsonl"))
    console.print(f"  Found {len(session_files)} session files")

    all_sessions: list[SessionAnalysis] = []
    for sf in session_files:
        session = parse_session(sf)
        if session:
            all_sessions.append(session)

    console.print(f"  Parsed {len(all_sessions)} sessions with identifiable tasks")

    # Filter to sessions with enough context-gathering calls and dbt model work
    qualifying = [
        s for s in all_sessions
        if len(s.context_calls) >= MIN_CONTEXT_CALLS and len(s.models_explored) > 0
    ]
    console.print(f"  Qualifying sessions (>= {MIN_CONTEXT_CALLS} context calls + dbt models): "
                  f"[bold]{len(qualifying)}[/bold]")

    if not qualifying:
        console.print("[yellow]No qualifying sessions found. Showing all sessions with dbt models.[/yellow]")
        qualifying = [s for s in all_sessions if len(s.models_explored) > 0]
        if not qualifying:
            console.print("[red]No sessions with dbt model references found.[/red]")
            sys.exit(0)

    # Build Ariadne index
    console.print("\n[bold]Phase 2:[/bold] Building Ariadne index from manifest...")
    import sqlite3
    db_path, tmpdir = build_ariadne_index(MANIFEST_PATH)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get all model names in the index
    index_model_names = {
        row[0] for row in conn.execute("SELECT name FROM models").fetchall()
    }
    console.print(f"  Indexed {len(index_model_names)} models")

    config = CapsuleConfig()
    builder = CapsuleBuilder(conn, config)

    # Compare each session with Ariadne
    console.print("\n[bold]Phase 3:[/bold] Comparing agent exploration vs Ariadne capsule...\n")
    comparisons: list[AriadneComparison] = []

    for session in qualifying:
        comparison = compare_with_ariadne(session, builder, index_model_names)
        if comparison:
            comparisons.append(comparison)

    conn.close()

    if not comparisons:
        console.print("[red]No successful comparisons. Check if manifest has matching models.[/red]")
        sys.exit(0)

    # ── Per-session results table ──────────────────────────────────────────────

    table = Table(title="Per-Session Analysis", border_style="green", show_lines=True)
    table.add_column("Session", style="dim", max_width=12, no_wrap=True)
    table.add_column("Task", max_width=50)
    table.add_column("Ctx\nCalls", justify="right", style="yellow")
    table.add_column("Agent\nModels", justify="right")
    table.add_column("Ariadne\nModels", justify="right", style="cyan")
    table.add_column("Overlap", justify="right")
    table.add_column("Overlap\n%", justify="right", style="bold")
    table.add_column("Saved\nCalls", justify="right", style="green")

    for c in comparisons:
        s = c.session
        task_short = s.task[:80] + ("..." if len(s.task) > 80 else "")
        agent_models_in_idx = s.models_explored & index_model_names

        overlap_style = ""
        if c.overlap_pct >= 75:
            overlap_style = "bold green"
        elif c.overlap_pct >= 50:
            overlap_style = "yellow"
        else:
            overlap_style = "red"

        table.add_row(
            s.session_id[:12],
            task_short,
            str(len(s.context_calls)),
            str(len(agent_models_in_idx)),
            str(len(c.ariadne_models & index_model_names)),
            str(len(c.overlap)),
            f"[{overlap_style}]{c.overlap_pct:.0f}%[/{overlap_style}]",
            str(c.potential_savings),
        )

    console.print(table)

    # ── Aggregate summary ──────────────────────────────────────────────────────

    avg_ctx_calls = sum(len(c.session.context_calls) for c in comparisons) / len(comparisons)
    avg_overlap = sum(c.overlap_pct for c in comparisons) / len(comparisons)
    total_savings = sum(c.potential_savings for c in comparisons)
    total_ctx_calls = sum(len(c.session.context_calls) for c in comparisons)
    savings_pct = (total_savings / total_ctx_calls * 100) if total_ctx_calls else 0

    summary = Table(title="Aggregate Summary", border_style="blue")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Sessions analyzed", str(len(comparisons)))
    summary.add_row("Avg context calls / session", f"{avg_ctx_calls:.1f}")
    summary.add_row("Total context calls", str(total_ctx_calls))
    summary.add_row("Avg model overlap %", f"{avg_overlap:.1f}%")
    summary.add_row("Total saveable calls", str(total_savings))
    summary.add_row("Overall savings %", f"{savings_pct:.1f}%")
    console.print(summary)

    # ── Detailed overlap examples ──────────────────────────────────────────────

    # Show top 3 sessions with best overlap for illustration
    best = sorted(comparisons, key=lambda c: c.overlap_pct, reverse=True)[:3]
    if best:
        detail = Table(title="Top Overlap Sessions - Detail", border_style="magenta")
        detail.add_column("Session", style="dim", max_width=12)
        detail.add_column("Agent Models", max_width=40)
        detail.add_column("Ariadne Models", max_width=40, style="cyan")
        detail.add_column("Overlap", max_width=40, style="green")

        for c in best:
            agent_in_idx = sorted(c.session.models_explored & index_model_names)
            ariadne_in_idx = sorted(c.ariadne_models & index_model_names)
            overlap_sorted = sorted(c.overlap)
            detail.add_row(
                c.session.session_id[:12],
                ", ".join(agent_in_idx[:8]) + ("..." if len(agent_in_idx) > 8 else ""),
                ", ".join(ariadne_in_idx[:8]) + ("..." if len(ariadne_in_idx) > 8 else ""),
                ", ".join(overlap_sorted[:8]) + ("..." if len(overlap_sorted) > 8 else ""),
            )
        console.print(detail)

    # ── All sessions overview (including non-qualifying) ───────────────────────

    overview = Table(title="All Sessions Overview", border_style="dim")
    overview.add_column("Metric", style="bold")
    overview.add_column("Count", justify="right")
    overview.add_row("Total session files", str(len(session_files)))
    overview.add_row("Sessions with tasks", str(len(all_sessions)))
    overview.add_row("Sessions with dbt models", str(len([s for s in all_sessions if s.models_explored])))
    overview.add_row(f"Sessions with >= {MIN_CONTEXT_CALLS} ctx calls + models", str(len(qualifying)))
    overview.add_row("Successful comparisons", str(len(comparisons)))

    # Distribution of context calls
    if all_sessions:
        ctx_counts = [len(s.context_calls) for s in all_sessions]
        overview.add_row("", "")
        overview.add_row("Context calls: min", str(min(ctx_counts)))
        overview.add_row("Context calls: max", str(max(ctx_counts)))
        overview.add_row("Context calls: median", str(sorted(ctx_counts)[len(ctx_counts) // 2]))

    console.print(overview)

    # Clean up
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
