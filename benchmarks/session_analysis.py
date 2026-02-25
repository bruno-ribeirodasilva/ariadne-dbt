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
import shutil
import sqlite3
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

# Task prefixes that indicate non-dbt sessions
SKIP_TASK_PREFIXES = [
    "<local",            # git worktree resumptions
    "[Requested interr", # tool resumptions (incomplete task context)
]


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
    focus_model: str | None
    ariadne_models: set[str]
    ariadne_pivot_names: list[str]
    overlap: set[str]
    overlap_pct: float
    potential_savings: int  # context calls that could have been skipped


# ── Session parsing ────────────────────────────────────────────────────────────


def _extract_models_from_input(tool_name: str, tool_input: dict) -> list[str]:
    """Extract dbt model names from a tool call's input."""
    models = set()

    # Collect all string values from input for path-based extraction
    string_vals: list[str] = []

    if tool_name == "Read":
        string_vals.append(tool_input.get("file_path", ""))

    elif tool_name == "Grep":
        string_vals.extend([
            tool_input.get("path", ""),
            tool_input.get("pattern", ""),
            tool_input.get("glob", ""),
        ])

    elif tool_name == "Glob":
        string_vals.extend([
            tool_input.get("path", ""),
            tool_input.get("pattern", ""),
        ])

    elif tool_name == "Bash":
        string_vals.append(tool_input.get("command", ""))

    elif tool_name in ("Edit", "Write"):
        string_vals.append(tool_input.get("file_path", ""))

    for text in string_vals:
        if not text:
            continue
        for m in MODEL_PATH_RE.findall(text):
            models.add(m)
        for m in REF_RE.findall(text):
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


def _is_dbt_relevant_task(task: str) -> bool:
    """Check if a task is likely about dbt model work (not a generic chat)."""
    for prefix in SKIP_TASK_PREFIXES:
        if task.startswith(prefix):
            return False
    return True


def _detect_focus_model(task: str, index_model_names: set[str]) -> str | None:
    """Try to extract a focus model name from the task text.

    Looks for known model names mentioned in the task, returning the longest match
    (most specific) as the focus model.
    """
    task_lower = task.lower()
    # Replace common separators with spaces for matching
    task_normalized = re.sub(r"[^a-z0-9_]", " ", task_lower)

    matches: list[str] = []
    for name in index_model_names:
        name_lower = name.lower()
        # Check if the model name appears as a word (or underscored token) in the task
        if name_lower in task_normalized or name_lower in task_lower:
            matches.append(name)

    if not matches:
        return None

    # Return the longest match (most specific model name)
    return max(matches, key=len)


def parse_session(filepath: Path) -> SessionAnalysis | None:
    """Parse a JSONL session file into a SessionAnalysis."""
    messages: list[dict] = []
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


def build_ariadne_index(manifest_path: Path) -> tuple[Path, str]:
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
    # Try to detect a focus model from the task text
    focus_model = _detect_focus_model(session.task, index_model_names)

    try:
        capsule = builder.build(
            task=session.task,
            focus_model=focus_model,
            token_budget=10000,
        )
    except Exception as e:
        console.print(f"  [dim red]Capsule build failed for {session.session_id}: {e}[/dim red]")
        return None

    # Collect all model names from Ariadne's capsule
    ariadne_models: set[str] = set()
    pivot_names: list[str] = []
    for pm in capsule.pivot_models:
        ariadne_models.add(pm.name)
        pivot_names.append(pm.name)
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
        focus_model=focus_model,
        ariadne_models=ariadne_models,
        ariadne_pivot_names=pivot_names,
        overlap=overlap,
        overlap_pct=overlap_pct,
        potential_savings=savings,
    )


# ── Display helpers ────────────────────────────────────────────────────────────


def _overlap_style(pct: float) -> str:
    if pct >= 75:
        return "bold green"
    elif pct >= 50:
        return "yellow"
    elif pct > 0:
        return "bright_red"
    return "red"


def _truncate(text: str, length: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) > length:
        return text[:length] + "..."
    return text


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

    # ── Phase 1: Parse sessions ───────────────────────────────────────────────

    console.print("\n[bold]Phase 1:[/bold] Parsing session transcripts...")
    session_files = sorted(SESSIONS_DIR.glob("*.jsonl"))
    console.print(f"  Found {len(session_files)} session files")

    all_sessions: list[SessionAnalysis] = []
    skipped_non_dbt = 0
    for sf in session_files:
        session = parse_session(sf)
        if session:
            if _is_dbt_relevant_task(session.task):
                all_sessions.append(session)
            else:
                skipped_non_dbt += 1

    console.print(f"  Parsed {len(all_sessions)} dbt-relevant sessions "
                  f"(skipped {skipped_non_dbt} non-dbt sessions)")

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

    # ── Phase 2: Build Ariadne index ──────────────────────────────────────────

    console.print("\n[bold]Phase 2:[/bold] Building Ariadne index from manifest...")
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

    # ── Phase 3: Compare ──────────────────────────────────────────────────────

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
    table.add_column("Task", max_width=45)
    table.add_column("Focus\nModel", max_width=20, style="cyan")
    table.add_column("Ctx\nCalls", justify="right", style="yellow")
    table.add_column("Agent\nModels", justify="right")
    table.add_column("Ariadne\nModels", justify="right", style="cyan")
    table.add_column("Overlap", justify="right")
    table.add_column("Overlap\n%", justify="right", style="bold")
    table.add_column("Saved\nCalls", justify="right", style="green")

    for c in comparisons:
        s = c.session
        task_short = _truncate(s.task, 60)
        agent_models_in_idx = s.models_explored & index_model_names
        style = _overlap_style(c.overlap_pct)

        table.add_row(
            s.session_id[:12],
            task_short,
            c.focus_model or "-",
            str(len(s.context_calls)),
            str(len(agent_models_in_idx)),
            str(len(c.ariadne_models & index_model_names)),
            str(len(c.overlap)),
            f"[{style}]{c.overlap_pct:.0f}%[/{style}]",
            str(c.potential_savings),
        )

    console.print(table)

    # ── Aggregate summary ──────────────────────────────────────────────────────

    n = len(comparisons)
    avg_ctx_calls = sum(len(c.session.context_calls) for c in comparisons) / n
    avg_overlap = sum(c.overlap_pct for c in comparisons) / n
    total_savings = sum(c.potential_savings for c in comparisons)
    total_ctx_calls = sum(len(c.session.context_calls) for c in comparisons)
    savings_pct = (total_savings / total_ctx_calls * 100) if total_ctx_calls else 0

    # Split by focus-model availability
    with_focus = [c for c in comparisons if c.focus_model]
    without_focus = [c for c in comparisons if not c.focus_model]

    summary = Table(title="Aggregate Summary", border_style="blue")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Sessions analyzed", str(n))
    summary.add_row("Avg context calls / session", f"{avg_ctx_calls:.1f}")
    summary.add_row("Total context calls", str(total_ctx_calls))
    summary.add_row("Avg model overlap %", f"{avg_overlap:.1f}%")
    summary.add_row("Total saveable calls", str(total_savings))
    summary.add_row("Overall savings %", f"{savings_pct:.1f}%")
    summary.add_row("", "")
    summary.add_row("Sessions WITH focus model detected", str(len(with_focus)))
    if with_focus:
        avg_focus = sum(c.overlap_pct for c in with_focus) / len(with_focus)
        focus_savings = sum(c.potential_savings for c in with_focus)
        focus_ctx = sum(len(c.session.context_calls) for c in with_focus)
        summary.add_row("  Avg overlap % (with focus)", f"{avg_focus:.1f}%")
        summary.add_row("  Saveable calls (with focus)", f"{focus_savings}/{focus_ctx}")
    summary.add_row("Sessions WITHOUT focus model", str(len(without_focus)))
    if without_focus:
        avg_nofocus = sum(c.overlap_pct for c in without_focus) / len(without_focus)
        nofocus_savings = sum(c.potential_savings for c in without_focus)
        nofocus_ctx = sum(len(c.session.context_calls) for c in without_focus)
        summary.add_row("  Avg overlap % (no focus)", f"{avg_nofocus:.1f}%")
        summary.add_row("  Saveable calls (no focus)", f"{nofocus_savings}/{nofocus_ctx}")

    console.print(summary)

    # ── Detailed overlap examples ──────────────────────────────────────────────

    # Show top 5 sessions with best overlap for illustration
    best = sorted(comparisons, key=lambda c: c.overlap_pct, reverse=True)[:5]
    if best and best[0].overlap_pct > 0:
        detail = Table(title="Top Overlap Sessions - Detail", border_style="magenta", show_lines=True)
        detail.add_column("Session", style="dim", max_width=12)
        detail.add_column("Focus", max_width=25, style="cyan")
        detail.add_column("Agent Models", max_width=35)
        detail.add_column("Ariadne Pivots", max_width=35, style="cyan")
        detail.add_column("Overlap", max_width=35, style="green")
        detail.add_column("%", justify="right", style="bold")

        for c in best:
            if c.overlap_pct == 0:
                continue
            agent_in_idx = sorted(c.session.models_explored & index_model_names)
            overlap_sorted = sorted(c.overlap)
            style = _overlap_style(c.overlap_pct)
            detail.add_row(
                c.session.session_id[:12],
                c.focus_model or "-",
                "\n".join(agent_in_idx[:6]) + ("\n..." if len(agent_in_idx) > 6 else ""),
                "\n".join(c.ariadne_pivot_names[:4]),
                "\n".join(overlap_sorted[:6]) + ("\n..." if len(overlap_sorted) > 6 else ""),
                f"[{style}]{c.overlap_pct:.0f}%[/{style}]",
            )
        console.print(detail)

    # ── Zero-overlap diagnosis ────────────────────────────────────────────────

    zero_overlap = [c for c in comparisons if c.overlap_pct == 0 and len(c.session.models_explored & index_model_names) > 0]
    if zero_overlap:
        diag = Table(title="Zero-Overlap Diagnosis (agent found models but Ariadne missed)",
                     border_style="red", show_lines=True)
        diag.add_column("Session", style="dim", max_width=12)
        diag.add_column("Task Snippet", max_width=40)
        diag.add_column("Agent Models (in index)", max_width=35)
        diag.add_column("Ariadne Pivots", max_width=35, style="cyan")
        diag.add_column("Reason", max_width=25, style="dim")

        for c in zero_overlap[:8]:
            agent_in_idx = sorted(c.session.models_explored & index_model_names)
            # Diagnose why
            if not c.ariadne_models & index_model_names:
                reason = "Ariadne found 0 models"
            elif not c.focus_model:
                reason = "No focus model detected"
            else:
                reason = "Focus model mismatch"
            diag.add_row(
                c.session.session_id[:12],
                _truncate(c.session.task, 55),
                "\n".join(agent_in_idx[:5]) + ("\n..." if len(agent_in_idx) > 5 else ""),
                "\n".join(c.ariadne_pivot_names[:3]),
                reason,
            )
        console.print(diag)

    # ── All sessions overview ─────────────────────────────────────────────────

    overview = Table(title="All Sessions Overview", border_style="dim")
    overview.add_column("Metric", style="bold")
    overview.add_column("Count", justify="right")
    overview.add_row("Total session files", str(len(session_files)))
    overview.add_row("Sessions with tasks (dbt-relevant)", str(len(all_sessions)))
    overview.add_row("Sessions with dbt model refs", str(len([s for s in all_sessions if s.models_explored])))
    overview.add_row(f"Sessions with >= {MIN_CONTEXT_CALLS} ctx calls + models", str(len(qualifying)))
    overview.add_row("Successful comparisons", str(n))

    # Distribution of context calls
    if all_sessions:
        ctx_counts = [len(s.context_calls) for s in all_sessions]
        overview.add_row("", "")
        overview.add_row("Context calls: min", str(min(ctx_counts)))
        overview.add_row("Context calls: max", str(max(ctx_counts)))
        overview.add_row("Context calls: median", str(sorted(ctx_counts)[len(ctx_counts) // 2]))

    console.print(overview)

    # ── Key Insight ───────────────────────────────────────────────────────────

    console.print(Panel(
        "[bold]Key Insights[/bold]\n\n"
        f"  Ariadne analyzed {n} real agent sessions that involved dbt model exploration.\n"
        f"  Average context-gathering calls per session: [yellow]{avg_ctx_calls:.1f}[/yellow]\n"
        f"  Average model overlap (Ariadne vs agent): [bold]{avg_overlap:.1f}%[/bold]\n"
        + (f"  With focus model: [bold green]{sum(c.overlap_pct for c in with_focus) / len(with_focus):.1f}%[/bold green] overlap ({len(with_focus)} sessions)\n" if with_focus else "")
        + (f"  Without focus model: [bold red]{sum(c.overlap_pct for c in without_focus) / len(without_focus):.1f}%[/bold red] overlap ({len(without_focus)} sessions)\n" if without_focus else "")
        + f"\n  Potential to save {total_savings} of {total_ctx_calls} context-gathering calls ({savings_pct:.1f}%)\n"
        "  across all analyzed sessions.\n\n"
        "  [dim]Note: Many sessions involve PR reviews or external tool lookups where\n"
        "  the task text alone is insufficient for Ariadne to determine relevant models.\n"
        "  Sessions with explicit model names in the task show much higher overlap.[/dim]",
        border_style="green",
    ))

    # Clean up
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
