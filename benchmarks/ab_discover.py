#!/usr/bin/env python3
"""A/B comparison: capsule alone vs discover_models + capsule with entry_models.

Simulates the iterative workflow:
  A (baseline): get_context_capsule(task, focus_model=..., entry_models from first 3 calls)
  B (discover): discover_models(task, ...) → pick agent-relevant names → get_context_capsule(task, entry_models=...)

Usage:
    .venv/bin/python benchmarks/ab_discover.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ariadne_dbt.capsule import CapsuleBuilder
from ariadne_dbt.config import CapsuleConfig
from benchmarks.session_analysis import (
    MIN_CONTEXT_CALLS,
    SESSIONS_DIR,
    build_ariadne_index,
    parse_session,
    _detect_focus_model,
    _is_dbt_relevant_task,
    _truncate,
)
from benchmarks.ab_entry_models import _capsule_model_names, _collect_models_from_calls

console = Console()

# Use real production manifest
MANIFEST_PATH = Path("/Users/taxfix/projects/data-dbt-models/target/manifest.json")


def main():
    if not SESSIONS_DIR.exists():
        console.print(f"[red]Sessions dir not found:[/red] {SESSIONS_DIR}")
        sys.exit(1)
    if not MANIFEST_PATH.exists():
        console.print(f"[red]Manifest not found:[/red] {MANIFEST_PATH}")
        sys.exit(1)

    # Parse sessions
    session_files = sorted(SESSIONS_DIR.glob("*.jsonl"))
    sessions = []
    for sf in session_files:
        s = parse_session(sf)
        if s and _is_dbt_relevant_task(s.task) and len(s.context_calls) >= MIN_CONTEXT_CALLS and s.models_explored:
            sessions.append(s)

    console.print(f"Qualifying sessions: [bold]{len(sessions)}[/bold]")

    # Build index
    db_path, tmpdir = build_ariadne_index(MANIFEST_PATH)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    index_names = {r[0] for r in conn.execute("SELECT name FROM models").fetchall()}
    console.print(f"Indexed models: {len(index_names)}")

    cfg = CapsuleConfig()
    builder = CapsuleBuilder(conn, cfg)

    # Detail table
    detail = Table(title="A/B: Capsule alone vs Discover+Capsule", border_style="blue", show_lines=True)
    detail.add_column("Session", style="dim", max_width=12)
    detail.add_column("Task", max_width=40)
    detail.add_column("Agent\nModels", justify="right")
    detail.add_column("A: Entry(3)\nOverlap", justify="right")
    detail.add_column("B: Discover\nOverlap", justify="right")
    detail.add_column("Δ", justify="right")

    results_a = []
    results_b = []

    for s in sessions:
        focus_model = _detect_focus_model(s.task, index_names)
        agent_in_idx = s.models_explored & index_names
        if not agent_in_idx:
            continue

        # ── A: Capsule with entry_models from first 3 calls (current best) ────
        early_models = _collect_models_from_calls(s.context_calls, 3, index_names)
        entry_a = sorted(early_models - {focus_model} if focus_model else early_models)
        try:
            cap_a = builder.build(
                task=s.task, focus_model=focus_model,
                entry_models=entry_a or None, token_budget=10000,
            )
        except Exception:
            continue
        models_a = _capsule_model_names(cap_a) & index_names
        overlap_a = agent_in_idx & models_a
        pct_a = len(overlap_a) / len(agent_in_idx) * 100

        # ── B: discover_models → intersect with agent models → capsule ────────
        # Simulate: agent calls discover_models, then picks relevant ones
        try:
            discovered = builder.discover(
                task=s.task, focus_model=focus_model,
                entry_models=entry_a or None, limit=40,
            )
        except Exception:
            results_a.append(pct_a)
            results_b.append(pct_a)
            continue

        discovered_names = {m["name"] for m in discovered}

        # Simulate agent picking models from discovery that it would actually need
        # (intersection with what agent explored = ideal pick from discovery)
        agent_picks = agent_in_idx & discovered_names
        # Also include the early entry_models
        all_entry_b = sorted((set(entry_a) | agent_picks) - {focus_model} if focus_model else set(entry_a) | agent_picks)

        try:
            cap_b = builder.build(
                task=s.task, focus_model=focus_model,
                entry_models=all_entry_b[:cfg.max_pivots] or None,
                token_budget=10000,
            )
        except Exception:
            results_a.append(pct_a)
            results_b.append(pct_a)
            continue

        models_b = _capsule_model_names(cap_b) & index_names
        # Also count discovered names as "covered" (agent sees them in discovery)
        models_b_total = models_b | discovered_names
        overlap_b = agent_in_idx & models_b_total
        pct_b = len(overlap_b) / len(agent_in_idx) * 100

        results_a.append(pct_a)
        results_b.append(pct_b)

        delta = pct_b - pct_a
        if delta > 0:
            delta_str = f"[green]+{delta:.0f}pp[/green]"
            b_str = f"[green]{pct_b:.0f}%[/green]"
        elif delta < 0:
            delta_str = f"[red]{delta:.0f}pp[/red]"
            b_str = f"[red]{pct_b:.0f}%[/red]"
        else:
            delta_str = "0pp"
            b_str = f"{pct_b:.0f}%"

        detail.add_row(
            s.session_id[:12],
            _truncate(s.task, 55),
            str(len(agent_in_idx)),
            f"{pct_a:.0f}%",
            b_str,
            delta_str,
        )

    console.print(detail)

    # Summary
    n = len(results_a)
    if n == 0:
        console.print("[red]No comparisons completed.[/red]")
        return

    avg_a = sum(results_a) / n
    avg_b = sum(results_b) / n
    improved = sum(1 for a, b in zip(results_a, results_b) if b > a)
    same = sum(1 for a, b in zip(results_a, results_b) if b == a)

    summary = Table(title="Summary", border_style="green")
    summary.add_column("Scenario", style="bold")
    summary.add_column("Avg Overlap", justify="right")
    summary.add_column("Δ vs A", justify="right")
    summary.add_column("Sessions Improved", justify="right")

    summary.add_row("A: Capsule + entry(3)", f"{avg_a:.1f}%", "-", "-")

    delta = avg_b - avg_a
    style = "[green]" if delta > 0 else "[red]"
    summary.add_row(
        "B: Discover + Capsule",
        f"{avg_b:.1f}%",
        f"{style}+{delta:.1f}pp[/{style[1:]}" if delta > 0 else f"{delta:.1f}pp",
        f"{improved}/{n} ({same} same)",
    )

    console.print(summary)

    # Split by focus model presence
    has_focus_a, has_focus_b = [], []
    no_focus_a, no_focus_b = [], []
    for i, s in enumerate(sessions[:n]):
        f = _detect_focus_model(s.task, index_names)
        if f:
            has_focus_a.append(results_a[i])
            has_focus_b.append(results_b[i])
        else:
            no_focus_a.append(results_a[i])
            no_focus_b.append(results_b[i])

    split = Table(title="Split: With vs Without Focus Model", border_style="magenta")
    split.add_column("Group", style="bold")
    split.add_column("N", justify="right")
    split.add_column("A: Entry(3)", justify="right")
    split.add_column("B: Discover", justify="right")
    split.add_column("Δ", justify="right")

    for label, ga, gb in [
        ("WITH focus model", has_focus_a, has_focus_b),
        ("WITHOUT focus (vague)", no_focus_a, no_focus_b),
    ]:
        if not ga:
            split.add_row(label, "0", "-", "-", "-")
            continue
        gn = len(ga)
        a_avg = sum(ga) / gn
        b_avg = sum(gb) / gn
        d = b_avg - a_avg
        style = "[green]" if d > 0 else "[red]"
        split.add_row(
            label, str(gn),
            f"{a_avg:.1f}%", f"{b_avg:.1f}%",
            f"{style}+{d:.1f}pp[/{style[1:]}" if d > 0 else f"{d:.1f}pp",
        )

    console.print(split)

    console.print(Panel(
        "[bold]What this measures[/bold]\n\n"
        "  A: Capsule with entry_models from the agent's first 3 context calls.\n"
        "  B: discover_models(limit=40) first, then capsule with agent-relevant\n"
        "     models picked from discovery as entry_models.\n\n"
        "  'B' counts models as 'covered' if they appear in either the capsule\n"
        "  OR the discovery list (since the agent sees the names in discovery).\n\n"
        "  This simulates the recommended 2-step workflow:\n"
        "  1. discover_models → see 40 model names cheaply\n"
        "  2. get_context_capsule(entry_models=[...]) → full details for key models",
        border_style="dim",
    ))

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    conn.close()


if __name__ == "__main__":
    main()
