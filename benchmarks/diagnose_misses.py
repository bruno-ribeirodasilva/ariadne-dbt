#!/usr/bin/env python3
"""Diagnose WHY the capsule misses models the agent actually needed.

Categories:
  1. DAG-reachable: in the DAG neighborhood of pivots (depth 4) but cut by token budget
  2. DAG-distant: reachable at depth 5-8 but not at depth 4
  3. DAG-disconnected: not reachable from any pivot at depth 8
  4. Reference models: agent read them for pattern/style reference, not direct dependency

Usage:
    .venv/bin/python benchmarks/diagnose_misses.py
"""

from __future__ import annotations

import sqlite3
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ariadne_dbt.capsule import CapsuleBuilder
from ariadne_dbt.config import CapsuleConfig
from ariadne_dbt.graph import GraphOps
from benchmarks.session_analysis import (
    MIN_CONTEXT_CALLS,
    SESSIONS_DIR,
    build_ariadne_index,
    parse_session,
    _detect_focus_model,
    _is_dbt_relevant_task,
    _truncate,
)

# Use the real production manifest for diagnosis
MANIFEST_PATH = Path("/Users/taxfix/projects/data-dbt-models/target/manifest.json")
from benchmarks.ab_entry_models import _capsule_model_names, _collect_models_from_calls

console = Console()


def _dag_reachable(graph: GraphOps, pivot_ids: list[str], target_uid: str, max_depth: int = 8) -> int | None:
    """Return the minimum DAG distance from any pivot to target, or None if unreachable."""
    min_dist = None
    for pid in pivot_ids:
        for uid, dist in graph.upstream(pid, depth=max_depth):
            if uid == target_uid and (min_dist is None or dist < min_dist):
                min_dist = dist
        for uid, dist in graph.downstream(pid, depth=max_depth):
            if uid == target_uid and (min_dist is None or dist < min_dist):
                min_dist = dist
    return min_dist


def main():
    if not SESSIONS_DIR.exists():
        console.print(f"[red]Sessions dir not found:[/red] {SESSIONS_DIR}")
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
    # Build name→uid lookup
    name_to_uid = {}
    for row in conn.execute("SELECT name, unique_id FROM models").fetchall():
        name_to_uid[row[0]] = row[1]

    console.print(f"Indexed models: {len(index_names)}")

    cfg = CapsuleConfig()
    builder = CapsuleBuilder(conn, cfg)
    graph = GraphOps(conn)

    # Accumulators
    total_missed = 0
    total_agent_models = 0
    category_counts = Counter()  # dag_close, dag_distant, disconnected
    category_examples: dict[str, list[str]] = {"dag_close": [], "dag_distant": [], "disconnected": []}

    # Per-session detail
    detail = Table(title="Miss Diagnosis per Session", border_style="blue", show_lines=True)
    detail.add_column("Session", style="dim", max_width=12)
    detail.add_column("Task", max_width=40)
    detail.add_column("Agent\nModels", justify="right")
    detail.add_column("In\nCapsule", justify="right")
    detail.add_column("Missed", justify="right")
    detail.add_column("DAG\nClose\n(≤4)", justify="right", style="yellow")
    detail.add_column("DAG\nDistant\n(5-8)", justify="right", style="cyan")
    detail.add_column("Discon-\nnected", justify="right", style="red")
    detail.add_column("Missed Model Names", max_width=40, style="dim")

    for s in sessions:
        focus_model = _detect_focus_model(s.task, index_names)
        agent_in_idx = s.models_explored & index_names
        if not agent_in_idx:
            continue

        # Build capsule with entry_models (from first 3 calls) to match A/B benchmark
        early_models = _collect_models_from_calls(s.context_calls, 3, index_names)
        entry_list = sorted(early_models - {focus_model} if focus_model else early_models)

        try:
            capsule = builder.build(
                task=s.task,
                focus_model=focus_model,
                entry_models=entry_list or None,
                token_budget=10000,
            )
        except Exception:
            continue

        capsule_models = _capsule_model_names(capsule) & index_names
        missed = agent_in_idx - capsule_models

        if not missed:
            continue

        total_missed += len(missed)
        total_agent_models += len(agent_in_idx)

        # Get pivot unique_ids for DAG reachability check
        pivot_uids = [pm.unique_id for pm in capsule.pivot_models]

        dag_close = 0
        dag_distant = 0
        disconnected = 0
        missed_names = []

        for model_name in sorted(missed):
            uid = name_to_uid.get(model_name)
            if not uid:
                disconnected += 1
                continue

            dist = _dag_reachable(graph, pivot_uids, uid, max_depth=8)
            if dist is not None and dist <= 4:
                dag_close += 1
                category_counts["dag_close"] += 1
                if len(category_examples["dag_close"]) < 10:
                    category_examples["dag_close"].append(f"{model_name} (dist={dist}, session={s.session_id[:8]})")
            elif dist is not None:
                dag_distant += 1
                category_counts["dag_distant"] += 1
                if len(category_examples["dag_distant"]) < 10:
                    category_examples["dag_distant"].append(f"{model_name} (dist={dist}, session={s.session_id[:8]})")
            else:
                disconnected += 1
                category_counts["disconnected"] += 1
                if len(category_examples["disconnected"]) < 10:
                    category_examples["disconnected"].append(f"{model_name} (session={s.session_id[:8]})")

            missed_names.append(model_name)

        detail.add_row(
            s.session_id[:12],
            _truncate(s.task, 55),
            str(len(agent_in_idx)),
            str(len(capsule_models & agent_in_idx)),
            str(len(missed)),
            str(dag_close) if dag_close else "-",
            str(dag_distant) if dag_distant else "-",
            str(disconnected) if disconnected else "-",
            ", ".join(missed_names[:5]) + ("..." if len(missed_names) > 5 else ""),
        )

    console.print(detail)

    # Summary
    summary = Table(title="Miss Category Summary", border_style="green")
    summary.add_column("Category", style="bold")
    summary.add_column("Count", justify="right")
    summary.add_column("% of Misses", justify="right")
    summary.add_column("What This Means", max_width=50)

    total = sum(category_counts.values()) or 1
    summary.add_row(
        "DAG-close (≤4 hops)",
        str(category_counts["dag_close"]),
        f"{category_counts['dag_close'] / total * 100:.0f}%",
        "In DAG neighborhood but cut by token budget or DAG depth setting",
    )
    summary.add_row(
        "DAG-distant (5-8 hops)",
        str(category_counts["dag_distant"]),
        f"{category_counts['dag_distant'] / total * 100:.0f}%",
        "Reachable but far from pivots; agent explored broadly",
    )
    summary.add_row(
        "Disconnected (>8 hops)",
        str(category_counts["disconnected"]),
        f"{category_counts['disconnected'] / total * 100:.0f}%",
        "No DAG path to any pivot; pattern/reference models or wrong pivots",
    )
    summary.add_row(
        "[bold]Total missed[/bold]",
        f"[bold]{total}[/bold]",
        "",
        f"Out of {total_agent_models} agent-explored models across all sessions",
    )

    console.print(summary)

    # Examples
    for cat, label in [
        ("dag_close", "DAG-Close Misses (budget/depth issue)"),
        ("dag_distant", "DAG-Distant Misses (broad exploration)"),
        ("disconnected", "Disconnected Misses (no DAG path)"),
    ]:
        examples = category_examples[cat]
        if examples:
            console.print(f"\n[bold]{label}:[/bold]")
            for ex in examples:
                console.print(f"  - {ex}")

    # Actionable recommendations
    console.print(Panel(
        "[bold]Recommendations based on miss patterns[/bold]\n\n"
        f"  DAG-close ({category_counts['dag_close']}/{total} = {category_counts['dag_close']/total*100:.0f}%):\n"
        "    → Increase token budget or upstream/downstream depth settings\n"
        "    → Better token allocation: give more budget to upstream models\n"
        "    → Lazy loading: return model names first, let agent request details\n\n"
        f"  DAG-distant ({category_counts['dag_distant']}/{total} = {category_counts['dag_distant']/total*100:.0f}%):\n"
        "    → Multi-pivot discovery: agents explore multiple unrelated areas\n"
        "    → Iterative capsule: call capsule multiple times as task evolves\n\n"
        f"  Disconnected ({category_counts['disconnected']}/{total} = {category_counts['disconnected']/total*100:.0f}%):\n"
        "    → These are likely reference/pattern models (agent reads for style)\n"
        "    → Or the capsule picked wrong pivots entirely\n"
        "    → Consider: 'similar_models' section with more models at minimal detail\n",
        border_style="yellow",
    ))

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    conn.close()


if __name__ == "__main__":
    main()
