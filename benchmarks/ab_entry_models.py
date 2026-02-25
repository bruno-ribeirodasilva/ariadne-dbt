#!/usr/bin/env python3
"""A/B comparison: capsule with vs without entry_models fed back.

Simulates the workflow where an agent discovers models early (e.g., from
a PR diff or initial file reads) and feeds them back via entry_models.

Two scenarios per session:
  A (baseline): get_context_capsule(task, focus_model=...)          — old behavior
  B (entry_models): get_context_capsule(task, entry_models=[...])   — new behavior
     where entry_models = models the agent discovered in first N context calls

Usage:
    .venv/bin/python benchmarks/ab_entry_models.py
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
    MANIFEST_PATH,
    MIN_CONTEXT_CALLS,
    SESSIONS_DIR,
    build_ariadne_index,
    parse_session,
    _detect_focus_model,
    _is_dbt_relevant_task,
    _truncate,
)

console = Console()

# How many early context calls to use for entry_models discovery.
# 3 ≈ agent reads PR diff + a couple of greps.
EARLY_CALL_COUNTS = [3, 5]


def _collect_models_from_calls(calls, limit, index_names):
    """Collect model names from the first `limit` context calls that exist in the index."""
    found = set()
    for tc in calls[:limit]:
        for m in tc.models_referenced:
            if m in index_names:
                found.add(m)
    return found


def _capsule_model_names(capsule):
    names = set()
    for pm in capsule.pivot_models:
        names.add(pm.name)
    for um in capsule.upstream_models:
        names.add(um.name)
    for dm in capsule.downstream_models:
        names.add(dm.name)
    for n in capsule.similar_models:
        names.add(n)
    return names


def main():
    if not SESSIONS_DIR.exists():
        console.print(f"[red]Sessions dir not found:[/red] {SESSIONS_DIR}")
        sys.exit(1)

    # ── Parse sessions ───────────────────────────────────────────────────────
    session_files = sorted(SESSIONS_DIR.glob("*.jsonl"))
    sessions = []
    for sf in session_files:
        s = parse_session(sf)
        if s and _is_dbt_relevant_task(s.task) and len(s.context_calls) >= MIN_CONTEXT_CALLS and s.models_explored:
            sessions.append(s)

    console.print(f"Qualifying sessions: [bold]{len(sessions)}[/bold]")

    # ── Build index ──────────────────────────────────────────────────────────
    db_path, tmpdir = build_ariadne_index(MANIFEST_PATH)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    index_names = {r[0] for r in conn.execute("SELECT name FROM models").fetchall()}
    console.print(f"Indexed models: {len(index_names)}")

    cfg = CapsuleConfig()
    builder = CapsuleBuilder(conn, cfg)

    # ── Run A/B per session ──────────────────────────────────────────────────

    # Per-session detail table
    detail = Table(
        title="A/B: Baseline vs entry_models fed back",
        border_style="blue",
        show_lines=True,
    )
    detail.add_column("Session", style="dim", max_width=12, no_wrap=True)
    detail.add_column("Task", max_width=40)
    detail.add_column("Focus", max_width=15, style="cyan")
    detail.add_column("Agent\nModels", justify="right")
    detail.add_column("Baseline\nOverlap", justify="right")
    for n_calls in EARLY_CALL_COUNTS:
        detail.add_column(f"Entry({n_calls})\nOverlap", justify="right")
    detail.add_column("Entry(all)\nOverlap", justify="right")
    detail.add_column("Conf", justify="center")

    # Accumulators for summary
    results = {
        "baseline": [],
        "all": [],
    }
    for n_calls in EARLY_CALL_COUNTS:
        results[f"early_{n_calls}"] = []

    for s in sessions:
        focus_model = _detect_focus_model(s.task, index_names)
        agent_in_idx = s.models_explored & index_names
        if not agent_in_idx:
            continue

        # ── A: Baseline (old behavior) ───────────────────────────────────────
        try:
            cap_a = builder.build(task=s.task, focus_model=focus_model, token_budget=10000)
        except Exception:
            continue
        models_a = _capsule_model_names(cap_a) & index_names
        overlap_a = agent_in_idx & models_a
        pct_a = len(overlap_a) / len(agent_in_idx) * 100

        # ── B variants: entry_models from first N calls ──────────────────────
        row_values = [
            s.session_id[:12],
            _truncate(s.task, 55),
            focus_model or "-",
            str(len(agent_in_idx)),
            f"{pct_a:.0f}%",
        ]
        results["baseline"].append(pct_a)

        last_confidence = cap_a.confidence
        for n_calls in EARLY_CALL_COUNTS:
            early_models = _collect_models_from_calls(s.context_calls, n_calls, index_names)
            entry_list = sorted(early_models - {focus_model} if focus_model else early_models)
            try:
                cap_b = builder.build(
                    task=s.task,
                    focus_model=focus_model,
                    entry_models=entry_list or None,
                    token_budget=10000,
                )
            except Exception:
                results[f"early_{n_calls}"].append(pct_a)
                row_values.append(f"{pct_a:.0f}%")
                continue
            models_b = _capsule_model_names(cap_b) & index_names
            overlap_b = agent_in_idx & models_b
            pct_b = len(overlap_b) / len(agent_in_idx) * 100
            results[f"early_{n_calls}"].append(pct_b)
            last_confidence = cap_b.confidence

            delta = pct_b - pct_a
            if delta > 0:
                row_values.append(f"[green]{pct_b:.0f}% (+{delta:.0f})[/green]")
            elif delta < 0:
                row_values.append(f"[red]{pct_b:.0f}% ({delta:.0f})[/red]")
            else:
                row_values.append(f"{pct_b:.0f}%")

        # ── B-all: entry_models from ALL context calls ───────────────────────
        all_models = _collect_models_from_calls(s.context_calls, len(s.context_calls), index_names)
        entry_all = sorted(all_models - {focus_model} if focus_model else all_models)
        try:
            cap_all = builder.build(
                task=s.task,
                focus_model=focus_model,
                entry_models=entry_all or None,
                token_budget=10000,
            )
        except Exception:
            results["all"].append(pct_a)
            row_values.append(f"{pct_a:.0f}%")
            row_values.append(last_confidence)
            detail.add_row(*row_values)
            continue

        models_all = _capsule_model_names(cap_all) & index_names
        overlap_all = agent_in_idx & models_all
        pct_all = len(overlap_all) / len(agent_in_idx) * 100
        results["all"].append(pct_all)
        last_confidence = cap_all.confidence

        delta = pct_all - pct_a
        if delta > 0:
            row_values.append(f"[green]{pct_all:.0f}% (+{delta:.0f})[/green]")
        elif delta < 0:
            row_values.append(f"[red]{pct_all:.0f}% ({delta:.0f})[/red]")
        else:
            row_values.append(f"{pct_all:.0f}%")

        row_values.append(last_confidence)
        detail.add_row(*row_values)

    console.print(detail)

    # ── Summary ──────────────────────────────────────────────────────────────

    n = len(results["baseline"])
    if n == 0:
        console.print("[red]No comparisons completed.[/red]")
        return

    summary = Table(title="Summary: Average Overlap %", border_style="green")
    summary.add_column("Scenario", style="bold")
    summary.add_column("Avg Overlap", justify="right")
    summary.add_column("Δ vs Baseline", justify="right")
    summary.add_column("Sessions Improved", justify="right")

    avg_base = sum(results["baseline"]) / n
    summary.add_row("Baseline (task text only)", f"{avg_base:.1f}%", "-", "-")

    for key in [f"early_{nc}" for nc in EARLY_CALL_COUNTS] + ["all"]:
        vals = results[key]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        delta = avg - avg_base
        improved = sum(1 for b, e in zip(results["baseline"], vals) if e > b)
        label = {
            "all": "entry_models = ALL context calls",
        }
        for nc in EARLY_CALL_COUNTS:
            label[f"early_{nc}"] = f"entry_models from first {nc} calls"

        style = "[green]" if delta > 0 else "[red]"
        summary.add_row(
            label[key],
            f"{avg:.1f}%",
            f"{style}+{delta:.1f}pp[/{style[1:]}" if delta > 0 else f"{delta:.1f}pp",
            f"{improved}/{n}",
        )

    console.print(summary)

    # ── Split by focus model presence ────────────────────────────────────────

    # Rebuild per-session for splitting
    has_focus = []
    no_focus = []
    for i, s in enumerate(sessions[:n]):
        focus = _detect_focus_model(s.task, index_names)
        row = {
            "baseline": results["baseline"][i],
            "all": results["all"][i],
        }
        for nc in EARLY_CALL_COUNTS:
            row[f"early_{nc}"] = results[f"early_{nc}"][i]
        if focus:
            has_focus.append(row)
        else:
            no_focus.append(row)

    split = Table(title="Split: With vs Without Focus Model in Task", border_style="magenta")
    split.add_column("Group", style="bold")
    split.add_column("N", justify="right")
    split.add_column("Baseline", justify="right")
    split.add_column(f"Entry({EARLY_CALL_COUNTS[0]})", justify="right")
    split.add_column("Entry(all)", justify="right")
    split.add_column("Δ (all)", justify="right")

    for label, group in [("WITH focus model", has_focus), ("WITHOUT focus model (vague)", no_focus)]:
        if not group:
            split.add_row(label, "0", "-", "-", "-", "-")
            continue
        gn = len(group)
        b = sum(r["baseline"] for r in group) / gn
        e3 = sum(r[f"early_{EARLY_CALL_COUNTS[0]}"] for r in group) / gn
        ea = sum(r["all"] for r in group) / gn
        d = ea - b
        style = "[green]" if d > 0 else "[red]"
        split.add_row(
            label,
            str(gn),
            f"{b:.1f}%",
            f"{e3:.1f}%",
            f"{ea:.1f}%",
            f"{style}+{d:.1f}pp[/{style[1:]}" if d > 0 else f"{d:.1f}pp",
        )

    console.print(split)

    console.print(Panel(
        "[bold]What this measures[/bold]\n\n"
        "  Baseline: capsule sees only the task text + focus_model from task.\n"
        f"  Entry({EARLY_CALL_COUNTS[0]}): capsule also gets models from the agent's first "
        f"{EARLY_CALL_COUNTS[0]} context calls.\n"
        "  Entry(all): capsule gets ALL models the agent found during context-gathering.\n\n"
        "  'Entry(all)' is the upper bound — it assumes the agent feeds back everything\n"
        f"  it discovered. 'Entry({EARLY_CALL_COUNTS[0]})' is realistic — "
        "it simulates reading a PR diff.\n\n"
        "  The key question: does the vague-task group improve enough to matter?",
        border_style="dim",
    ))

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    conn.close()


if __name__ == "__main__":
    main()
