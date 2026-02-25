#!/usr/bin/env python3
"""Standalone Ariadne benchmark script.

No pytest-benchmark dependency required.

Usage:
    python benchmarks/run.py tests/fixtures/manifest_real.json

Output:
    Rich table with P50/P95/P99/Max for each operation, plus:
    - Index build: total time + models/sec throughput
    - Token reduction: capsule tokens vs naive tokens (percentage saved)
    - Memory: peak RSS during indexing (via resource.getrusage)
"""

from __future__ import annotations

import resource
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project src to path if running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ariadne_dbt.capsule import CapsuleBuilder
from ariadne_dbt.config import CapsuleConfig
from ariadne_dbt.graph import GraphOps
from ariadne_dbt.indexer import Indexer
from ariadne_dbt.patterns import PatternExtractor
from ariadne_dbt.search import HybridSearch

console = Console()

ROUNDS = 10


def _timeit(fn: Callable, rounds: int = ROUNDS) -> list[float]:
    """Run fn `rounds` times and return list of elapsed seconds."""
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _percentile(data: list[float], pct: float) -> float:
    data = sorted(data)
    k = (len(data) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


def _ms(s: float) -> str:
    return f"{s * 1000:.1f}ms"


def main(manifest_path: Path) -> None:
    if not manifest_path.exists():
        console.print(f"[red]Error:[/red] manifest not found at {manifest_path}")
        sys.exit(1)

    console.print(Panel(
        f"[bold]Ariadne Benchmark[/bold]\n"
        f"Manifest: [cyan]{manifest_path}[/cyan]\n"
        f"Rounds per operation: [cyan]{ROUNDS}[/cyan]",
        border_style="blue",
    ))

    results: list[tuple[str, list[float], str]] = []

    # ── 1. Index build ────────────────────────────────────────────────────────
    console.print("  [dim]Benchmarking index build...[/dim]")
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "ariadne.db"

        t0 = time.perf_counter()
        with Indexer(db_path) as idx:
            idx.index_manifest(manifest_path)
        index_time = time.perf_counter() - t0

        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_delta_mb = (rss_after - rss_before) / (1024 * 1024)  # macOS returns bytes

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        model_count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        throughput = model_count / index_time if index_time > 0 else 0

        # Warm up connection for subsequent benchmarks
        conn.close()

    console.print(f"    Index: {model_count} models in {index_time * 1000:.0f}ms "
                  f"({throughput:.0f} models/sec), +{rss_delta_mb:.1f} MB RSS")

    # Rebuild DB for benchmarks
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "ariadne.db"
        with Indexer(db_path) as idx:
            idx.index_manifest(manifest_path)

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        model_count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]

        # ── 2. Capsule build ──────────────────────────────────────────────────
        console.print("  [dim]Benchmarking capsule build...[/dim]")
        cfg = CapsuleConfig()
        builder = CapsuleBuilder(conn, cfg)
        capsule_times = _timeit(
            lambda: builder.build(task="debug failing test on revenue model", token_budget=8000)
        )
        results.append(("Capsule build", capsule_times, "<500ms"))

        # ── 3. Search query ───────────────────────────────────────────────────
        console.print("  [dim]Benchmarking search query...[/dim]")
        search = HybridSearch(conn)
        search_times = _timeit(
            lambda: search.search("revenue customer order", intent="explore", limit=10)
        )
        results.append(("Search query", search_times, "<100ms"))

        # ── 4. Lineage traversal ──────────────────────────────────────────────
        console.print("  [dim]Benchmarking lineage traversal...[/dim]")
        graph = GraphOps(conn)
        top = graph.high_centrality_models(limit=1)
        if top:
            uid = top[0][0]
            lineage_times = _timeit(lambda: (
                graph.upstream(uid, depth=3),
                graph.downstream(uid, depth=3),
            ))
            results.append(("Lineage traversal (depth=3)", lineage_times, "<50ms"))

        # ── 5. Pattern extraction ─────────────────────────────────────────────
        console.print("  [dim]Benchmarking pattern extraction...[/dim]")
        extractor = PatternExtractor(conn)
        pattern_times = _timeit(lambda: (
            extractor.get_stats(),
            extractor.get_patterns(),
        ))
        results.append(("Pattern extraction", pattern_times, "<200ms"))

        # ── Token reduction ───────────────────────────────────────────────────
        capsule = builder.build(task="explore the project", token_budget=10000)
        rows = conn.execute(
            "SELECT COALESCE(description,'') || ' ' || COALESCE(raw_code,'') FROM models"
        ).fetchall()
        naive_tokens = sum(len(r[0].split()) for r in rows)
        reduction_pct = (1 - capsule.token_estimate / naive_tokens) * 100 if naive_tokens else 0

        conn.close()

    # ── Results table ─────────────────────────────────────────────────────────
    table = Table(title="Ariadne Benchmark Results", border_style="green")
    table.add_column("Operation", style="bold")
    table.add_column("P50", justify="right")
    table.add_column("P95", justify="right", style="yellow")
    table.add_column("P99", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Target", justify="right", style="dim")
    table.add_column("Pass", justify="center")

    _TARGET_MS = {
        "Capsule build": 500,
        "Search query": 100,
        "Lineage traversal (depth=3)": 50,
        "Pattern extraction": 200,
    }

    for op, times, target in results:
        p50 = _percentile(times, 50)
        p95 = _percentile(times, 95)
        p99 = _percentile(times, 99)
        mx = max(times)
        target_ms = _TARGET_MS.get(op, 9999)
        passed = "[green]✓[/green]" if p95 * 1000 < target_ms else "[red]✗[/red]"
        table.add_row(op, _ms(p50), _ms(p95), _ms(p99), _ms(mx), target, passed)

    console.print()
    console.print(table)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = Table(title="Summary", border_style="blue")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Models indexed", str(model_count))
    summary.add_row("Index build time", f"{index_time * 1000:.0f}ms")
    summary.add_row("Throughput", f"{throughput:.0f} models/sec")
    summary.add_row("Peak RSS delta", f"{rss_delta_mb:.1f} MB")
    summary.add_row("Capsule tokens", str(capsule.token_estimate))
    summary.add_row("Naive tokens", str(naive_tokens))
    summary.add_row("Token reduction", f"{reduction_pct:.1f}%")
    console.print(summary)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("Usage: python benchmarks/run.py <manifest_path>")
        sys.exit(1)
    main(Path(sys.argv[1]))
