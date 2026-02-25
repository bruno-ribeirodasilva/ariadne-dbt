"""CLI commands: init, serve, sync."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import load_config
from .generator import ContextGenerator
from .indexer import Indexer

app = typer.Typer(
    name="ariadne",
    help="Intelligent context server for AI agents working with dbt projects (Ariadne).",
    add_completion=False,
)
console = Console()


@app.command()
def init(
    project_root: Optional[Path] = typer.Argument(
        None, help="Path to dbt project root (default: current directory)"
    ),
    targets: list[str] = typer.Option(
        ["claude_code"],
        "--target",
        "-t",
        help="Agent targets to generate rules for (claude_code, cursor, windsurf)",
    ),
    skip_generate: bool = typer.Option(
        False, "--skip-generate", help="Skip .md file generation"
    ),
) -> None:
    """Initialize Ariadne for a dbt project.

    This command will:
    1. Parse manifest.json (and catalog.json / run_results.json if present)
    2. Build the SQLite index
    3. Generate .claude/CLAUDE.md, .claude/skills/, .claude/context/ files
    """
    cfg = load_config(project_root)

    if not cfg.manifest_path.exists():
        console.print(
            f"[red]Error:[/red] manifest.json not found at {cfg.manifest_path}\n"
            "Run [bold]dbt compile[/bold] or [bold]dbt build[/bold] first.",
        )
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Ariadne — init[/bold]\n"
        f"Project root: [cyan]{cfg.dbt_project_root}[/cyan]\n"
        f"Manifest: [cyan]{cfg.manifest_path}[/cyan]\n"
        f"Index: [cyan]{cfg.absolute_index_path}[/cyan]",
        border_style="blue",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Step 1: Index
        task = progress.add_task("Indexing manifest.json...", total=None)
        with Indexer(cfg.absolute_index_path) as idx:
            idx.index_manifest(cfg.manifest_path)

            if cfg.catalog_path.exists():
                progress.update(task, description="Indexing catalog.json...")
                idx.index_catalog(cfg.catalog_path)

            if cfg.run_results_path.exists():
                progress.update(task, description="Indexing run_results.json...")
                idx.index_run_results(cfg.run_results_path)

            conn = idx.conn

            # Print stats
            model_count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
            source_count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            test_count = conn.execute("SELECT COUNT(*) FROM tests").fetchone()[0]

            # Step 2: Generate .md files
            if not skip_generate:
                progress.update(task, description="Generating context files...")
                generator = ContextGenerator(conn)
                written = generator.generate_all(cfg.dbt_project_root, targets=targets)
            else:
                written = []

        progress.update(task, description="Done!", completed=True)

    # Summary table
    table = Table(title="Index Summary", border_style="green")
    table.add_column("Item", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("Models indexed", str(model_count))
    table.add_row("Sources indexed", str(source_count))
    table.add_row("Tests indexed", str(test_count))
    table.add_row("catalog.json", "yes" if cfg.catalog_path.exists() else "no (optional)")
    table.add_row("run_results.json", "yes" if cfg.run_results_path.exists() else "no (optional)")
    console.print(table)

    if written:
        console.print("\n[green]Generated files:[/green]")
        for f in written:
            rel = f.relative_to(cfg.dbt_project_root) if f.is_relative_to(cfg.dbt_project_root) else f
            console.print(f"  [cyan]{rel}[/cyan]")

    console.print("\n[bold green]Next steps:[/bold green]")
    console.print("  1. Start the MCP server: [bold]ariadne serve[/bold]")
    console.print("  2. Add to Claude Code .claude/settings.json:")
    console.print("""     {
       "mcpServers": {
         "ariadne": {
           "command": "ariadne",
           "args": ["serve"]
         }
       }
     }""")
    console.print("  3. After running dbt compile: [bold]ariadne sync[/bold]")


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Override host"),
    port: Optional[int] = typer.Option(None, help="Override port"),
    transport: str = typer.Option("stdio", "--transport", "-t",
                                   help="MCP transport: stdio (default) or sse"),
) -> None:
    """Start the MCP server.

    For Claude Code and Cursor, use stdio transport (default).
    For browser-based clients, use sse transport.
    """
    cfg = load_config()

    if not cfg.absolute_index_path.exists():
        console.print(
            "[red]Error:[/red] Index not found. Run [bold]ariadne init[/bold] first."
        )
        raise typer.Exit(1)

    from .server import create_server
    mcp_server = create_server(cfg)

    if transport == "sse":
        h = host or cfg.server.host
        p = port or cfg.server.port
        console.print(f"Starting SSE server on [cyan]http://{h}:{p}[/cyan]")
        mcp_server.run(transport="sse", host=h, port=p)
    else:
        # stdio — used by Claude Code, Cursor, etc.
        mcp_server.run(transport="stdio")


@app.command()
def sync(
    project_root: Optional[Path] = typer.Argument(None),
    skip_generate: bool = typer.Option(False, "--skip-generate"),
) -> None:
    """Re-index and update .md files after dbt compile.

    Run this whenever you run `dbt compile` or `dbt build` to keep the
    context engine synchronized.
    """
    cfg = load_config(project_root)

    if not cfg.manifest_path.exists():
        console.print(
            f"[red]Error:[/red] manifest.json not found at {cfg.manifest_path}\n"
            "Run [bold]dbt compile[/bold] first."
        )
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Syncing index...", total=None)

        with Indexer(cfg.absolute_index_path) as idx:
            idx.index_manifest(cfg.manifest_path)
            if cfg.catalog_path.exists():
                progress.update(task, description="Updating catalog data...")
                idx.index_catalog(cfg.catalog_path)
            if cfg.run_results_path.exists():
                progress.update(task, description="Updating run results...")
                idx.index_run_results(cfg.run_results_path)

            if not skip_generate:
                progress.update(task, description="Updating context files...")
                generator = ContextGenerator(idx.conn)
                cfg_targets = cfg.generator.targets
                generator.generate_all(cfg.dbt_project_root, targets=cfg_targets)

        progress.update(task, description="Sync complete!", completed=True)

    console.print("[bold green]Sync complete.[/bold green] Index and context files are up to date.")


@app.command()
def stats(
    project_root: Optional[Path] = typer.Argument(None),
) -> None:
    """Show index statistics for the current project."""
    cfg = load_config(project_root)

    if not cfg.absolute_index_path.exists():
        console.print(
            "[red]Error:[/red] Index not found. Run [bold]ariadne init[/bold] first."
        )
        raise typer.Exit(1)

    import sqlite3
    conn = sqlite3.connect(str(cfg.absolute_index_path))
    conn.row_factory = sqlite3.Row

    from .patterns import PatternExtractor
    extractor = PatternExtractor(conn)
    project_stats = extractor.get_stats()

    table = Table(title=f"Project: {project_stats.project_name or 'Unknown'}", border_style="blue")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Adapter", project_stats.adapter_type)
    table.add_row("Total models", str(project_stats.model_count))
    table.add_row("  Staging", str(project_stats.staging_count))
    table.add_row("  Intermediate", str(project_stats.intermediate_count))
    table.add_row("  Marts", str(project_stats.marts_count))
    table.add_row("Sources", str(project_stats.source_count))
    table.add_row("Tests", str(project_stats.test_count))
    table.add_row("Column coverage", f"{project_stats.test_coverage_pct}%")
    table.add_row("Project macros", str(project_stats.project_macro_count))
    table.add_row("Exposures", str(project_stats.exposure_count))
    console.print(table)
    conn.close()


@app.command()
def usage(
    project_root: Optional[Path] = typer.Argument(None),
    days: int = typer.Option(30, "--days", "-d", help="Look-back window in days"),
    recent: bool = typer.Option(False, "--recent", "-r", help="Show 20 most recent queries"),
) -> None:
    """Show Ariadne usage statistics.

    Displays how the MCP server has been used: call counts by tool and intent,
    average token estimates, top queried models, and daily trends.
    """
    import sqlite3 as _sqlite3

    from .usage import UsageLogger

    cfg = load_config(project_root)

    if not cfg.absolute_index_path.exists():
        console.print(
            "[red]Error:[/red] Index not found. Run [bold]ariadne init[/bold] first."
        )
        raise typer.Exit(1)

    conn = _sqlite3.connect(str(cfg.absolute_index_path))
    conn.row_factory = _sqlite3.Row
    logger = UsageLogger(conn)

    if recent:
        rows = logger.recent_queries(limit=20)
        table = Table(title="Recent Queries (last 20)", border_style="blue")
        table.add_column("ID", style="dim")
        table.add_column("Time (UTC)", style="dim")
        table.add_column("Tool")
        table.add_column("Intent")
        table.add_column("Task / Query", max_width=50)
        table.add_column("Tokens", justify="right")
        table.add_column("ms", justify="right")
        table.add_column("★", justify="center")
        for r in rows:
            table.add_row(
                str(r["id"]),
                (r["ts"] or "")[:16].replace("T", " "),
                r["tool_name"],
                r["intent"] or "",
                (r["task_text"] or "")[:50],
                str(r["token_estimate"] or ""),
                str(r["duration_ms"] or ""),
                str(r["rating"] or ""),
            )
        console.print(table)
        conn.close()
        return

    s = logger.get_stats(days=days)
    conn.close()

    if s["total_calls"] == 0:
        console.print(
            f"[yellow]No usage data for the last {days} days.[/yellow]\n"
            "Start the MCP server and run some queries to populate the log."
        )
        return

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = Table(
        title=f"Ariadne Usage — last {days} days",
        border_style="green",
        show_header=False,
    )
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Total calls", str(s["total_calls"]))
    if s["avg_token_estimate"]:
        summary.add_row("Avg token estimate", f"{s['avg_token_estimate']:,}")
    if s["avg_duration_ms"]:
        summary.add_row("Avg duration", f"{s['avg_duration_ms']} ms")
    if s["avg_rating"]:
        summary.add_row("Avg rating", f"{s['avg_rating']} / 5.0")
    console.print(summary)

    # ── By tool ───────────────────────────────────────────────────────────────
    if s["by_tool"]:
        t = Table(title="Calls by tool", border_style="blue")
        t.add_column("Tool")
        t.add_column("Calls", justify="right")
        for tool, count in sorted(s["by_tool"].items(), key=lambda x: -x[1]):
            t.add_row(tool, str(count))
        console.print(t)

    # ── By intent ─────────────────────────────────────────────────────────────
    if s["by_intent"]:
        t = Table(title="Capsule calls by intent", border_style="blue")
        t.add_column("Intent")
        t.add_column("Calls", justify="right")
        for intent, count in sorted(s["by_intent"].items(), key=lambda x: -x[1]):
            t.add_row(intent, str(count))
        console.print(t)

    # ── Top queried models ────────────────────────────────────────────────────
    if s["top_models"]:
        t = Table(title="Top queried models", border_style="blue")
        t.add_column("Model")
        t.add_column("Calls", justify="right")
        for entry in s["top_models"]:
            t.add_row(entry["model"], str(entry["calls"]))
        console.print(t)

    # ── Daily calls ───────────────────────────────────────────────────────────
    if len(s["daily_calls"]) > 1:
        t = Table(title="Daily calls", border_style="dim")
        t.add_column("Date")
        t.add_column("Calls", justify="right")
        max_calls = max(r["calls"] for r in s["daily_calls"])
        for row in s["daily_calls"]:
            bar_len = round(row["calls"] / max_calls * 20) if max_calls else 0
            t.add_row(row["date"], f"{'█' * bar_len} {row['calls']}")
        console.print(t)
