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
