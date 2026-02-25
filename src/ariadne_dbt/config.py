"""TOML configuration loader for dbt Context Engine."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-reuse-import]


CONFIG_FILENAME = "ariadne.toml"
DEFAULT_INDEX_PATH = ".ariadne/index.db"
DEFAULT_TARGET_DIR = "target"
DEFAULT_TOKEN_BUDGET = 10_000


@dataclass
class IntentDepth:
    upstream: int = 1
    downstream: int = 1


@dataclass
class CapsuleConfig:
    default_token_budget: int = DEFAULT_TOKEN_BUDGET
    max_pivots: int = 3
    intent_depths: dict[str, IntentDepth] = field(default_factory=lambda: {
        "debug":       IntentDepth(upstream=2, downstream=1),
        "add_feature": IntentDepth(upstream=1, downstream=2),
        "refactor":    IntentDepth(upstream=1, downstream=3),
        "test":        IntentDepth(upstream=0, downstream=0),
        "document":    IntentDepth(upstream=1, downstream=1),
        "explore":     IntentDepth(upstream=1, downstream=1),
    })


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    watch: bool = True


@dataclass
class GeneratorConfig:
    generate_claude_md: bool = True
    generate_skills: bool = True
    generate_context_snapshots: bool = True
    targets: list[str] = field(default_factory=lambda: ["claude_code"])


@dataclass
class EngineConfig:
    dbt_project_root: Path = field(default_factory=Path.cwd)
    target_dir: Path = field(default_factory=lambda: Path("target"))
    index_path: Path = field(default_factory=lambda: Path(DEFAULT_INDEX_PATH))
    server: ServerConfig = field(default_factory=ServerConfig)
    capsule: CapsuleConfig = field(default_factory=CapsuleConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    @property
    def manifest_path(self) -> Path:
        return self.dbt_project_root / self.target_dir / "manifest.json"

    @property
    def catalog_path(self) -> Path:
        return self.dbt_project_root / self.target_dir / "catalog.json"

    @property
    def run_results_path(self) -> Path:
        return self.dbt_project_root / self.target_dir / "run_results.json"

    @property
    def absolute_index_path(self) -> Path:
        p = self.index_path
        if not p.is_absolute():
            p = self.dbt_project_root / p
        return p


def load_config(search_root: Path | None = None) -> EngineConfig:
    """Load configuration from TOML file, falling back to defaults.

    Searches for ariadne.toml in ``search_root`` (or cwd) and up to
    the filesystem root.
    """
    start = Path(search_root or Path.cwd()).resolve()
    config_path: Path | None = None

    candidate = start
    while True:
        maybe = candidate / CONFIG_FILENAME
        if maybe.exists():
            config_path = maybe
            break
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    raw: dict = {}
    if config_path is not None:
        with config_path.open("rb") as f:
            raw = tomllib.load(f)

    project_root = config_path.parent if config_path else start

    project_section = raw.get("project", {})
    server_section = raw.get("server", {})
    capsule_section = raw.get("capsule", {})
    generator_section = raw.get("generator", {})

    # Intent depths
    intent_depths_raw = capsule_section.pop("intent_depths", {})
    default_depths = CapsuleConfig().intent_depths
    for intent, vals in intent_depths_raw.items():
        default_depths[intent] = IntentDepth(**vals)

    target_dir_str = project_section.get("target_dir", DEFAULT_TARGET_DIR)
    index_path_str = project_section.get("index_path", DEFAULT_INDEX_PATH)

    # dbt_project_root override
    root_override = project_section.get("dbt_project_root")
    if root_override:
        dbt_root = Path(root_override).expanduser().resolve()
    else:
        dbt_root = _find_dbt_project_root(project_root)

    return EngineConfig(
        dbt_project_root=dbt_root,
        target_dir=Path(target_dir_str),
        index_path=Path(index_path_str),
        server=ServerConfig(**{k: v for k, v in server_section.items()}),
        capsule=CapsuleConfig(
            default_token_budget=capsule_section.get("default_token_budget", DEFAULT_TOKEN_BUDGET),
            max_pivots=capsule_section.get("max_pivots", 3),
            intent_depths=default_depths,
        ),
        generator=GeneratorConfig(**{k: v for k, v in generator_section.items()}),
    )


def _find_dbt_project_root(start: Path) -> Path:
    """Walk up from ``start`` looking for dbt_project.yml."""
    candidate = start.resolve()
    while True:
        if (candidate / "dbt_project.yml").exists():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            return start  # fallback: use start
        candidate = parent
