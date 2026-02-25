"""Pydantic data models for dbt Context Engine."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ─── Core dbt nodes ──────────────────────────────────────────────────────────


class ColumnInfo(BaseModel):
    name: str
    data_type: str = ""
    description: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    is_primary_key: bool = False
    is_foreign_key: bool = False
    tests: list[str] = Field(default_factory=list)  # test type names (not_null, unique, …)


class ModelNode(BaseModel):
    unique_id: str
    name: str
    fqn: list[str] = Field(default_factory=list)
    package_name: str = ""
    database: str = ""
    db_schema: str = ""
    alias: str = ""
    file_path: str = ""
    raw_code: str = ""
    compiled_code: str = ""
    language: str = "sql"
    description: str = ""
    layer: str = "other"  # staging | intermediate | marts | other
    materialization: str = "view"
    tags: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    depends_on_nodes: list[str] = Field(default_factory=list)  # upstream unique_ids
    refs: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    columns: list[ColumnInfo] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    # From catalog.json
    row_count: int | None = None
    bytes: int | None = None
    last_modified: str | None = None
    # Computed
    upstream_count: int = 0
    downstream_count: int = 0
    centrality: float = 0.0


class SourceNode(BaseModel):
    unique_id: str
    name: str
    source_name: str
    schema_name: str = ""
    database: str = ""
    description: str = ""
    loader: str = ""
    freshness_warn: dict[str, Any] | None = None
    freshness_error: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    columns: list[ColumnInfo] = Field(default_factory=list)


class TestNode(BaseModel):
    unique_id: str
    name: str
    test_type: str  # not_null | unique | accepted_values | relationships | generic | singular
    model_id: str | None = None
    column_name: str = ""
    depends_on: list[str] = Field(default_factory=list)
    severity: str = "error"
    # From run_results.json
    last_status: str | None = None
    last_execution_time: float | None = None
    last_failures: int | None = None


class MacroNode(BaseModel):
    unique_id: str
    name: str
    package_name: str = ""
    file_path: str = ""
    description: str = ""
    arguments: list[dict[str, Any]] = Field(default_factory=list)
    macro_sql: str = ""


class ExposureNode(BaseModel):
    unique_id: str
    name: str
    label: str = ""
    type: str = ""
    url: str = ""
    description: str = ""
    owner_name: str = ""
    owner_email: str = ""
    depends_on: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


# ─── Search results ───────────────────────────────────────────────────────────


class SearchResult(BaseModel):
    unique_id: str
    name: str
    layer: str
    description: str
    bm25_score: float = 0.0
    centrality: float = 0.0
    layer_boost: float = 0.0
    name_bonus: float = 0.0
    score: float = 0.0


# ─── Capsule output ───────────────────────────────────────────────────────────


class SkeletonColumn(BaseModel):
    name: str
    data_type: str = ""
    description: str = ""
    tests: list[str] = Field(default_factory=list)


class FullModelContext(BaseModel):
    """PIVOT level — full detail."""
    unique_id: str
    name: str
    layer: str
    materialization: str
    file_path: str
    compiled_sql: str
    description: str
    columns: list[SkeletonColumn]
    tags: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class SkeletonModelContext(BaseModel):
    """ADJACENT level — schema only."""
    unique_id: str
    name: str
    layer: str
    materialization: str
    columns: list[dict[str, str]]  # [{"name": ..., "type": ...}]


class MinimalModelContext(BaseModel):
    """DISTANT level — just awareness."""
    unique_id: str
    name: str
    layer: str
    column_count: int
    key_columns: list[str] = Field(default_factory=list)


class ColumnLineage(BaseModel):
    source_model: str
    source_column: str
    transformation: str = "direct"  # direct | renamed | derived | aggregated
    depth: int = 0


class ContextCapsule(BaseModel):
    """The primary output of get_context_capsule."""
    task: str
    intent: str
    pivot_models: list[FullModelContext] = Field(default_factory=list)
    upstream_models: list[SkeletonModelContext] = Field(default_factory=list)
    downstream_models: list[MinimalModelContext] = Field(default_factory=list)
    relevant_tests: list[dict[str, Any]] = Field(default_factory=list)
    relevant_macros: list[dict[str, Any]] = Field(default_factory=list)
    relevant_sources: list[dict[str, Any]] = Field(default_factory=list)
    project_patterns: dict[str, Any] = Field(default_factory=dict)
    similar_models: list[str] = Field(default_factory=list)
    session_context: dict[str, Any] = Field(default_factory=dict)
    token_estimate: int = 0
    token_budget: int = 10000


# ─── Project statistics ───────────────────────────────────────────────────────


class ProjectStats(BaseModel):
    project_name: str = ""
    adapter_type: str = ""
    dbt_schema_version: str = ""
    model_count: int = 0
    staging_count: int = 0
    intermediate_count: int = 0
    marts_count: int = 0
    other_count: int = 0
    source_count: int = 0
    source_schema_count: int = 0
    test_count: int = 0
    test_coverage_pct: int = 0
    macro_count: int = 0
    project_macro_count: int = 0
    exposure_count: int = 0


class NamingPatterns(BaseModel):
    staging_pattern: str = "stg_{source}__{entity}"
    staging_example: str = ""
    staging_materialization: str = "view"
    intermediate_pattern: str = "int_{entity}_{verb}"
    intermediate_materialization: str = "view"
    marts_pattern: str = "fct_{entity} | dim_{entity}"
    marts_materialization: str = "table"
    yaml_pattern: str = "__{folder_name}_models.yml"
    naming_summary: str = ""
    directory_summary: str = ""
    yaml_requirements: str = ""


class ProjectPatterns(BaseModel):
    naming: NamingPatterns = Field(default_factory=NamingPatterns)
    has_staging: bool = True
    has_intermediate: bool = False
    has_marts: bool = True
    common_tags: list[str] = Field(default_factory=list)
    common_materializations: dict[str, str] = Field(default_factory=dict)
    test_coverage_by_layer: dict[str, float] = Field(default_factory=dict)
    most_tested_columns: list[str] = Field(default_factory=list)
