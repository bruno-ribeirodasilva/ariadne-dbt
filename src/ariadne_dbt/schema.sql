-- dbt Context Engine SQLite Schema
-- All artifact data from manifest.json, catalog.json, run_results.json

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ─── Index metadata ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS index_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ─── Models ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS models (
    unique_id          TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    fqn                TEXT,              -- fully-qualified name (array as JSON)
    package_name       TEXT,
    database           TEXT,
    db_schema          TEXT,
    alias              TEXT,
    file_path          TEXT,              -- original_file_path in manifest
    raw_code           TEXT,
    compiled_code      TEXT,
    language           TEXT DEFAULT 'sql',
    description        TEXT DEFAULT '',
    layer              TEXT,              -- staging | intermediate | marts | other
    materialization    TEXT,
    tags               TEXT DEFAULT '[]', -- JSON array
    meta               TEXT DEFAULT '{}', -- JSON object
    config             TEXT DEFAULT '{}', -- JSON object
    depends_on_nodes   TEXT DEFAULT '[]', -- JSON array of upstream unique_ids
    refs               TEXT DEFAULT '[]', -- JSON array of ref() calls
    sources            TEXT DEFAULT '[]', -- JSON array of source() calls
    -- From catalog.json (optional)
    row_count          INTEGER,
    bytes              INTEGER,
    last_modified      TEXT,
    -- Computed
    upstream_count     INTEGER DEFAULT 0,
    downstream_count   INTEGER DEFAULT 0,
    centrality         REAL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_models_name  ON models(name);
CREATE INDEX IF NOT EXISTS idx_models_layer ON models(layer);

-- ─── Columns ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS columns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id     TEXT NOT NULL REFERENCES models(unique_id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    data_type    TEXT DEFAULT '',
    description  TEXT DEFAULT '',
    meta         TEXT DEFAULT '{}',
    tags         TEXT DEFAULT '[]',
    is_primary_key   INTEGER DEFAULT 0,   -- 1 if has unique+not_null tests
    is_foreign_key   INTEGER DEFAULT 0,   -- 1 if has relationships test
    UNIQUE (model_id, name)
);

CREATE INDEX IF NOT EXISTS idx_columns_model ON columns(model_id);

-- ─── Sources ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sources (
    unique_id        TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    source_name      TEXT NOT NULL,
    schema_name      TEXT,
    database         TEXT,
    description      TEXT DEFAULT '',
    loader           TEXT,
    freshness_warn   TEXT,               -- JSON freshness config
    freshness_error  TEXT,
    tags             TEXT DEFAULT '[]',
    meta             TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS source_columns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL REFERENCES sources(unique_id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    data_type   TEXT DEFAULT '',
    description TEXT DEFAULT '',
    UNIQUE (source_id, name)
);

-- ─── Tests ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tests (
    unique_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    test_type       TEXT NOT NULL,  -- not_null | unique | accepted_values | relationships | generic | singular
    model_id        TEXT REFERENCES models(unique_id) ON DELETE CASCADE,
    column_name     TEXT DEFAULT '',
    depends_on      TEXT DEFAULT '[]',  -- JSON
    severity        TEXT DEFAULT 'error',
    -- From run_results.json (optional)
    last_status     TEXT,   -- pass | fail | warn | error
    last_execution_time REAL,
    last_failures   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_tests_model  ON tests(model_id);
CREATE INDEX IF NOT EXISTS idx_tests_column ON tests(model_id, column_name);

-- ─── Macros ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS macros (
    unique_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    package_name TEXT,
    file_path    TEXT,
    description  TEXT DEFAULT '',
    arguments    TEXT DEFAULT '[]',   -- JSON array
    macro_sql    TEXT
);

-- ─── Exposures ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS exposures (
    unique_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    label        TEXT,
    type         TEXT,   -- dashboard | notebook | analysis | ml | application
    url          TEXT,
    description  TEXT DEFAULT '',
    owner_name   TEXT,
    owner_email  TEXT,
    depends_on   TEXT DEFAULT '[]',  -- JSON array of model unique_ids
    tags         TEXT DEFAULT '[]'
);

-- ─── DAG edges ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS edges (
    parent_id TEXT NOT NULL,   -- upstream node unique_id
    child_id  TEXT NOT NULL,   -- downstream node unique_id
    PRIMARY KEY (parent_id, child_id)
);

CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges(parent_id);
CREATE INDEX IF NOT EXISTS idx_edges_child  ON edges(child_id);

-- ─── Column lineage (populated on demand via SQLGlot) ────────────────────────

CREATE TABLE IF NOT EXISTS column_lineage (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    target_model_id  TEXT NOT NULL,
    target_column    TEXT NOT NULL,
    source_model_id  TEXT NOT NULL,
    source_column    TEXT NOT NULL,
    transformation   TEXT DEFAULT 'direct',  -- direct | renamed | derived | aggregated
    computed_at      TEXT NOT NULL,
    UNIQUE (target_model_id, target_column, source_model_id, source_column)
);

CREATE INDEX IF NOT EXISTS idx_collin_target ON column_lineage(target_model_id, target_column);
CREATE INDEX IF NOT EXISTS idx_collin_source ON column_lineage(source_model_id, source_column);

-- ─── Session events (v1.0, created now to avoid schema migrations) ────────────

CREATE TABLE IF NOT EXISTS session_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    event_type   TEXT NOT NULL,  -- investigation | decision | validation | change
    tool_name    TEXT,
    payload      TEXT DEFAULT '{}',  -- JSON
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_events ON session_events(session_id);

-- ─── Full-text search index ───────────────────────────────────────────────────

-- FTS5 virtual table over model content
-- Column weights set in queries: name=5x, description=3x, column_names=2x, sql=1x, tags=1x
CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
    unique_id UNINDEXED,
    name,
    description,
    column_names,
    sql_text,
    tags,
    tokenize = 'porter ascii'
);
