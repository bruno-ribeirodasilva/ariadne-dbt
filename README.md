# dbt Context Engine

> **Intelligent context server for AI agents working with dbt projects.**
> One tool call. Everything pre-filtered and ranked.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

## The Problem

Analytics engineers using AI agents (Claude Code, Cursor, etc.) with large dbt projects face a **context paradox**: the agent needs enormous project-specific context (DAG structure, column definitions, business logic, naming conventions) to be effective, but context windows can't hold it all.

- dbt projects at scale have 200–10,000+ models; `manifest.json` can reach 75 MB
- Without help, agents make 8+ tool calls just to understand a model before starting work
- AI agents pick wrong models, hallucinate business logic, miss downstream impact

## The Solution

**Without ariadne (today):**
1. Agent reads `dbt_project.yml`
2. Agent lists `models/` directory
3. Agent reads 3–5 model files
4. Agent searches for relevant models
5. Agent reads upstream model files
6. Agent reads test YAML files
7. Agent reads macro files
8. **NOW** it can start working → **8+ tool calls, thousands of tokens wasted**

**With ariadne:**
1. Agent reads `.claude/CLAUDE.md` (auto-loaded — project identity, conventions, key models)
2. Agent calls `get_context_capsule("add monthly revenue metric")`
3. **NOW** it can start working → **1 explicit tool call. Everything pre-filtered.**

## Features

- **`get_context_capsule`** — ONE call returns pivot models (full SQL), upstream/downstream schemas, relevant tests, macros, sources, and project patterns. Token-budgeted to fit your context window.
- **Hybrid search** — BM25 full-text search + graph centrality re-ranking over names, descriptions, columns, and SQL
- **Intent detection** — auto-detects `debug` / `add_feature` / `refactor` / `test` / `document` and adjusts DAG traversal depth accordingly
- **Auto-generated static context** — `init` command generates `.claude/CLAUDE.md`, `.claude/skills/new_model.md`, `.claude/skills/debug_test.md`, `.claude/context/dag_summary.md` — project-specific, not generic
- **Warehouse-agnostic** — reads only dbt artifacts (`manifest.json`, `catalog.json`). No credentials, no warehouse connection. Works with any dbt adapter.
- **Open-source (MIT)** — free forever

## Quick Start

### 1. Install

```bash
pip install ariadne
# or with uv:
uvx ariadne
```

### 2. Initialize

```bash
cd ~/my-dbt-project
dbt compile   # generate manifest.json
ariadne init
```

This will:
- Index your `manifest.json` into a local SQLite database (`.ariadne/index.db`)
- Generate `.claude/CLAUDE.md` — project identity, conventions, agent instructions
- Generate `.claude/skills/new_model.md` — project-specific model creation playbook
- Generate `.claude/skills/debug_test.md` — project-specific debugging playbook
- Generate `.claude/context/dag_summary.md` — top-level DAG overview

### 3. Configure Claude Code

Add to `.claude/settings.json` in your dbt project:

```json
{
  "mcpServers": {
    "ariadne": {
      "command": "ariadne",
      "args": ["serve"]
    }
  }
}
```

### 4. Configure Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "ariadne": {
      "command": "ariadne",
      "args": ["serve"]
    }
  }
}
```

### 5. Keep in sync

```bash
dbt compile && ariadne sync
```

## MCP Tools

### Primary Tool

| Tool | Description |
|------|-------------|
| `get_context_capsule(task, focus_model?, token_budget?)` | **THE main tool.** Returns pre-filtered, token-budgeted context for any task. |

### Core Tools

| Tool | Description |
|------|-------------|
| `get_model_details(model_name)` | Full details: SQL, columns, tests, lineage, file path |
| `get_lineage(model_name, direction?, depth?)` | DAG traversal — upstream, downstream, or both |
| `search_models(query, limit?, layer?)` | Hybrid BM25 + centrality search |
| `refresh_index(full?)` | Re-index from latest `dbt compile` artifacts |

## How `get_context_capsule` Works

```
INPUT: task="add a monthly revenue metric"

1. INTENT DETECTION: "add" + "metric" → add_feature
2. HYBRID SEARCH: FTS5 BM25 over model names/descriptions/columns/SQL
   + graph centrality re-ranking → top 2-3 pivot models
3. DAG TRAVERSAL: intent-depth traversal (add_feature: 1 up, 2 down)
4. SKELETONIZATION:
   - Pivot models:  full SQL + all columns + tests      (~450 tokens each)
   - Adjacent:      column names + types only           (~90 tokens each)
   - Distant:       name + column count only            (~25 tokens each)
5. TOKEN BUDGETING: trim to budget, lowest-relevance nodes removed first
6. ASSEMBLE: structured JSON response

OUTPUT: ~8,000 tokens (vs. ~25,000+ for naive "read all referenced models")
```

**Token reduction: 50–65%** for typical dbt tasks.

## Generated Files

After `ariadne init`, your dbt project gets:

```
my-dbt-project/
├── .claude/
│   ├── CLAUDE.md              # Project identity + conventions + agent instructions
│   ├── memory.md              # Cross-session memory (update manually or via v1.0 session tracking)
│   ├── skills/
│   │   ├── new_model.md       # How to create a model in THIS project (with YOUR naming)
│   │   └── debug_test.md      # How to debug tests in THIS project
│   └── context/
│       └── dag_summary.md     # Top-level DAG overview (layers, counts, key paths)
├── .ariadne/
│   └── index.db               # SQLite index (add to .gitignore)
└── ariadne.toml    # Optional config
```

## Configuration

Create `ariadne.toml` in your dbt project root (see `ariadne.toml.example`):

```toml
[capsule]
default_token_budget = 10000
max_pivots = 3

[generator]
targets = ["claude_code", "cursor"]
```

## vs. Official dbt MCP Server

| Capability | Official dbt MCP | ariadne |
|---|---|---|
| **Context selection** | Raw — agent must discover what's relevant | **Intelligent** — pre-filtered, ranked |
| **Token efficiency** | Returns full definitions regardless of relevance | **Token-budgeted** — skeletonizes adjacent models |
| **Search** | No search | **Hybrid** BM25 + graph centrality |
| **Intent awareness** | None | **Auto-detects** intent, adjusts traversal |
| **Static context** | None | **Auto-generates** CLAUDE.md + skills + DAG summary |
| **Warehouse dependency** | Requires dbt Core/Cloud connection | **None** — artifacts only |
| **Cost** | Free (local) | **Free** (MIT, fully local) |

**Use both together:** ariadne handles context selection. The official server handles execution (run models, execute SQL). They're complementary.

## Development

```bash
git clone https://github.com/brunoribeirodasilva/ariadne-dbt
cd ariadne
pip install -e ".[dev]"
pytest
```

## Roadmap

### v1.0
- Column-level lineage via SQLGlot (any adapter, fully local)
- Session memory + anti-pattern detection
- PR Trail: branch-scoped investigation logs, auto-generated HANDOFF.md and PR descriptions
- 9 additional MCP tools (impact analysis, test coverage, column lineage, project patterns, session context)
- Incremental indexing (hash-based delta)
- Cursor + Windsurf rules generation

### v2.0+
- dbt Cloud API integration
- Cross-project support (dbt Mesh)
- Git-aware context (what changed since last commit)

## License

MIT. See [LICENSE](LICENSE).

## Inspiration

- [Vexp](https://vexp.dev/) — graph-RAG capsule architecture, skeletonization, token budgeting
- [OpenClaw](https://openclaw.ai/) — append-only memory logs, HANDOFF.md pattern
- [OneContext](https://github.com/TheAgentContextLab/OneContext) — trajectory recording, cross-session persistence
