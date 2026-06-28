# Memory-DB — Minimal AI Memory

A lightweight vector-based memory system for AI agents. Store, retrieve, and delete memories via semantic search. You own the embedding model; we handle the vectors.

## Why Memory-DB?

**Minimal tokens, maximum model freedom.**

- **~150 tokens/turn** — 4 tools with single-line descriptions
- **No knowledge graph, no FTS, no dashboard** — the model decides how to use memory, not the system
- **6 files, 3 dependencies** — lightweight footprint, easy to understand and modify

**You provide the embedding API.**

- Use llama.cpp, Ollama, OpenAI, or any OpenAI-compatible endpoint
- Switch models anytime — just change the `EMBEDDING_API_URL`
- Rebuild vectors with `memory-db-manage rebuild` when switching models

We removed everything that constrains the model. The memory system should be a tool, not a framework.

## Architecture

```
┌──────────────┐      ┌─────────────┐       ┌──────────────┐
│  MCP Server   │─────▶│    Qdrant   │◀──────│              │
│  (3 tools)    │◀─────│  :6333      │       │ llama.cpp    │
└──────────────┘      └─────────────┘       │  :8081       │
                                          │  /v1/embed   │
                                          └──────────────┘
```

One vector store, one embedding API. 4 MCP tools for AI agents, a management CLI for ops.

## Installation

```bash
# Clone the repo
git clone https://github.com/cunzai97/Memory-DB.git
cd Memory-DB

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Start Qdrant (required)
docker compose up -d
```

Now you can start the MCP server:

```bash
memory-db  # starts MCP server
```

Or use the admin CLI:

```bash
memory-db-manage list  # list all memories
```

## Quick Start

### Prerequisites

- Qdrant running on `:6333` (Docker Compose or standalone)
- Embedding API on `:8081` (your llama.cpp instance)

```bash
docker compose up -d          # starts Qdrant only
pip install -e .              # installs memory-db + CLI
memory-db                     # starts MCP server
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `EMBEDDING_API_URL` | `http://localhost:8081/v1/embeddings` | Embedding API (OpenAI-compatible) |

## MCP 配置

### Claude Code — 全局配置

```bash
claude mcp add memory-db \
  -e PYTHONPATH=/path/to/Memory-DB/src \
  -e EMBEDDING_API_URL=http://localhost:8081/v1/embeddings \
  -e QDRANT_HOST=localhost \
  -e QDRANT_PORT=6333 \
  -- /path/to/Memory-DB/venv/bin/python3 -m memory_simple.server
```

验证：`claude mcp list`（应显示 `✓ Connected`）。

### Hermes — 全局配置

编辑 `~/.hermes/config.yaml`，在 `mcp_servers` 下添加：

```yaml
mcp_servers:
  memory-db:
    command: /path/to/Memory-DB/venv/bin/python3
    args: ["-m", "memory_simple.server"]
    timeout: 120
    env:
      PYTHONPATH: /path/to/Memory-DB/src
      EMBEDDING_API_URL: http://localhost:8081/v1/embeddings
      QDRANT_HOST: localhost
      QDRANT_PORT: "6333"
```

重启 Hermes。

## MCP Tools

### `store_memory(content, tags?, dedup_threshold=0.85)`

Store a memory (text → vector). Dedup threshold ≥ 0.85 replaces semantically similar memories; set to 0 to disable. Use tags for categorization: `["user-preference"]`, `["project-decision"]`, etc. Returns `{id, deduped}`.

```
store_memory(content="Memory-DB 项目重构经验：从 80+ Python 文件精简到 6 个核心文件，只保留 Qdrant + embedding API。MCP 工具约 130 tokens/turn（单行描述）。", tags=["project-refactor"])
→ {"id": "a1b2c3d4-...", "deduped": false}

store_memory(content="Python是一门动态类型的编程语言")
→ {"id": "e5f6g7h8-...", "deduped": true}  // replaced the duplicate
```

### `get_memories(query, limit=5, min_score=0.5)`

Search memories by cosine similarity. min_score=0.5 (default), 0.8+ for strict matching, <0.3 is noise. Each hit increments `recall_count`. Returns sorted list of `{id, content, score, tags?, recall_count}` or `[]`.

```
get_memories(query="动态类型")
→ [{"id": "...", "content": "Python是动态类型语言", "score": 0.79,
     "recall_count": 1, "last_recalled_at": "2026-06-27T...", ...}]

# Broader search with lower threshold
get_memories(query="动态类型", min_score=0.2)
```

### `update_memory(memory_id, content?, oldText?, newText?, tags?)`

Update a memory's content and/or tags by ID. Supports two modes:
- **Full replace**: `content="..."` — replaces entire content; vector is re-encoded.
- **Partial replace**: `oldText="match" newText="replace"` — finds exact substring and substitutes; vector is re-encoded.

At least one of content, tags, or (oldText+newText) must be provided. `content` and `(oldText+newText)` are mutually exclusive. Returns `{updated: true, id, changes, update_type}`.

```
update_memory(memory_id="a1b2c3d4-...", content="updated text")
→ {"updated": true, "id": "a1b2c3d4-...", "changes": {"content": true}, "update_type": "full_replace"}

update_memory(memory_id="a1b2c3d4-...", oldText="old", newText="new")
→ {"updated": true, "id": "a1b2c3d4-...", "changes": {"content": true}, "update_type": "partial_replace"}

update_memory(memory_id="a1b2c3d4-...", tags=["new-tag"])
→ {"updated": true, "id": "a1b2c3d4-...", "changes": {"tags": true}, "update_type": "tags_only"}
```

### `delete_memory(memory_id)`

Delete a memory by ID. Returns `{deleted: true, id}` or `{deleted: false, id, error}`.

```
delete_memory(memory_id="a1b2c3d4-...")
→ {"deleted": true, "id": "a1b2c3d4-..."}
```

### Embedding API Limit

Content must be <1024 tokens (embedding API limit). Longer text causes a 400 error.

### Token Cost

MCP tool definitions (descriptions + JSON schemas) cost **~150 tokens per turn** in the system prompt — about 600 chars of description text across all four tools, each as a single line. Behavioral instructions are removed from tool descriptions and placed in CLAUDE.md / system prompts instead. This is a one-time overhead added to every request, not cumulative.

## Management CLI

Admin operations via terminal — not exposed to MCP tools.

```bash
# List all memories (no search)
memory-db-manage list [--limit N]

# Export to JSON backup (preserves raw text, independent of vectors)
memory-db-manage export --path backups/memories.json

# Import from JSON (re-encodes with current embedding model)
memory-db-manage import --path backups/memories.json

# Rebuild index — re-encode all memories with the same or a new model
memory-db-manage rebuild [--embedding-url http://new-host:port/v1/embeddings]

# Purge unused memories (entropy reduction)
memory-db-manage purge --min-recall-count 0 --unused-days 30   # dry-run by default
memory-db-manage purge --min-recall-count 0 --unused-days 30 --execute  # actually delete

# Delete all (destructive, requires confirmation)
memory-db-manage delete-all [--force]
```

### Switching Embedding Models

When you change your embedding model, existing vectors become stale. Two options:

1. **Rebuild in place** — keeps metadata and recall stats, replaces vectors only:
   ```bash
   memory-db-manage rebuild --embedding-url http://new-host:9090/v1/embeddings
   ```

2. **Export → Import** — full text backup:
   ```bash
   memory-db-manage export --path backups/old-model.json
   # ... switch model ...
   memory-db-manage import --path backups/old-model.json
   ```

### Entropy Reduction (Purging Unused Memories)

Memories that are never recalled accumulate over time. Use `purge` to clean up:

```bash
# Preview what would be deleted (dry-run by default)
memory-db-manage purge --min-recall-count 0 --unused-days 30

# Actually delete memories with recall_count=0 that haven't been recalled in 30+ days
memory-db-manage purge --min-recall-count 0 --unused-days 30 --execute
```

## Memory Payload Schema

Each stored memory carries this payload in Qdrant:

```json
{
  "id": "<uuid>",
  "content": "原始文本",
  "created_at": "2026-06-27T14:33:24+00:00",
  "tags": ["rust", "systems"],
  "recall_count": 3,
  "last_recalled_at": "2026-06-27T15:00:00+00:00"
}
```

`tags` is optional. `recall_count` / `last_recalled_at` auto-tracked on every search hit — useful for identifying never-recalled memories during entropy reduction.

## Project Structure

```
src/memory_simple/
├── embedding.py   # Embedding API client (httpx)
├── service.py     # MemoryService — core store/get/update
├── admin.py       # MemoryAdmin — backup/import/rebuild/purge
├── server.py      # MCP server — exposes 3 tools
└── manage.py      # CLI — admin operations
```

**Dependencies:** `mcp`, `qdrant-client`, `httpx` — that's it.
