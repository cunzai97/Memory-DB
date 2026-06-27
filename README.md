# Memory-DB — Minimal AI Memory

A lightweight vector-based memory system for AI agents. Store, retrieve, and delete memories via semantic search. You own the embedding model; we handle the vectors.

## Why Memory-DB?

**Minimal tokens, maximum model freedom.**

- **~236 tokens/turn** — 3 tools with concise descriptions, no verbose instructions
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

One vector store, one embedding API. 3 MCP tools for AI agents, a management CLI for ops.

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

Save a memory. Returns `{id, deduped}`.

If the text is semantically similar to an existing memory (cosine ≥ threshold), the old one is replaced — entropy reduction built in. Set `dedup_threshold=0` to disable.

```
store_memory(content="Memory-DB 项目重构经验：从 80+ Python 文件精简到 6 个核心文件。移除了 knowledge graph (FalkorDB)、FTS5、dashboard、sessions，只保留 Qdrant + embedding API。MCP 工具从复杂变简单：store_memory、get_memories、delete_memory，约 236 tokens/turn。关键设计：语义去重（余弦相似度 ≥ 0.85）、recall_count 自动追踪、min_score 过滤、空内容校验 + 友好错误提示。依赖精简到：mcp、qdrant-client、httpx。", tags=["project", "refactor", "mcp", "memory-system"])
→ {"id": "a1b2c3d4-...", "deduped": false}

store_memory(content="Python是一门动态类型的编程语言")
→ {"id": "e5f6g7h8-...", "deduped": true}  // replaced the duplicate
```

### `get_memories(query, limit=5, min_score=0.5)`

Semantic search (cosine similarity). Returns top matches sorted by score descending. Results below `min_score` are filtered out — use 0.8+ for strict matching, lower (e.g. 0.2) for broader search. Each hit auto-increments `recall_count` and updates `last_recalled_at`.

```
get_memories(query="动态类型")
→ [{"id": "...", "content": "Python是动态类型语言", "score": 0.79,
     "recall_count": 1, "last_recalled_at": "2026-06-27T...", ...}]

# Broader search with lower threshold
get_memories(query="动态类型", min_score=0.2)
```

### `delete_memory(memory_id)`

Delete by ID. Returns `{deleted: true/false}` — false if the memory doesn't exist.

```
delete_memory(memory_id="a1b2c3d4-...")
→ {"deleted": true, "id": "a1b2c3d4-..."}
```

### Token Cost

MCP tool definitions (descriptions + JSON schemas) cost **~236 tokens per turn** in the system prompt — about 464 chars of description text across all three tools. Self-explanatory parameters (`content`, `tags`, `query`, `limit`) are left out; only non-obvious ones (`dedup_threshold`, `min_score`) get explained. This is a one-time overhead added to every request, not cumulative.

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
├── service.py     # MemoryService — core store/get/delete
├── admin.py       # MemoryAdmin — backup/import/rebuild/purge
├── server.py      # MCP server — exposes 3 tools
└── manage.py      # CLI — admin operations
```

**Dependencies:** `mcp`, `qdrant-client`, `httpx` — that's it.
