# Memory-DB — CLAUDE.md

A lightweight vector-based memory system for AI agents. Qdrant (vector storage) + your embedding API (llama.cpp :8081). 3 MCP tools for AI agents, a management CLI for ops.

## The Harness

**Assertive pushback is non-negotiable. See global `~/.claude/CLAUDE.md` § The Harness.**

## Architecture

```
MCP Server (3 tools) ───▶ Qdrant :6333  (vector storage + cosine search)
       │
       ▼
llama.cpp :8081 /v1/embeddings  (your model, your vectors)
```

## The 3 Tools (MCP)

### `store_memory(content, tags?, dedup_threshold=0.85)`
Save a memory. Dedup threshold default 0.85 — semantically similar memories replace old ones (entropy reduction). Returns `{id, deduped: true/false}`. Empty content raises ValueError.

### `get_memories(query, limit=5, min_score=0.5)`
Semantic search (cosine similarity). Results below `min_score` are filtered out — use 0.8+ for strict matching, lower for broader search. Each hit increments `recall_count`. Returns sorted list of `{id, content, score, tags?, recall_count, last_recalled_at}`.

### `delete_memory(memory_id)`
Delete a memory by ID. Returns `{deleted: true/false, id}` — false if not found.

## Management CLI (not exposed to MCP)

```bash
memory-db-manage list [--limit N]                           # list all memories
memory-db-manage export --path backups/memories.json        # JSON backup (raw text)
memory-db-manage import --path backups/memories.json        # restore + re-encode
memory-db-manage rebuild                                    # re-encode with current model
memory-db-manage rebuild --embedding-url http://new:8081/v1/embeddings  # switch model
memory-db-manage purge --min-recall-count 0 --unused-days 30  # dry-run by default
memory-db-manage purge --min-recall-count 0 --unused-days 30 --execute  # actually delete
memory-db-manage delete-all                                 # destructive, requires confirm
```

## Setup Verification

```bash
docker ps --filter "name=memory-db"   # should see qdrant :6333
curl http://localhost:8081/v1/embeddings -X POST -H 'Content-Type: application/json' -d '{"input":["test"],"model":""}' 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'dim={len(d[\"data\"][0][\"embedding\"])}')"
```

MCP failures are **silent** — always verify tool availability at session start.

## Common Pitfalls

- **MCP failures are silent.** If `get_memories` isn't available, check Docker and embedding API.
- **Subagents can't use MCP tools.** Never delegate memory operations to background agents.
