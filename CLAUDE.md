# Memory-DB — CLAUDE.md

Vector memory for AI agents: Qdrant (storage) + llama.cpp :8081 (embeddings). 3 MCP tools. Content must be <1024 tokens.

## The Harness

**Assertive pushback is non-negotiable. See global `~/.claude/CLAUDE.md` § The Harness.**

## The 3 Tools (MCP)

### `store_memory(content, tags?, dedup_threshold=0.85)`
Store text → vector. Dedup ≥ 0.85 replaces similar memories; set to 0 to disable. Returns `{id, deduped}`.

### `get_memories(query, limit=5, min_score=0.5)`
Search by cosine similarity. Each hit increments `recall_count`. Returns sorted `[{id, content, score, ...}]` or `[]`.

### `update_memory(memory_id, content?, tags?)`
Update content/tags by ID. At least one required; content change re-encodes vector. Returns `{updated: true, id, changes}`.

## Management CLI (not exposed to MCP)

```bash
memory-db-manage list [--limit N]           # list all memories
memory-db-manage export --path backups.json # JSON backup
memory-db-manage import --path backups.json # restore + re-encode
memory-db-manage rebuild                    # re-encode with current model
memory-db-manage purge ...                  # dry-run by default, add --execute to delete
memory-db-manage delete-all                 # destructive, requires confirm
```

## Common Pitfalls

- **MCP failures are silent** — always verify tool availability at session start.
- **Subagents can't use MCP tools** — never delegate memory operations to background agents.
