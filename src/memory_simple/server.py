"""MCP server — 3 tools: store_memory, get_memories, delete_memory."""

import asyncio
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from memory_simple.embedding import health_check
from memory_simple.service import MemoryService

logger = logging.getLogger(__name__)

# Initialize MCP server and service
mcp = FastMCP("memory-db")
service = MemoryService()


@mcp.tool()
async def store_memory(
    content: str,
    tags: list[str] | None = None,
    dedup_threshold: float = 0.85,
) -> dict[str, Any]:
    """Store a memory (text → vector). Dedup threshold ≥ 0.85 replaces semantically similar memories. Set to 0 to disable dedup. Use tags for categorization: ["user-preference"], ["project-decision"], etc. Returns: {id, deduped}."""
    try:
        return await service.store_memory(
            content=content,
            tags=tags, dedup_threshold=dedup_threshold
        )
    except ValueError as e:
        return {
            "error": str(e),
            "hint": 'store_memory(content="text", tags=["tag"])',
        }
    except Exception as e:
        logger.error("store_memory failed: %s", e)
        return {"error": f"MEMORY_LAYER_DEGRADED: {e}"}


@mcp.tool()
async def get_memories(
    query: str,
    limit: int = 5,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Search memories by cosine similarity. min_score=0.5 (default), 0.8+ for strict matching, <0.3 is noise. Each hit increments recall_count. Returns: sorted [{id, content, score, tags?, recall_count}] or []."""
    try:
        return await service.get_memories(query=query, limit=limit, min_score=min_score)
    except Exception as e:
        logger.error("get_memories failed: %s", e)
        return [{
            "error": f"MEMORY_LAYER_DEGRADED: {e}",
            "hint": 'get_memories(query="text", limit=5, min_score=0.5)',
        }]


@mcp.tool()
async def delete_memory(memory_id: str) -> dict[str, Any]:
    """Delete a memory by ID. Verify with get_memories first to confirm the right ID. Returns: {deleted, id}."""
    try:
        return await service.delete_memory(memory_id=memory_id)
    except Exception as e:
        logger.error("delete_memory failed: %s", e)
        return {
            "error": f"MEMORY_LAYER_DEGRADED: {e}",
            "hint": 'delete_memory(memory_id="uuid")',
        }


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # P0: Health check embedding API at startup
    async def check_health():
        ok, msg = await health_check(service.embedding_url)
        if ok:
            logger.info("Embedding API: %s", msg)
        else:
            logger.error("Embedding API check failed: %s", msg)
            logger.error("Server will start but operations will fail.")
            # Don't exit — let the server start, but warn loudly

    asyncio.run(check_health())

    logger.info("Starting Memory-DB MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
