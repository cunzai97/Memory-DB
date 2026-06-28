"""MCP server — 3 tools: store_memory, get_memories, update_memory."""

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
    """Store a memory (text → vector). Dedup threshold ≥ 0.85 replaces similar memories; set to 0 to disable. Returns: {id, deduped}. Must be <1024 tokens."""
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
    """Search memories by cosine similarity. min_score=0.5 default; each hit increments recall_count. Returns: sorted [{id, content, score, ...}] or []."""
    try:
        return await service.get_memories(query=query, limit=limit, min_score=min_score)
    except Exception as e:
        logger.error("get_memories failed: %s", e)
        return [{
            "error": f"MEMORY_LAYER_DEGRADED: {e}",
            "hint": 'get_memories(query="text", limit=5, min_score=0.5)',
        }]


@mcp.tool()
async def update_memory(
    memory_id: str,
    content: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Update a memory's content/tags by ID. At least one required; content change re-encodes vector. Returns: {updated: true, id, changes}."""
    try:
        return await service.update_memory(
            memory_id=memory_id,
            content=content,
            tags=tags,
        )
    except ValueError as e:
        return {"error": str(e), "hint": 'update_memory(memory_id="uuid", content="new text")'}
    except Exception as e:
        logger.error("update_memory failed: %s", e)
        return {
            "error": f"MEMORY_LAYER_DEGRADED: {e}",
            "hint": 'update_memory(memory_id="uuid", content="new text", tags=["tag"])',
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
