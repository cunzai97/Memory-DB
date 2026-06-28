"""Memory service — core operations for MCP tools: store, get, delete."""

import asyncio
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from memory_simple.embedding import encode as _encode

logger = logging.getLogger(__name__)


def _payload_to_dict(point: Any, include_score: bool = False) -> dict[str, Any]:
    """Convert a Qdrant point to a plain dict."""
    payload = point.payload or {}
    result: dict[str, Any] = {
        "id": str(point.id),
        "content": payload.get("content", ""),
        "created_at": payload.get("created_at"),
        "tags": payload.get("tags"),
        "recall_count": payload.get("recall_count", 0),
        "last_recalled_at": payload.get("last_recalled_at"),
    }
    if include_score:
        result["score"] = point.score
    return result


class MemoryService:
    """Memory service backed by Qdrant. Embeddings via local API.

    Core methods only — store, get, update. For admin operations, use MemoryAdmin.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection: str = "memory_embeddings",
        vector_size: int = 1024,
        embedding_url: str | None = None,
    ):
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.collection = collection
        self.vector_size = vector_size
        self.embedding_url = embedding_url
        # trust_env=False: don't let system proxy env vars break localhost connections
        self.client = AsyncQdrantClient(
            host=self.host,
            port=self.port,
            trust_env=False,
            check_compatibility=False,
        )
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_collection(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:  # prevent race on concurrent calls
            if self._initialized:  # double-check after lock
                return
        try:
            collections = await self.client.get_collections()
            exists = any(c.name == self.collection for c in collections.collections)
            if not exists:
                logger.info("Creating Qdrant collection '%s' (size %d)", self.collection, self.vector_size)
                await self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=self.vector_size, distance=models.Distance.COSINE
                    ),
                )
            self._initialized = True
        except Exception as e:
            logger.error("Failed to initialize Qdrant: %s", e)
            raise

    async def store_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        dedup_threshold: float = 0.85,
    ) -> dict[str, Any]:
        """Store a memory. Returns its ID.

        If dedup_threshold > 0, checks for semantically duplicate memories first.
        A match ≥ threshold replaces the old one (entropy reduction).

        Note: content must be <1024 tokens (embedding API limit).
        """
        content = content.strip()
        if not content:
            raise ValueError("content must be non-empty")
        await self._ensure_collection()
        vector = await _encode(content, url=self.embedding_url)
        entity_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "id": entity_id,
            "content": content,
            "created_at": datetime.now(UTC).isoformat(),
            "recall_count": 0,
            "last_recalled_at": None,
        }
        if tags is not None:
            payload["tags"] = tags

        deduped = False
        # Dedup check: search for near-duplicates before storing
        if dedup_threshold > 0:
            dup_results = await self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            if dup_results.points and dup_results.points[0].score >= dedup_threshold:
                # Replace the duplicate
                old_id = str(dup_results.points[0].id)
                await self.client.delete(
                    collection_name=self.collection,
                    points_selector=models.PointIdsList(points=[old_id]),
                )
                logger.info("Dedup: replaced %s (score=%.4f) with new memory", old_id, dup_results.points[0].score)
                deduped = True

        await self.client.upsert(
            collection_name=self.collection,
            points=[models.PointStruct(id=entity_id, vector=vector, payload=payload)],
        )
        return {"id": entity_id, "deduped": deduped}

    async def get_memories(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Search memories by semantic similarity.

        Args:
            query: Search text (encoded via embedding API).
            limit: Maximum results to return (default 5).
            min_score: Minimum cosine similarity threshold (0.0–1.0). Results below this are filtered out.
        """
        await self._ensure_collection()
        vector = await _encode(query, url=self.embedding_url)

        results = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        # Record recall stats and filter by min_score (entropy reduction signal)
        if results.points:
            now_iso = datetime.now(UTC).isoformat()
            try:
                for point in results.points:
                    payload = point.payload or {}
                    current_count = int(payload.get("recall_count", 0) or 0)
                    await self.client.set_payload(
                        collection_name=self.collection,
                        payload={
                            "recall_count": current_count + 1,
                            "last_recalled_at": now_iso,
                        },
                        points=[str(point.id)],
                    )
            except Exception as e:
                logger.warning("Failed to update recall stats: %s", e)

        # Filter by min_score threshold
        return [
            _payload_to_dict(point, include_score=True)
            for point in results.points
            if point.score >= min_score
        ]

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update a memory by ID. Supports full replace (content) or partial replace (old_text→new_text).

        Args:
            memory_id: The point ID to update.
            content: New content text (optional). Full replace; vector is re-encoded.
            old_text: Exact substring to find in current content for partial replace.
            new_text: Replacement text for old_text. Must be used together with old_text.
            tags: New tags list (optional). If provided, only payload is updated.

        Returns:
            {updated: True, id, changes: {content, tags}, update_type} or
            {updated: False, id, error: "not_found" / "no_fields_provided" / ...}

        Note: content must be <1024 tokens (embedding API limit).
        """
        await self._ensure_collection()

        # Check existence first
        try:
            retrieved = await self.client.retrieve(
                collection_name=self.collection,
                ids=[memory_id],
            )
        except Exception:
            return {"updated": False, "id": memory_id, "error": "not_found"}
        if not retrieved:
            return {"updated": False, "id": memory_id, "error": "not_found"}

        if content is None and tags is None and old_text is None and new_text is None:
            return {"updated": False, "id": memory_id, "error": "no_fields_provided"}

        # Conflict: content and (old_text/new_text) are mutually exclusive
        if content is not None and (old_text is not None or new_text is not None):
            return {"updated": False, "id": memory_id, "error": "conflict"}

        # Incomplete partial replace
        if (old_text is not None and new_text is None) or (new_text is not None and old_text is None):
            return {"updated": False, "id": memory_id, "error": "incomplete_partial_replace"}

        # Build update payload
        update_payload: dict[str, Any] = {}
        if content is not None:
            content = content.strip()
            if not content:
                raise ValueError("content must be non-empty")
            update_payload["content"] = content

        if tags is not None:
            update_payload["tags"] = tags

        # If content changed (full or partial), re-encode vector to keep semantic consistency
        if content is not None:
            new_vector = await _encode(content, url=self.embedding_url)
            original_payload = retrieved[0].payload or {}
            merged_payload = {**original_payload, **update_payload}
            await self.client.upsert(
                collection_name=self.collection,
                points=[models.PointStruct(
                    id=memory_id,
                    vector=new_vector,
                    payload=merged_payload,
                )],
            )
        elif old_text is not None and new_text is not None:
            # Partial replace: find old_text in current content, substitute
            original_payload = retrieved[0].payload or {}
            current_content = str(original_payload.get("content", ""))
            if old_text not in current_content:
                return {"updated": False, "id": memory_id, "error": "old_text_not_found"}
            new_content = current_content.replace(old_text, new_text)
            if new_content == current_content:
                return {"updated": False, "id": memory_id, "error": "no_change"}
            new_vector = await _encode(new_content.strip(), url=self.embedding_url)
            merged_payload = {**original_payload, "content": new_content.strip()}
            if tags is not None:
                merged_payload["tags"] = tags
            await self.client.upsert(
                collection_name=self.collection,
                points=[models.PointStruct(
                    id=memory_id,
                    vector=new_vector,
                    payload=merged_payload,
                )],
            )
        else:
            # Only tags changed — use set_payload (no vector change)
            await self.client.set_payload(
                collection_name=self.collection,
                payload={"tags": tags},
                points=[memory_id],
            )

        changes = {}
        if content is not None:
            changes["content"] = True
        if tags is not None:
            changes["tags"] = True

        update_type = "partial_replace" if (old_text is not None and new_text is not None) else (
            "full_replace" if content is not None else "tags_only")
        return {"updated": True, "id": memory_id, "changes": changes, "update_type": update_type}

    async def delete_memory(
        self,
        memory_id: str,
    ) -> dict[str, Any]:
        """Delete a memory by ID.

        Args:
            memory_id: The point ID to delete.

        Returns:
            {deleted: True, id} or {deleted: False, id, error: "not_found"}
        """
        await self._ensure_collection()

        # Check existence first
        try:
            retrieved = await self.client.retrieve(
                collection_name=self.collection,
                ids=[memory_id],
            )
        except Exception:
            return {"deleted": False, "id": memory_id, "error": "not_found"}
        if not retrieved:
            return {"deleted": False, "id": memory_id, "error": "not_found"}

        # Delete the point
        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=[memory_id]),
        )

        return {"deleted": True, "id": memory_id}
