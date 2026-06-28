"""Memory admin operations — backup, export, import, rebuild, purge. Not exposed to MCP."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from qdrant_client.http import models

from memory_simple.embedding import encode as _encode
from memory_simple.service import MemoryService, _payload_to_dict

logger = logging.getLogger(__name__)


class MemoryAdmin:
    """Admin operations for memory management. Uses MemoryService internally.

    Methods here are for CLI / scripts, not MCP tools.
    """

    def __init__(self, service: MemoryService | None = None):
        self.service = service or MemoryService()

    @property
    def client(self):
        return self.service.client

    @property
    def collection(self):
        return self.service.collection

    async def _ensure_collection(self) -> None:
        await self.service._ensure_collection()

    async def list_memories(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List all memories (no search). For admin/export use."""
        await self._ensure_collection()

        results = await self.client.query_points(
            collection_name=self.collection,
            query=None,  # no vector search, just list
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        return [_payload_to_dict(point) for point in results.points]

    async def export_to_json(self, path: str) -> dict[str, Any]:
        """Export all memories as plain JSON. For backup."""
        await self._ensure_collection()
        all_memories = []
        offset = 0
        batch_size = 100

        while True:
            batch = await self.list_memories(limit=batch_size, offset=offset)
            if not batch:
                break
            all_memories.extend(batch)
            offset += batch_size

        export_data: dict[str, Any] = {
            "exported_at": datetime.now(UTC).isoformat(),
            "total_count": len(all_memories),
            "memories": all_memories,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        return {"path": path, "total_count": len(all_memories)}

    async def import_from_json(
        self,
        path: str,
        dedup_threshold: float = 0.85,
    ) -> dict[str, Any]:
        """Import memories from a JSON export file.

        Re-encodes all texts with the current embedding model.
        """
        await self._ensure_collection()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {"imported": 0, "error": "file_not_found", "path": path}
        except json.JSONDecodeError as e:
            return {"imported": 0, "error": f"invalid_json: {e}", "path": path}

        memories = data.get("memories", [])
        imported = 0
        skipped = 0
        deduped_count = 0

        for m in memories:
            content = m.get("content", "")
            if not content:
                skipped += 1
                continue
            tags = m.get("tags") or None
            result = await self.service.store_memory(
                content=content,
                tags=tags,
                dedup_threshold=dedup_threshold,
            )
            imported += 1
            if result.get("deduped"):
                deduped_count += 1

        return {
            "imported": imported,
            "skipped": skipped,
            "deduped": deduped_count,
            "path": path,
        }

    async def rebuild_index(
        self,
        new_embedding_url: str | None = None,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Re-encode all memories with the current (or a new) embedding model.

        Used when switching embedding models — keeps raw text intact,
        replaces vectors only.
        """
        await self._ensure_collection()
        url = new_embedding_url or self.service.embedding_url

        # Read all points with their ORIGINAL payloads directly from Qdrant
        # (not through _payload_to_dict which drops payload fields like "id")
        all_points: list[tuple[str, dict[str, Any]]] = []  # (point_id, original_payload)
        offset = 0
        while True:
            results = await self.client.query_points(
                collection_name=self.collection,
                query=None,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not results.points:
                break
            for point in results.points:
                all_points.append((str(point.id), point.payload or {}))
            offset += batch_size

        total = len(all_points)
        rebuilt = 0
        errors = 0

        for point_id, original_payload in all_points:
            try:
                content = original_payload.get("content", "")
                if not content:
                    logger.warning("Skipping empty content: %s", point_id)
                    continue
                vector = await _encode(content, url=url)
                # Upsert with new vector but keep original payload intact
                await self.client.upsert(
                    collection_name=self.collection,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=original_payload,
                        )
                    ],
                )
                rebuilt += 1
            except Exception as e:
                logger.error("Rebuild error for %s: %s", point_id, e)
                errors += 1

        if new_embedding_url and url != self.service.embedding_url:
            self.service.embedding_url = url
            logger.info("Updated embedding_url to %s", url)

        return {
            "total": total,
            "rebuilt": rebuilt,
            "errors": errors,
        }

    async def delete_all(self) -> dict[str, Any]:
        """Delete all memories. Destructive — no undo."""
        await self._ensure_collection()

        selector = models.FilterSelector(filter=models.Filter())

        await self.client.delete(
            collection_name=self.collection,
            points_selector=selector,
        )
        return {"deleted": True}

    async def purge_unused(
        self,
        min_recall_count: int | None = None,
        unused_days: int | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Purge unused memories based on recall stats.

        Args:
            min_recall_count: Only delete memories with recall_count <= this value.
            unused_days: Only delete memories not recalled in X days.
            dry_run: If True (default), only count what would be deleted.

        Returns:
            {"deleted": N, "dry_run": true/false, "criteria": {...}}
        """
        await self._ensure_collection()

        # Fetch all memories and filter in Python
        # (Qdrant doesn't support range queries on payload fields easily)
        all_memories: list[dict[str, Any]] = []
        offset = 0
        batch_size = 100
        while True:
            batch = await self.list_memories(limit=batch_size, offset=offset)
            if not batch:
                break
            all_memories.extend(batch)
            offset += batch_size

        # Filter by criteria
        to_delete: list[str] = []
        now = datetime.now(UTC)

        for m in all_memories:

            # recall_count filter
            if min_recall_count is not None:
                rc = m.get("recall_count", 0)
                if isinstance(rc, dict):  # handle old bad data
                    rc = 0
                if rc > min_recall_count:
                    continue

            # unused_days filter
            if unused_days is not None:
                lra = m.get("last_recalled_at")
                if lra is not None:
                    try:
                        last_recalled = datetime.fromisoformat(lra.replace("Z", "+00:00"))
                        days_since_recall = (now - last_recalled).days
                        if days_since_recall < unused_days:
                            continue
                    except Exception:
                        pass
                else:
                    # never recalled — check created_at instead
                    created = m.get("created_at")
                    if created:
                        try:
                            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                            days_since_created = (now - created_dt).days
                            if days_since_created < unused_days:
                                continue
                        except Exception:
                            pass

            to_delete.append(m["id"])

        criteria = {
            "min_recall_count": min_recall_count,
            "unused_days": unused_days,
        }

        if dry_run:
            return {
                "deleted": 0,
                "would_delete": len(to_delete),
                "dry_run": True,
                "criteria": criteria,
            }

        # Batch delete
        if to_delete:
            await self.client.delete(
                collection_name=self.collection,
                points_selector=models.PointIdsList(points=to_delete),
            )

        return {
            "deleted": len(to_delete),
            "dry_run": False,
            "criteria": criteria,
        }
