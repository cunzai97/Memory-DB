"""Tests for admin operations — list, export, import, rebuild, purge."""

import json
import os
import tempfile
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_simple.admin import MemoryAdmin
from memory_simple.service import MemoryService


@pytest.fixture
def mock_qdrant():
    """Replace AsyncQdrantClient with a fully async-mocked instance."""
    client = AsyncMock()
    collections_resp = MagicMock()
    collections_resp.collections = []
    client.get_collections.return_value = collections_resp

    # Default: query_points returns empty (for list_memories)
    response = MagicMock()
    response.points = []
    client.query_points.return_value = response

    with patch("memory_simple.service.AsyncQdrantClient", return_value=client):
        yield client


@pytest.fixture
def mock_encode():
    """Replace the embedding API call."""
    async def fake_encode(text, url=None):
        return [0.1] * 1024

    with patch("memory_simple.service._encode", new=fake_encode):
        with patch("memory_simple.admin._encode", new=fake_encode):
            yield


@pytest.fixture
def admin(mock_qdrant, mock_encode) -> MemoryAdmin:
    svc = MemoryService()
    return MemoryAdmin(service=svc)


# ── list_memories ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_memories_empty(admin, mock_qdrant):
    results = await admin.list_memories()
    assert results == []


@pytest.mark.asyncio
async def test_list_memories_with_data(admin, mock_qdrant):
    mock_point = MagicMock()
    mock_point.id = "test-id"
    mock_point.payload = {
        "content": "Test memory",
        "created_at": "2026-01-01T00:00:00Z",
        "tags": ["test"],
        "recall_count": 2,
        "last_recalled_at": "2026-06-01T00:00:00Z",
    }

    response = MagicMock()
    response.points = [mock_point]
    mock_qdrant.query_points.return_value = response

    results = await admin.list_memories(limit=10)
    assert len(results) == 1
    assert results[0]["id"] == "test-id"
    assert results[0]["content"] == "Test memory"


# ── export_to_json ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_export_to_json(admin, mock_qdrant):
    mock_point = MagicMock()
    mock_point.id = "export-id"
    mock_point.payload = {
        "content": "Export test",
        "created_at": "2026-01-01T00:00:00Z",
        "project_id": "test",
        "recall_count": 0,
    }

    response1 = MagicMock()
    response1.points = [mock_point]
    response2 = MagicMock()
    response2.points = []  # Second call returns empty (end of pagination)

    mock_qdrant.query_points.side_effect = [response1, response2]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        export_path = f.name

    try:
        result = await admin.export_to_json(export_path)
        assert result["total_count"] == 1
        assert result["path"] == export_path

        # Verify file content
        with open(export_path) as f:
            data = json.load(f)
        assert data["total_count"] == 1
        assert len(data["memories"]) == 1
        assert data["memories"][0]["content"] == "Export test"
    finally:
        os.unlink(export_path)


# ── import_from_json ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_import_from_json(admin, mock_qdrant):
    import_data = {
        "exported_at": "2026-01-01T00:00:00Z",
        "total_count": 2,
        "memories": [
            {"id": "old-1", "content": "First memory"},
            {"id": "old-2", "content": "Second memory"},
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(import_data, f)
        import_path = f.name

    try:
        result = await admin.import_from_json(import_path, dedup_threshold=0)
        assert result["imported"] == 2
        assert result["path"] == import_path

        # Verify upsert was called twice
        assert mock_qdrant.upsert.call_count == 2
    finally:
        os.unlink(import_path)


# ── rebuild_index ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rebuild_index(admin, mock_qdrant):
    # Simulate original payload with all fields including "id" in payload
    mock_point = MagicMock()
    mock_point.id = "rebuild-id"
    mock_point.payload = {
        "id": "rebuild-id",  # payload also stores id
        "content": "Rebuild test",
        "created_at": "2026-01-01T00:00:00Z",
        "recall_count": 5,
        "last_recalled_at": "2026-06-01T00:00:00Z",
    }

    response1 = MagicMock()
    response1.points = [mock_point]
    response2 = MagicMock()
    response2.points = []  # End pagination

    mock_qdrant.query_points.side_effect = [response1, response2]

    result = await admin.rebuild_index()

    assert result["total"] == 1
    assert result["rebuilt"] == 1
    assert result["errors"] == 0

    # Verify upsert was called with original payload preserved (including "id")
    mock_qdrant.upsert.assert_called_once()
    call_kwargs = mock_qdrant.upsert.call_args.kwargs
    points = call_kwargs["points"]
    assert len(points) == 1
    assert points[0].id == "rebuild-id"
    # Original payload must be preserved intact — this is the key test
    payload = points[0].payload
    assert payload["id"] == "rebuild-id"  # id in payload should NOT be lost
    assert payload["recall_count"] == 5
    assert payload["created_at"] == "2026-01-01T00:00:00Z"
    # Vector should be the mocked encoding result (1024d)
    assert len(points[0].vector) == 1024


@pytest.mark.asyncio
async def test_rebuild_index_with_new_url(admin, mock_qdrant):
    mock_point = MagicMock()
    mock_point.id = "rebuild-id"
    mock_point.payload = {
        "content": "Rebuild with new model",
        "recall_count": 0,
    }

    response1 = MagicMock()
    response1.points = [mock_point]
    response2 = MagicMock()
    response2.points = []

    mock_qdrant.query_points.side_effect = [response1, response2]

    result = await admin.rebuild_index(new_embedding_url="http://new:9090/v1/embeddings")

    assert result["rebuilt"] == 1
    # embedding_url should be updated on the service instance
    assert admin.service.embedding_url == "http://new:9090/v1/embeddings"


@pytest.mark.asyncio
async def test_rebuild_index_skips_empty_content(admin, mock_qdrant):
    mock_point = MagicMock()
    mock_point.id = "empty-id"
    mock_point.payload = {
        "content": "",
        "recall_count": 0,
    }

    response1 = MagicMock()
    response1.points = [mock_point]
    response2 = MagicMock()
    response2.points = []

    mock_qdrant.query_points.side_effect = [response1, response2]

    result = await admin.rebuild_index()

    assert result["total"] == 1
    assert result["rebuilt"] == 0  # empty content skipped
    mock_qdrant.upsert.assert_not_called()


# ── delete_all ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_all(admin, mock_qdrant):
    result = await admin.delete_all()
    assert result["deleted"] is True
    mock_qdrant.delete.assert_called_once()


# ── purge_unused ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_purge_dry_run(admin, mock_qdrant):
    # Memory with recall_count=0, never recalled
    mock_point = MagicMock()
    mock_point.id = "unused-id"
    mock_point.payload = {
        "content": "Unused memory",
        "created_at": "2025-01-01T00:00:00Z",  # old
        "recall_count": 0,
        "last_recalled_at": None,
    }

    response1 = MagicMock()
    response1.points = [mock_point]
    response2 = MagicMock()
    response2.points = []  # End pagination

    mock_qdrant.query_points.side_effect = [response1, response2]

    result = await admin.purge_unused(min_recall_count=0, unused_days=30, dry_run=True)

    assert result["dry_run"] is True
    assert result["would_delete"] == 1
    assert result["deleted"] == 0
    mock_qdrant.delete.assert_not_called()


@pytest.mark.asyncio
async def test_purge_execute(admin, mock_qdrant):
    mock_point = MagicMock()
    mock_point.id = "unused-id"
    mock_point.payload = {
        "content": "Unused memory",
        "created_at": "2025-01-01T00:00:00Z",
        "recall_count": 0,
        "last_recalled_at": None,
    }

    response1 = MagicMock()
    response1.points = [mock_point]
    response2 = MagicMock()
    response2.points = []

    mock_qdrant.query_points.side_effect = [response1, response2]

    result = await admin.purge_unused(min_recall_count=0, unused_days=30, dry_run=False)

    assert result["dry_run"] is False
    assert result["deleted"] == 1
    mock_qdrant.delete.assert_called_once()


@pytest.mark.asyncio
async def test_purge_respects_recall_count(admin, mock_qdrant):
    # Memory with recall_count=5, should NOT be purged with min_recall_count=0
    mock_point = MagicMock()
    mock_point.id = "used-id"
    mock_point.payload = {
        "content": "Used memory",
        "created_at": "2025-01-01T00:00:00Z",
        "recall_count": 5,
        "last_recalled_at": "2026-01-01T00:00:00Z",
    }

    response1 = MagicMock()
    response1.points = [mock_point]
    response2 = MagicMock()
    response2.points = []

    mock_qdrant.query_points.side_effect = [response1, response2]

    result = await admin.purge_unused(min_recall_count=0, dry_run=True)

    assert result["would_delete"] == 0


@pytest.mark.asyncio
async def test_purge_respects_unused_days(admin, mock_qdrant):
    # Memory recalled recently, should NOT be purged
    mock_point = MagicMock()
    mock_point.id = "recent-id"
    mock_point.payload = {
        "content": "Recent memory",
        "created_at": "2026-01-01T00:00:00Z",
        "recall_count": 0,
        "last_recalled_at": datetime.now(UTC).isoformat(),  # now
    }

    response1 = MagicMock()
    response1.points = [mock_point]
    response2 = MagicMock()
    response2.points = []

    mock_qdrant.query_points.side_effect = [response1, response2]

    result = await admin.purge_unused(min_recall_count=0, unused_days=30, dry_run=True)

    assert result["would_delete"] == 0
