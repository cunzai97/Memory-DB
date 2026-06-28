"""Tests for the simplified memory service (mock Qdrant + embedding API)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory_simple.service import MemoryService


@pytest.fixture(autouse=True)
def mock_qdrant():
    """Replace AsyncQdrantClient with a fully async-mocked instance."""
    client = AsyncMock()
    collections_resp = MagicMock()
    collections_resp.collections = []
    client.get_collections.return_value = collections_resp
    # query_points for dedup check returns empty (no duplicates)
    dup_response = MagicMock()
    dup_response.points = []
    client.query_points.return_value = dup_response
    with patch("memory_simple.service.AsyncQdrantClient", return_value=client):
        yield client


@pytest.fixture(autouse=True)
def mock_encode():
    """Replace the embedding API call."""
    async def fake_encode(text, url=None):
        return [0.1] * 1024

    with patch("memory_simple.service._encode", new=fake_encode):
        yield


@pytest.fixture
def svc(mock_qdrant, mock_encode) -> MemoryService:
    return MemoryService()


@pytest.mark.asyncio
async def test_store_memory(svc, mock_qdrant):
    result = await svc.store_memory(content="Hello world")
    assert "id" in result
    assert result["deduped"] is False  # no duplicates to replace
    call_args = mock_qdrant.upsert.call_args
    points = call_args.kwargs["points"]
    assert len(points) == 1
    payload = points[0].payload
    assert payload["content"] == "Hello world"
    assert payload["recall_count"] == 0
    assert payload["last_recalled_at"] is None


@pytest.mark.asyncio
async def test_store_memory_with_tags(svc, mock_qdrant):
    await svc.store_memory(content="Test", tags=["a", "b"])
    call_args = mock_qdrant.upsert.call_args
    payload = call_args.kwargs["points"][0].payload
    assert payload["tags"] == ["a", "b"]


@pytest.mark.asyncio
async def test_get_memories(svc, mock_qdrant):
    from qdrant_client.http.models import ScoredPoint

    mock_point = MagicMock(spec=ScoredPoint)
    mock_point.id = "test-id"
    mock_point.score = 0.95
    mock_point.payload = {
        "content": "Hello world",
        "created_at": "2026-01-01T00:00:00Z",
        "tags": ["a"],
        "recall_count": 3,
        "last_recalled_at": "2026-06-01T00:00:00Z",
    }

    response = MagicMock()
    response.points = [mock_point]
    mock_qdrant.query_points.return_value = response

    results = await svc.get_memories(query="hello")
    assert len(results) == 1
    assert results[0]["id"] == "test-id"
    assert results[0]["content"] == "Hello world"
    assert results[0]["score"] == 0.95
    assert results[0]["recall_count"] == 3
    assert results[0]["last_recalled_at"] == "2026-06-01T00:00:00Z"

    # Verify recall stats were updated via set_payload (manual increment, not patch operator)
    mock_qdrant.set_payload.assert_called_once()
    call_kwargs = mock_qdrant.set_payload.call_args.kwargs
    assert isinstance(call_kwargs["payload"]["recall_count"], int)


@pytest.mark.asyncio
async def test_get_memories_empty(svc, mock_qdrant):
    response = MagicMock()
    response.points = []
    mock_qdrant.query_points.return_value = response

    results = await svc.get_memories(query="hello")
    assert results == []
    # set_payload should NOT be called when no results
    mock_qdrant.set_payload.assert_not_called()


@pytest.mark.asyncio
async def test_get_memories_min_score(svc, mock_qdrant):
    from qdrant_client.http.models import ScoredPoint

    high_point = MagicMock(spec=ScoredPoint)
    high_point.id = "high-id"
    high_point.score = 0.92
    high_point.payload = {"content": "High score", "recall_count": 0}

    low_point = MagicMock(spec=ScoredPoint)
    low_point.id = "low-id"
    low_point.score = 0.35
    low_point.payload = {"content": "Low score", "recall_count": 0}

    response = MagicMock()
    response.points = [high_point, low_point]
    mock_qdrant.query_points.return_value = response

    # min_score=0.8 should filter out the low one
    results = await svc.get_memories(query="test", min_score=0.8)
    assert len(results) == 1
    assert results[0]["id"] == "high-id"
    assert results[0]["score"] == 0.92


@pytest.mark.asyncio
async def test_update_memory_content_only(svc, mock_qdrant):
    mock_point = MagicMock()
    mock_point.payload = {
        "content": "old content",
        "created_at": "2026-01-01T00:00:00Z",
        "recall_count": 0,
    }
    mock_qdrant.retrieve.return_value = [mock_point]

    result = await svc.update_memory(memory_id="test-id", content="new content")
    assert result["updated"] is True
    assert result["id"] == "test-id"
    assert result["changes"]["content"] is True

    # Should upsert with new vector (not delete)
    mock_qdrant.upsert.assert_called_once()
    call_kwargs = mock_qdrant.upsert.call_args.kwargs
    point = call_kwargs["points"][0]
    assert point.id == "test-id"
    assert point.payload["content"] == "new content"
    assert len(point.vector) == 1024  # re-encoded


@pytest.mark.asyncio
async def test_update_memory_tags_only(svc, mock_qdrant):
    mock_point = MagicMock()
    mock_point.payload = {
        "content": "unchanged content",
        "created_at": "2026-01-01T00:00:00Z",
        "recall_count": 0,
    }
    mock_qdrant.retrieve.return_value = [mock_point]

    result = await svc.update_memory(memory_id="test-id", tags=["new-tag"])
    assert result["updated"] is True
    assert result["changes"]["tags"] is True

    # Should use set_payload (no vector change)
    mock_qdrant.set_payload.assert_called_once()
    call_kwargs = mock_qdrant.set_payload.call_args.kwargs
    assert call_kwargs["payload"]["tags"] == ["new-tag"]
    assert call_kwargs["points"] == ["test-id"]


@pytest.mark.asyncio
async def test_update_memory_both(svc, mock_qdrant):
    mock_point = MagicMock()
    mock_point.payload = {
        "content": "old",
        "created_at": "2026-01-01T00:00:00Z",
        "recall_count": 0,
    }
    mock_qdrant.retrieve.return_value = [mock_point]

    result = await svc.update_memory(
        memory_id="test-id", content="new", tags=["a", "b"]
    )
    assert result["updated"] is True
    assert result["changes"]["content"] is True
    assert result["changes"]["tags"] is True

    # Should upsert with new vector (content changed)
    mock_qdrant.upsert.assert_called_once()
    call_kwargs = mock_qdrant.upsert.call_args.kwargs
    point = call_kwargs["points"][0]
    assert point.payload["content"] == "new"
    assert point.payload["tags"] == ["a", "b"]


@pytest.mark.asyncio
async def test_update_memory_not_found(svc, mock_qdrant):
    mock_qdrant.retrieve.return_value = []

    result = await svc.update_memory(memory_id="no-such-id", content="x")
    assert result["updated"] is False
    assert result["id"] == "no-such-id"
    assert result["error"] == "not_found"
    mock_qdrant.upsert.assert_not_called()
    mock_qdrant.set_payload.assert_not_called()


@pytest.mark.asyncio
async def test_update_memory_no_fields(svc, mock_qdrant):
    mock_qdrant.retrieve.return_value = [MagicMock()]

    result = await svc.update_memory(memory_id="test-id")
    assert result["updated"] is False
    assert result["error"] == "no_fields_provided"
    mock_qdrant.upsert.assert_not_called()
    mock_qdrant.set_payload.assert_not_called()


@pytest.mark.asyncio
async def test_update_memory_empty_content_rejected(svc, mock_qdrant):
    mock_qdrant.retrieve.return_value = [MagicMock()]

    with pytest.raises(ValueError, match="non-empty"):
        await svc.update_memory(memory_id="test-id", content="   ")

    with pytest.raises(ValueError, match="non-empty"):
        await svc.update_memory(memory_id="test-id", content="")


@pytest.mark.asyncio
async def test_store_memory_dedup(svc, mock_qdrant):
    # Simulate a near-duplicate found during dedup check
    dup_point = MagicMock()
    dup_point.id = "old-id"
    dup_point.score = 0.92  # above default threshold of 0.85

    dup_response = MagicMock()
    dup_response.points = [dup_point]
    mock_qdrant.query_points.return_value = dup_response

    result = await svc.store_memory(content="Almost the same")
    assert result["deduped"] is True
    # Old duplicate should be deleted before new one is upserted
    mock_qdrant.delete.assert_called_once()
    del_call = mock_qdrant.delete.call_args.kwargs
    assert "old-id" in del_call["points_selector"].points


@pytest.mark.asyncio
async def test_store_memory_empty_content_rejected(svc, mock_qdrant):
    with pytest.raises(ValueError, match="non-empty"):
        await svc.store_memory(content="")

    with pytest.raises(ValueError, match="non-empty"):
        await svc.store_memory(content="   ")
