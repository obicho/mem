"""Tests for Memory client."""

import os
import shutil
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_embedding_service():
    """Mock the embedding service with content-dependent embeddings."""
    with patch("app.client.EmbeddingService") as mock:
        instance = MagicMock()
        # Generate different embeddings based on content hash to avoid false duplicates
        def generate_embedding(text):
            import hashlib
            hash_bytes = hashlib.sha256(text.encode()).digest()
            # Create a 1536-dim embedding from hash bytes (cycling through)
            embedding = []
            for i in range(1536):
                embedding.append(hash_bytes[i % 32] / 255.0)
            return embedding
        instance.embed_text.side_effect = generate_embedding
        mock.return_value = instance
        yield instance


@pytest.fixture
def temp_persist_dir(tmp_path):
    """Create a temporary directory for ChromaDB."""
    persist_dir = tmp_path / "test_chroma"
    persist_dir.mkdir()
    yield str(persist_dir)


@pytest.fixture
def memory_client(mock_embedding_service, temp_persist_dir):
    """Create a Memory client with mocked embeddings."""
    from app.client import Memory
    return Memory(
        api_key="test-key",
        persist_dir=temp_persist_dir,
        collection_name="test_memories",
    )


def test_add_memory(memory_client):
    """Test adding a memory."""
    result = memory_client.add("Test memory content", user_id="test_user")

    assert result["status"] == "added"
    assert "memory_id" in result


def test_add_memory_with_metadata(memory_client):
    """Test adding a memory with custom metadata."""
    result = memory_client.add(
        "Test memory content",
        user_id="test_user",
        metadata={"type": "note", "priority": "high"},
    )

    assert result["status"] == "added"

    # Verify metadata was stored
    memory = memory_client.get(result["memory_id"])
    assert memory is not None
    assert memory["metadata"]["type"] == "note"
    assert memory["metadata"]["priority"] == "high"


def test_add_empty_content_raises_error(memory_client):
    """Test that adding empty content raises an error."""
    with pytest.raises(ValueError, match="Content cannot be empty"):
        memory_client.add("")

    with pytest.raises(ValueError, match="Content cannot be empty"):
        memory_client.add("   ")


def test_search_memories(memory_client):
    """Test searching memories."""
    # Add some memories
    memory_client.add("I love pizza", user_id="test_user")
    memory_client.add("Meeting at 3pm", user_id="test_user")

    # Search
    results = memory_client.search("food preferences", user_id="test_user")

    assert len(results) == 2
    assert all("id" in r for r in results)
    assert all("content" in r for r in results)
    assert all("score" in r for r in results)


def test_search_with_user_filter(memory_client):
    """Test that search filters by user_id."""
    memory_client.add("User A memory", user_id="user_a")
    memory_client.add("User B memory", user_id="user_b")

    results = memory_client.search("memory", user_id="user_a")

    assert len(results) == 1
    assert results[0]["content"] == "User A memory"


def test_get_memory(memory_client):
    """Test getting a specific memory."""
    result = memory_client.add("Test content", user_id="test_user")
    memory_id = result["memory_id"]

    memory = memory_client.get(memory_id)

    assert memory is not None
    assert memory["id"] == memory_id
    assert memory["content"] == "Test content"


def test_get_nonexistent_memory(memory_client):
    """Test getting a memory that doesn't exist."""
    memory = memory_client.get("nonexistent_id")
    assert memory is None


def test_get_all_memories(memory_client):
    """Test getting all memories."""
    memory_client.add("Memory 1", user_id="test_user")
    memory_client.add("Memory 2", user_id="test_user")
    memory_client.add("Memory 3", user_id="other_user")

    # Get all for test_user
    memories = memory_client.get_all(user_id="test_user")
    assert len(memories) == 2

    # Get all without filter
    all_memories = memory_client.get_all()
    assert len(all_memories) == 3


def test_update_memory(memory_client):
    """Test updating a memory."""
    result = memory_client.add("Original content", user_id="test_user")
    memory_id = result["memory_id"]

    update_result = memory_client.update(memory_id, "Updated content")

    assert update_result["status"] == "updated"

    memory = memory_client.get(memory_id)
    assert memory["content"] == "Updated content"
    assert "updated_at" in memory["metadata"]


def test_update_nonexistent_memory_raises_error(memory_client):
    """Test that updating a nonexistent memory raises an error."""
    with pytest.raises(ValueError, match="Memory not found"):
        memory_client.update("nonexistent_id", "New content")


def test_delete_memory(memory_client):
    """Test deleting a memory."""
    result = memory_client.add("To be deleted", user_id="test_user")
    memory_id = result["memory_id"]

    delete_result = memory_client.delete(memory_id)

    assert delete_result["status"] == "deleted"
    assert memory_client.get(memory_id) is None


def test_delete_nonexistent_memory_raises_error(memory_client):
    """Test that deleting a nonexistent memory raises an error."""
    with pytest.raises(ValueError, match="Memory not found"):
        memory_client.delete("nonexistent_id")


def test_delete_all_memories(memory_client):
    """Test deleting all memories for a user."""
    memory_client.add("Memory 1", user_id="test_user")
    memory_client.add("Memory 2", user_id="test_user")
    memory_client.add("Memory 3", user_id="other_user")

    result = memory_client.delete_all(user_id="test_user")

    assert result["deleted"] == 2
    assert memory_client.count(user_id="test_user") == 0
    assert memory_client.count(user_id="other_user") == 1


def test_count_memories(memory_client):
    """Test counting memories."""
    assert memory_client.count() == 0

    memory_client.add("Memory 1", user_id="test_user")
    memory_client.add("Memory 2", user_id="test_user")

    assert memory_client.count() == 2
    assert memory_client.count(user_id="test_user") == 2
    assert memory_client.count(user_id="other_user") == 0


def test_duplicate_detection_exact_match(memory_client):
    """Test that exact duplicate content is detected."""
    content = "This is a unique memory content"

    # Add first memory
    result1 = memory_client.add(content, user_id="test_user")
    assert result1["status"] == "added"

    # Try to add the same content again
    result2 = memory_client.add(content, user_id="test_user")
    assert result2["status"] == "duplicate"
    assert result2["memory_id"] == result1["memory_id"]

    # Should still only have 1 memory
    assert memory_client.count() == 1


def test_duplicate_detection_case_insensitive(memory_client):
    """Test that duplicate detection is case-insensitive."""
    content1 = "Hello World"
    content2 = "hello world"

    result1 = memory_client.add(content1, user_id="test_user")
    assert result1["status"] == "added"

    # Same content with different case should be detected as duplicate
    result2 = memory_client.add(content2, user_id="test_user")
    assert result2["status"] == "duplicate"


def test_duplicate_detection_disabled(memory_client):
    """Test that duplicate detection can be disabled."""
    content = "Duplicate content test"

    result1 = memory_client.add(content, user_id="test_user", dedupe=False)
    assert result1["status"] == "added"

    # With dedupe disabled, should add as new memory
    result2 = memory_client.add(content, user_id="test_user", dedupe=False)
    assert result2["status"] == "added"
    assert result2["memory_id"] != result1["memory_id"]

    # Should have 2 memories now
    assert memory_client.count() == 2


def test_find_duplicates(memory_client):
    """Test the find_duplicates method."""
    content = "Find duplicates test content"

    # Initially no duplicates
    duplicates = memory_client.find_duplicates(content)
    assert len(duplicates) == 0

    # Add the content
    memory_client.add(content, user_id="test_user")

    # Now should find duplicate
    duplicates = memory_client.find_duplicates(content)
    assert len(duplicates) == 1
    assert duplicates[0]["match_type"] == "exact"
    assert duplicates[0]["similarity"] == 1.0
