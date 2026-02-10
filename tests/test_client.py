"""Tests for Memory client."""

import json
import os
import shutil
import pytest
from unittest.mock import MagicMock, patch

from app.core.chat_summarizer import ChatSummaryResult
from app.core.chunker import detect_content_type, ContentType


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


# --- Chat summarization integration tests ---

CHAT_CONTENT = (
    "alice: We should use PostgreSQL for the new project.\n"
    "bob: Agreed. The deadline is next Friday.\n"
    "alice: I think React is better than Vue for this.\n"
    "bob: Let's aim for 99.9% uptime.\n"
    "alice: I prefer deploying on Fridays.\n"
    "bob: TODO: set up CI pipeline by Wednesday.\n"
)

SUMMARY_RESULT = ChatSummaryResult(
    outcome="The team decided to use PostgreSQL and React, targeting 99.9% uptime with CI setup by Wednesday.",
    facts=["The project deadline is next Friday."],
    decisions=["The team decided to use PostgreSQL for the new project."],
    opinions=["Alice thinks React is better than Vue."],
    tasks=["Bob will set up the CI pipeline by Wednesday."],
    goals=["The team aims for 99.9% uptime."],
    preferences=["Alice prefers deploying on Fridays."],
)


def test_add_chat_triggers_summarization(memory_client):
    """Chat content triggers extraction with outcome and items are stored."""
    with patch.object(memory_client.chat_summarizer, "extract", return_value=SUMMARY_RESULT):
        result = memory_client.add(
            CHAT_CONTENT,
            user_id="test_user",
            content_type="chat",
        )

    assert result["status"] == "added"
    assert "extracted_items" in result
    assert "outcome" in result
    assert result["outcome"] == SUMMARY_RESULT.outcome

    extracted = result["extracted_items"]
    # 1 outcome + 6 category items = 7
    assert len(extracted) == 7
    added_items = [i for i in extracted if i["status"] == "added"]
    assert len(added_items) == 7

    categories = {i["category"] for i in extracted}
    assert categories == {"outcome", "facts", "decisions", "opinions", "tasks", "goals", "preferences"}


def test_add_non_chat_does_not_trigger_summarization(memory_client):
    """Document content should not trigger chat extraction."""
    doc_content = (
        "This is a formal document about project architecture.\n\n"
        "The system uses a microservices pattern with REST APIs.\n\n"
        "Each service is independently deployable."
    )
    with patch.object(memory_client.chat_summarizer, "extract") as mock_extract:
        result = memory_client.add(
            doc_content,
            user_id="test_user",
            content_type="document",
        )

    assert result["status"] == "added"
    assert "extracted_items" not in result
    mock_extract.assert_not_called()


def test_chat_summarization_failure_does_not_block_storage(memory_client):
    """Raw chat chunks are stored even when summarization fails."""
    with patch.object(
        memory_client.chat_summarizer,
        "extract",
        side_effect=RuntimeError("LLM unavailable"),
    ):
        result = memory_client.add(
            CHAT_CONTENT,
            user_id="test_user",
            content_type="chat",
        )

    # Raw storage succeeded
    assert result["status"] == "added"
    # No extracted items key since it failed gracefully
    assert "extracted_items" not in result
    # Verify raw content is searchable
    assert memory_client.count() >= 1


def test_extracted_items_have_group_id_linking(memory_client):
    """Extracted items share group_id with raw chunks and have correct metadata."""
    with patch.object(memory_client.chat_summarizer, "extract", return_value=SUMMARY_RESULT):
        result = memory_client.add(
            CHAT_CONTENT,
            user_id="test_user",
            content_type="chat",
        )

    extracted = result["extracted_items"]
    # Get the first added item
    added_item = next(i for i in extracted if i["status"] == "added")

    # Retrieve from storage and verify metadata
    memory = memory_client.get(added_item["memory_id"])
    assert memory is not None
    meta = memory["metadata"]
    assert meta["content_type"] == "chat_extract"
    assert meta["extract_category"] in {
        "outcome", "facts", "decisions", "opinions", "tasks", "goals", "preferences",
    }
    assert meta["source_content_type"] == "chat"
    assert "group_id" in meta

    # Verify group_id links to raw chunks — all extracted items share the same group_id
    group_id = meta["group_id"]
    for item in extracted:
        if item["status"] == "added":
            m = memory_client.get(item["memory_id"])
            assert m["metadata"]["group_id"] == group_id


# --- Content type detection tests ---

def test_detect_iso_datetime_dash_speaker_as_chat():
    """ISO date + time + em-dash + speaker pattern is detected as chat."""
    text = (
        "2026-01-26 16:42 — Alex: We need to decide which channel to use.\n\n"
        "2026-01-26 16:45 — Maya: LINE is way more common here.\n\n"
        "2026-01-26 16:48 — Ken: LINE is better for early pilots.\n"
    )
    assert detect_content_type(text) == ContentType.CHAT


def test_detect_iso_datetime_hyphen_speaker_as_chat():
    """ISO date + time + regular dash + speaker pattern is detected as chat."""
    text = (
        "2026-01-26 16:42 - Alex: First message.\n"
        "2026-01-26 16:45 - Maya: Second message.\n"
    )
    assert detect_content_type(text) == ContentType.CHAT


# --- Search collapse tests ---

def test_search_collapses_chat_results_to_outcome(memory_client):
    """Search returns outcome instead of raw chat chunks, with related_memories."""
    with patch.object(memory_client.chat_summarizer, "extract", return_value=SUMMARY_RESULT):
        add_result = memory_client.add(
            CHAT_CONTENT,
            user_id="test_user",
            content_type="chat",
        )

    assert "outcome" in add_result

    # Search should return the outcome, not the raw chunks
    results = memory_client.search("PostgreSQL", user_id="test_user")
    assert len(results) >= 1

    # Find the chat-related result — it should be the outcome
    chat_results = [
        r for r in results
        if r["metadata"].get("content_type") == "chat_extract"
        and r["metadata"].get("extract_category") == "outcome"
    ]
    assert len(chat_results) == 1
    outcome_result = chat_results[0]

    # Should have related_memories linking to sibling items
    assert "related_memories" in outcome_result
    assert len(outcome_result["related_memories"]) > 0

    # Related memories should not include the outcome itself
    related_ids = {r["id"] for r in outcome_result["related_memories"]}
    assert outcome_result["id"] not in related_ids


def test_search_does_not_collapse_non_chat_results(memory_client):
    """Non-chat results pass through search without collapsing."""
    memory_client.add("PostgreSQL is a relational database.", user_id="test_user")
    memory_client.add("PostgreSQL supports JSON columns.", user_id="test_user")

    results = memory_client.search("PostgreSQL", user_id="test_user")
    assert len(results) == 2
    # Neither should have related_memories
    for r in results:
        assert "related_memories" not in r


def test_get_related_memories(memory_client):
    """get_related_memories returns all siblings for a group_id."""
    with patch.object(memory_client.chat_summarizer, "extract", return_value=SUMMARY_RESULT):
        add_result = memory_client.add(
            CHAT_CONTENT,
            user_id="test_user",
            content_type="chat",
        )

    # Get the group_id from any extracted item
    extracted = add_result["extracted_items"]
    first_added = next(i for i in extracted if i["status"] == "added")
    mem = memory_client.get(first_added["memory_id"])
    group_id = mem["metadata"]["group_id"]

    related = memory_client.get_related_memories(group_id)
    # Should include raw chunk(s) + outcome + 6 category items
    assert len(related) >= 7

    # All should share the same group_id
    for r in related:
        assert r["metadata"]["group_id"] == group_id


# --- List/table tests ---

from app.core.list_classifier import ListClassification


MOCK_LIST_CLASSIFICATION = ListClassification(
    category="suppliers",
    schema="name, location",
    key_field="name",
)


def test_add_table_detected_and_classified(memory_client):
    """Markdown table is detected and stored with classification."""
    table_content = """| Supplier | Location |
|----------|----------|
| NSL Industry | Thailand |
| Asahi-Thai | Thailand |"""

    with patch.object(
        memory_client.list_classifier, "classify", return_value=MOCK_LIST_CLASSIFICATION
    ):
        with patch.object(
            memory_client.list_classifier, "normalize_category", return_value="suppliers"
        ):
            result = memory_client.add(table_content, user_id="test_user")

    assert result["status"] == "added"
    assert result["content_type"] == "table"
    assert result["list_category"] == "suppliers"

    # Verify stored metadata
    mem = memory_client.get(result["memory_id"])
    assert mem["metadata"]["content_type"] == "table"
    assert mem["metadata"]["list_category"] == "suppliers"
    assert mem["metadata"]["list_format"] == "markdown_table"


def test_add_bullet_list_detected(memory_client):
    """Bullet list is detected and stored without chunking."""
    list_content = """- First supplier
- Second supplier
- Third supplier
- Fourth supplier"""

    with patch.object(
        memory_client.list_classifier,
        "classify",
        return_value=ListClassification(category="suppliers", schema="name", key_field="name"),
    ):
        with patch.object(
            memory_client.list_classifier, "normalize_category", return_value="suppliers"
        ):
            result = memory_client.add(list_content, user_id="test_user")

    assert result["status"] == "added"
    assert result["content_type"] == "list"

    # Content should be stored as-is (no chunking)
    mem = memory_client.get(result["memory_id"])
    assert "First supplier" in mem["content"]
    assert "Fourth supplier" in mem["content"]


def test_get_lists_by_category(memory_client):
    """get_lists returns lists filtered by category."""
    table1 = """| Name | Country |
|------|---------|
| Acme | USA |"""

    table2 = """| Name | Country |
|------|---------|
| Beta | Canada |"""

    with patch.object(
        memory_client.list_classifier,
        "classify",
        return_value=ListClassification(category="suppliers", schema="name, country", key_field="name"),
    ):
        with patch.object(
            memory_client.list_classifier, "normalize_category", return_value="suppliers"
        ):
            memory_client.add(table1, user_id="test_user")
            memory_client.add(table2, user_id="test_user")

    lists = memory_client.get_lists(category="suppliers")
    assert len(lists) == 2


def test_merge_tables(memory_client):
    """merge_lists combines tables and dedupes by key field."""
    table1 = """| Name | Location |
|------|----------|
| NSL | Thailand |
| Asahi | Thailand |"""

    table2 = """| Name | Location |
|------|----------|
| NSL | Thailand |
| DALI | Thailand |"""

    with patch.object(
        memory_client.list_classifier,
        "classify",
        return_value=ListClassification(category="suppliers", schema="name, location", key_field="name"),
    ):
        with patch.object(
            memory_client.list_classifier, "normalize_category", return_value="suppliers"
        ):
            memory_client.add(table1, user_id="test_user")
            memory_client.add(table2, user_id="test_user")

    result = memory_client.merge_lists(category="suppliers")

    assert result["item_count"] == 3  # NSL, Asahi, DALI (NSL deduped)
    assert len(result["source_ids"]) == 2
    assert "NSL" in result["merged_content"]
    assert "Asahi" in result["merged_content"]
    assert "DALI" in result["merged_content"]


def test_merge_bullet_lists(memory_client):
    """merge_lists combines bullet lists and dedupes."""
    list1 = """- Apple
- Banana
- Cherry"""

    list2 = """- Banana
- Date
- Elderberry"""

    with patch.object(
        memory_client.list_classifier,
        "classify",
        return_value=ListClassification(category="fruits", schema="name", key_field="name"),
    ):
        with patch.object(
            memory_client.list_classifier, "normalize_category", return_value="fruits"
        ):
            memory_client.add(list1, user_id="test_user")
            memory_client.add(list2, user_id="test_user")

    result = memory_client.merge_lists(category="fruits")

    assert result["item_count"] == 5  # Apple, Banana, Cherry, Date, Elderberry
    assert "Apple" in result["merged_content"]
    assert "Date" in result["merged_content"]


def test_get_list_categories(memory_client):
    """get_list_categories returns all unique categories."""
    table = """| Name | Price |
|------|-------|
| Item | $10 |"""

    with patch.object(
        memory_client.list_classifier,
        "classify",
        return_value=ListClassification(category="products", schema="name, price", key_field="name"),
    ):
        with patch.object(
            memory_client.list_classifier, "normalize_category", return_value="products"
        ):
            memory_client.add(table, user_id="test_user")

    categories = memory_client.get_list_categories()
    assert "products" in categories


# --- Excel tests ---

import io
import pandas as pd


@pytest.fixture
def sample_excel_bytes():
    """Create sample Excel bytes for testing."""
    df = pd.DataFrame({
        "Supplier": ["NSL Industry", "Asahi-Thai"],
        "Location": ["Thailand", "Thailand"],
    })
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, sheet_name="Suppliers")
    return buffer.getvalue()


@pytest.fixture
def sample_excel_bytes_2():
    """Create another sample Excel with overlapping and new data."""
    df = pd.DataFrame({
        "Supplier": ["NSL Industry", "DALI Corp"],  # NSL is duplicate
        "Location": ["Thailand", "Japan"],
    })
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, sheet_name="Suppliers")
    return buffer.getvalue()


def test_add_excel_creates_table_memories(memory_client, sample_excel_bytes):
    """Test that add_excel creates table memories for each sheet."""
    with patch.object(
        memory_client.list_classifier,
        "classify",
        return_value=ListClassification(category="suppliers", schema="supplier, location", key_field="supplier"),
    ):
        with patch.object(
            memory_client.list_classifier, "normalize_category", return_value="suppliers"
        ):
            result = memory_client.add_excel(
                file_bytes=sample_excel_bytes,
                user_id="test_user",
            )

    assert result["status"] == "added"
    assert result["sheet_count"] == 1
    assert result["total_rows"] == 2
    assert result["new_count"] == 1
    assert len(result["memory_ids"]) == 1

    # Verify memory was stored
    mem_id = result["memory_ids"][0]
    memory = memory_client.get(mem_id)
    assert memory is not None
    assert memory["metadata"]["content_type"] == "table"
    assert memory["metadata"]["list_format"] == "excel"
    assert memory["metadata"]["list_category"] == "suppliers"


def test_add_excel_auto_merge_appends_rows(memory_client, sample_excel_bytes, sample_excel_bytes_2):
    """Test that auto_merge appends rows to existing lists."""
    with patch.object(
        memory_client.list_classifier,
        "classify",
        return_value=ListClassification(category="suppliers", schema="supplier, location", key_field="supplier"),
    ):
        with patch.object(
            memory_client.list_classifier, "normalize_category", return_value="suppliers"
        ):
            # Add first Excel
            result1 = memory_client.add_excel(
                file_bytes=sample_excel_bytes,
                user_id="test_user",
            )
            first_id = result1["memory_ids"][0]

            # Add second Excel with auto_merge
            result2 = memory_client.add_excel(
                file_bytes=sample_excel_bytes_2,
                user_id="test_user",
                auto_merge=True,
            )

    assert result2["status"] == "added"
    assert result2["merged_count"] == 1
    assert result2["new_count"] == 0

    # Check sheet result details
    sheet_result = result2["sheets"][0]
    assert sheet_result["status"] == "merged"
    assert sheet_result["rows_added"] == 1  # DALI Corp
    assert sheet_result["rows_skipped"] == 1  # NSL Industry (duplicate)

    # Verify merged content
    memory = memory_client.get(first_id)
    assert "NSL Industry" in memory["content"]
    assert "Asahi-Thai" in memory["content"]
    assert "DALI Corp" in memory["content"]


def test_add_excel_dedupe_by_file_hash(memory_client, sample_excel_bytes):
    """Test that duplicate files are detected by hash."""
    with patch.object(
        memory_client.list_classifier,
        "classify",
        return_value=ListClassification(category="suppliers", schema="supplier, location", key_field="supplier"),
    ):
        with patch.object(
            memory_client.list_classifier, "normalize_category", return_value="suppliers"
        ):
            # Add first time
            result1 = memory_client.add_excel(
                file_bytes=sample_excel_bytes,
                user_id="test_user",
            )
            assert result1["status"] == "added"

            # Add same file again
            result2 = memory_client.add_excel(
                file_bytes=sample_excel_bytes,
                user_id="test_user",
            )

    assert result2["status"] == "duplicate"
    assert result2["match_type"] == "exact_file"


def test_add_excel_classifies_each_sheet(memory_client):
    """Test that each sheet is classified with category/schema."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df = pd.DataFrame({"Name": ["Alice"], "Email": ["alice@test.com"]})
        df.to_excel(writer, index=False, sheet_name="Contacts")

    with patch.object(
        memory_client.list_classifier,
        "classify",
        return_value=ListClassification(category="contacts", schema="name, email", key_field="email"),
    ) as mock_classify:
        with patch.object(
            memory_client.list_classifier, "normalize_category", return_value="contacts"
        ):
            result = memory_client.add_excel(
                file_bytes=buffer.getvalue(),
                user_id="test_user",
            )

    # Verify classify was called
    mock_classify.assert_called()

    # Verify category was set
    mem = memory_client.get(result["memory_ids"][0])
    assert mem["metadata"]["list_category"] == "contacts"
    assert mem["metadata"]["list_schema"] == "name, email"
    assert mem["metadata"]["list_key_field"] == "email"


def test_add_excel_no_merge_when_disabled(memory_client, sample_excel_bytes, sample_excel_bytes_2):
    """Test that auto_merge=False creates new lists instead of merging."""
    with patch.object(
        memory_client.list_classifier,
        "classify",
        return_value=ListClassification(category="suppliers", schema="supplier, location", key_field="supplier"),
    ):
        with patch.object(
            memory_client.list_classifier, "normalize_category", return_value="suppliers"
        ):
            # Add first Excel
            result1 = memory_client.add_excel(
                file_bytes=sample_excel_bytes,
                user_id="test_user",
            )

            # Add second Excel with auto_merge disabled
            result2 = memory_client.add_excel(
                file_bytes=sample_excel_bytes_2,
                user_id="test_user",
                auto_merge=False,
                dedupe=False,  # Disable file hash dedupe too
            )

    assert result2["status"] == "added"
    assert result2["new_count"] == 1
    assert result2["merged_count"] == 0

    # Should have two separate memories
    lists = memory_client.get_lists(category="suppliers")
    assert len(lists) == 2


def test_add_excel_stores_file(memory_client, sample_excel_bytes):
    """Test that Excel file is stored in files directory."""
    with patch.object(
        memory_client.list_classifier,
        "classify",
        return_value=ListClassification(category="suppliers", schema="supplier, location", key_field="supplier"),
    ):
        with patch.object(
            memory_client.list_classifier, "normalize_category", return_value="suppliers"
        ):
            result = memory_client.add_excel(
                file_bytes=sample_excel_bytes,
                user_id="test_user",
                filename="test_suppliers.xlsx",
            )

    assert "file_path" in result
    assert result["file_path"].endswith("test_suppliers.xlsx")

    # Verify metadata contains file info
    mem = memory_client.get(result["memory_ids"][0])
    assert mem["metadata"]["excel_file_path"] == result["file_path"]
    assert "excel_file_hash" in mem["metadata"]


def test_add_excel_handles_csv(memory_client):
    """Test that CSV files are processed correctly."""
    csv_content = "Name,Department\nAlice,Engineering\nBob,Sales\n"
    csv_bytes = csv_content.encode("utf-8")

    with patch.object(
        memory_client.list_classifier,
        "classify",
        return_value=ListClassification(category="employees", schema="name, department", key_field="name"),
    ):
        with patch.object(
            memory_client.list_classifier, "normalize_category", return_value="employees"
        ):
            result = memory_client.add_excel(
                file_bytes=csv_bytes,
                user_id="test_user",
                filename="employees.csv",
            )

    assert result["status"] == "added"
    assert result["sheet_count"] == 1
    assert result["total_rows"] == 2

    mem = memory_client.get(result["memory_ids"][0])
    assert "Alice" in mem["content"]
    assert "Engineering" in mem["content"]
