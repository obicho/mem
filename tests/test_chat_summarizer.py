"""Tests for ChatSummarizer service."""

import json
import pytest
from unittest.mock import MagicMock, patch

from mem.core.chat_summarizer import (
    ChatExtractedItem,
    ChatSummarizer,
    ChatSummaryResult,
    _strip_markdown_fences,
)


SAMPLE_LLM_RESPONSE = json.dumps({
    "outcome": "The team decided to use PostgreSQL and aims to reduce response time by 40%.",
    "facts": ["Acme Corp has 500 employees.", "The project deadline is March 15."],
    "decisions": ["The team decided to use PostgreSQL for the database."],
    "opinions": ["Sarah thinks the new design is too complex."],
    "tasks": ["John will prepare the quarterly report by Friday."],
    "goals": ["The team aims to reduce response time by 40%."],
    "preferences": ["The client prefers weekly status updates over daily ones."],
})


@pytest.fixture
def mock_openai_client():
    """Mock the OpenAI client for ChatSummarizer."""
    with patch("mem.core.chat_summarizer.OpenAI") as mock_cls:
        client_instance = MagicMock()
        mock_cls.return_value = client_instance

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = SAMPLE_LLM_RESPONSE
        client_instance.chat.completions.create.return_value = response

        yield client_instance


@pytest.fixture
def summarizer(mock_openai_client):
    """Create a ChatSummarizer with mocked OpenAI."""
    return ChatSummarizer(api_key="test-key")


def test_extract_returns_all_categories(summarizer):
    """Verify outcome and all 6 categories are populated from a rich chat."""
    chat = (
        "alice: Acme Corp has 500 employees.\n"
        "bob: The project deadline is March 15.\n"
        "alice: Let's use PostgreSQL for the database.\n"
        "bob: I think the new design is too complex.\n"
        "alice: John, please prepare the quarterly report by Friday.\n"
        "bob: We should aim to reduce response time by 40%.\n"
        "alice: The client prefers weekly status updates.\n"
    )
    result = summarizer.extract(chat)

    assert isinstance(result, ChatSummaryResult)
    assert result.outcome != ""
    assert len(result.facts) == 2
    assert len(result.decisions) == 1
    assert len(result.opinions) == 1
    assert len(result.tasks) == 1
    assert len(result.goals) == 1
    assert len(result.preferences) == 1


def test_extract_all_items_flattened(summarizer):
    """Verify all_items() returns a flat list with correct types."""
    chat = "alice: We decided on PostgreSQL.\nbob: Sounds good."
    result = summarizer.extract(chat)

    items = result.all_items()
    assert isinstance(items, list)
    assert all(isinstance(item, ChatExtractedItem) for item in items)
    # Total should be 2+1+1+1+1+1 = 7 from SAMPLE_LLM_RESPONSE
    assert len(items) == 7

    categories = {item.category for item in items}
    assert categories == {"facts", "decisions", "opinions", "tasks", "goals", "preferences"}


def test_extract_empty_chat_raises():
    """ValueError on empty input."""
    with patch("mem.core.chat_summarizer.OpenAI"):
        summarizer = ChatSummarizer(api_key="test-key")

    with pytest.raises(ValueError, match="Chat text cannot be empty"):
        summarizer.extract("")

    with pytest.raises(ValueError, match="Chat text cannot be empty"):
        summarizer.extract("   ")


def test_extract_handles_markdown_fences(mock_openai_client):
    """JSON wrapped in code fences should still parse."""
    fenced_response = f"```json\n{SAMPLE_LLM_RESPONSE}\n```"
    mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
        fenced_response
    )

    summarizer = ChatSummarizer(api_key="test-key")
    result = summarizer.extract("alice: hello\nbob: hi")

    assert len(result.facts) == 2
    assert len(result.decisions) == 1


def test_strip_markdown_fences_plain():
    """Plain JSON passes through unchanged."""
    raw = '{"facts": []}'
    assert _strip_markdown_fences(raw) == raw


def test_strip_markdown_fences_with_json_tag():
    """Fences with ```json tag are stripped."""
    raw = '```json\n{"facts": []}\n```'
    assert _strip_markdown_fences(raw) == '{"facts": []}'


def test_strip_markdown_fences_without_tag():
    """Fences without language tag are stripped."""
    raw = '```\n{"facts": []}\n```'
    assert _strip_markdown_fences(raw) == '{"facts": []}'


def test_chat_summary_result_empty():
    """ChatSummaryResult with all empty lists returns no items."""
    result = ChatSummaryResult()
    assert result.outcome == ""
    assert result.all_items() == []


def test_chat_summary_result_partial():
    """ChatSummaryResult with only some categories populated."""
    result = ChatSummaryResult(facts=["The sky is blue."], tasks=["Buy milk."])
    items = result.all_items()
    assert len(items) == 2
    assert items[0].category == "facts"
    assert items[0].content == "The sky is blue."
    assert items[1].category == "tasks"
    assert items[1].content == "Buy milk."
