"""Tests for ListClassifier service and list detection."""

import json
import pytest
from unittest.mock import MagicMock, patch

from app.core.list_classifier import (
    ListClassifier,
    ListClassification,
    detect_list_or_table,
)


# --- Detection tests ---

def test_detect_markdown_table():
    """Markdown table is detected."""
    content = """| Supplier | Location | Specialty |
|----------|----------|-----------|
| NSL Industry | Thailand | Hot forging |
| Asahi-Thai | Thailand | Brass manufacturing |"""

    result = detect_list_or_table(content)
    assert result is not None
    assert result["content_type"] == "table"
    assert result["list_format"] == "markdown_table"


def test_detect_bullet_list():
    """Bullet list is detected."""
    content = """- First item
- Second item
- Third item
- Fourth item"""

    result = detect_list_or_table(content)
    assert result is not None
    assert result["content_type"] == "list"
    assert result["list_format"] == "bullet"


def test_detect_numbered_list():
    """Numbered list is detected."""
    content = """1. First task
2. Second task
3. Third task
4. Fourth task"""

    result = detect_list_or_table(content)
    assert result is not None
    assert result["content_type"] == "list"
    assert result["list_format"] == "numbered"


def test_detect_asterisk_bullet_list():
    """Asterisk bullet list is detected."""
    content = """* Item one
* Item two
* Item three"""

    result = detect_list_or_table(content)
    assert result is not None
    assert result["content_type"] == "list"
    assert result["list_format"] == "bullet"


def test_detect_numbered_with_parens():
    """Numbered list with parentheses is detected."""
    content = """1) First thing
2) Second thing
3) Third thing"""

    result = detect_list_or_table(content)
    assert result is not None
    assert result["content_type"] == "list"
    assert result["list_format"] == "numbered"


def test_detect_prose_not_list():
    """Regular prose is not detected as list."""
    content = """This is a paragraph of text that talks about various things.
It has multiple lines but they are not list items.
Some text might have a - dash but that doesn't make it a list."""

    result = detect_list_or_table(content)
    assert result is None


def test_detect_short_content_not_list():
    """Very short content is not detected as list."""
    content = "- Single item"
    result = detect_list_or_table(content)
    assert result is None


def test_detect_mixed_content_not_list():
    """Content with some bullets but mostly prose is not a list."""
    content = """This is an introduction paragraph.

Here's more text explaining things.

- One bullet point

And then more prose continues here with
multiple lines of regular text."""

    result = detect_list_or_table(content)
    assert result is None


# --- Classification tests ---

SAMPLE_CLASSIFICATION = json.dumps({
    "category": "suppliers",
    "schema": "name, location, specialty",
    "key_field": "name",
})


@pytest.fixture
def mock_openai_client():
    """Mock the OpenAI client for ListClassifier."""
    with patch("app.core.list_classifier.OpenAI") as mock_cls:
        client_instance = MagicMock()
        mock_cls.return_value = client_instance

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = SAMPLE_CLASSIFICATION
        client_instance.chat.completions.create.return_value = response

        yield client_instance


@pytest.fixture
def classifier(mock_openai_client):
    """Create a ListClassifier with mocked OpenAI."""
    return ListClassifier(api_key="test-key")


def test_classify_returns_classification(classifier):
    """Classify returns ListClassification with expected fields."""
    content = """| Supplier | Location |
|----------|----------|
| NSL | Thailand |"""

    result = classifier.classify(content)

    assert isinstance(result, ListClassification)
    assert result.category == "suppliers"
    assert result.schema == "name, location, specialty"
    assert result.key_field == "name"


def test_classify_handles_markdown_fences(mock_openai_client):
    """Classification handles JSON wrapped in code fences."""
    mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
        f"```json\n{SAMPLE_CLASSIFICATION}\n```"
    )

    classifier = ListClassifier(api_key="test-key")
    result = classifier.classify("some list content")

    assert result.category == "suppliers"


def test_normalize_category_exact_match(classifier, mock_openai_client):
    """Exact match returns the existing category."""
    result = classifier.normalize_category("suppliers", ["suppliers", "contacts"])
    assert result == "suppliers"


def test_normalize_category_calls_llm_for_fuzzy(classifier, mock_openai_client):
    """Non-exact match calls LLM for normalization."""
    mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
        "suppliers"
    )

    result = classifier.normalize_category("vendor_list", ["suppliers", "contacts"])
    assert result == "suppliers"


def test_normalize_category_empty_existing(classifier):
    """Empty existing categories returns new category as-is."""
    result = classifier.normalize_category("new_category", [])
    assert result == "new_category"
