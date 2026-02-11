"""Tests for API endpoints."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from mem.main import app
from mem.config import Settings


# Mock settings for tests
@pytest.fixture
def mock_settings():
    return Settings(
        openai_api_key="test-openai-key",
        api_key="test-api-key",
        chroma_persist_dir="./test_chroma_data",
    )


@pytest.fixture
def client(mock_settings):
    with patch("mem.config.get_settings", return_value=mock_settings):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-api-key"}


def test_root_endpoint(client):
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Memory Layer API" in data["data"]["name"]


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_unauthorized_without_api_key(client):
    """Test that endpoints require API key."""
    response = client.get("/api/v1/emails")
    assert response.status_code == 422  # Missing required header


def test_unauthorized_with_wrong_api_key(client):
    """Test that wrong API key is rejected."""
    response = client.get(
        "/api/v1/emails",
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


@patch("mem.api.routes.emails.get_db")
def test_list_emails_empty(mock_get_db, client, auth_headers):
    """Test listing emails when empty."""
    mock_db = MagicMock()
    mock_db.list_emails.return_value = ([], 0)
    mock_get_db.return_value = mock_db

    response = client.get("/api/v1/emails", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["emails"] == []
    assert data["data"]["total"] == 0


@patch("mem.api.routes.search.get_db")
@patch("mem.api.routes.search.embed_text")
def test_search_endpoint(mock_embed, mock_get_db, client, auth_headers):
    """Test search endpoint."""
    mock_embed.return_value = [0.1] * 1536  # Mock embedding
    mock_db = MagicMock()
    mock_db.search.return_value = []
    mock_get_db.return_value = mock_db

    response = client.post(
        "/api/v1/search",
        headers=auth_headers,
        json={"query": "test query", "n_results": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["query"] == "test query"


def test_search_empty_query(client, auth_headers):
    """Test search with empty query is rejected."""
    response = client.post(
        "/api/v1/search",
        headers=auth_headers,
        json={"query": "   ", "n_results": 5},
    )
    assert response.status_code == 400


def test_upload_invalid_file_type(client, auth_headers):
    """Test uploading non-email file is rejected."""
    response = client.post(
        "/api/v1/emails",
        headers=auth_headers,
        files={"file": ("test.txt", b"text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "EML or MSG" in response.json()["detail"]


# --- Inbound email tests ---

@pytest.fixture
def mock_memory_client():
    """Mock Memory client for inbound email tests."""
    mock = MagicMock()
    mock.add.return_value = {"memory_id": "test-memory-id", "status": "added"}
    return mock


@pytest.fixture
def inbound_client(mock_settings, mock_memory_client):
    """Test client with mocked Memory dependency."""
    from mem.dependencies import get_memory_client

    with patch("mem.config.get_settings", return_value=mock_settings):
        app.dependency_overrides[get_memory_client] = lambda: mock_memory_client
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()


def test_inbound_email_success(inbound_client, mock_memory_client, auth_headers):
    """Test inbound email endpoint adds email to memory."""
    payload = {
        "sender": "alice@example.com",
        "recipients": ["bob@example.com"],
        "subject": "Meeting tomorrow",
        "body": "Let's meet at 3pm to discuss the project.",
        "user_id": "user123",
    }

    response = inbound_client.post(
        "/api/v1/emails/inbound",
        headers=auth_headers,
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["status"] == "added"
    assert data["data"]["memory_id"] == "test-memory-id"

    # Verify memory.add was called with correct args
    mock_memory_client.add.assert_called_once()
    call_kwargs = mock_memory_client.add.call_args.kwargs
    assert call_kwargs["user_id"] == "user123"
    assert call_kwargs["content_type"] == "email"
    assert "Meeting tomorrow" in call_kwargs["content"]


def test_inbound_email_generates_message_id(inbound_client, mock_memory_client, auth_headers):
    """Test inbound email generates message_id when not provided."""
    payload = {
        "sender": "alice@example.com",
        "body": "Hello world",
    }

    response = inbound_client.post(
        "/api/v1/emails/inbound",
        headers=auth_headers,
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["email_id"].startswith("inbound-")


def test_inbound_email_empty_body_rejected(inbound_client, auth_headers):
    """Test inbound email with empty body is rejected."""
    payload = {
        "sender": "alice@example.com",
        "body": "   ",
    }

    response = inbound_client.post(
        "/api/v1/emails/inbound",
        headers=auth_headers,
        json=payload,
    )
    assert response.status_code == 400
    assert "body cannot be empty" in response.json()["detail"]


def test_inbound_email_missing_sender_rejected(inbound_client, auth_headers):
    """Test inbound email without sender is rejected."""
    payload = {
        "body": "Hello world",
    }

    response = inbound_client.post(
        "/api/v1/emails/inbound",
        headers=auth_headers,
        json=payload,
    )
    assert response.status_code == 422  # Pydantic validation error
