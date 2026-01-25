"""Pydantic request/response models."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Attachment(BaseModel):
    """Email attachment metadata."""

    filename: str
    content_type: Optional[str] = None
    size: int


class EmailDocument(BaseModel):
    """Parsed email document."""

    message_id: str
    subject: Optional[str] = None
    body_text: Optional[str] = None
    body_html: Optional[str] = None
    sender: Optional[str] = None
    to: List[str] = Field(default_factory=list)
    cc: List[str] = Field(default_factory=list)
    bcc: List[str] = Field(default_factory=list)
    date: Optional[datetime] = None
    thread_id: Optional[str] = None
    attachments: List[Attachment] = Field(default_factory=list)


class EmailChunk(BaseModel):
    """Chunked email content for embedding."""

    chunk_id: str
    email_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Response from email ingestion."""

    email_id: str
    chunks_created: int
    status: str


class SearchRequest(BaseModel):
    """Search request body."""

    query: str
    n_results: int = Field(default=10, ge=1, le=100)
    filters: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Single search result."""

    email_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search response with ranked results."""

    query: str
    results: List[SearchResult]
    total: int


class EmailSummary(BaseModel):
    """Summary of an email for list responses."""

    email_id: str
    subject: Optional[str] = None
    sender: Optional[str] = None
    date: Optional[datetime] = None
    snippet: Optional[str] = None


class EmailListResponse(BaseModel):
    """Paginated list of emails."""

    emails: List[EmailSummary]
    total: int
    limit: int
    offset: int


class APIResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = True
    data: Any = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
