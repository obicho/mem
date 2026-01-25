"""Text chunking for long emails."""

import re
import uuid
from typing import Any, Dict, List

from app.models.schemas import EmailChunk, EmailDocument

# Approximate tokens per chunk (targeting ~500 tokens, ~4 chars per token)
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_OVERLAP = 200


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple boundary detection."""
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _create_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Create text chunks with sentence-boundary awareness."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    sentences = _split_into_sentences(text)
    chunks = []
    current_chunk: List[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Calculate overlap
            overlap_sentences: List[str] = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s)
                else:
                    break

            current_chunk = overlap_sentences
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_email(
    email_doc: EmailDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[EmailChunk]:
    """Chunk an email document for embedding."""
    text = email_doc.body_text or ""

    # If no plain text, try to extract from HTML (basic stripping)
    if not text and email_doc.body_html:
        text = re.sub(r"<[^>]+>", " ", email_doc.body_html)
        text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Add subject as context prefix
    if email_doc.subject:
        text = f"Subject: {email_doc.subject}\n\n{text}"

    chunks_text = _create_chunks(text, chunk_size, overlap)

    email_chunks = []
    for idx, chunk_text in enumerate(chunks_text):
        chunk_id = f"{email_doc.message_id}_{idx}"

        metadata: Dict[str, Any] = {
            "subject": email_doc.subject,
            "sender": email_doc.sender,
            "to": email_doc.to,
            "chunk_index": idx,
            "total_chunks": len(chunks_text),
        }
        if email_doc.date:
            metadata["date"] = email_doc.date.isoformat()
        if email_doc.thread_id:
            metadata["thread_id"] = email_doc.thread_id

        email_chunks.append(
            EmailChunk(
                chunk_id=chunk_id,
                email_id=email_doc.message_id,
                content=chunk_text,
                chunk_index=idx,
                metadata=metadata,
            )
        )

    return email_chunks


def generate_chunk_id() -> str:
    """Generate a unique chunk ID."""
    return str(uuid.uuid4())
