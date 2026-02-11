"""Auto-detecting text chunking for emails, chats, and documents."""

import re
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

# Default chunk settings by content type
CHUNK_SETTINGS = {
    "email": {"size": 1500, "overlap": 150},
    "chat": {"size": 1500, "overlap": 2},  # overlap = number of speaker turns carried over
    "document": {"size": 2000, "overlap": 200},
}


class ContentType(Enum):
    """Detected content types."""

    EMAIL = "email"
    CHAT = "chat"
    DOCUMENT = "document"


def detect_content_type(text: str, metadata: dict[str, Any] | None = None) -> ContentType:
    """Auto-detect content type from text and metadata.

    Args:
        text: The content text
        metadata: Optional metadata dict with hints

    Returns:
        Detected ContentType
    """
    metadata = metadata or {}

    # 1. Check explicit metadata hints first (highest confidence)
    if metadata.get("content_type"):
        try:
            return ContentType(metadata["content_type"])
        except ValueError:
            pass

    # 2. Check metadata fields that indicate type
    # Chat indicators
    chat_fields = {"channel_id", "thread_ts", "slack_ts", "room_id", "chat_id"}
    if any(field in metadata for field in chat_fields):
        return ContentType.CHAT

    # Email indicators
    email_fields = {"message_id", "subject", "mail_from", "mail_to"}
    if any(field in metadata for field in email_fields):
        return ContentType.EMAIL

    # 3. Structural pattern detection
    # Email headers pattern
    email_header_pattern = r"^(From|To|Subject|Date|Cc|Bcc):\s*.+"
    if re.search(email_header_pattern, text, re.MULTILINE | re.IGNORECASE):
        return ContentType.EMAIL

    # Email quoted reply pattern
    if re.search(r"^>+ .+", text, re.MULTILINE) or re.search(
        r"On .+ wrote:", text, re.IGNORECASE
    ):
        return ContentType.EMAIL

    # Chat patterns
    # Timestamp + username pattern: "[10:30] alice: " or "10:30 AM alice:"
    chat_timestamp_pattern = r"^\[?\d{1,2}:\d{2}(?:\s*[AP]M)?\]?\s+\w+:"
    if re.search(chat_timestamp_pattern, text, re.MULTILINE):
        return ContentType.CHAT

    # ISO date + time + separator + speaker: "2026-01-26 16:42 — Alex:" or "2026-01-26 16:42 - Alex:"
    iso_chat_pattern = r"^\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?\s*[\u2014\u2013\-]+\s*\w+:"
    if re.search(iso_chat_pattern, text, re.MULTILINE):
        return ContentType.CHAT

    # Slack/Discord style: "<@user>" or "@user:"
    if re.search(r"<@\w+>|^@\w+:", text, re.MULTILINE):
        return ContentType.CHAT

    # Multiple short lines with speaker labels
    lines = text.strip().split("\n")
    if len(lines) >= 3:
        speaker_pattern = r"^\w+:"
        speaker_lines = sum(1 for line in lines if re.match(speaker_pattern, line.strip()))
        if speaker_lines / len(lines) > 0.5:
            return ContentType.CHAT

    # 4. Check for document indicators before chat heuristics
    # Paragraph breaks suggest document
    if "\n\n" in text or "\n\r\n" in text:
        return ContentType.DOCUMENT

    # 5. Heuristics based on content characteristics
    # Very short, single line = likely chat
    if len(text) < 200 and "\n" not in text.strip():
        return ContentType.CHAT

    # Short average line length with many lines AND no paragraph breaks = chat
    non_empty_lines = [line for line in lines if line.strip()]
    if non_empty_lines and len(non_empty_lines) > 5:
        avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        if avg_line_length < 60:
            return ContentType.CHAT

    # 6. Default to document
    return ContentType.DOCUMENT


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict]:
        """Chunk text into segments.

        Args:
            text: The text to chunk
            metadata: Optional metadata

        Returns:
            List of chunk dicts with 'content' and 'metadata' keys
        """
        pass


class EmailChunker(ChunkingStrategy):
    """Chunking strategy optimized for emails."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SETTINGS["email"]["size"],
        overlap: int = CHUNK_SETTINGS["email"]["overlap"],
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict]:
        metadata = metadata or {}

        # Pre-process: clean email content
        cleaned = self._preprocess_email(text)

        if not cleaned.strip():
            return []

        # Add subject context if available
        subject = metadata.get("subject", "")
        if subject:
            cleaned = f"Subject: {subject}\n\n{cleaned}"

        # Chunk with sentence boundaries
        chunks_text = self._split_into_chunks(cleaned)

        # Build chunk objects
        chunks = []
        for idx, chunk_text in enumerate(chunks_text):
            chunk_metadata = {
                **metadata,
                "chunk_index": idx,
                "total_chunks": len(chunks_text),
                "content_type": "email",
            }
            chunks.append({"content": chunk_text, "metadata": chunk_metadata})

        return chunks

    def _preprocess_email(self, text: str) -> str:
        """Remove quotes, signatures, and noise from email."""
        lines = text.split("\n")
        cleaned_lines = []
        in_signature = False

        for line in lines:
            # Detect signature start
            if line.strip() in ("--", "___", "---", "Best,", "Thanks,", "Regards,"):
                in_signature = True

            # Skip signature lines
            if in_signature:
                continue

            # Skip quoted lines (> prefix)
            if re.match(r"^>+\s*", line):
                continue

            # Skip "On ... wrote:" lines
            if re.match(r"^On .+ wrote:$", line.strip(), re.IGNORECASE):
                continue

            # Skip forwarded message headers
            if re.match(r"^-+ ?(Forwarded|Original) (message|Message) ?-+$", line.strip()):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into chunks respecting sentence boundaries."""
        if len(text) <= self.chunk_size:
            return [text] if text else []

        sentences = self._split_sentences(text)
        chunks = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Keep overlap sentences
                overlap_chunk, overlap_len = self._get_overlap(current_chunk)
                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(sentence)
            current_length += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap(self, sentences: list[str]) -> tuple[list[str], int]:
        """Get overlap sentences from end of chunk."""
        overlap_sentences: list[str] = []
        overlap_length = 0

        for s in reversed(sentences):
            if overlap_length + len(s) <= self.overlap:
                overlap_sentences.insert(0, s)
                overlap_length += len(s)
            else:
                break

        return overlap_sentences, overlap_length


class ChatChunker(ChunkingStrategy):
    """Chunking strategy optimized for chat messages.

    Groups speaker turns into windows of ``min_turns`` to ``max_turns``
    exchanges. Consecutive windows share ``overlap_turns`` trailing turns
    so that each chunk starts with enough context to be coherent on its own.
    A hard ``chunk_size`` character limit forces a split even if the turn
    count hasn't been reached.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SETTINGS["chat"]["size"],
        min_turns: int = 5,
        max_turns: int = 7,
        overlap_turns: int = CHUNK_SETTINGS["chat"]["overlap"],
    ):
        self.chunk_size = chunk_size
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.overlap_turns = overlap_turns

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict]:
        metadata = metadata or {}

        # Parse speaker turns from text
        turns = self._parse_turns(text)

        if not turns:
            # Fall back to simple chunking if parsing fails
            return self._simple_chunk(text, metadata)

        # If everything fits in one window, return as a single chunk
        total_len = sum(len(t) for t in turns) + len(turns) - 1
        if len(turns) <= self.max_turns and total_len <= self.chunk_size:
            return [{
                "content": "\n".join(turns),
                "metadata": {
                    **metadata,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "content_type": "chat",
                    "message_count": len(turns),
                },
            }]

        # Group turns into overlapping windows
        windows = self._create_windows(turns)

        # Build chunk objects
        chunks = []
        for idx, window in enumerate(windows):
            chunk_text = "\n".join(window)
            chunk_metadata = {
                **metadata,
                "chunk_index": idx,
                "total_chunks": len(windows),
                "content_type": "chat",
                "message_count": len(window),
            }
            chunks.append({"content": chunk_text, "metadata": chunk_metadata})

        return chunks

    def _parse_turns(self, text: str) -> list[str]:
        """Parse individual speaker turns from chat text.

        A turn is one or more consecutive lines that belong to the same
        message (multi-line messages are kept together).
        """
        lines = text.strip().split("\n")
        turns: list[str] = []
        current_turn: list[str] = []

        # Patterns that indicate start of new speaker turn
        turn_start_patterns = [
            r"^\[?\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?\]?\s+\w+:",  # timestamp + user
            r"^\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?\s*[\u2014\u2013\-]+\s*\w+:",  # ISO date + dash + user
            r"^<@?\w+>:",  # Slack style
            r"^\w{2,20}:\s",  # Simple "user: message"
            r"^\[?\d{4}-\d{2}-\d{2}",  # ISO date prefix
        ]

        for line in lines:
            is_new_turn = any(
                re.match(pattern, line.strip()) for pattern in turn_start_patterns
            )

            if is_new_turn and current_turn:
                turns.append("\n".join(current_turn))
                current_turn = []

            if line.strip():
                current_turn.append(line)

        if current_turn:
            turns.append("\n".join(current_turn))

        return turns

    def _create_windows(self, turns: list[str]) -> list[list[str]]:
        """Group speaker turns into overlapping conversation windows.

        Each window contains ``max_turns`` turns (or fewer for the last
        window). Consecutive windows share ``overlap_turns`` trailing turns
        from the previous window so the next chunk starts with context.
        A hard ``chunk_size`` character limit forces an early split.
        """
        if not turns:
            return []

        windows: list[list[str]] = []
        start = 0

        while start < len(turns):
            window: list[str] = []
            window_len = 0

            for i in range(start, len(turns)):
                turn_len = len(turns[i])
                # Force split if adding this turn exceeds size (but always
                # include at least min_turns if possible)
                if window and (
                    len(window) >= self.max_turns
                    or (window_len + turn_len + 1 > self.chunk_size
                        and len(window) >= self.min_turns)
                ):
                    break
                window.append(turns[i])
                window_len += turn_len + 1  # +1 for newline

            windows.append(window)

            # Advance past the turns we consumed, minus overlap
            consumed = len(window)
            start += max(consumed - self.overlap_turns, 1)

        return windows

    def _simple_chunk(self, text: str, metadata: dict[str, Any]) -> list[dict]:
        """Fallback simple chunking by lines when turn parsing fails."""
        lines = text.strip().split("\n")
        chunks = []
        current_chunk: list[str] = []
        current_length = 0

        for line in lines:
            if current_length + len(line) > self.chunk_size and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**metadata, "content_type": "chat"},
                })
                current_chunk = []
                current_length = 0

            current_chunk.append(line)
            current_length += len(line) + 1

        if current_chunk:
            chunks.append({
                "content": "\n".join(current_chunk),
                "metadata": {**metadata, "content_type": "chat"},
            })

        return chunks


class DocumentChunker(ChunkingStrategy):
    """Chunking strategy for general documents."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SETTINGS["document"]["size"],
        overlap: int = CHUNK_SETTINGS["document"]["overlap"],
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict]:
        metadata = metadata or {}

        if not text.strip():
            return []

        # Try paragraph-based splitting first
        chunks_text = self._split_by_paragraphs(text)

        # Fall back to sentence-based if paragraphs are too large
        if any(len(chunk) > self.chunk_size * 1.5 for chunk in chunks_text):
            chunks_text = self._split_by_sentences(text)

        chunks = []
        for idx, chunk_text in enumerate(chunks_text):
            chunk_metadata = {
                **metadata,
                "chunk_index": idx,
                "total_chunks": len(chunks_text),
                "content_type": "document",
            }
            chunks.append({"content": chunk_text, "metadata": chunk_metadata})

        return chunks

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split by paragraphs, merging small ones."""
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text] if text.strip() else []

        chunks = []
        current_chunk: list[str] = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para)

            if current_length + para_len > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(para)
            current_length += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split by sentences with overlap."""
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_endings.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Overlap
                overlap_chunk: list[str] = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.overlap:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break

                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(sentence)
            current_length += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


# Strategy registry
CHUNKING_STRATEGIES: dict[ContentType, ChunkingStrategy] = {
    ContentType.EMAIL: EmailChunker(),
    ContentType.CHAT: ChatChunker(),
    ContentType.DOCUMENT: DocumentChunker(),
}


def chunk_content(
    text: str,
    metadata: dict[str, Any] | None = None,
    content_type: str | ContentType | None = None,
) -> list[dict]:
    """Auto-detect content type and chunk accordingly.

    Args:
        text: The text to chunk
        metadata: Optional metadata dict
        content_type: Optional explicit content type (auto-detected if not provided)

    Returns:
        List of chunk dicts with 'content' and 'metadata' keys
    """
    if not text or not text.strip():
        return []

    metadata = metadata or {}

    # Resolve content type
    if content_type is None:
        detected_type = detect_content_type(text, metadata)
    elif isinstance(content_type, str):
        try:
            detected_type = ContentType(content_type)
        except ValueError:
            detected_type = ContentType.DOCUMENT
    else:
        detected_type = content_type

    # Get strategy and chunk
    strategy = CHUNKING_STRATEGIES.get(detected_type, CHUNKING_STRATEGIES[ContentType.DOCUMENT])
    return strategy.chunk(text, metadata)


def generate_chunk_id() -> str:
    """Generate a unique chunk ID."""
    return str(uuid.uuid4())
