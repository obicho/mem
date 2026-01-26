"""PDF parsing service for extracting text from PDF files."""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pypdf import PdfReader


@dataclass
class ParsedPDF:
    """Parsed PDF document."""

    content: str
    page_count: int
    metadata: dict
    pages: list[str]  # Text per page


class PDFParser:
    """Service for parsing PDF files and extracting text."""

    def parse(
        self,
        file_path: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
    ) -> ParsedPDF:
        """Parse a PDF file and extract text.

        Args:
            file_path: Path to the PDF file.
            file_bytes: Raw PDF bytes (alternative to file_path).

        Returns:
            ParsedPDF with extracted content and metadata.

        Raises:
            ValueError: If no input is provided.
        """
        if file_path:
            reader = PdfReader(file_path)
        elif file_bytes:
            reader = PdfReader(io.BytesIO(file_bytes))
        else:
            raise ValueError("Must provide either file_path or file_bytes")

        # Extract text from each page
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text.strip())

        # Combine all text
        full_content = "\n\n".join(pages)

        # Extract PDF metadata
        pdf_metadata = {}
        if reader.metadata:
            if reader.metadata.title:
                pdf_metadata["title"] = reader.metadata.title
            if reader.metadata.author:
                pdf_metadata["author"] = reader.metadata.author
            if reader.metadata.subject:
                pdf_metadata["subject"] = reader.metadata.subject
            if reader.metadata.creator:
                pdf_metadata["creator"] = reader.metadata.creator

        return ParsedPDF(
            content=full_content,
            page_count=len(reader.pages),
            metadata=pdf_metadata,
            pages=pages,
        )

    def parse_with_page_chunks(
        self,
        file_path: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
    ) -> list[dict]:
        """Parse PDF and return chunks by page.

        Each page becomes a separate chunk with page number in metadata.

        Args:
            file_path: Path to the PDF file.
            file_bytes: Raw PDF bytes.

        Returns:
            List of chunk dicts with 'content' and 'metadata' keys.
        """
        parsed = self.parse(file_path=file_path, file_bytes=file_bytes)

        chunks = []
        for i, page_text in enumerate(parsed.pages):
            if not page_text.strip():
                continue

            chunks.append({
                "content": page_text,
                "metadata": {
                    "page_number": i + 1,
                    "total_pages": parsed.page_count,
                    **parsed.metadata,
                },
            })

        return chunks
