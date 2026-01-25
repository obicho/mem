"""ChromaDB client and operations."""

from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.models.schemas import EmailChunk, EmailDocument, EmailSummary, SearchResult

COLLECTION_NAME = "emails"


class ChromaDBClient:
    """Client for ChromaDB operations."""

    def __init__(self, persist_dir: Optional[str] = None):
        settings = get_settings()
        persist_path = persist_dir or settings.chroma_persist_dir

        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Email chunks for semantic search"},
        )

    def add_email(
        self,
        email_doc: EmailDocument,
        chunks: List[EmailChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Add email chunks to the collection."""
        if not chunks or not embeddings:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            metadata: Dict[str, Any] = {
                "email_id": chunk.email_id,
                "chunk_index": chunk.chunk_index,
            }
            # Add email-level metadata
            if email_doc.subject:
                metadata["subject"] = email_doc.subject
            if email_doc.sender:
                metadata["sender"] = email_doc.sender
            if email_doc.date:
                metadata["date"] = email_doc.date.isoformat()
            if email_doc.thread_id:
                metadata["thread_id"] = email_doc.thread_id
            if email_doc.to:
                metadata["to"] = ",".join(email_doc.to)

            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def get_email(self, email_id: str) -> Optional[Dict[str, Any]]:
        """Get all chunks for an email by ID."""
        results = self.collection.get(
            where={"email_id": email_id},
            include=["documents", "metadatas"],
        )

        if not results["ids"]:
            return None

        chunks = []
        metadata = {}
        for i, chunk_id in enumerate(results["ids"]):
            chunk_metadata = results["metadatas"][i] if results["metadatas"] else {}
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": chunk_metadata,
                }
            )
            # Extract email-level metadata from first chunk
            if i == 0:
                metadata = {
                    k: v
                    for k, v in chunk_metadata.items()
                    if k not in ("chunk_index",)
                }

        return {
            "email_id": email_id,
            "chunks": sorted(chunks, key=lambda x: x["metadata"].get("chunk_index", 0)),
            "metadata": metadata,
        }

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar chunks."""
        where_clause = None
        if filters:
            where_clause = self._build_where_clause(filters)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # Convert distance to similarity score (assuming L2 distance)
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 / (1 + distance)  # Convert distance to similarity

                metadata = (
                    results["metadatas"][0][i]
                    if results["metadatas"] and results["metadatas"][0]
                    else {}
                )

                search_results.append(
                    SearchResult(
                        email_id=metadata.get("email_id", ""),
                        chunk_id=chunk_id,
                        content=results["documents"][0][i]
                        if results["documents"] and results["documents"][0]
                        else "",
                        score=score,
                        metadata=metadata,
                    )
                )

        return search_results

    def delete_email(self, email_id: str) -> bool:
        """Delete all chunks for an email."""
        results = self.collection.get(
            where={"email_id": email_id},
        )

        if not results["ids"]:
            return False

        self.collection.delete(ids=results["ids"])
        return True

    def list_emails(
        self,
        limit: int = 20,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[EmailSummary], int]:
        """List unique emails with pagination."""
        where_clause = None
        if filters:
            where_clause = self._build_where_clause(filters)

        # Get all matching chunks
        results = self.collection.get(
            where=where_clause,
            include=["metadatas", "documents"],
        )

        # Group by email_id and get unique emails
        email_map: Dict[str, Dict[str, Any]] = {}
        if results["ids"]:
            for i, _ in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                email_id = metadata.get("email_id", "")

                if email_id and email_id not in email_map:
                    document = (
                        results["documents"][i] if results["documents"] else ""
                    )
                    snippet = document[:200] + "..." if len(document) > 200 else document

                    email_map[email_id] = {
                        "email_id": email_id,
                        "subject": metadata.get("subject"),
                        "sender": metadata.get("sender"),
                        "date": metadata.get("date"),
                        "snippet": snippet,
                    }

        # Sort by date descending (if available)
        emails = sorted(
            email_map.values(),
            key=lambda x: x.get("date") or "",
            reverse=True,
        )

        total = len(emails)
        paginated = emails[offset : offset + limit]

        summaries = [
            EmailSummary(
                email_id=e["email_id"],
                subject=e.get("subject"),
                sender=e.get("sender"),
                date=e.get("date"),
                snippet=e.get("snippet"),
            )
            for e in paginated
        ]

        return summaries, total

    def _build_where_clause(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause from filters."""
        conditions = []

        if "sender" in filters:
            conditions.append({"sender": filters["sender"]})

        if "date_from" in filters:
            conditions.append({"date": {"$gte": filters["date_from"]}})

        if "date_to" in filters:
            conditions.append({"date": {"$lte": filters["date_to"]}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}


_db_client: Optional[ChromaDBClient] = None


def get_db_client() -> ChromaDBClient:
    """Get or create the ChromaDB client singleton."""
    global _db_client
    if _db_client is None:
        _db_client = ChromaDBClient()
    return _db_client
