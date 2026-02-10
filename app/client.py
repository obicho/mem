"""Memory client - mem0-style interface for AI agent memory."""

import hashlib
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.chat_summarizer import ChatSummarizer
from app.core.chunker import chunk_content, detect_content_type, ContentType
from app.core.embeddings import EmbeddingService
from app.core.excel_parser import ExcelParser
from app.core.feedback import Feedback, FeedbackStore
from app.core.list_classifier import ListClassifier, ListClassification, detect_list_or_table
from app.core.pdf_parser import PDFParser
from app.core.vision import VisionService


class Memory:
    """
    A mem0-style memory client for AI agents.

    Usage:
        from app.client import Memory

        # Initialize
        m = Memory(api_key="your-openai-key")

        # Add memories
        m.add("User prefers dark mode", user_id="alice")
        m.add("Meeting scheduled for Friday", user_id="alice", metadata={"type": "calendar"})

        # Search memories
        results = m.search("user preferences", user_id="alice")

        # Get all memories for a user
        memories = m.get_all(user_id="alice")

        # Delete a memory
        m.delete(memory_id="...")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        collection_name: str = "memories",
        persist_dir: str = "./chroma_data",
        embedding_model: str = "text-embedding-3-small",
        image_dir: str = "./images",
        files_dir: str = "./files",
        feedback_db: str = "./feedback.db",
        vision_model: str = "gpt-4o-mini",
    ):
        """
        Initialize the Memory client.

        Args:
            api_key: OpenAI API key for embeddings. If not provided, uses OPENAI_API_KEY env var.
            collection_name: Name of the ChromaDB collection.
            persist_dir: Directory for persistent storage.
            embedding_model: OpenAI embedding model to use.
            image_dir: Directory for storing uploaded images.
            files_dir: Directory for storing uploaded documents.
            feedback_db: Path to feedback SQLite database.
            vision_model: OpenAI vision model for image captioning.
        """
        self.embedding_service = EmbeddingService(api_key=api_key, model=embedding_model)
        self.vision_service = VisionService(api_key=api_key, model=vision_model)
        self.chat_summarizer = ChatSummarizer(api_key=api_key, model=vision_model)
        self.list_classifier = ListClassifier(api_key=api_key, model=vision_model)
        self.pdf_parser = PDFParser()
        self.excel_parser = ExcelParser()
        self.feedback_store = FeedbackStore(db_path=feedback_db)
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir = Path(files_dir)
        self.files_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "AI agent memories"},
        )

    def add(
        self,
        content: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        content_type: Optional[str] = None,
        chunking: bool = True,
        dedupe: bool = True,
        dedupe_threshold: float = 0.98,
    ) -> Dict[str, Any]:
        """
        Add a memory with auto-detection chunking and duplicate detection.

        Args:
            content: The text content to remember.
            user_id: Optional user identifier for filtering.
            agent_id: Optional agent identifier for filtering.
            run_id: Optional run/session identifier.
            metadata: Optional additional metadata.
            memory_id: Optional custom memory ID. Auto-generated if not provided.
            content_type: Optional content type hint ("email", "chat", "document").
                         Auto-detected if not provided.
            chunking: Whether to chunk long content. Default True.
            dedupe: Whether to check for duplicates. Default True.
            dedupe_threshold: Similarity threshold for duplicate detection (0-1). Default 0.95.

        Returns:
            Dict with memory_id(s) and status. If duplicate found, returns existing memory ID
            with status "duplicate".
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        # Check for duplicates if enabled
        if dedupe:
            duplicates = self.find_duplicates(content, threshold=dedupe_threshold, limit=1)
            if duplicates:
                existing = duplicates[0]
                return {
                    "memory_id": existing["id"],
                    "status": "duplicate",
                    "similarity": existing["similarity"],
                    "match_type": existing["match_type"],
                }

        # Build base metadata
        base_metadata: Dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(),
            "content_hash": self._compute_content_hash(content),
        }
        if user_id:
            base_metadata["user_id"] = user_id
        if agent_id:
            base_metadata["agent_id"] = agent_id
        if run_id:
            base_metadata["run_id"] = run_id
        if metadata:
            base_metadata.update(metadata)

        # Check for list/table content - don't chunk, classify instead
        list_info = detect_list_or_table(content)
        if list_info:
            return self._add_list(
                content=content,
                list_info=list_info,
                base_metadata=base_metadata,
                memory_id=memory_id,
                dedupe=dedupe,
                dedupe_threshold=dedupe_threshold,
            )

        # Chunk content if enabled
        if chunking:
            chunks = chunk_content(content, base_metadata, content_type)
        else:
            # No chunking - store as single memory
            detected_type = detect_content_type(content, base_metadata) if not content_type else ContentType(content_type)
            chunks = [{
                "content": content,
                "metadata": {**base_metadata, "content_type": detected_type.value},
            }]

        if not chunks:
            raise ValueError("Content produced no chunks")

        # Generate group ID for linking chunks
        group_id = self._generate_id(content) if len(chunks) > 1 else None

        memory_ids = []
        for i, chunk in enumerate(chunks):
            # Generate or use provided memory ID
            if memory_id and len(chunks) == 1:
                chunk_id = memory_id
            else:
                chunk_id = self._generate_id(chunk["content"])

            # Build chunk metadata
            chunk_metadata = chunk["metadata"].copy()
            if group_id:
                chunk_metadata["group_id"] = group_id

            # Generate embedding
            embedding = self.embedding_service.embed_text(chunk["content"])

            # Store in ChromaDB
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk["content"]],
                metadatas=[chunk_metadata],
            )
            memory_ids.append(chunk_id)

        # Build base result
        if len(memory_ids) == 1:
            result: Dict[str, Any] = {
                "memory_id": memory_ids[0],
                "status": "added",
            }
        else:
            result = {
                "memory_ids": memory_ids,
                "group_id": group_id,
                "chunks_created": len(memory_ids),
                "status": "added",
            }

        # Chat extraction: extract structured items from chat content
        detected_type = chunks[0]["metadata"].get("content_type") if chunks else None
        if detected_type == ContentType.CHAT.value:
            try:
                summary = self.chat_summarizer.extract(content)
                # Use the group_id from chunks, or generate one to link extracts to raw chunks
                link_group_id = group_id or self._generate_id(content)
                if group_id is None and len(memory_ids) == 1:
                    # Retroactively set group_id on the single raw chunk
                    existing_meta = chunks[0]["metadata"].copy()
                    existing_meta["group_id"] = link_group_id
                    self.collection.update(
                        ids=[memory_ids[0]],
                        metadatas=[existing_meta],
                    )

                extracted_items: List[Dict[str, Any]] = []

                # Store outcome as a memory if present
                if summary.outcome:
                    outcome_result = self._upsert_extracted_item(
                        content=summary.outcome,
                        category="outcome",
                        group_id=link_group_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        dedupe=dedupe,
                        dedupe_threshold=dedupe_threshold,
                    )
                    extracted_items.append(outcome_result)
                    result["outcome"] = summary.outcome

                for item in summary.all_items():
                    item_result = self._upsert_extracted_item(
                        content=item.content,
                        category=item.category,
                        group_id=link_group_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        dedupe=dedupe,
                        dedupe_threshold=dedupe_threshold,
                    )
                    extracted_items.append(item_result)

                result["extracted_items"] = extracted_items
            except Exception:
                # Summarization failure must not block raw storage
                pass

        return result

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity.

        Args:
            query: The search query.
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            run_id: Filter by run ID.
            limit: Maximum number of results.
            filters: Additional ChromaDB filters.

        Returns:
            List of matching memories with scores.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Build where clause
        where_clause = self._build_where_clause(user_id, agent_id, run_id, filters)

        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Search - fetch extra results to allow re-ranking
        fetch_limit = min(limit * 2, limit + 20)  # Fetch more for re-ranking
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_limit,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        memories = []
        if results["ids"] and results["ids"][0]:
            memory_ids = results["ids"][0]

            # Get feedback scores for all results
            feedback_scores = self.feedback_store.get_memory_scores(memory_ids)

            for i, memory_id in enumerate(memory_ids):
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity_score = 1 / (1 + distance)

                # Blend similarity score with feedback score
                # feedback_score is -1 to 1, we scale it to a boost factor
                feedback_score = feedback_scores.get(memory_id, 0.0)
                # Apply feedback as a multiplicative boost (0.8x to 1.2x)
                feedback_boost = 1.0 + (feedback_score * 0.2)
                final_score = similarity_score * feedback_boost

                memories.append({
                    "id": memory_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "score": final_score,
                    "similarity_score": similarity_score,
                    "feedback_score": feedback_score,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

            # Re-rank by final score and limit results
            memories.sort(key=lambda x: x["score"], reverse=True)
            memories = memories[:limit]

        # Collapse chat-related results: replace raw chat chunks and
        # individual extracts with a single outcome entry per group_id,
        # carrying links to the sibling memories.
        memories = self._collapse_chat_results(memories)

        return memories

    def _collapse_chat_results(
        self, memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collapse chat-related results into outcome entries.

        For each ``group_id`` that appears in chat / chat_extract results,
        keep only the outcome memory (or the highest-scored sibling when
        no outcome was stored yet) and attach ``related_ids`` so the UI
        can link to the remaining pieces.

        Non-chat results pass through unchanged.
        """
        collapsed: List[Dict[str, Any]] = []
        seen_group_ids: dict[str, int] = {}  # group_id → index in collapsed

        for mem in memories:
            meta = mem.get("metadata", {})
            content_type = meta.get("content_type", "")
            group_id = meta.get("group_id", "")

            is_chat_related = content_type in ("chat", "chat_extract")

            if not is_chat_related or not group_id:
                collapsed.append(mem)
                continue

            is_outcome = meta.get("extract_category") == "outcome"

            if group_id not in seen_group_ids:
                # First time seeing this group — fetch all siblings once
                related = self.get_related_memories(group_id)
                related_ids = [
                    {"id": r["id"], "content_type": r["metadata"].get("content_type", ""),
                     "extract_category": r["metadata"].get("extract_category", "")}
                    for r in related if r["id"] != mem["id"]
                ]

                # If this isn't the outcome, try to find and swap in the outcome
                if not is_outcome:
                    outcome_mem = next(
                        (r for r in related
                         if r["metadata"].get("extract_category") == "outcome"),
                        None,
                    )
                    if outcome_mem:
                        # Replace content with outcome, keep the best score
                        mem = {
                            **mem,
                            "id": outcome_mem["id"],
                            "content": outcome_mem["content"],
                            "metadata": outcome_mem["metadata"],
                        }
                        related_ids = [
                            {"id": r["id"],
                             "content_type": r["metadata"].get("content_type", ""),
                             "extract_category": r["metadata"].get("extract_category", "")}
                            for r in related if r["id"] != outcome_mem["id"]
                        ]

                mem["related_memories"] = related_ids
                seen_group_ids[group_id] = len(collapsed)
                collapsed.append(mem)
            # else: already have an entry for this group — skip duplicate

        return collapsed

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID.

        Args:
            memory_id: The memory ID.

        Returns:
            Memory dict or None if not found.
        """
        results = self.collection.get(
            ids=[memory_id],
            include=["documents", "metadatas"],
        )

        if not results["ids"]:
            return None

        return {
            "id": memory_id,
            "content": results["documents"][0] if results["documents"] else "",
            "metadata": results["metadatas"][0] if results["metadatas"] else {},
        }

    def get_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get all memories, optionally filtered.

        Args:
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            run_id: Filter by run ID.
            limit: Maximum number of results.

        Returns:
            List of memories.
        """
        where_clause = self._build_where_clause(user_id, agent_id, run_id)

        results = self.collection.get(
            where=where_clause,
            limit=limit,
            include=["documents", "metadatas"],
        )

        memories = []
        if results["ids"]:
            for i, memory_id in enumerate(results["ids"]):
                memories.append({
                    "id": memory_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })

        return memories

    def update(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing memory.

        Args:
            memory_id: The memory ID to update.
            content: New content.
            metadata: Optional new metadata (merged with existing).

        Returns:
            Dict with status.
        """
        existing = self.get(memory_id)
        if not existing:
            raise ValueError(f"Memory not found: {memory_id}")

        # Merge metadata
        new_metadata = existing.get("metadata", {}).copy()
        new_metadata["updated_at"] = datetime.utcnow().isoformat()
        if metadata:
            new_metadata.update(metadata)

        # Generate new embedding
        embedding = self.embedding_service.embed_text(content)

        # Update in ChromaDB
        self.collection.update(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[new_metadata],
        )

        return {
            "memory_id": memory_id,
            "status": "updated",
        }

    def delete(self, memory_id: str) -> Dict[str, Any]:
        """
        Delete a memory.

        Args:
            memory_id: The memory ID to delete.

        Returns:
            Dict with status.
        """
        existing = self.get(memory_id)
        if not existing:
            raise ValueError(f"Memory not found: {memory_id}")

        self.collection.delete(ids=[memory_id])

        return {
            "memory_id": memory_id,
            "status": "deleted",
        }

    def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Delete all memories, optionally filtered.

        Args:
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            run_id: Filter by run ID.

        Returns:
            Dict with count of deleted memories.
        """
        where_clause = self._build_where_clause(user_id, agent_id, run_id)

        results = self.collection.get(where=where_clause)

        if not results["ids"]:
            return {"deleted": 0}

        self.collection.delete(ids=results["ids"])

        return {"deleted": len(results["ids"])}

    def count(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> int:
        """
        Count memories, optionally filtered.

        Args:
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            run_id: Filter by run ID.

        Returns:
            Number of memories.
        """
        where_clause = self._build_where_clause(user_id, agent_id, run_id)

        if where_clause:
            results = self.collection.get(where=where_clause)
            return len(results["ids"]) if results["ids"] else 0
        else:
            return self.collection.count()

    def add_image(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        dedupe: bool = True,
    ) -> Dict[str, Any]:
        """
        Add an image to memory by generating a caption and embedding it.

        Args:
            image_path: Path to the image file.
            image_bytes: Raw image bytes (alternative to image_path).
            user_id: Optional user identifier for filtering.
            agent_id: Optional agent identifier for filtering.
            run_id: Optional run/session identifier.
            metadata: Optional additional metadata.
            filename: Optional filename for the stored image.
            dedupe: Whether to check for duplicate images. Default True.

        Returns:
            Dict with memory_id, image_path, and status. If duplicate found,
            returns existing memory ID with status "duplicate".
        """
        if not image_path and not image_bytes:
            raise ValueError("Must provide either image_path or image_bytes")

        # Get image bytes for hashing
        if image_path:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
        else:
            img_bytes = image_bytes

        # Check for duplicate image by hash
        if dedupe:
            image_hash = hashlib.sha256(img_bytes).hexdigest()[:32]
            try:
                hash_results = self.collection.get(
                    where={"image_hash": image_hash},
                    include=["documents", "metadatas"],
                )
                if hash_results["ids"]:
                    return {
                        "memory_id": hash_results["ids"][0],
                        "status": "duplicate",
                        "match_type": "exact_image",
                    }
            except Exception:
                pass  # image_hash field might not exist

        # Generate caption using vision service
        caption = self.vision_service.caption_image(
            image_path=image_path,
            image_bytes=image_bytes,
        )

        # Store image file
        memory_id = self._generate_id(caption)
        if image_path:
            ext = Path(image_path).suffix
            stored_filename = filename or f"{memory_id}{ext}"
            stored_path = self.image_dir / stored_filename
            shutil.copy2(image_path, stored_path)
        else:
            ext = ".jpg"  # Default for bytes
            stored_filename = filename or f"{memory_id}{ext}"
            stored_path = self.image_dir / stored_filename
            with open(stored_path, "wb") as f:
                f.write(image_bytes)

        # Build metadata
        image_hash = hashlib.sha256(img_bytes).hexdigest()[:32]
        mem_metadata: Dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(),
            "content_type": "image",
            "content_hash": self._compute_content_hash(caption),
            "image_hash": image_hash,
            "image_path": str(stored_path),
            "image_filename": stored_filename,
        }
        if user_id:
            mem_metadata["user_id"] = user_id
        if agent_id:
            mem_metadata["agent_id"] = agent_id
        if run_id:
            mem_metadata["run_id"] = run_id
        if metadata:
            mem_metadata.update(metadata)

        # Generate embedding from caption
        embedding = self.embedding_service.embed_text(caption)

        # Store in ChromaDB
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[caption],  # Store caption as the document
            metadatas=[mem_metadata],
        )

        return {
            "memory_id": memory_id,
            "image_path": str(stored_path),
            "caption": caption,
            "status": "added",
        }

    def search_image(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images by uploading a query image.

        Args:
            image_path: Path to the query image file.
            image_bytes: Raw query image bytes (alternative to image_path).
            user_id: Filter by user ID.
            agent_id: Filter by agent ID.
            run_id: Filter by run ID.
            limit: Maximum number of results.
            filters: Additional ChromaDB filters.

        Returns:
            List of matching memories with scores.
        """
        if not image_path and not image_bytes:
            raise ValueError("Must provide either image_path or image_bytes")

        # Generate caption for query image
        query_caption = self.vision_service.caption_image(
            image_path=image_path,
            image_bytes=image_bytes,
        )

        # Search using the caption
        return self.search(
            query=query_caption,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit,
            filters=filters,
        )

    def add_pdf(
        self,
        file_path: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        dedupe: bool = True,
    ) -> Dict[str, Any]:
        """
        Add a PDF document to memory by extracting and chunking its text.

        Args:
            file_path: Path to the PDF file.
            file_bytes: Raw PDF bytes (alternative to file_path).
            user_id: Optional user identifier for filtering.
            agent_id: Optional agent identifier for filtering.
            run_id: Optional run/session identifier.
            metadata: Optional additional metadata.
            filename: Optional filename for the stored PDF.
            dedupe: Whether to check for duplicate PDFs. Default True.

        Returns:
            Dict with memory_ids, file_path, page_count, and status. If duplicate found,
            returns existing group ID with status "duplicate".
        """
        if not file_path and not file_bytes:
            raise ValueError("Must provide either file_path or file_bytes")

        # Get file bytes for hashing
        if file_path:
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
        else:
            pdf_bytes = file_bytes

        # Check for duplicate PDF by file hash
        if dedupe:
            file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:32]
            try:
                hash_results = self.collection.get(
                    where={"file_hash": file_hash},
                    include=["metadatas"],
                )
                if hash_results["ids"]:
                    existing_meta = hash_results["metadatas"][0] if hash_results["metadatas"] else {}
                    return {
                        "memory_id": hash_results["ids"][0],
                        "group_id": existing_meta.get("group_id"),
                        "status": "duplicate",
                        "match_type": "exact_file",
                    }
            except Exception:
                pass  # file_hash field might not exist

        # Parse PDF
        parsed = self.pdf_parser.parse(file_path=file_path, file_bytes=file_bytes)

        if not parsed.content.strip():
            raise ValueError("PDF contains no extractable text")

        # Generate group ID for linking chunks
        group_id = self._generate_id(parsed.content[:100])

        # Store PDF file
        if file_path:
            stored_filename = filename or Path(file_path).name
        else:
            stored_filename = filename or f"{group_id}.pdf"

        stored_path = self.files_dir / stored_filename
        if file_path:
            shutil.copy2(file_path, stored_path)
        else:
            with open(stored_path, "wb") as f:
                f.write(file_bytes)

        # Build base metadata
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:32]
        base_metadata: Dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(),
            "content_type": "pdf",
            "file_hash": file_hash,
            "file_path": str(stored_path),
            "file_filename": stored_filename,
            "page_count": parsed.page_count,
            "group_id": group_id,
        }
        if user_id:
            base_metadata["user_id"] = user_id
        if agent_id:
            base_metadata["agent_id"] = agent_id
        if run_id:
            base_metadata["run_id"] = run_id
        if parsed.metadata:
            base_metadata["pdf_metadata"] = str(parsed.metadata)
        if metadata:
            base_metadata.update(metadata)

        # Chunk the content using document chunker
        chunks = chunk_content(parsed.content, base_metadata, content_type="document")

        # If no chunks from chunker, create one chunk per page
        if not chunks:
            chunks = []
            for i, page_text in enumerate(parsed.pages):
                if page_text.strip():
                    chunks.append({
                        "content": page_text,
                        "metadata": {
                            **base_metadata,
                            "page_number": i + 1,
                        },
                    })

        if not chunks:
            raise ValueError("PDF produced no chunks")

        # Store each chunk
        memory_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_id(chunk["content"])

            # Add chunk-specific metadata
            chunk_metadata = chunk["metadata"].copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)

            # Generate embedding
            embedding = self.embedding_service.embed_text(chunk["content"])

            # Store in ChromaDB
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk["content"]],
                metadatas=[chunk_metadata],
            )
            memory_ids.append(chunk_id)

        return {
            "memory_ids": memory_ids,
            "group_id": group_id,
            "file_path": str(stored_path),
            "page_count": parsed.page_count,
            "chunks_created": len(memory_ids),
            "status": "added",
        }

    def add_excel(
        self,
        file_path: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        dedupe: bool = True,
        auto_merge: bool = True,
    ) -> Dict[str, Any]:
        """Add an Excel/CSV file to memory.

        Each sheet becomes a table memory. If auto_merge=True and a list
        with the same category exists, appends new rows (deduped by key_field).

        Args:
            file_path: Path to the Excel/CSV file.
            file_bytes: Raw file bytes (alternative to file_path).
            user_id: Optional user identifier for filtering.
            agent_id: Optional agent identifier for filtering.
            run_id: Optional run/session identifier.
            metadata: Optional additional metadata.
            filename: Optional filename for the stored file.
            dedupe: Whether to check for duplicate files by hash. Default True.
            auto_merge: Whether to merge with existing lists of same category. Default True.

        Returns:
            Dict with memory_ids, sheet results, and status. If duplicate found,
            returns existing info with status "duplicate".
        """
        if not file_path and not file_bytes:
            raise ValueError("Must provide either file_path or file_bytes")

        # Get file bytes for hashing
        if file_path:
            with open(file_path, "rb") as f:
                excel_bytes = f.read()
        else:
            excel_bytes = file_bytes

        # Check for duplicate file by hash
        if dedupe:
            file_hash = hashlib.sha256(excel_bytes).hexdigest()[:32]
            try:
                hash_results = self.collection.get(
                    where={"excel_file_hash": file_hash},
                    include=["metadatas"],
                )
                if hash_results["ids"]:
                    existing_meta = hash_results["metadatas"][0] if hash_results["metadatas"] else {}
                    return {
                        "memory_id": hash_results["ids"][0],
                        "status": "duplicate",
                        "match_type": "exact_file",
                        "file_hash": file_hash,
                    }
            except Exception:
                pass  # excel_file_hash field might not exist

        # Parse Excel file
        parsed = self.excel_parser.parse(file_path=file_path, file_bytes=file_bytes)

        if not parsed.sheets:
            raise ValueError("Excel file contains no data")

        # Store Excel file
        if file_path:
            stored_filename = filename or Path(file_path).name
        else:
            ext = ".xlsx"
            stored_filename = filename or f"{parsed.file_hash}{ext}"

        stored_path = self.files_dir / stored_filename
        if file_path:
            shutil.copy2(file_path, stored_path)
        else:
            with open(stored_path, "wb") as f:
                f.write(file_bytes)

        # Process each sheet
        sheet_results = []
        memory_ids = []
        total_rows = 0
        merged_count = 0
        new_count = 0

        for sheet in parsed.sheets:
            # Classify the sheet content
            try:
                classification = self.list_classifier.classify(sheet.markdown)
                existing_categories = self.get_list_categories()
                category = self.list_classifier.normalize_category(
                    classification.category,
                    existing_categories,
                )
            except Exception:
                category = "uncategorized"
                classification = ListClassification(
                    category=category,
                    schema="",
                    key_field="",
                )

            # Check for existing list with same category to merge into
            merged = False
            if auto_merge and category != "uncategorized":
                existing_lists = self.get_lists(category=category, user_id=user_id, limit=1)
                if existing_lists:
                    # Merge into existing list
                    merge_result = self._merge_into_existing_list(
                        existing_id=existing_lists[0]["id"],
                        new_content=sheet.markdown,
                        key_field=classification.key_field,
                        new_rows=sheet.rows,
                        excel_metadata={
                            "excel_sheet_name": sheet.name,
                            "excel_file_hash": parsed.file_hash,
                            "excel_file_path": str(stored_path),
                        },
                    )
                    sheet_results.append({
                        "sheet_name": sheet.name,
                        "category": category,
                        "row_count": sheet.row_count,
                        "status": "merged",
                        "memory_id": existing_lists[0]["id"],
                        "rows_added": merge_result["rows_added"],
                        "rows_skipped": merge_result["rows_skipped"],
                    })
                    memory_ids.append(existing_lists[0]["id"])
                    merged = True
                    merged_count += 1

            if not merged:
                # Create new list memory
                base_metadata: Dict[str, Any] = {
                    "created_at": datetime.utcnow().isoformat(),
                    "content_hash": self._compute_content_hash(sheet.markdown),
                    "content_type": "table",
                    "list_format": "excel",
                    "list_category": category,
                    "list_schema": classification.schema,
                    "list_key_field": classification.key_field,
                    "excel_sheet_name": sheet.name,
                    "excel_file_hash": parsed.file_hash,
                    "excel_file_path": str(stored_path),
                    "excel_row_count": sheet.row_count,
                }
                if user_id:
                    base_metadata["user_id"] = user_id
                if agent_id:
                    base_metadata["agent_id"] = agent_id
                if run_id:
                    base_metadata["run_id"] = run_id
                if metadata:
                    base_metadata.update(metadata)

                mem_id = self._generate_id(sheet.markdown)
                embedding = self.embedding_service.embed_text(sheet.markdown)

                self.collection.add(
                    ids=[mem_id],
                    embeddings=[embedding],
                    documents=[sheet.markdown],
                    metadatas=[base_metadata],
                )

                sheet_results.append({
                    "sheet_name": sheet.name,
                    "category": category,
                    "row_count": sheet.row_count,
                    "status": "added",
                    "memory_id": mem_id,
                })
                memory_ids.append(mem_id)
                new_count += 1

            total_rows += sheet.row_count

        return {
            "memory_ids": memory_ids,
            "file_path": str(stored_path),
            "sheet_count": parsed.sheet_count,
            "total_rows": total_rows,
            "sheets": sheet_results,
            "merged_count": merged_count,
            "new_count": new_count,
            "status": "added",
        }

    def _merge_into_existing_list(
        self,
        existing_id: str,
        new_content: str,
        key_field: str,
        new_rows: list[list[str]],
        excel_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge new table rows into existing list memory.

        Args:
            existing_id: ID of the existing list memory.
            new_content: New markdown table content.
            key_field: Field to use for deduplication.
            new_rows: List of new rows to add.
            excel_metadata: Excel-specific metadata to add.

        Returns:
            Dict with rows_added and rows_skipped counts.
        """
        import re

        existing = self.get(existing_id)
        if not existing:
            raise ValueError(f"Memory not found: {existing_id}")

        existing_content = existing["content"]
        existing_meta = existing["metadata"]

        # Parse existing table
        existing_lines = existing_content.strip().split("\n")
        existing_header: Optional[list[str]] = None
        existing_rows: list[list[str]] = []
        key_index = 0

        for line in existing_lines:
            if not line.strip() or re.match(r"^\|?[\s\-:|]+\|?$", line.strip()):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if existing_header is None:
                existing_header = cells
                # Find key field index
                for idx, h in enumerate(existing_header):
                    if h.lower().replace(" ", "_") == key_field.lower().replace(" ", "_"):
                        key_index = idx
                        break
                continue
            existing_rows.append(cells)

        if not existing_header:
            return {"rows_added": 0, "rows_skipped": len(new_rows)}

        # Get existing key values
        existing_keys = set()
        for row in existing_rows:
            if key_index < len(row):
                existing_keys.add(row[key_index].lower().strip())

        # Add new rows that don't exist
        rows_added = 0
        rows_skipped = 0
        for row in new_rows:
            if key_index < len(row):
                key_value = row[key_index].lower().strip()
                if key_value in existing_keys:
                    rows_skipped += 1
                    continue
                existing_keys.add(key_value)
            existing_rows.append(row)
            rows_added += 1

        # Rebuild merged table
        lines = []
        lines.append("| " + " | ".join(existing_header) + " |")
        lines.append("| " + " | ".join(["---"] * len(existing_header)) + " |")
        for row in existing_rows:
            # Pad row if needed
            padded = row + [""] * (len(existing_header) - len(row))
            escaped = [cell.replace("|", "\\|") for cell in padded[:len(existing_header)]]
            lines.append("| " + " | ".join(escaped) + " |")

        merged_content = "\n".join(lines)

        # Update metadata
        new_meta = existing_meta.copy()
        new_meta["updated_at"] = datetime.utcnow().isoformat()
        new_meta["content_hash"] = self._compute_content_hash(merged_content)
        new_meta["excel_row_count"] = len(existing_rows)
        # Track source files
        source_hashes = new_meta.get("excel_source_hashes", "")
        if excel_metadata.get("excel_file_hash"):
            if source_hashes:
                source_hashes += "," + excel_metadata["excel_file_hash"]
            else:
                source_hashes = excel_metadata["excel_file_hash"]
            new_meta["excel_source_hashes"] = source_hashes

        # Generate new embedding
        embedding = self.embedding_service.embed_text(merged_content)

        # Update in ChromaDB
        self.collection.update(
            ids=[existing_id],
            embeddings=[embedding],
            documents=[merged_content],
            metadatas=[new_meta],
        )

        return {"rows_added": rows_added, "rows_skipped": rows_skipped}

    def feedback(
        self,
        query: str,
        memory_id: str,
        signal: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record feedback on a search result.

        Args:
            query: The search query that produced this result.
            memory_id: The memory ID being rated.
            signal: Feedback signal - "positive" or "negative".
            user_id: Optional user identifier.

        Returns:
            Dict with feedback_id and status.
        """
        if signal not in ("positive", "negative"):
            raise ValueError("Signal must be 'positive' or 'negative'")

        feedback_record = Feedback(
            query=query,
            memory_id=memory_id,
            signal=signal,
            user_id=user_id,
        )

        feedback_id = self.feedback_store.add(feedback_record)

        return {
            "feedback_id": feedback_id,
            "status": "recorded",
        }

    def get_feedback_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get feedback statistics.

        Args:
            user_id: Optional user ID to filter by.

        Returns:
            Dict with feedback statistics.
        """
        return self.feedback_store.get_stats(user_id=user_id)

    def get_related_memories(self, group_id: str) -> List[Dict[str, Any]]:
        """
        Get all memories sharing a group_id.

        Args:
            group_id: The group ID linking related memories.

        Returns:
            List of memory dicts with id, content, and metadata.
        """
        if not group_id:
            return []

        try:
            results = self.collection.get(
                where={"group_id": group_id},
                include=["documents", "metadatas"],
            )
        except Exception:
            return []

        memories = []
        if results["ids"]:
            for i, memory_id in enumerate(results["ids"]):
                memories.append({
                    "id": memory_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        return memories

    def _add_list(
        self,
        content: str,
        list_info: Dict[str, Any],
        base_metadata: Dict[str, Any],
        memory_id: Optional[str] = None,
        dedupe: bool = True,
        dedupe_threshold: float = 0.98,
    ) -> Dict[str, Any]:
        """Store a list/table as a single memory with classification.

        Lists and tables are stored without chunking to preserve structure.
        They are classified by category for later retrieval and merging.

        Args:
            content: The list/table text.
            list_info: Detection result with content_type and list_format.
            base_metadata: Base metadata dict.
            memory_id: Optional custom memory ID.
            dedupe: Whether to check for duplicates.
            dedupe_threshold: Similarity threshold for dedup.

        Returns:
            Dict with memory_id, list_category, and status.
        """
        # Classify the list
        try:
            classification = self.list_classifier.classify(content)

            # Normalize category against existing ones
            existing_categories = self.get_list_categories()
            category = self.list_classifier.normalize_category(
                classification.category,
                existing_categories,
            )
        except Exception:
            # Fallback if classification fails
            category = "uncategorized"
            classification = ListClassification(
                category=category,
                schema="",
                key_field="",
            )

        # Build list-specific metadata
        list_metadata = {
            **base_metadata,
            **list_info,
            "list_category": category,
            "list_schema": classification.schema,
            "list_key_field": classification.key_field,
        }

        # Generate ID
        mem_id = memory_id or self._generate_id(content)

        # Store embedding and content
        embedding = self.embedding_service.embed_text(content)

        self.collection.add(
            ids=[mem_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[list_metadata],
        )

        return {
            "memory_id": mem_id,
            "status": "added",
            "content_type": list_info["content_type"],
            "list_category": category,
            "list_schema": classification.schema,
        }

    def get_list_categories(self) -> List[str]:
        """Get all existing list categories.

        Returns:
            List of unique category names.
        """
        try:
            results = self.collection.get(
                where={"content_type": {"$in": ["list", "table"]}},
                include=["metadatas"],
            )
        except Exception:
            return []

        categories = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                if meta and "list_category" in meta:
                    categories.add(meta["list_category"])

        return list(categories)

    def get_lists(
        self,
        category: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get all lists/tables, optionally filtered by category.

        Args:
            category: Optional category to filter by.
            user_id: Optional user ID to filter by.
            limit: Maximum number of results.

        Returns:
            List of list/table memories with content and metadata.
        """
        where_conditions = []
        where_conditions.append({"content_type": {"$in": ["list", "table"]}})

        if category:
            where_conditions.append({"list_category": category})
        if user_id:
            where_conditions.append({"user_id": user_id})

        if len(where_conditions) == 1:
            where_clause = where_conditions[0]
        else:
            where_clause = {"$and": where_conditions}

        try:
            results = self.collection.get(
                where=where_clause,
                include=["documents", "metadatas"],
                limit=limit,
            )
        except Exception:
            return []

        lists = []
        if results["ids"]:
            for i, mem_id in enumerate(results["ids"]):
                lists.append({
                    "id": mem_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })

        return lists

    def merge_lists(
        self,
        category: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Merge all lists of a category into a combined result.

        For tables: Combines rows, deduplicates by key_field.
        For bullet/numbered lists: Combines items, deduplicates by content.

        Args:
            category: The list category to merge.
            user_id: Optional user ID to filter by.

        Returns:
            Dict with merged_content, source_ids, and item_count.
        """
        lists = self.get_lists(category=category, user_id=user_id)

        if not lists:
            return {
                "merged_content": "",
                "source_ids": [],
                "item_count": 0,
            }

        source_ids = [lst["id"] for lst in lists]

        # Determine format from first list
        first_format = lists[0]["metadata"].get("list_format", "bullet")
        key_field = lists[0]["metadata"].get("list_key_field", "")

        if first_format == "markdown_table":
            merged = self._merge_tables(lists, key_field)
        else:
            merged = self._merge_bullet_lists(lists)

        return {
            "merged_content": merged["content"],
            "source_ids": source_ids,
            "item_count": merged["count"],
        }

    def _merge_tables(
        self,
        lists: List[Dict[str, Any]],
        key_field: str,
    ) -> Dict[str, Any]:
        """Merge multiple markdown tables into one.

        Args:
            lists: List of table memories.
            key_field: Field to use for deduplication.

        Returns:
            Dict with content and count.
        """
        import re

        all_rows: List[List[str]] = []
        header: Optional[List[str]] = None
        seen_keys: set = set()
        key_index = 0

        for lst in lists:
            content = lst["content"]
            lines = content.strip().split("\n")

            for i, line in enumerate(lines):
                if not line.strip() or re.match(r"^\|?[\s\-:|]+\|?$", line.strip()):
                    continue

                cells = [c.strip() for c in line.strip().strip("|").split("|")]

                if header is None:
                    header = cells
                    # Find key field index
                    for idx, h in enumerate(header):
                        if h.lower().replace(" ", "_") == key_field.lower().replace(" ", "_"):
                            key_index = idx
                            break
                    continue

                # Skip if this is another header row (same as first)
                if cells == header:
                    continue

                # Dedupe by key field
                if key_index < len(cells):
                    key_value = cells[key_index]
                    if key_value in seen_keys:
                        continue
                    seen_keys.add(key_value)

                all_rows.append(cells)

        if not header:
            return {"content": "", "count": 0}

        # Build merged table
        lines = []
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in all_rows:
            # Pad row if needed
            while len(row) < len(header):
                row.append("")
            lines.append("| " + " | ".join(row[:len(header)]) + " |")

        return {"content": "\n".join(lines), "count": len(all_rows)}

    def _merge_bullet_lists(
        self,
        lists: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge multiple bullet/numbered lists into one.

        Args:
            lists: List of list memories.

        Returns:
            Dict with content and count.
        """
        import re

        items: List[str] = []
        seen: set = set()

        for lst in lists:
            content = lst["content"]
            lines = content.strip().split("\n")

            for line in lines:
                # Match bullet or numbered items
                match = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$", line)
                if match:
                    item_text = match.group(1).strip()
                    # Normalize for dedup
                    normalized = item_text.lower().strip()
                    if normalized not in seen:
                        seen.add(normalized)
                        items.append(item_text)

        # Rebuild as bullet list
        merged = "\n".join(f"- {item}" for item in items)
        return {"content": merged, "count": len(items)}

    def _upsert_extracted_item(
        self,
        content: str,
        category: str,
        group_id: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        dedupe: bool = True,
        dedupe_threshold: float = 0.98,
    ) -> Dict[str, Any]:
        """Store a single extracted item from chat summarization.

        Args:
            content: The extracted item text.
            category: Extraction category (facts, decisions, etc.).
            group_id: Shared group ID linking to raw chat chunks.
            user_id: Optional user identifier.
            agent_id: Optional agent identifier.
            run_id: Optional run identifier.
            dedupe: Whether to check for duplicates.
            dedupe_threshold: Similarity threshold for dedup.

        Returns:
            Dict with memory_id, category, content, and status.
        """
        if dedupe:
            duplicates = self.find_duplicates(content, threshold=dedupe_threshold, limit=1)
            if duplicates:
                return {
                    "memory_id": duplicates[0]["id"],
                    "category": category,
                    "content": content,
                    "status": "duplicate",
                }

        item_metadata: Dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(),
            "content_hash": self._compute_content_hash(content),
            "content_type": "chat_extract",
            "extract_category": category,
            "source_content_type": "chat",
            "group_id": group_id,
        }
        if user_id:
            item_metadata["user_id"] = user_id
        if agent_id:
            item_metadata["agent_id"] = agent_id
        if run_id:
            item_metadata["run_id"] = run_id

        memory_id = self._generate_id(content)
        embedding = self.embedding_service.embed_text(content)

        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[item_metadata],
        )

        return {
            "memory_id": memory_id,
            "category": category,
            "content": content,
            "status": "added",
        }

    def _generate_id(self, content: str) -> str:
        """Generate a unique memory ID."""
        unique_str = f"{content}{datetime.utcnow().isoformat()}{uuid.uuid4()}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]

    def _compute_content_hash(self, content: str) -> str:
        """Compute a deterministic hash of content for duplicate detection."""
        # Normalize content: strip whitespace, lowercase
        normalized = content.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def find_duplicates(
        self,
        content: str,
        threshold: float = 0.95,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find potential duplicate memories for the given content.

        Uses a hybrid approach:
        1. Exact match via content hash
        2. Semantic similarity via embeddings

        Args:
            content: The content to check for duplicates.
            threshold: Similarity threshold (0-1). Default 0.95.
            limit: Maximum number of duplicates to return.

        Returns:
            List of potential duplicate memories with similarity scores.
        """
        if not content or not content.strip():
            return []

        duplicates = []
        content_hash = self._compute_content_hash(content)

        # 1. Check for exact hash match
        try:
            hash_results = self.collection.get(
                where={"content_hash": content_hash},
                include=["documents", "metadatas"],
            )
            if hash_results["ids"]:
                for i, memory_id in enumerate(hash_results["ids"]):
                    duplicates.append({
                        "id": memory_id,
                        "content": hash_results["documents"][i] if hash_results["documents"] else "",
                        "metadata": hash_results["metadatas"][i] if hash_results["metadatas"] else {},
                        "similarity": 1.0,
                        "match_type": "exact",
                    })
                # Return early if exact match found
                return duplicates[:limit]
        except Exception:
            # content_hash field might not exist in older memories
            pass

        # 2. Check for semantic similarity
        query_embedding = self.embedding_service.embed_text(content)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 / (1 + distance)

                if similarity >= threshold:
                    duplicates.append({
                        "id": memory_id,
                        "content": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "similarity": similarity,
                        "match_type": "semantic",
                    })

        return duplicates[:limit]

    def _build_where_clause(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause from filters."""
        conditions = []

        if user_id:
            conditions.append({"user_id": user_id})
        if agent_id:
            conditions.append({"agent_id": agent_id})
        if run_id:
            conditions.append({"run_id": run_id})
        if extra_filters:
            for key, value in extra_filters.items():
                conditions.append({key: value})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}


# Convenience function for quick initialization
def memory(
    api_key: Optional[str] = None,
    collection_name: str = "memories",
    persist_dir: str = "./chroma_data",
) -> Memory:
    """
    Create a Memory client instance.

    Usage:
        from app.client import memory

        m = memory()
        m.add("User likes coffee", user_id="alice")
        results = m.search("beverage preferences", user_id="alice")
    """
    return Memory(
        api_key=api_key,
        collection_name=collection_name,
        persist_dir=persist_dir,
    )
