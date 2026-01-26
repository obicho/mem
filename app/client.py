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

from app.core.chunker import chunk_content, detect_content_type, ContentType
from app.core.embeddings import EmbeddingService
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
            vision_model: OpenAI vision model for image captioning.
        """
        self.embedding_service = EmbeddingService(api_key=api_key, model=embedding_model)
        self.vision_service = VisionService(api_key=api_key, model=vision_model)
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

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
    ) -> Dict[str, Any]:
        """
        Add a memory with auto-detection chunking.

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

        Returns:
            Dict with memory_id(s) and status.
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        # Build base metadata
        base_metadata: Dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(),
        }
        if user_id:
            base_metadata["user_id"] = user_id
        if agent_id:
            base_metadata["agent_id"] = agent_id
        if run_id:
            base_metadata["run_id"] = run_id
        if metadata:
            base_metadata.update(metadata)

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

        if len(memory_ids) == 1:
            return {
                "memory_id": memory_ids[0],
                "status": "added",
            }
        else:
            return {
                "memory_ids": memory_ids,
                "group_id": group_id,
                "chunks_created": len(memory_ids),
                "status": "added",
            }

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

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        memories = []
        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 / (1 + distance)

                memories.append({
                    "id": memory_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "score": score,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return memories

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

        Returns:
            Dict with memory_id, image_path, and status.
        """
        if not image_path and not image_bytes:
            raise ValueError("Must provide either image_path or image_bytes")

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
        mem_metadata: Dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(),
            "content_type": "image",
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

    def _generate_id(self, content: str) -> str:
        """Generate a unique memory ID."""
        unique_str = f"{content}{datetime.utcnow().isoformat()}{uuid.uuid4()}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]

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
