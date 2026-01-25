"""Memory client - mem0-style interface for AI agent memory."""

import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.embeddings import EmbeddingService


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
    ):
        """
        Initialize the Memory client.

        Args:
            api_key: OpenAI API key for embeddings. If not provided, uses OPENAI_API_KEY env var.
            collection_name: Name of the ChromaDB collection.
            persist_dir: Directory for persistent storage.
            embedding_model: OpenAI embedding model to use.
        """
        self.embedding_service = EmbeddingService(api_key=api_key, model=embedding_model)

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
    ) -> Dict[str, Any]:
        """
        Add a memory.

        Args:
            content: The text content to remember.
            user_id: Optional user identifier for filtering.
            agent_id: Optional agent identifier for filtering.
            run_id: Optional run/session identifier.
            metadata: Optional additional metadata.
            memory_id: Optional custom memory ID. Auto-generated if not provided.

        Returns:
            Dict with memory_id and status.
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        # Generate memory ID
        if not memory_id:
            memory_id = self._generate_id(content)

        # Build metadata
        mem_metadata: Dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(),
        }
        if user_id:
            mem_metadata["user_id"] = user_id
        if agent_id:
            mem_metadata["agent_id"] = agent_id
        if run_id:
            mem_metadata["run_id"] = run_id
        if metadata:
            mem_metadata.update(metadata)

        # Generate embedding
        embedding = self.embedding_service.embed_text(content)

        # Store in ChromaDB
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[mem_metadata],
        )

        return {
            "memory_id": memory_id,
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
