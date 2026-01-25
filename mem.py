"""
Mem - Persistent Memory Layer for AI Agents

A mem0-style interface for storing and retrieving memories with semantic search.

Usage:
    from mem import Memory

    # Initialize
    m = Memory()

    # Add memories
    m.add("User prefers dark mode", user_id="alice")
    m.add("Meeting scheduled for Friday at 2pm", user_id="alice", metadata={"type": "calendar"})

    # Search memories
    results = m.search("user interface preferences", user_id="alice")
    for r in results:
        print(f"{r['content']} (score: {r['score']:.2f})")

    # Get all memories for a user
    all_memories = m.get_all(user_id="alice")

    # Update a memory
    m.update(memory_id="...", content="User prefers light mode now")

    # Delete a memory
    m.delete(memory_id="...")

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key for embeddings
"""

from app.client import Memory, memory

__all__ = ["Memory", "memory"]
__version__ = "0.1.0"
