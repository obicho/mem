"""Feedback storage for improving search quality."""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional


@dataclass
class Feedback:
    """A single feedback record."""

    query: str
    memory_id: str
    signal: Literal["positive", "negative"]
    user_id: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class FeedbackStore:
    """Store and retrieve search feedback using SQLite."""

    def __init__(self, db_path: str = "./feedback.db"):
        """Initialize the feedback store.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    signal TEXT NOT NULL CHECK (signal IN ('positive', 'negative')),
                    user_id TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_memory_id
                ON feedback(memory_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_query
                ON feedback(query)
            """)
            conn.commit()

    def add(self, feedback: Feedback) -> int:
        """Add a feedback record.

        Args:
            feedback: The feedback to store.

        Returns:
            The ID of the inserted record.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback (query, memory_id, signal, user_id, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    feedback.query,
                    feedback.memory_id,
                    feedback.signal,
                    feedback.user_id,
                    feedback.timestamp,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_feedback_for_memory(self, memory_id: str) -> dict:
        """Get aggregated feedback for a specific memory.

        Args:
            memory_id: The memory ID to get feedback for.

        Returns:
            Dict with positive and negative counts.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN signal = 'positive' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN signal = 'negative' THEN 1 ELSE 0 END) as negative
                FROM feedback
                WHERE memory_id = ?
                """,
                (memory_id,),
            )
            row = cursor.fetchone()
            return {
                "positive": row[0] or 0,
                "negative": row[1] or 0,
            }

    def get_feedback_for_query(self, query: str, limit: int = 100) -> list[dict]:
        """Get all feedback for a specific query.

        Args:
            query: The search query.
            limit: Maximum number of records to return.

        Returns:
            List of feedback records.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT memory_id, signal, user_id, timestamp
                FROM feedback
                WHERE query = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (query, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_memory_scores(self, memory_ids: list[str]) -> dict[str, float]:
        """Get feedback-based scores for multiple memories.

        Args:
            memory_ids: List of memory IDs.

        Returns:
            Dict mapping memory_id to feedback score (-1 to 1).
        """
        if not memory_ids:
            return {}

        placeholders = ",".join("?" * len(memory_ids))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                SELECT
                    memory_id,
                    SUM(CASE WHEN signal = 'positive' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN signal = 'negative' THEN 1 ELSE 0 END) as negative
                FROM feedback
                WHERE memory_id IN ({placeholders})
                GROUP BY memory_id
                """,
                memory_ids,
            )

            scores = {}
            for row in cursor.fetchall():
                memory_id, positive, negative = row
                total = positive + negative
                if total > 0:
                    # Score from -1 (all negative) to 1 (all positive)
                    scores[memory_id] = (positive - negative) / total
                else:
                    scores[memory_id] = 0.0

            # Set 0 for memories with no feedback
            for mid in memory_ids:
                if mid not in scores:
                    scores[mid] = 0.0

            return scores

    def get_stats(self, user_id: Optional[str] = None) -> dict:
        """Get overall feedback statistics.

        Args:
            user_id: Optional user ID to filter by.

        Returns:
            Dict with feedback statistics.
        """
        with sqlite3.connect(self.db_path) as conn:
            if user_id:
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN signal = 'positive' THEN 1 ELSE 0 END) as positive,
                        SUM(CASE WHEN signal = 'negative' THEN 1 ELSE 0 END) as negative,
                        COUNT(DISTINCT query) as unique_queries,
                        COUNT(DISTINCT memory_id) as unique_memories
                    FROM feedback
                    WHERE user_id = ?
                    """,
                    (user_id,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN signal = 'positive' THEN 1 ELSE 0 END) as positive,
                        SUM(CASE WHEN signal = 'negative' THEN 1 ELSE 0 END) as negative,
                        COUNT(DISTINCT query) as unique_queries,
                        COUNT(DISTINCT memory_id) as unique_memories
                    FROM feedback
                    """
                )

            row = cursor.fetchone()
            return {
                "total_feedback": row[0] or 0,
                "positive": row[1] or 0,
                "negative": row[2] or 0,
                "unique_queries": row[3] or 0,
                "unique_memories": row[4] or 0,
            }
