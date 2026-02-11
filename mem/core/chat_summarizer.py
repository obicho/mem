"""Chat summarizer service for extracting structured items from chat conversations."""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI


EXTRACT_CATEGORIES = ("facts", "decisions", "opinions", "tasks", "goals", "preferences")

SYSTEM_PROMPT = """\
You are a precise information extractor. Given a chat conversation, do two things:

1. Write a short "outcome" summarizing the conclusion or result of the \
conversation in 1-2 sentences. If the conversation has no clear outcome, \
write a brief summary of what was discussed. The outcome must be self-contained \
and understandable without reading the original chat.

2. Extract standalone, self-contained sentences into these categories:

- facts: Objective statements or pieces of information mentioned.
- decisions: Choices or resolutions that were made.
- opinions: Subjective viewpoints or assessments expressed.
- tasks: Action items, to-dos, or assignments.
- goals: Objectives, targets, or aspirations discussed.
- preferences: Likes, dislikes, or preferred ways of doing things.

Rules:
1. Each extracted item MUST be a complete, self-contained sentence that makes \
sense without the original conversation context.
2. Include the subject (who) in each sentence when known.
3. Do NOT duplicate items across categories.
4. If a category has no items, return an empty list for it.
5. Return ONLY valid JSON with these exact keys: outcome, facts, decisions, \
opinions, tasks, goals, preferences. "outcome" is a string; each category \
value is a list of strings.
"""


@dataclass
class ChatExtractedItem:
    """A single extracted item from a chat conversation."""

    category: str
    content: str


@dataclass
class ChatSummaryResult:
    """Structured extraction result from a chat conversation."""

    outcome: str = ""
    facts: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    opinions: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)

    def all_items(self) -> list[ChatExtractedItem]:
        """Flatten all categories into a single list of extracted items."""
        items: list[ChatExtractedItem] = []
        for category in EXTRACT_CATEGORIES:
            for content in getattr(self, category):
                items.append(ChatExtractedItem(category=category, content=content))
        return items


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = text.strip()
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


class ChatSummarizer:
    """Service for extracting structured items from chat conversations using an LLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        """Initialize the chat summarizer.

        Args:
            api_key: OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
            model: LLM model to use for extraction.
        """
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def extract(self, chat_text: str) -> ChatSummaryResult:
        """Extract structured items from a chat conversation.

        Args:
            chat_text: The raw chat conversation text.

        Returns:
            ChatSummaryResult with categorized extracted items.

        Raises:
            ValueError: If chat_text is empty.
        """
        if not chat_text or not chat_text.strip():
            raise ValueError("Chat text cannot be empty")

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chat_text},
            ],
        )

        raw_output = response.choices[0].message.content or "{}"
        cleaned = _strip_markdown_fences(raw_output)
        data = json.loads(cleaned)

        return ChatSummaryResult(
            outcome=data.get("outcome", ""),
            facts=data.get("facts", []),
            decisions=data.get("decisions", []),
            opinions=data.get("opinions", []),
            tasks=data.get("tasks", []),
            goals=data.get("goals", []),
            preferences=data.get("preferences", []),
        )
