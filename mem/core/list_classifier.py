"""List classifier service for categorizing lists and tables."""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


CLASSIFY_PROMPT = """\
Analyze this list or table and extract:
1. category: A short, lowercase, snake_case name for what this list contains.
   Examples: "suppliers", "action_items", "contacts", "meeting_attendees",
   "product_features", "pricing", "team_members"
2. schema: The fields/columns present as comma-separated names.
   Examples: "name, email, company" or "task, assignee, deadline"
3. key_field: Which field uniquely identifies items (for deduping when merging).
   Pick the most unique field, often "name", "email", "id", or "task".

Return ONLY valid JSON: {"category": "...", "schema": "...", "key_field": "..."}
"""

NORMALIZE_PROMPT = """\
Given a new list category and existing categories in the system, determine if
the new category matches any existing one (even if worded differently).

Existing categories: {existing}
New category: {new}

If the new category is semantically the same as an existing one, return the
existing category name. If it's truly different, return the new category.

Return ONLY the category name, nothing else.
"""


@dataclass
class ListClassification:
    """Result of list classification."""
    category: str
    schema: str
    key_field: str


def detect_list_or_table(content: str) -> Optional[dict]:
    """Detect if content is a list or markdown table.

    Args:
        content: The text content to analyze.

    Returns:
        Dict with content_type and list_format if detected, None otherwise.
    """
    lines = content.strip().split("\n")
    non_empty = [line for line in lines if line.strip()]

    if len(non_empty) < 2:
        return None

    # 1. Markdown table: lines with | separators
    # Must have header row, separator row (with ---), and data rows
    table_lines = [l for l in non_empty if "|" in l]
    separator_lines = [l for l in table_lines if re.match(r"^\|?[\s\-:|]+\|?$", l.strip())]

    if len(table_lines) >= 3 and len(separator_lines) >= 1:
        # Check it's actually a table structure, not just text with |
        pipe_counts = [l.count("|") for l in table_lines]
        if min(pipe_counts) >= 2 and max(pipe_counts) - min(pipe_counts) <= 2:
            return {"content_type": "table", "list_format": "markdown_table"}

    # 2. Bullet list: lines starting with -, *, •
    bullet_pattern = r"^\s*[-*•]\s+.+"
    bullet_lines = [l for l in non_empty if re.match(bullet_pattern, l)]
    if len(bullet_lines) >= 3 and len(bullet_lines) / len(non_empty) >= 0.6:
        return {"content_type": "list", "list_format": "bullet"}

    # 3. Numbered list: lines starting with 1., 2., 1), 2), etc.
    numbered_pattern = r"^\s*\d+[.)]\s+.+"
    numbered_lines = [l for l in non_empty if re.match(numbered_pattern, l)]
    if len(numbered_lines) >= 3 and len(numbered_lines) / len(non_empty) >= 0.6:
        return {"content_type": "list", "list_format": "numbered"}

    return None


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = text.strip()
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


class ListClassifier:
    """Service for classifying lists and tables into categories."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        """Initialize the list classifier.

        Args:
            api_key: OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
            model: LLM model to use for classification.
        """
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def classify(self, content: str) -> ListClassification:
        """Classify a list or table content.

        Args:
            content: The list/table text to classify.

        Returns:
            ListClassification with category, schema, and key_field.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": content},
            ],
        )

        raw_output = response.choices[0].message.content or "{}"
        cleaned = _strip_markdown_fences(raw_output)
        data = json.loads(cleaned)

        return ListClassification(
            category=data.get("category", "unknown"),
            schema=data.get("schema", ""),
            key_field=data.get("key_field", ""),
        )

    def normalize_category(
        self,
        new_category: str,
        existing_categories: list[str],
    ) -> str:
        """Normalize a category name against existing categories.

        Args:
            new_category: The new category to normalize.
            existing_categories: List of existing category names.

        Returns:
            The canonical category name (existing match or new).
        """
        if not existing_categories:
            return new_category

        # Check for exact match first
        if new_category in existing_categories:
            return new_category

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": NORMALIZE_PROMPT.format(
                        existing=", ".join(existing_categories),
                        new=new_category,
                    ),
                },
                {"role": "user", "content": "Return the canonical category name."},
            ],
        )

        result = response.choices[0].message.content or new_category
        return result.strip().lower().replace(" ", "_")
